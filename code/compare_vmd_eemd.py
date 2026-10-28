"""Matched subject-level comparison of VMD and EEMD ECG features."""

from pathlib import Path
import os
import re
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
from PyEMD import EEMD
from vmdpy import VMD
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FS = 500
MAX_BEATS = 300
WINDOW = 300


def load_signal(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    delimiters = [",", "\t", ";", "|"]
    delimiter = max(delimiters, key=lambda d: max((line.count(d) for line in lines), default=0))
    start = next((i for i, line in enumerate(lines)
                  if sum(pd.notna(pd.to_numeric(v.strip(), errors="coerce"))
                         for v in line.split(delimiter)) >= 2), None)
    if start is None:
        raise ValueError("no numeric sample row")
    raw = pd.read_csv(path, header=None, sep=re.escape(delimiter), skiprows=start,
                      engine="python", on_bad_lines="skip")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    usable = numeric.columns[numeric.notna().sum() >= max(3, len(numeric) * .8)]
    if len(usable) == 0:
        raise ValueError("no numeric signal column")
    series = numeric[usable].std().idxmax()
    return numeric[series].interpolate(limit_direction="both").dropna().to_numpy(float)


def beat_windows(path: Path) -> list[np.ndarray]:
    signal = load_signal(path)
    if len(signal) < FS * 3:
        return []
    cleaned = nk.ecg_clean(signal, sampling_rate=FS, method="neurokit")
    _, peaks = nk.ecg_peaks(cleaned, sampling_rate=FS)
    indices = peaks.get("ECG_R_Peaks", [])
    windows = []
    for index in indices[1:min(len(indices) - 1, MAX_BEATS + 1)]:
        start, end = int(index - WINDOW), int(index + WINDOW)
        if start >= 0 and end <= len(cleaned):
            windows.append(cleaned[start:end].astype(np.float64))
    return windows


def common_features(signal: np.ndarray) -> list[float]:
    coeffs = pywt.wavedec(signal, "db4", level=3)
    values = [float(signal.mean()), float(signal.std())]
    values.extend(v for c in coeffs for v in (float(c.mean()), float(c.std())))
    return values


def vmd_features(signal: np.ndarray) -> list[float]:
    modes, _, _ = VMD(signal, 2000, 0, 3, 0, 1, 1e-7)
    return [value for mode in modes for value in
            (float(mode.mean()), float(mode.std()), float(np.sum(mode ** 2)))]


def eemd_features(signal: np.ndarray) -> list[float]:
    decomposition = EEMD(trials=5, parallel=False)
    imfs = decomposition.eemd(signal, max_imf=3)
    values = []
    for i in range(3):
        imf = imfs[i] if i < len(imfs) else np.zeros_like(signal)
        values.extend((float(imf.mean()), float(imf.std()), float(np.sum(imf ** 2))))
    return values


def build_dataset(root: Path, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups = [], [], []
    for label, folder in ((1, root / "diabetes"), (0, root / "normal")):
        for path in sorted(folder.glob("*.csv")):
            windows = beat_windows(path)
            print(f"{method.upper()} {path.name}: {len(windows)} beats", flush=True)
            for signal in windows:
                features.append(common_features(signal) +
                                (vmd_features(signal) if method == "vmd" else eemd_features(signal)))
                labels.append(label)
                groups.append(path.name)
    if not features:
        raise RuntimeError("No usable ECG beats found")
    return np.asarray(features), np.asarray(labels), np.asarray(groups)


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str) -> dict[str, float]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    prediction = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(X, y, groups), 1):
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=.05,
                                     subsample=.8, eval_metric="logloss",
                                     random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        model.fit(X[train], y[train])
        prediction[test] = model.predict(X[test])
        print(f"{method.upper()} fold {fold}/3 complete", flush=True)
    result = {
        "method": method.upper(),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision_diabetes": precision_score(y, prediction, zero_division=0),
        "sensitivity_recall": recall_score(y, prediction, zero_division=0),
        "specificity": recall_score(1 - y, 1 - prediction, zero_division=0),
        "f1_diabetes": f1_score(y, prediction, zero_division=0),
    }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    output = Path(__file__).resolve().parent / "vmd_eemd_comparison.csv"
    rows = []
    for method in ("vmd", "eemd"):
        X, y, groups = build_dataset(root, method)
        print(f"{method.upper()}: {len(y)} beats, {len(np.unique(groups))} recordings, {X.shape[1]} features", flush=True)
        rows.append(evaluate(X, y, groups, method))
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
