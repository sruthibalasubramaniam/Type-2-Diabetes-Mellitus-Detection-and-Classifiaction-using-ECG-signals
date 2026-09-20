"""Train the Sym4 + EMD ECG classifier using CSV files stored on this computer.

Example:
    python diabetes_sym4_emd_local.py --diabetes-dir data/diabetes --normal-dir data/normal --signal-column 1

Each directory must contain CSV recordings for one class.  A file is treated as
one ECG record; the signal column may be given by its zero-based column index
(best for headerless CSVs) or by its heading (best for headed CSVs).
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots instead of requiring a notebook/display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from PyEMD import EMD
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

WINDOW_BEFORE = 90
WINDOW_AFTER = 120
WINDOW_LENGTH = WINDOW_BEFORE + WINDOW_AFTER
MAX_IMFS = 5
EPS = 1e-12
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Sym4 + EMD ECG classifier")
    parser.add_argument("--diabetes-dir", type=Path, required=True,
                        help="Folder containing diabetes ECG CSV recordings")
    parser.add_argument("--normal-dir", type=Path, required=True,
                        help="Folder containing normal/control ECG CSV recordings")
    parser.add_argument("--signal-column", default="auto",
                        help="CSV signal column name or zero-based index (e.g. 1). Default: auto")
    parser.add_argument("--output-dir", type=Path, default=Path("sym4_emd_output"),
                        help="Directory for extracted features, results, and plots")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel EMD workers; use 1 if memory is limited")
    parser.add_argument("--peak-height", type=float, default=0.5)
    parser.add_argument("--peak-distance", type=int, default=150,
                        help="Minimum R-peak distance in samples")
    return parser.parse_args()


def denoise_sym4(data: np.ndarray, threshold_factor: float = 0.04) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2:
        return x.copy()
    wavelet = pywt.Wavelet("sym4")
    level = pywt.dwt_max_level(len(x), wavelet.dec_len)
    if level <= 0:
        return x.copy()
    coeffs = pywt.wavedec(x, "sym4", level=level)
    for i in range(1, len(coeffs)):
        scale = np.max(np.abs(coeffs[i]))
        if scale > 0:
            coeffs[i] = pywt.threshold(coeffs[i], threshold_factor * scale, mode="soft")
    return pywt.waverec(coeffs, "sym4")[:len(x)]


def safe_zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    return np.zeros_like(x) if not np.isfinite(std) or std < EPS else (x - np.mean(x)) / std


def read_signal(file_path: Path, signal_column: str) -> np.ndarray:
    """Read headed or headerless CSV data and return the selected numeric column."""
    # Some recordings begin with a date line and then a header, while others
    # start immediately with samples.  Locate the first actual sample row so a
    # date/header cannot make pandas interpret the entire file as one column.
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    delimiters = [",", "\t", ";", "|"]
    delimiter = max(delimiters, key=lambda candidate: max((line.count(candidate) for line in lines), default=0))
    data_start = None
    for line_number, line in enumerate(lines):
        fields = [item.strip() for item in line.split(delimiter)]
        numeric_count = sum(pd.notna(pd.to_numeric(item, errors="coerce")) for item in fields)
        if len(fields) >= 2 and numeric_count >= 2:
            data_start = line_number
            break
    if data_start is None:
        raise ValueError("could not find a numeric sample row")

    raw = pd.read_csv(file_path, header=None, sep=re.escape(delimiter), skiprows=data_start,
                      engine="python", on_bad_lines="skip")
    if raw.empty:
        raise ValueError("file is empty")

    if signal_column != "auto" and signal_column.lstrip("-").isdigit():
        index = int(signal_column)
        if not -len(raw.columns) <= index < len(raw.columns):
            raise ValueError(f"column index {index} is outside 0..{len(raw.columns) - 1}")
        series = pd.to_numeric(raw.iloc[:, index], errors="coerce")
    else:
        if signal_column != "auto":
            # Use the last pre-data row as a possible heading row.  The local
            # files label ECG here even though some data rows have one extra PPG
            # column, so resolve the heading to its position instead of relying
            # on pandas' strict header-width rules.
            possible_headers = [line.split(delimiter) for line in lines[:data_start]
                                if delimiter in line]
            headers = [item.strip() for item in possible_headers[-1]] if possible_headers else []
            if signal_column not in headers:
                raise ValueError(f"column {signal_column!r} not found; use an index such as --signal-column 1")
            series = pd.to_numeric(raw.iloc[:, headers.index(signal_column)], errors="coerce")
        else:
            numeric = raw.apply(pd.to_numeric, errors="coerce")
            usable = numeric.columns[numeric.notna().sum() >= max(3, len(numeric) * 0.8)]
            if len(usable) == 0:
                raise ValueError("no mostly numeric column found; pass --signal-column")
            # Prefer the numeric column with greatest variation; never choose a time string.
            series = numeric[usable].std().idxmax()
            series = numeric[series]

    series = series.interpolate(limit_direction="both").ffill().bfill().dropna()
    if len(series) < WINDOW_LENGTH:
        raise ValueError(f"only {len(series)} usable samples; at least {WINDOW_LENGTH} are required")
    return series.to_numpy(dtype=np.float64)


def process_folder(folder: Path, label: int, signal_column: str, peak_height: float,
                   peak_distance: int) -> pd.DataFrame:
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in: {folder}")
    rows: list[dict] = []
    for file_path in files:
        try:
            normalized = safe_zscore(denoise_sym4(read_signal(file_path, signal_column)))
            peaks, _ = find_peaks(normalized, height=peak_height, distance=peak_distance)
            kept = 0
            for peak in peaks:
                start, end = int(peak - WINDOW_BEFORE), int(peak + WINDOW_AFTER)
                if start < 0 or end > len(normalized):
                    continue
                beat = normalized[start:end]
                rows.append({"record_id": f"{label}_{file_path.name}", "file": file_path.name,
                             "r_peak": int(peak), "window_signal": beat.astype(np.float32),
                             "standard_deviation": float(np.std(beat)), "diabetes": label})
                kept += 1
            print(f"{file_path.name}: {len(peaks)} peaks, {kept} complete beat windows")
        except (ValueError, pd.errors.ParserError) as error:
            print(f"Skipping {file_path.name}: {error}")
    return pd.DataFrame(rows)


def entropy(x: np.ndarray) -> float:
    power = x * x
    total = power.sum()
    if total <= EPS or len(x) <= 1:
        return 0.0
    p = power / total
    return float(-(p * np.log2(p + EPS)).sum() / np.log2(len(x)))


def emd_features(signal: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    try:
        emd = EMD()
        emd.emd(signal, max_imf=MAX_IMFS)
        imfs, residue = emd.get_imfs_and_residue()
    except Exception:
        imfs, residue = np.empty((0, len(signal))), signal
    signal_energy = float(np.sum(signal ** 2)) + EPS
    result: dict[str, float] = {}
    for i in range(MAX_IMFS):
        prefix = f"emd_imf{i + 1}"
        imf = np.asarray(imfs[i], dtype=np.float64) if i < len(imfs) else np.zeros_like(signal)
        std = np.std(imf)
        skew = float(stats.skew(imf, bias=False)) if std >= EPS else 0.0
        kurt = float(stats.kurtosis(imf, fisher=True, bias=False)) if std >= EPS else 0.0
        result.update({f"{prefix}_energy": float(np.mean(imf ** 2)),
                       f"{prefix}_relative_energy": float(np.sum(imf ** 2) / signal_energy),
                       f"{prefix}_rms": float(np.sqrt(np.mean(imf ** 2))),
                       f"{prefix}_std": float(std), f"{prefix}_mean_abs": float(np.mean(abs(imf))),
                       f"{prefix}_entropy": entropy(imf), f"{prefix}_skewness": skew if np.isfinite(skew) else 0.0,
                       f"{prefix}_kurtosis": kurt if np.isfinite(kurt) else 0.0,
                       f"{prefix}_zcr": float(np.mean(np.signbit(imf[1:]) != np.signbit(imf[:-1])))})
    result["emd_residue_energy"] = float(np.mean(np.asarray(residue) ** 2))
    result["emd_residue_std"] = float(np.std(residue))
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_diabetes": precision_score(y_true, y_pred, zero_division=0),
            "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0, "f1_diabetes": f1_score(y_true, y_pred, zero_division=0)}


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diabetes = process_folder(args.diabetes_dir, 1, args.signal_column, args.peak_height, args.peak_distance)
    normal = process_folder(args.normal_dir, 0, args.signal_column, args.peak_height, args.peak_distance)
    df = pd.concat([diabetes, normal], ignore_index=True)
    if df.empty or df.diabetes.nunique() != 2:
        raise RuntimeError("No usable beat windows from both classes. Check folders and --signal-column.")
    groups_per_class = df.groupby("diabetes").record_id.nunique()
    splits = min(5, int(groups_per_class.min()))
    if splits < 2:
        raise RuntimeError("At least two CSV records with usable peaks are needed for each class.")
    print(f"\nUsable beat windows: {len(df)}; records per class: {groups_per_class.to_dict()}")
    feature_df = pd.DataFrame(Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(emd_features)(x) for x in df.window_signal))
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = np.hstack([feature_df.to_numpy(float), df[["standard_deviation"]].to_numpy(float)])
    names = list(feature_df.columns) + ["standard_deviation"]
    y, groups = df.diabetes.to_numpy(int), df.record_id.to_numpy(str)
    train, test = next(StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE).split(X, y, groups))
    # Avoid an invalid KNN configuration when testing a very small dataset.
    k_neighbors = min(5, len(train))
    models = {"KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=k_neighbors))]),
              "SVM_RBF": Pipeline([("scale", StandardScaler()), ("model", SVC(class_weight="balanced", random_state=RANDOM_STATE))]),
              "Decision_Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
              "Random_Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)}
    results, predictions = [], {}
    for name, model in models.items():
        model.fit(X[train], y[train]); pred = model.predict(X[test]); predictions[name] = pred
        results.append({"model": name, **metrics(y[test], pred)})
        print(f"\n{name}\n{classification_report(y[test], pred, target_names=['Normal', 'Diabetes'], zero_division=0)}")
    pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False).to_csv(args.output_dir / "model_results.csv", index=False)
    export = pd.concat([df.drop(columns="window_signal").reset_index(drop=True), feature_df], axis=1)
    export["split"] = "train"; export.loc[test, "split"] = "test"
    export.to_csv(args.output_dir / "ecg_features_sym4_emd.csv", index=False)
    for name, pred in predictions.items():
        plt.figure(figsize=(5, 4)); sns.heatmap(confusion_matrix(y[test], pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Diabetes"], yticklabels=["Normal", "Diabetes"])
        plt.title(name); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(args.output_dir / f"confusion_matrix_{name}.png", dpi=150); plt.close()
    print(f"\nSaved all outputs to: {args.output_dir.resolve()}")
    print(f"Feature count: {len(names)}")


if __name__ == "__main__":
    main()
