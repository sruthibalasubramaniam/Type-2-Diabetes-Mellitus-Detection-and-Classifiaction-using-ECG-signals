"""Run the complete wavelet, feature, and model comparison.

The evaluation is grouped by recording, so beats from one ECG recording never
appear in both a training and a test fold.  Results and readable PNG tables are
written to ``full_comparison_output`` by default.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from PyEMD import EEMD, EMD
from scipy import stats
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from vmdpy import VMD


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diabetes_sym4_emd_local import (EPS, MAX_IMFS, WINDOW_AFTER,
                                     WINDOW_BEFORE, read_signal, safe_zscore)

RANDOM_STATE = 42
MODEL_NAMES = ("SVM_RBF", "Random_Forest", "KNN", "Decision_Tree")
FEATURE_METHODS = ("ica", "emd", "eemd", "vmd")
WAVELETS = ("sym4", "db4")


def denoise(signal: np.ndarray, wavelet_name: str) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 2:
        return x.copy()
    wavelet = pywt.Wavelet(wavelet_name)
    level = pywt.dwt_max_level(len(x), wavelet.dec_len)
    if level <= 0:
        return x.copy()
    coeffs = pywt.wavedec(x, wavelet_name, level=level)
    for index in range(1, len(coeffs)):
        scale = np.max(np.abs(coeffs[index]))
        if scale > EPS:
            coeffs[index] = pywt.threshold(coeffs[index], 0.04 * scale, mode="soft")
    return pywt.waverec(coeffs, wavelet_name)[:len(x)]


def process_folder(folder: Path, label: int, wavelet: str) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.csv")):
        try:
            signal = safe_zscore(denoise(read_signal(path, "auto"), wavelet))
            peaks, _ = find_peaks(signal, height=0.5, distance=150)
            kept = 0
            for peak in peaks:
                start, end = int(peak - WINDOW_BEFORE), int(peak + WINDOW_AFTER)
                if start < 0 or end > len(signal):
                    continue
                beat = signal[start:end].astype(np.float32)
                rows.append({"record_id": f"{label}_{path.name}",
                             "window_signal": beat,
                             "standard_deviation": float(np.std(beat)),
                             "diabetes": label})
                kept += 1
            print(f"  {wavelet} {path.name}: {len(peaks)} peaks, {kept} windows", flush=True)
        except (ValueError, pd.errors.ParserError) as error:
            print(f"  Skipping {path.name}: {error}", flush=True)
    return pd.DataFrame(rows)


def entropy(x: np.ndarray) -> float:
    power = x * x
    total = power.sum()
    if total <= EPS or len(x) <= 1:
        return 0.0
    probabilities = power / total
    return float(-(probabilities * np.log2(probabilities + EPS)).sum() / np.log2(len(x)))


def component_features(components: np.ndarray, prefix: str, signal: np.ndarray, start_index: int = 1) -> dict[str, float]:
    signal_energy = float(np.sum(signal ** 2)) + EPS
    result = {}
    for i in range(components.shape[0]):
        component = np.asarray(components[i], dtype=np.float64)
        std = float(np.std(component))
        skew = float(stats.skew(component, bias=False)) if std >= EPS else 0.0
        kurtosis = float(stats.kurtosis(component, fisher=True, bias=False)) if std >= EPS else 0.0
        number = i + start_index
        result.update({f"{prefix}{number}_energy": float(np.mean(component ** 2)),
                       f"{prefix}{number}_relative_energy": float(np.sum(component ** 2) / signal_energy),
                       f"{prefix}{number}_rms": float(np.sqrt(np.mean(component ** 2))),
                       f"{prefix}{number}_std": std,
                       f"{prefix}{number}_mean_abs": float(np.mean(np.abs(component))),
                       f"{prefix}{number}_entropy": entropy(component),
                       f"{prefix}{number}_skewness": skew if np.isfinite(skew) else 0.0,
                       f"{prefix}{number}_kurtosis": kurtosis if np.isfinite(kurtosis) else 0.0,
                       f"{prefix}{number}_zcr": float(np.mean(np.signbit(component[1:]) != np.signbit(component[:-1])))})
    return result


def emd_features(signal: np.ndarray, ensemble: bool = False) -> dict[str, float]:
    try:
        decomposition = EEMD(trials=5, parallel=False) if ensemble else EMD()
        imfs = decomposition.eemd(signal, max_imf=MAX_IMFS) if ensemble else decomposition.emd(signal, max_imf=MAX_IMFS)
        residue = signal - np.sum(imfs, axis=0) if len(imfs) else signal
    except Exception:
        imfs, residue = np.empty((0, len(signal))), signal
    result = component_features(np.asarray(imfs)[:MAX_IMFS], "eemd_imf" if ensemble else "emd_imf", signal)
    # Keep the feature shape fixed if a decomposition returns fewer IMFs.
    missing = MAX_IMFS - min(len(imfs), MAX_IMFS)
    if missing:
        result.update(component_features(np.zeros((missing, len(signal))),
                                         "eemd_imf" if ensemble else "emd_imf", signal,
                                         start_index=min(len(imfs), MAX_IMFS) + 1))
    result["residue_energy"] = float(np.mean(np.asarray(residue) ** 2))
    result["residue_std"] = float(np.std(residue))
    return result


def vmd_features(signal: np.ndarray) -> dict[str, float]:
    try:
        modes, _, _ = VMD(signal, 2000, 0, 3, 0, 1, 1e-7)
    except Exception:
        modes = np.zeros((3, len(signal)))
    return component_features(np.asarray(modes)[:3], "vmd_mode", signal)


def metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
            "precision_diabetes": precision_score(y_true, prediction, zero_division=0),
            "sensitivity_recall": recall_score(y_true, prediction, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1_diabetes": f1_score(y_true, prediction, zero_division=0)}


def models(train_size: int) -> dict:
    return {
        "SVM_RBF": Pipeline([("scale", StandardScaler()),
                             ("model", SVC(class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random_Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=RANDOM_STATE),
        "KNN": Pipeline([("scale", StandardScaler()),
                          ("model", KNeighborsClassifier(n_neighbors=min(5, train_size)))]),
        "Decision_Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    }


def evaluate(df: pd.DataFrame, method: str, feature_cache: dict) -> list[dict]:
    raw = np.vstack(df.window_signal.to_numpy()).astype(float)
    std = df[["standard_deviation"]].to_numpy(float)
    y = df.diabetes.to_numpy(int)
    groups = df.record_id.to_numpy(str)
    if method == "ica":
        X_base = raw
    else:
        if method not in feature_cache:
            print(f"  Extracting {method.upper()} features for {len(raw)} beats...", flush=True)
            if method == "emd":
                rows = Parallel(n_jobs=-1, prefer="processes")(delayed(emd_features)(signal, False) for signal in raw)
            elif method == "eemd":
                rows = Parallel(n_jobs=-1, prefer="processes")(delayed(emd_features)(signal, True) for signal in raw)
            else:
                rows = Parallel(n_jobs=-1, prefer="processes")(delayed(vmd_features)(signal) for signal in raw)
            feature_cache[method] = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        X_base = feature_cache[method]

    splits = min(5, int(df.groupby("diabetes").record_id.nunique().min()))
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    predictions = {name: np.zeros(len(df), dtype=int) for name in MODEL_NAMES}
    for fold, (train, test) in enumerate(splitter.split(X_base, y, groups), 1):
        if method == "ica":
            components = min(48, len(train), raw.shape[1])
            ica = FastICA(n_components=components, random_state=RANDOM_STATE, max_iter=1000)
            X_train = np.hstack([ica.fit_transform(raw[train]), std[train]])
            X_test = np.hstack([ica.transform(raw[test]), std[test]])
        else:
            X_train, X_test = X_base[train], X_base[test]
        for name, model in models(len(train)).items():
            model.fit(X_train, y[train])
            predictions[name][test] = model.predict(X_test)
        print(f"  {method.upper()} fold {fold}/{splits} complete", flush=True)

    rows = []
    for name in MODEL_NAMES:
        rows.append({"model": name, **metric_row(y, predictions[name])})
    return rows


def formatted_table(data: pd.DataFrame, title: str, path: Path, widths=None) -> None:
    display = data.copy()
    for column in display.columns:
        if column not in {"model", "Denoising", "Feature extraction", "Preprocessing", "Best model"}:
            if pd.api.types.is_numeric_dtype(display[column]):
                display[column] = display[column].map(lambda value: f"{value:.2%}")
    figure, axis = plt.subplots(figsize=(max(13, len(display.columns) * 1.55), max(3.0, len(display) * 0.48 + 1.7)))
    axis.axis("off")
    table = axis.table(cellText=display.values, colLabels=display.columns, cellLoc="center", loc="center",
                       colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b7c3d0")
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f8")
    axis.set_title(title, fontsize=15, weight="bold", pad=18)
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diabetes-dir", type=Path, default=ROOT / "data" / "diabetes")
    parser.add_argument("--normal-dir", type=Path, default=ROOT / "data" / "normal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "full_comparison_output")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for wavelet in WAVELETS:
        print(f"\n=== {wavelet.upper()} denoising ===", flush=True)
        df = pd.concat([process_folder(args.diabetes_dir, 1, wavelet),
                        process_folder(args.normal_dir, 0, wavelet)], ignore_index=True)
        if df.empty or df.diabetes.nunique() != 2:
            raise RuntimeError(f"No usable windows for {wavelet}")
        print(f"  {len(df)} beat windows from {df.record_id.nunique()} recordings", flush=True)
        feature_cache = {}
        for method in FEATURE_METHODS:
            for result in evaluate(df, method, feature_cache):
                all_results.append({"Denoising": wavelet.upper(),
                                    "Feature extraction": method.upper(),
                                    "Preprocessing": "Z-score + R-peaks + 210-sample windows",
                                    **result})

    results = pd.DataFrame(all_results)
    metric_columns = ["accuracy", "balanced_accuracy", "precision_diabetes", "sensitivity_recall", "specificity", "f1_diabetes"]
    results.to_csv(args.output_dir / "all_32_results.csv", index=False)
    best_by_combo = results.sort_values("balanced_accuracy", ascending=False).groupby(
        ["Denoising", "Feature extraction"], as_index=False, sort=False).first()
    best_by_combo.to_csv(args.output_dir / "eight_combination_summary.csv", index=False)

    summary_columns = ["Denoising", "Feature extraction", "Preprocessing", "model"] + metric_columns
    formatted_table(best_by_combo[summary_columns].rename(columns={"model": "Best model"}),
                    "Eight Denoising + Feature-Extraction Combinations", args.output_dir / "eight_combination_summary.png")

    for (wavelet, method), group in results.groupby(["Denoising", "Feature extraction"], sort=False):
        table_columns = ["model"] + metric_columns
        formatted_table(group[table_columns], f"{wavelet} + {method}: Model Comparison",
                        args.output_dir / f"comparison_{wavelet.lower()}_{method.lower()}.png")

    best = results.sort_values(["balanced_accuracy", "f1_diabetes", "accuracy"], ascending=False).iloc[0]
    pd.DataFrame([best]).to_csv(args.output_dir / "best_combination.csv", index=False)
    best_display = pd.DataFrame([{
        "Denoising": best["Denoising"], "Feature extraction": best["Feature extraction"],
        "Preprocessing": best["Preprocessing"], "Best model": best["model"],
        **{column: best[column] for column in metric_columns},
    }])
    formatted_table(best_display, "Best Overall Configuration (ranked by balanced accuracy)",
                    args.output_dir / "best_combination.png")
    print("\n=== BEST CONFIGURATION ===", flush=True)
    print(best_display.to_string(index=False), flush=True)
    print(f"\nSaved results and tables to: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
