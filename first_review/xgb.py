import argparse

import warnings

from pathlib import Path



import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import pywt

import seaborn as sns

from joblib import Parallel, delayed

from scipy.signal import find_peaks

from sklearn.decomposition import FastICA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,

                             f1_score, precision_score, recall_score)

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier



from diabetes_sym4_emd_local import (EPS, RANDOM_STATE, WINDOW_AFTER, WINDOW_BEFORE,

                                      WINDOW_LENGTH, emd_features, read_signal, safe_zscore)





def arguments(wavelet: str = "sym4", feature_method: str = "ica") -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=f"Local ECG classifier: {wavelet.upper()} + {feature_method.upper()}")

    parser.add_argument("--diabetes-dir", type=Path, default=Path("./data/diabetes"))

    parser.add_argument("--normal-dir", type=Path, default=Path("./data/normal"))

    parser.add_argument("--signal-column", default="auto", help="CSV column name or zero-based index, e.g. 1")

    parser.add_argument("--output-dir", type=Path, default=Path(f"{wavelet}_{feature_method}_output"))

    parser.add_argument("--n-jobs", type=int, default=-1, help="Use 1 if memory is limited")

    parser.add_argument("--peak-height", type=float, default=0.5)

    parser.add_argument("--peak-distance", type=int, default=150)

    parser.add_argument("--ica-components", type=int, default=48, help="Maximum number of ICA components")

   

    args, _ = parser.parse_known_args()

    return args





def denoise(data: np.ndarray, wavelet_name: str = "sym4", threshold_factor: float = 0.04) -> np.ndarray:

    x = np.asarray(data, dtype=np.float64)

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

            coeffs[index] = pywt.threshold(coeffs[index], threshold_factor * scale, mode="soft")

    return pywt.waverec(coeffs, wavelet_name)[:len(x)]





def process_folder(folder: Path, label: int, signal_column: str, wavelet: str = "sym4",

                   peak_height: float = 0.5, peak_distance: int = 150) -> pd.DataFrame:

    if not folder.is_dir():

        raise FileNotFoundError(f"Folder not found: {folder}")

    rows = []

    for path in sorted(folder.glob("*.csv")):

        try:

            signal = safe_zscore(denoise(read_signal(path, signal_column), wavelet))

            peaks, _ = find_peaks(signal, height=peak_height, distance=peak_distance)

            kept = 0

            for peak in peaks:

                start, end = int(peak - WINDOW_BEFORE), int(peak + WINDOW_AFTER)

                if start < 0 or end > len(signal):

                    continue

                beat = signal[start:end]

                rows.append({"record_id": f"{label}_{path.name}", "file": path.name,

                             "r_peak": int(peak), "window_signal": beat.astype(np.float32),

                             "standard_deviation": float(np.std(beat)), "diabetes": label})

                kept += 1

            print(f"{path.name}: {len(peaks)} peaks, {kept} complete beat windows")

        except (ValueError, pd.errors.ParserError) as error:

            print(f"Skipping {path.name}: {error}")

    return pd.DataFrame(rows)





def metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:

    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()

    return {"accuracy": accuracy_score(y_true, prediction),

            "balanced_accuracy": balanced_accuracy_score(y_true, prediction),

            "precision_diabetes": precision_score(y_true, prediction, zero_division=0),

            "sensitivity_recall": recall_score(y_true, prediction, zero_division=0),

            "specificity": tn / (tn + fp) if tn + fp else 0.0,

            "f1_diabetes": f1_score(y_true, prediction, zero_division=0)}





def run_variant(wavelet: str = "sym4", feature_method: str = "ica") -> None:

    args = arguments(wavelet, feature_method)

    warnings.filterwarnings("ignore")

    args.output_dir.mkdir(parents=True, exist_ok=True)

   

    diabetic = process_folder(args.diabetes_dir, 1, args.signal_column, wavelet, args.peak_height, args.peak_distance)

    normal = process_folder(args.normal_dir, 0, args.signal_column, wavelet, args.peak_height, args.peak_distance)

    df = pd.concat([diabetic, normal], ignore_index=True)

   

    if df.empty or df.diabetes.nunique() != 2:

        raise RuntimeError("Both folders need usable ECG beat windows.")

       

    groups_per_class = df.groupby("diabetes").record_id.nunique()

    n_splits = min(5, int(groups_per_class.min()))

    if n_splits < 2:

        raise RuntimeError("At least two usable CSV recordings are needed per class.")

       

    y, groups = df.diabetes.to_numpy(int), df.record_id.to_numpy(str)

   

    # Pre-extract raw signals/features depending on method

    raw_signals = np.vstack(df.window_signal.to_numpy()).astype(np.float64)

    std_feats = df[["standard_deviation"]].to_numpy(float)



    if feature_method == "emd":

        print("Extracting EMD features...")

        emd_df = pd.DataFrame(Parallel(n_jobs=args.n_jobs, backend="loky")(

            delayed(emd_features)(signal) for signal in df.window_signal))

        emd_df = emd_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        X_base = np.hstack([emd_df.to_numpy(float), std_feats])

    elif feature_method == "ica":

        X_base = raw_signals  # ICA transform will be fitted strictly within each CV fold

    else:

        raise ValueError(f"Unsupported feature method: {feature_method}")



    # Full 5-Fold Stratified Group Cross-Validation

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)



    model_names = ["KNN", "SVM_RBF", "Decision_Tree", "Random_Forest", "XGBoost"]

    cv_predictions = {name: np.zeros(len(df)) for name in model_names}



    print(f"\nRunning Leak-Free {n_splits}-Fold Cross-Validation ({feature_method.upper()})...")



    for fold, (train, test) in enumerate(sgkf.split(X_base, y, groups)):

        print(f"--- Fold {fold + 1}/{n_splits} ---")



        if feature_method == "ica":

            # Leak-Free ICA: Fit FastICA ONLY on train split

            components = min(args.ica_components, train.shape[0], raw_signals.shape[1])

            ica = FastICA(n_components=components, random_state=RANDOM_STATE, max_iter=1000)

           

            X_train_ica = ica.fit_transform(raw_signals[train])

            X_test_ica = ica.transform(raw_signals[test])



            X_tr = np.hstack([X_train_ica, std_feats[train]])

            X_te = np.hstack([X_test_ica, std_feats[test]])

        else:

            X_tr, X_te = X_base[train], X_base[test]



        y_tr, y_te = y[train], y[test]



        # Model Definitions per fold

        models = {

            "KNN": Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=min(5, len(train))))]),

            "SVM_RBF": Pipeline([("scale", StandardScaler()), ("model", SVC(class_weight="balanced", random_state=RANDOM_STATE))]),

            "Decision_Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),

            "Random_Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),

            "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)

        }



        for name, model in models.items():

            model.fit(X_tr, y_tr)

            cv_predictions[name][test] = model.predict(X_te)



    # Evaluate Accumative Overall Cross-Validation Results

    results = []

    evaluations = []

   

    print("\n==========================================")

    print("   OVERALL LEAK-FREE CROSS-VALIDATION RESULTS")

    print("==========================================")



    for name in model_names:

        preds = cv_predictions[name]

        result = {"model": name, **metric_row(y, preds)}

        results.append(result)

        evaluations.append((result, preds))



        # Save Individual Confusion Matrix Plots

        plt.figure(figsize=(5, 4))

        sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues",

                    xticklabels=["Normal", "Diabetes"], yticklabels=["Normal", "Diabetes"])

        plt.title(f"{wavelet.upper()} + {feature_method.upper()}: {name}")

        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()

        plt.savefig(args.output_dir / f"confusion_matrix_{name}.png", dpi=150); plt.close()



    result_table = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)

    result_table.to_csv(args.output_dir / "model_results.csv", index=False)

   

    best_result, best_prediction = max(evaluations, key=lambda item: item[0]["balanced_accuracy"])

    pd.DataFrame([best_result]).to_csv(args.output_dir / "best_model_summary.csv", index=False)



    # Plot Best Model Confusion Matrix

    plt.figure(figsize=(6, 5))

    sns.heatmap(confusion_matrix(y, best_prediction), annot=True, fmt="d", cmap="Blues",

                xticklabels=["Normal", "Diabetes"], yticklabels=["Normal", "Diabetes"])

    plt.title(f"Best Model: {wavelet.upper()} + {feature_method.upper()} — {best_result['model']}")

    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.tight_layout()

    plt.savefig(args.output_dir / "best_confusion_matrix.png", dpi=180); plt.close()



    print(result_table.to_string(index=False))

    print(f"\n{wavelet.upper()} + {feature_method.upper()} complete. Results saved to: {args.output_dir.resolve()}")

    print(f"Best model: {best_result['model']} (balanced accuracy: {best_result['balanced_accuracy']:.2%})")





if __name__ == "__main__":

    run_variant(wavelet="sym4", feature_method="ica")