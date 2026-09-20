import os
import glob
import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from vmdpy import VMD
from PyEMD import EEMD
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# ==========================================
# 1. ROBUST FILE LOADER & CLEANING
# ==========================================

def load_ecg_data(file_path):
    """
    Safely parses text, CSV, or NumPy signal files. Automatically isolates the 
    highest-variance numeric column (the AC signal channel) and discards text headers.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        return np.load(file_path).squeeze()

    try:
        df_raw = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception:
        df_raw = pd.read_csv(file_path, sep=r'\s+', on_bad_lines='skip', engine='python')

    # Convert all columns to numeric, coercing headers or metadata to NaN
    df_numeric = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).dropna(how='all', axis=0)

    if df_numeric.empty:
        raise ValueError(f"No valid numeric data found in {file_path}")

    # If multi-column, pick column with highest variance (typically the raw signal)
    if df_numeric.shape[1] > 1:
        signal_col = df_numeric.var().idxmax()
        data = df_numeric[signal_col].dropna().values
    else:
        data = df_numeric.iloc[:, 0].dropna().values

    return np.asarray(data, dtype=np.float64).squeeze()

# ==========================================
# 2. SIGNAL DECOMPOSITION FEATURE EXTRACTION
# ==========================================

def compute_vmd_features(signal, K=3, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Extracts Variational Mode Decomposition (VMD) mode metrics.
    """
    vmd_feats = {}
    try:
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        for i in range(K):
            mode = u[i]
            vmd_feats[f'vmd_mean_m{i}'] = np.mean(mode)
            vmd_feats[f'vmd_std_m{i}'] = np.std(mode)
            vmd_feats[f'vmd_energy_m{i}'] = np.sum(mode ** 2)
    except Exception:
        for i in range(K):
            vmd_feats[f'vmd_mean_m{i}'] = 0.0
            vmd_feats[f'vmd_std_m{i}'] = 0.0
            vmd_feats[f'vmd_energy_m{i}'] = 0.0
    return vmd_feats


def compute_eemd_features(signal, num_imfs=3):
    """
    Extracts Ensemble Empirical Mode Decomposition (EEMD) IMF metrics.
    Disables parallel workers to prevent VS Code deadlock.
    """
    eemd_feats = {}
    try:
        eemd = EEMD(trials=5, parallel=False)
        imfs = eemd.eemd(signal)
        for i in range(min(num_imfs, imfs.shape[0])):
            imf = imfs[i]
            eemd_feats[f'eemd_mean_imf{i}'] = np.mean(imf)
            eemd_feats[f'eemd_std_imf{i}'] = np.std(imf)
            eemd_feats[f'eemd_energy_imf{i}'] = np.sum(imf ** 2)
        for i in range(imfs.shape[0], num_imfs):
            eemd_feats[f'eemd_mean_imf{i}'] = 0.0
            eemd_feats[f'eemd_std_imf{i}'] = 0.0
            eemd_feats[f'eemd_energy_imf{i}'] = 0.0
    except Exception:
        for i in range(num_imfs):
            eemd_feats[f'eemd_mean_imf{i}'] = 0.0
            eemd_feats[f'eemd_std_imf{i}'] = 0.0
            eemd_feats[f'eemd_energy_imf{i}'] = 0.0
    return eemd_feats


def extract_features_from_file(file_path, sampling_rate=500, max_beats=300, method='vmd'):
    """
    Denoises signal, detects R-peaks, and extracts features per beat window.
    Method options: 'vmd', 'eemd', or 'wavelet'
    """
    try:
        raw_signal = load_ecg_data(file_path)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

    if len(raw_signal) < sampling_rate * 3:
        return None

    try:
        cleaned = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate, method="neurokit")
        _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
    except Exception:
        return None

    rpeak_indices = rpeaks.get("ECG_R_Peaks", [])
    if len(rpeak_indices) < 3:
        return None

    window_features = []
    half_win = int(0.3 * sampling_rate) # 300ms window before & after R-peak

    for i in range(1, min(len(rpeak_indices) - 1, max_beats + 1)):
        idx = rpeak_indices[i]
        start, end = idx - half_win, idx + half_win
        if start < 0 or end > len(cleaned):
            continue

        beat_window = cleaned[start:end]
        
        # Base Features
        feat_dict = {
            'beat_std': np.std(beat_window),
            'beat_mean': np.mean(beat_window),
            'rr_interval': (rpeak_indices[i] - rpeak_indices[i-1]) / sampling_rate
        }

        # 1. Discrete Wavelet Transform (db4)
        coeffs = pywt.wavedec(beat_window, 'db4', level=3)
        for lvl, (m, s) in enumerate(zip([np.mean(c) for c in coeffs], [np.std(c) for c in coeffs])):
            feat_dict[f'wavelet_mean_l{lvl}'] = m
            feat_dict[f'wavelet_std_l{lvl}'] = s

        # 2. Selected Mode Decomposition Method
        if method == 'vmd':
            feat_dict.update(compute_vmd_features(beat_window, K=3))
        elif method == 'eemd':
            feat_dict.update(compute_eemd_features(beat_window, num_imfs=3))

        window_features.append(feat_dict)

    return pd.DataFrame(window_features) if window_features else None

# ==========================================
# 3. DATASET BUILDING
# ==========================================

def build_dataset(data_dir, sampling_rate=500, max_beats=300, method='vmd'):
    all_dfs = []
    file_list = []
    for ext in ("*.txt", "*.csv", "*.npy"):
        file_list.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    print(f"Found {len(file_list)} total candidate files under '{data_dir}'.")

    for file_path in file_list:
        path_lower = file_path.lower()
        if "flutter_assets" in path_lower:
            continue
            
        if "diabetes" in path_lower:
            label = "diabetes"
        elif "normal" in path_lower:
            label = "normal"
        else:
            continue

        df_feat = extract_features_from_file(file_path, sampling_rate=sampling_rate, max_beats=max_beats, method=method)
        if df_feat is not None:
            df_feat['label'] = label
            df_feat['subject_id'] = os.path.basename(file_path)
            all_dfs.append(df_feat)
            print(f"Loaded ({label}): {os.path.basename(file_path)} [{len(df_feat)} beats]")

    if not all_dfs:
        raise ValueError("No valid features extracted from dataset directory.")

    dataset = pd.concat(all_dfs, ignore_index=True)
    return dataset

# ==========================================
# 4. LEAK-FREE EVALUATION PIPELINE
# ==========================================

if __name__ == "__main__":
    DATA_DIR = "./data"      # Directory containing 'diabetes' and 'normal' subfolders
    SAMPLING_RATE = 500      # Sensor sampling frequency
    DECOMP_METHOD = 'vmd'    # Select feature set: 'vmd', 'eemd', or 'wavelet'

    print(f"--- Starting Pipeline using [{DECOMP_METHOD.upper()}] Features ---")
    df = build_dataset(DATA_DIR, sampling_rate=SAMPLING_RATE, max_beats=300, method=DECOMP_METHOD)
    
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=['label', 'subject_id'])
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    groups = df['subject_id'].values

    print(f"\nExtracted Class Breakdown: {dict(zip(le.classes_, np.bincount(y)))}")

    # 3-Fold Stratified Group Split (Ensures Subject Separation & Balanced Classes)
    sgkf = StratifiedGroupKFold(n_splits=3)
    
    valid_fold_found = False
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        y_tr, y_te = y[train_idx], y[test_idx]
        # Ensure both classes are present in both splits
        if len(np.unique(y_tr)) > 1 and len(np.unique(y_te)) > 1:
            valid_fold_found = True
            break

    if not valid_fold_found:
        raise ValueError("Could not find a valid split containing both classes in training and testing sets.")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train Set Distribution: {dict(zip(le.classes_, np.bincount(y_train)))}")
    print(f"Test Set Distribution:  {dict(zip(le.classes_, np.bincount(y_test)))}\n")

    # Feature Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    print("Training XGBoost Classifier...")
    model.fit(X_train_scaled, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test_scaled)

    print("\n--- Final Performance Metrics ---")
    print(f"Beat-Level Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Patient / Subject-Level Majority Voting
    df_test = X_test.copy()
    df_test['y_true'] = y_test
    df_test['y_pred'] = y_pred
    df_test['subject_id'] = groups[test_idx]

    subject_eval = df_test.groupby('subject_id')[['y_true', 'y_pred']].mean().round()

    print("\n--- Patient-Level Performance (Majority Vote per File) ---")
    print(f"Subject Accuracy: {accuracy_score(subject_eval['y_true'], subject_eval['y_pred']):.4f}")
    print(classification_report(subject_eval['y_true'], subject_eval['y_pred'], target_names=le.classes_))