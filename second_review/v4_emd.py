import os
import glob
import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from PyEMD import EMD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. ROBUST FILE LOADER & PREPROCESSING
# ==========================================

def load_ecg_data(file_path):
    """
    Safely parses text, CSV, or NumPy signal files. Isolates the highest-variance 
    numeric column (the AC signal channel) and discards headers.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        return np.load(file_path).squeeze()

    try:
        df_raw = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception:
        df_raw = pd.read_csv(file_path, sep=r'\s+', on_bad_lines='skip', engine='python')

    df_numeric = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).dropna(how='all', axis=0)

    if df_numeric.empty:
        raise ValueError(f"No valid numeric data found in {file_path}")

    if df_numeric.shape[1] > 1:
        signal_col = df_numeric.var().idxmax()
        data = df_numeric[signal_col].dropna().values
    else:
        data = df_numeric.iloc[:, 0].dropna().values

    return np.asarray(data, dtype=np.float64).squeeze()


def preprocess_ecg_signal(raw_signal, sampling_rate=500):
    """
    Applies baseline wander removal and bandpass filtering via NeuroKit2.
    """
    cleaned_signal = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate, method="neurokit")
    return cleaned_signal

# ==========================================
# 2. EMD & WAVELET FEATURE EXTRACTION
# ==========================================

def compute_emd_features(signal, num_imfs=3):
    """
    Extracts Empirical Mode Decomposition (EMD) Intrinsic Mode Functions (IMFs).
    Computes mean, standard deviation, and energy for the first 3 IMFs.
    """
    emd_feats = {}
    try:
        emd = EMD()
        imfs = emd.emd(signal)
        
        for i in range(min(num_imfs, imfs.shape[0])):
            imf = imfs[i]
            emd_feats[f'emd_mean_imf{i}'] = np.mean(imf)
            emd_feats[f'emd_std_imf{i}'] = np.std(imf)
            emd_feats[f'emd_energy_imf{i}'] = np.sum(imf ** 2)
            
        # Pad remaining IMFs if decomposition returned fewer than requested
        for i in range(imfs.shape[0], num_imfs):
            emd_feats[f'emd_mean_imf{i}'] = 0.0
            emd_feats[f'emd_std_imf{i}'] = 0.0
            emd_feats[f'emd_energy_imf{i}'] = 0.0
    except Exception:
        for i in range(num_imfs):
            emd_feats[f'emd_mean_imf{i}'] = 0.0
            emd_feats[f'emd_std_imf{i}'] = 0.0
            emd_feats[f'emd_energy_imf{i}'] = 0.0
            
    return emd_feats


def extract_features_emd(file_path, sampling_rate=500, max_beats_per_file=300):
    """
    Extracts Wavelet (db4) + EMD + HRV features per beat window.
    """
    try:
        raw_signal = load_ecg_data(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if raw_signal is None or len(raw_signal) < sampling_rate * 3:
        return None

    try:
        cleaned_signal = preprocess_ecg_signal(raw_signal, sampling_rate=sampling_rate)
        signals, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)
    except Exception:
        return None

    rpeak_indices = rpeaks.get("ECG_R_Peaks", [])
    if len(rpeak_indices) < 3:
        return None

    window_features = []
    half_win = int(0.3 * sampling_rate)
    total_beats_to_process = min(len(rpeak_indices) - 1, max_beats_per_file + 1)

    for i in range(1, total_beats_to_process):
        idx = rpeak_indices[i]
        start = idx - half_win
        end = idx + half_win

        if start < 0 or end > len(cleaned_signal):
            continue

        beat_window = cleaned_signal[start:end]

        # 1. Basic Statistical & HRV Features
        feat_dict = {
            'beat_std': np.std(beat_window),
            'beat_mean': np.mean(beat_window),
            'rr_interval': (rpeak_indices[i] - rpeak_indices[i-1]) / sampling_rate
        }

        # 2. Discrete Wavelet Transform (db4)
        coeffs = pywt.wavedec(beat_window, 'sym4', level=3)
        for level, (m, s) in enumerate(zip([np.mean(c) for c in coeffs], [np.std(c) for c in coeffs])):
            feat_dict[f'wavelet_mean_l{level}'] = m
            feat_dict[f'wavelet_std_l{level}'] = s

        # 3. Standard Empirical Mode Decomposition (EMD)
        emd_feats = compute_emd_features(beat_window, num_imfs=3)
        feat_dict.update(emd_feats)

        window_features.append(feat_dict)

    if not window_features:
        return None

    return pd.DataFrame(window_features)

# ==========================================
# 3. DATASET BUILDING & VISUALIZATION
# ==========================================

def build_dataset(data_dir, sampling_rate=500):
    all_dfs = []
    file_list = []
    for ext in ("*.txt", "*.csv", "*.npy"):
        file_list.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    print(f"Found {len(file_list)} total candidate files across directories.")

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

        df_feat = extract_features_emd(file_path, sampling_rate=sampling_rate, max_beats_per_file=300)
        
        if df_feat is not None:
            df_feat['label'] = label
            df_feat['subject_id'] = os.path.basename(file_path)
            all_dfs.append(df_feat)
            print(f"Loaded ({label}): {os.path.basename(file_path)} [{len(df_feat)} beats]")

    if not all_dfs:
        raise ValueError("No valid features extracted from dataset directory.")

    dataset = pd.concat(all_dfs, ignore_index=True)
    return dataset


def plot_correlation_map(df, top_n=15):
    """
    Plots a feature correlation heatmap focusing on the top N features 
    most correlated with the target label.
    """
    df_corr = df.copy()
    if 'subject_id' in df_corr.columns:
        df_corr = df_corr.drop(columns=['subject_id'])
    
    if df_corr['label'].dtype == 'object':
        df_corr['label'] = df_corr['label'].map({'diabetes': 1, 'normal': 0})

    corr_matrix = df_corr.corr()

    top_corr_features = corr_matrix['label'].abs().sort_values(ascending=False).head(top_n).index
    filtered_corr = corr_matrix.loc[top_corr_features, top_corr_features]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        filtered_corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5
    )
    plt.title(f"Top {top_n} Feature Correlation Heatmap (EMD + Wavelet)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig("feature_correlation_map_emd.png", dpi=300)
    print("\nSaved correlation heatmap to 'feature_correlation_map_emd.png'")
    plt.show()

# ==========================================
# 4. GUARANTEED BOTH-CLASS SPLITTING
# ==========================================

def subject_level_stratified_split(df, test_ratio=0.25):
    """
    Splits subjects strictly at the file level while guaranteeing that both 
    classes are present in both training and testing splits.
    """
    subjects_df = df[['subject_id', 'label']].drop_duplicates()
    
    train_subs, test_subs = train_test_split(
        subjects_df['subject_id'],
        test_size=test_ratio,
        stratify=subjects_df['label'],
        random_state=42
    )

    train_mask = df['subject_id'].isin(train_subs)
    test_mask = df['subject_id'].isin(test_subs)

    return train_mask, test_mask

# ==========================================
# 5. MAIN TRAINING PIPELINE
# ==========================================

if __name__ == "__main__":
    DATA_DIR = "./data"     # Adjust to match your dataset directory
    SAMPLING_RATE = 500

    print("--- Starting Wavelet (`db4`) + EMD Pipeline ---")
    df = build_dataset(DATA_DIR, sampling_rate=SAMPLING_RATE)
    
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Plot Correlation Heatmap
    plot_correlation_map(df, top_n=15)

    X = df.drop(columns=['label', 'subject_id'])
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # Subject-level stratified split
    train_mask, test_mask = subject_level_stratified_split(df, test_ratio=0.25)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    test_groups = df.loc[test_mask, 'subject_id'].values

    print(f"\nExtracted Class Breakdown: {dict(zip(le.classes_, np.bincount(y)))}")
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

    # Beat-Level Evaluation
    y_pred = model.predict(X_test_scaled)

    print("\n--- Final Performance Metrics (EMD + db4) ---")
    print(f"Beat-Level Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Patient / Subject-Level Evaluation (Majority Voting)
    df_test = X_test.copy()
    df_test['y_true'] = y_test
    df_test['y_pred'] = y_pred
    df_test['subject_id'] = test_groups

    subject_eval = df_test.groupby('subject_id')[['y_true', 'y_pred']].mean().round()

    print("\n--- Patient-Level Performance (Majority Vote per File) ---")
    print(f"Subject Accuracy: {accuracy_score(subject_eval['y_true'], subject_eval['y_pred']):.4f}")
    print(classification_report(subject_eval['y_true'], subject_eval['y_pred'], target_names=le.classes_))