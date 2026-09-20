import os
import glob
import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# ==========================================
# 1. FIXED SIGNAL PREPROCESSING & LOADING
# ==========================================

def load_ecg_data(file_path):
    """
    Loads CSV/TXT data while keeping column structure intact to isolate 
    the correct signal channel without flattening timestamps into signal data.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        return np.load(file_path).squeeze()

    # Load with fallback delimiter detection
    try:
        df_raw = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    except Exception:
        df_raw = pd.read_csv(file_path, sep=r'\s+', on_bad_lines='skip', engine='python')

    # Convert all values to float, coercing header text to NaN
    df_numeric = df_raw.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.dropna(how='all', axis=1).dropna(how='all', axis=0)

    if df_numeric.empty:
        raise ValueError(f"No numeric data found in {file_path}")

    # If multi-column (e.g., Timestamp, ECG, PPG), pick column with highest variance (the signal)
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


def extract_features_from_file(file_path, sampling_rate=500, max_beats_per_file=300):
    """
    Processes a single ECG file, extracts features, and caps maximum heartbeats
    per subject to prevent class/recording duration imbalance.
    """
    try:
        raw_signal = load_ecg_data(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if raw_signal is None or len(raw_signal) < sampling_rate * 3:
        return None

    # Step 1: Denoise Signal
    try:
        cleaned_signal = preprocess_ecg_signal(raw_signal, sampling_rate=sampling_rate)
    except Exception as e:
        print(f"Cleaning failed for {file_path}: {e}")
        return None

    # Step 2: Peak Detection
    try:
        signals, rpeaks = nk.ecg_peaks(cleaned_signal, sampling_rate=sampling_rate)
    except Exception:
        return None

    rpeak_indices = rpeaks.get("ECG_R_Peaks", [])
    if len(rpeak_indices) < 3:
        return None

    # Step 3: Beat-level Window Extraction (Capped to max_beats_per_file)
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

        # Discrete Wavelet Transform Features (db4)
        coeffs = pywt.wavedec(beat_window, 'db4', level=3)
        wavelet_means = [np.mean(c) for c in coeffs]
        wavelet_stds = [np.std(c) for c in coeffs]

        feat_dict = {
            'beat_std': np.std(beat_window),
            'beat_mean': np.mean(beat_window),
            'rr_interval': (rpeak_indices[i] - rpeak_indices[i-1]) / sampling_rate
        }
        
        for level, (m, s) in enumerate(zip(wavelet_means, wavelet_stds)):
            feat_dict[f'wavelet_mean_l{level}'] = m
            feat_dict[f'wavelet_std_l{level}'] = s

        window_features.append(feat_dict)

    if not window_features:
        return None

    return pd.DataFrame(window_features)

# ==========================================
# 2. DATASET BUILDING (LEAKAGE-FREE)
# ==========================================

def build_dataset(data_dir, sampling_rate=500):
    all_dfs = []
    
    extensions = ("*.txt", "*.csv", "*.npy")
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    print(f"Found {len(file_list)} total candidate files across directories.")

    if not file_list:
        raise FileNotFoundError(
            f"No signal files matching {extensions} found under '{data_dir}'."
        )

    for file_path in file_list:
        path_lower = file_path.lower()
        
        # Skip flutter assets or build folders
        if "flutter_assets" in path_lower:
            continue
            
        # Determine class based on folder structure
        if "diabetes" in path_lower:
            label = "diabetes"
        elif "normal" in path_lower:
            label = "normal"
        else:
            continue

        df_feat = extract_features_from_file(file_path, sampling_rate=sampling_rate, max_beats_per_file=300)
        
        if df_feat is not None:
            df_feat['label'] = label
            df_feat['subject_id'] = os.path.basename(file_path)  # Subject/file identifier
            all_dfs.append(df_feat)
            print(f"Successfully processed ({label}): {os.path.basename(file_path)} [{len(df_feat)} beats]")
        else:
            print(f"Skipped file: {file_path}")

    if not all_dfs:
        raise ValueError("No valid features extracted from dataset files.")

    dataset = pd.concat(all_dfs, ignore_index=True)
    return dataset

# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================

if __name__ == "__main__":
    DATA_DIR = "./data"   # Adjust to match directory containing 'diabetes' and 'normal' folders
    SAMPLING_RATE = 500   # Sampling frequency of MAX86150 sensor

    print("Extracting features and building dataset...")
    df = build_dataset(DATA_DIR, sampling_rate=SAMPLING_RATE)
    
    df.fillna(df.median(numeric_only=True), inplace=True)

    X = df.drop(columns=['label', 'subject_id'])
    y_raw = df['label'].values
    groups = df['subject_id'].values

    # Encode target labels explicitly
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    unique_classes = np.unique(y)
    print(f"\nExtracted Class Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    if len(unique_classes) < 2:
        raise ValueError("Error: Dataset missing one class. Verify file path structures.")

    # Step 1: Stratified Subject-Level Group Split (Prevents Data Leakage)
    sgkf = StratifiedGroupKFold(n_splits=5)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution:  {np.bincount(y_test)}")

    # Step 2: Feature Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Train Classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    print("\nTraining classifier...")
    model.fit(X_train_scaled, y_train)

    # Step 4: Evaluate Model
    y_pred = model.predict(X_test_scaled)

    print("\n--- Final Performance Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))