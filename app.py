"""Streamlit dashboard for the deployed second_review_final ECG classifier."""

from __future__ import annotations

import gzip
import io
import sys
import tempfile
from urllib.request import urlopen
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from diabetes_sym4_emd_local import (  # noqa: E402
    RANDOM_STATE,
    WINDOW_AFTER,
    WINDOW_BEFORE,
    read_signal,
    safe_zscore,
)
from second_review_final import denoise  # noqa: E402


# The bundled recordings live beside the app in the project-level data folder.
DEFAULT_DIABETES_DIR = ROOT / "data" / "diabetes"
DEFAULT_NORMAL_DIR = ROOT / "data" / "normal"
ONLINE_REFERENCE_URL = "https://zenodo.org/records/6522747/files/16265.ecg.gz?download=1"
ONLINE_REFERENCE_PAGE = "https://zenodo.org/records/6522747"
REFERENCE_IMAGE = ROOT / "assets" / "normal_ecg_pqrst_reference.png"
PEAK_HEIGHT = 0.5
PEAK_DISTANCE = 150
ICA_COMPONENTS = 48


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    return safe_zscore(denoise(signal, "sym4"))


def extract_windows(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = preprocess_signal(signal)
    peaks, _ = find_peaks(normalized, height=PEAK_HEIGHT, distance=PEAK_DISTANCE)
    windows, kept_peaks = [], []
    for peak in peaks:
        start = int(peak - WINDOW_BEFORE)
        end = int(peak + WINDOW_AFTER)
        if start >= 0 and end <= len(normalized):
            windows.append(normalized[start:end].astype(np.float32))
            kept_peaks.append(int(peak))
    if not windows:
        raise ValueError("No complete heartbeat windows were found in this ECG.")
    return np.vstack(windows), np.asarray(kept_peaks)


def load_training_windows(folder: Path, label: int) -> list[tuple[np.ndarray, int]]:
    rows = []
    for path in sorted(folder.glob("*.csv")):
        try:
            signal = read_signal(path, "auto")
            windows, _ = extract_windows(signal)
            rows.extend((window, label) for window in windows)
        except (ValueError, pd.errors.ParserError):
            continue
    return rows


@st.cache_resource(show_spinner="Preparing the trained ECG classifier...")
def train_classifier() -> dict:
    records = load_training_windows(DEFAULT_DIABETES_DIR, 1)
    records += load_training_windows(DEFAULT_NORMAL_DIR, 0)
    if not records or {label for _, label in records} != {0, 1}:
        raise RuntimeError("Training data must contain usable CSV recordings for both classes.")

    raw = np.vstack([window for window, _ in records]).astype(np.float64)
    y = np.asarray([label for _, label in records], dtype=int)
    std = np.std(raw, axis=1, keepdims=True)
    components = min(ICA_COMPONENTS, raw.shape[0], raw.shape[1])
    ica = FastICA(n_components=components, random_state=RANDOM_STATE, max_iter=1000)
    ica_features = ica.fit_transform(raw)
    X = np.hstack([ica_features, std])
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", SVC(class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    model.fit(X, y)
    return {"ica": ica, "model": model, "records": len(records), "components": components}


def predict_signal(signal: np.ndarray, artifacts: dict) -> tuple[int, np.ndarray, np.ndarray]:
    windows, peaks = extract_windows(signal)
    std = np.std(windows, axis=1, keepdims=True)
    transformed = artifacts["ica"].transform(windows.astype(np.float64))
    predictions = artifacts["model"].predict(np.hstack([transformed, std])).astype(int)
    result = int(np.mean(predictions) >= 0.5)
    return result, predictions, peaks


def make_overlay(uploaded: np.ndarray, reference: np.ndarray | None, n_windows: int = 1) -> tuple[pd.DataFrame, int]:
    uploaded_windows, _ = extract_windows(uploaded)
    count = min(n_windows, len(uploaded_windows))
    def plot_normalize(windows: np.ndarray) -> np.ndarray:
        """Normalize each displayed beat independently without changing inference input."""
        normalized = [safe_zscore(np.asarray(window, dtype=np.float64)) for window in windows[:count]]
        joined = []
        for index, window in enumerate(normalized):
            if index:
                joined.append(np.full(12, np.nan))
            joined.append(window)
        return np.concatenate(joined)

    uploaded_plot = plot_normalize(uploaded_windows)
    data = {"Uploaded ECG": uploaded_plot}
    if reference is not None:
        try:
            reference_windows, _ = extract_windows(reference)
            reference_plot = plot_normalize(np.vstack([
                reference_windows[index % len(reference_windows)] for index in range(count)
            ]))
            data["Normal sinus/PQRST reference"] = reference_plot
        except ValueError:
            pass
    return pd.DataFrame(data), count


def make_static_chart(chart: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(12, 5), dpi=120)
    x = np.arange(len(chart))
    axis.plot(x, chart["Uploaded ECG"], color="#1746ff", linewidth=2.0, label="Uploaded ECG")
    if "Normal sinus/PQRST reference" in chart:
        axis.plot(x, chart["Normal sinus/PQRST reference"], color="#f000d0", linewidth=2.0, label="Normal ECG")
    finite_values = chart.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    lower, upper = float(np.min(finite_values)), float(np.max(finite_values))
    padding = max((upper - lower) * 0.12, 0.5)
    axis.set_xlim(0, len(chart) - 1)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel("Samples")
    axis.set_ylabel("Normalized amplitude")
    axis.set_title("Comparison of ECG Segments: Uploaded vs Normal")
    axis.grid(True, color="#d9dce3", linewidth=0.8)
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    return figure


@st.cache_data(ttl=86400, show_spinner="Loading online normal ECG reference...")
def load_online_reference() -> np.ndarray:
    # Prefer the public reference, but keep the chart useful when the remote
    # file changes format or is unavailable. The bundled normal recordings use
    # the same signal parser as the training pipeline.
    try:
        with urlopen(ONLINE_REFERENCE_URL, timeout=30) as response:
            compressed = response.read()
        raw = gzip.decompress(compressed)
        frame = pd.read_csv(io.BytesIO(raw))
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        preferred = [column for column in numeric.columns
                     if str(column).lower() in {"channel 1", "channel_1", "ecg"}]
        candidates = preferred or list(numeric.columns)
        for column in candidates:
            values = numeric[column].dropna().to_numpy(dtype=np.float64)
            if len(values) >= WINDOW_BEFORE + WINDOW_AFTER:
                return values
    except Exception:
        pass

    for path in sorted(DEFAULT_NORMAL_DIR.glob("*.csv")):
        try:
            return read_signal(path, "auto")
        except (ValueError, pd.errors.ParserError):
            continue
    raise ValueError("No usable online or bundled normal ECG reference was found.")


def main() -> None:
    st.set_page_config(page_title="T2DM ECG Detection", page_icon="🫀", layout="wide")
    st.markdown("<style>.block-container{max-width:1180px;padding-top:2rem}</style>", unsafe_allow_html=True)
    st.title("T2DM Detection and Classification")
    st.caption("ECG-based screening support using the Sym4 + ICA SVM-RBF pipeline.")
    st.info("This is a research decision-support tool. Its output is not a medical diagnosis.")

    uploaded = st.file_uploader("Upload an ECG recording", type=["csv"], help="CSV must contain at least one numeric signal column.")
    if uploaded is None:
        st.markdown("Upload a CSV file to begin analysis.")
        return

    st.caption(f"Selected file: {uploaded.name}")
    st.markdown("The uploaded ECG and normal reference will be shown on the same axes using different colors.")
    start_diagnosis = st.button("▶ Start diagnosis", type="primary", use_container_width=False)
    if not start_diagnosis:
        st.info("Click **Start diagnosis** to validate the signal and run the model.")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_file.write(uploaded.getvalue())
            upload_path = Path(temp_file.name)
        try:
            signal = read_signal(upload_path, "auto")
        finally:
            upload_path.unlink(missing_ok=True)
        artifacts = train_classifier()
        with st.spinner("Analyzing ECG signal..."):
            result, beat_predictions, peaks = predict_signal(signal, artifacts)
        try:
            reference = load_online_reference()
        except Exception as reference_error:
            reference = None
            st.warning(f"Online normal ECG reference unavailable: {reference_error}")
    except Exception as error:
        st.error(f"The ECG could not be analyzed: {error}")
        return

    st.success("Signal validated and analyzed.")
    left, right = st.columns([1, 2])
    with left:
        label = "Diabetes" if result else "No Diabetes"
        st.metric("Predicted class", label)
        st.metric("Usable heartbeat windows", len(beat_predictions))
        st.metric("Detected R-peaks", len(peaks))
        st.caption(f"Model trained on {artifacts['records']:,} beat windows using {artifacts['components']} ICA components.")
    with right:
        chart, window_count = make_overlay(signal, reference, n_windows=1)
        st.pyplot(make_static_chart(chart), clear_figure=True)
        st.caption(f"Showing {window_count} detected heartbeat window(s), each {WINDOW_BEFORE + WINDOW_AFTER} samples long.")
        if REFERENCE_IMAGE.exists():
            st.image(str(REFERENCE_IMAGE), caption="Public-domain normal sinus rhythm PQRST reference")

    st.caption("Blue: uploaded ECG. Magenta: sample-aligned online normal sinus/PQRST reference. The representative heartbeat is independently centered and z-score normalized for visual comparison; this display-only normalization does not change the model input. The chart is static and locked to a fixed zoom level.")
    st.markdown(f"Clinical waveform context: [NCBI guide to P, QRS, and T components](https://www.ncbi.nlm.nih.gov/books/NBK594493/) · [Online MIT normal-sinus ECG sample]({ONLINE_REFERENCE_PAGE}) · [PhysioNet database](https://archive.physionet.org/physiobank/database/nsrdb/)")


if __name__ == "__main__":
    main()
