# ECG diabetes classification

This repository contains the final ECG classification program, its reusable signal-processing module, earlier experiments, research material, and saved experiment results.

## Run the final program

Open a terminal in `code/` so the default paths resolve to the dataset folders and results folder there:

```powershell
cd code
python second_review_final.py
```

The program expects CSV recordings in `code/data/diabetes/` and `code/data/normal/`. Set custom locations or an output folder with `--diabetes-dir`, `--normal-dir`, and `--output-dir`. For limited memory, pass `--n-jobs 1`.

Install the Python packages used by the final program with:

```powershell
python -m pip install -r requirements_sym4_emd.txt
```

## Run the dashboard

The dashboard uses `code/second_review_final.py` as the source-of-truth pipeline:
Sym4 denoising, z-score normalization, R-peak detection, 210-sample beat windows,
leakage-safe ICA features, and the SVM-RBF classifier. It trains the deployable
classifier from the bundled labelled recordings on first use, then caches it for
the session.

```powershell
cd project
python -m pip install -r requirements_app.txt
streamlit run app.py
```

The reported 84.41% accuracy is the existing grouped cross-validation result. The
dashboard does not fabricate a probability because the final SVC was not calibrated.
It displays a class prediction and overlays the first usable uploaded heartbeat with
the first detected heartbeat window for visual context only. The chart is rendered as
a static Matplotlib figure with fixed axes so the review panel cannot accidentally pan
or zoom it while scrolling.
The comparison reference is downloaded at runtime from the public [MIT normal-sinus ECG sample on Zenodo](https://zenodo.org/records/6522747)
and cached for one day. The PQRST context is documented through the [NCBI ECG guide](https://www.ncbi.nlm.nih.gov/books/NBK594493/),
with the source database documented by [PhysioNet](https://archive.physionet.org/physiobank/database/nsrdb/).
The dashboard also keeps a local public-domain PQRST diagram at
`assets/normal_ecg_pqrst_reference.png`, sourced from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:SinusRhythmLabels.svg).

Note: `research/notebooks/Diabetes_ecg_SYM4_EMD.ipynb` documents a different
Sym4+EMD experiment. It is retained as research material and is not used by the
dashboard.

## Repository map

| Location | Contents |
| --- | --- |
| `code/second_review_final.py` | Final program to run |
| `code/diabetes_sym4_emd_local.py` | Shared signal processing and feature extraction imported by the final program |
| `code/diabetes_*_local.py` | Earlier and alternative local implementations |
| `code/v*.py` | Earlier numbered experiments |
| `code/data/` | Local ECG input data and application/runtime assets; data files are not committed |
| `code/*_output/` | Saved results from prior experiments |
| `code/*.ipynb` | Research notebooks |
| `code/Diabetes_ECG_Research_Paper_v1.docx` | Research paper draft |
| `../ppt/` | Review presentation decks stored beside this repository |

Run scripts from `code/` unless you provide explicit paths. Several scripts use paths relative to the current working directory.
