"""Run the local ECG experiment with DB4 wavelet denoising and EMD features."""
from diabetes_variant_local import run_variant

if __name__ == "__main__":
    run_variant("db4", "emd")
