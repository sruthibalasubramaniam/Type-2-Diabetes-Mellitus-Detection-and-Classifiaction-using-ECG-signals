"""Create a readable 32-row table for every experiment configuration."""

from pathlib import Path

import pandas as pd

from run_full_comparison import formatted_table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "full_comparison_output"

results = pd.read_csv(OUTPUT / "all_32_results.csv")
results.insert(0, "Combination", results["Denoising"] + " + " + results["Feature extraction"])
results = results.sort_values(["Denoising", "Feature extraction", "balanced_accuracy"],
                              ascending=[True, True, False])
columns = ["Combination", "Preprocessing", "model", "accuracy", "balanced_accuracy",
           "precision_diabetes", "sensitivity_recall", "specificity", "f1_diabetes"]
formatted_table(
    results[columns],
    "Performance Comparison of All 32 Configurations",
    OUTPUT / "all_32_combinations_performance.png",
)
print(OUTPUT / "all_32_combinations_performance.png")
