"""Create the final eight-combination performance table image."""

from pathlib import Path

import pandas as pd

from run_full_comparison import formatted_table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "full_comparison_output"

results = pd.read_csv(OUTPUT / "eight_combination_summary.csv")
results.insert(0, "Combination", results["Denoising"] + " + " + results["Feature extraction"])
columns = ["Combination", "model", "accuracy", "balanced_accuracy",
           "precision_diabetes", "sensitivity_recall", "specificity", "f1_diabetes"]
results = results.sort_values("balanced_accuracy", ascending=False)
formatted_table(
    results[columns],
    "Performance Comparison of All Eight Combinations",
    OUTPUT / "all_8_combinations_performance.png",
)
print(OUTPUT / "all_8_combinations_performance.png")
