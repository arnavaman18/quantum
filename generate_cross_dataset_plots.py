"""
Generate Cross-Dataset Synthesis Plots from Saved CSV Logs.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import LOGS_DIR, FIGURES_DIR
from src.visualization.plots import (
    plot_cross_dataset_comparison,
    plot_kta_vs_accuracy_scatter,
)


def main():
    csv_path = LOGS_DIR / "cross_dataset_summary.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    primary_kernels = [
        "global",
        "local_1qubit",
        "local_2qubit",
        "multiscale",
        "hierarchical",
        "classical_rbf",
        "classical_linear",
    ]

    print("Generating cross-dataset comparison plots...")
    # 1. Balanced Accuracy (Primary Kernels)
    plot_cross_dataset_comparison(
        records,
        metric_name="balanced_accuracy",
        metric_label="Test Balanced Accuracy",
        selected_kernels=primary_kernels,
        save_path=FIGURES_DIR / "cross_dataset_balanced_accuracy.png",
    )

    # 2. Standard Accuracy (Primary Kernels)
    plot_cross_dataset_comparison(
        records,
        metric_name="accuracy",
        metric_label="Test Accuracy",
        selected_kernels=primary_kernels,
        save_path=FIGURES_DIR / "cross_dataset_accuracy.png",
    )

    # 3. Centered KTA (Quantum Kernels)
    quantum_kernels = ["global", "local_1qubit", "local_2qubit", "local_4qubit", "multiscale", "hierarchical"]
    plot_cross_dataset_comparison(
        records,
        metric_name="kta",
        metric_label="Centered Kernel Target Alignment (KTA)",
        selected_kernels=quantum_kernels,
        save_path=FIGURES_DIR / "cross_dataset_kta.png",
    )

    # 4. Effective Rank (Quantum Kernels)
    plot_cross_dataset_comparison(
        records,
        metric_name="effective_rank",
        metric_label="Effective Rank",
        selected_kernels=quantum_kernels,
        save_path=FIGURES_DIR / "cross_dataset_effective_rank.png",
    )

    # 5. KTA vs Accuracy Scatter
    kta_acc_records = []
    for r in records:
        if pd.notna(r["kta"]):
            kta_acc_records.append({
                "dataset": r["dataset"],
                "kernel": r["kernel"],
                "kta": float(r["kta"]),
                "balanced_accuracy": float(r["balanced_accuracy"]),
                "accuracy": float(r["accuracy"]),
            })

    plot_kta_vs_accuracy_scatter(
        kta_acc_records,
        save_path=FIGURES_DIR / "kta_vs_accuracy_scatter.png",
    )

    print("Cross-dataset plots generated successfully in figures/")


if __name__ == "__main__":
    main()
