"""
Master Execution Script for Quantum Kernel Benchmarking on MedMNIST.
Usage:
    python run_experiments.py --all
    python run_experiments.py --dataset pneumoniamnist
    python run_experiments.py --dataset breastmnist
    python run_experiments.py --ablation sample_size
"""
import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATASETS_CONFIG, LOGS_DIR, FIGURES_DIR, REPORTS_DIR
from src.benchmark import run_benchmark_for_dataset
from src.visualization.plots import (
    plot_cross_dataset_comparison,
    plot_kta_vs_accuracy_scatter,
)


def run_all_benchmarks():
    """Run benchmark across all 4 MedMNIST datasets."""
    print("=================================================================")
    print("STARTING FULL QUANTUM KERNEL BENCHMARK ACROSS 4 MEDMNIST DATASETS")
    print("=================================================================")

    datasets = ["pneumoniamnist", "breastmnist", "retinamnist", "bloodmnist"]
    all_results = {}
    summary_records = []
    kta_acc_records = []

    for d_key in datasets:
        res = run_benchmark_for_dataset(d_key, run_alpha_sweep=True)
        all_results[d_key] = res

        # Extract summary records for cross-dataset comparisons
        kernel_results = res["kernel_results"]
        for k_name, k_data in kernel_results.items():
            summary_records.append({
                "dataset": res["dataset_name"],
                "kernel": k_name,
                "balanced_accuracy": k_data["balanced_accuracy"],
                "accuracy": k_data["accuracy"],
                "macro_f1": k_data["macro_f1"],
                "effective_rank": k_data["effective_rank"],
                "effective_rank_ratio": k_data["effective_rank_ratio"],
                "kta": k_data["kta"],
                "offdiag_var": k_data["offdiag_var"],
            })
            if k_data["kta"] is not None:
                kta_acc_records.append({
                    "dataset": d_key,
                    "kernel": k_name,
                    "kta": k_data["kta"],
                    "balanced_accuracy": k_data["balanced_accuracy"],
                    "accuracy": k_data["accuracy"],
                })

    # Save summary dataframe
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(LOGS_DIR / "cross_dataset_summary.csv", index=False)
    print("\nCross-Dataset Summary Table saved to results/logs/cross_dataset_summary.csv")

    # Generate Cross-Dataset Visualizations
    print("Generating cross-dataset synthesis figures...")
    plot_cross_dataset_comparison(
        summary_records,
        metric_name="balanced_accuracy",
        metric_label="Test Balanced Accuracy",
        save_path=FIGURES_DIR / "cross_dataset_balanced_accuracy.png",
    )
    plot_cross_dataset_comparison(
        summary_records,
        metric_name="accuracy",
        metric_label="Test Standard Accuracy",
        save_path=FIGURES_DIR / "cross_dataset_accuracy.png",
    )
    plot_cross_dataset_comparison(
        summary_records,
        metric_name="kta",
        metric_label="Centered Kernel Target Alignment (KTA)",
        save_path=FIGURES_DIR / "cross_dataset_kta.png",
    )
    plot_cross_dataset_comparison(
        summary_records,
        metric_name="effective_rank",
        metric_label="Effective Rank",
        save_path=FIGURES_DIR / "cross_dataset_effective_rank.png",
    )
    plot_kta_vs_accuracy_scatter(
        kta_acc_records,
        save_path=FIGURES_DIR / "kta_vs_accuracy_scatter.png",
    )

    print("\nALL BENCHMARKS COMPLETED SUCCESSFULLY!")


def run_sample_size_ablation(dataset_key: str = "pneumoniamnist"):
    """Evaluate performance scaling across sample sizes N in [100, 250, 500]."""
    print(f"\n=======================================================")
    print(f"RUNNING SAMPLE SIZE ABLATION ON {dataset_key.upper()}")
    print(f"=======================================================")

    sample_sizes = [100, 250, 500]
    records = []

    for n_tr in sample_sizes:
        print(f"\n--- Testing Train Sample Size N = {n_tr} ---")
        res = run_benchmark_for_dataset(
            dataset_key,
            n_train=n_tr,
            n_test=150,
            run_alpha_sweep=False,
            save_artifacts=False,
        )
        for k_name in ["global", "local_2qubit", "multiscale"]:
            k_data = res["kernel_results"][k_name]
            records.append({
                "sample_size": n_tr,
                "kernel": k_name,
                "balanced_accuracy": k_data["balanced_accuracy"],
                "accuracy": k_data["accuracy"],
                "effective_rank": k_data["effective_rank"],
                "kta": k_data["kta"],
                "offdiag_var": k_data["offdiag_var"],
            })

    df = pd.DataFrame(records)
    df.to_csv(LOGS_DIR / f"{dataset_key}_sample_size_ablation.csv", index=False)
    print(f"\nSample size ablation results saved to results/logs/{dataset_key}_sample_size_ablation.csv")
    print(df.to_string())


def main():
    parser = argparse.ArgumentParser(description="Quantum Kernel Benchmarking on MedMNIST")
    parser.add_argument("--all", action="store_true", help="Run benchmark on all 4 MedMNIST datasets")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS_CONFIG.keys()), help="Run benchmark on single dataset")
    parser.add_argument("--ablation", type=str, choices=["sample_size"], help="Run specified ablation")
    args = parser.parse_args()

    if args.all or (not args.dataset and not args.ablation):
        run_all_benchmarks()
    elif args.dataset:
        run_benchmark_for_dataset(args.dataset)
    elif args.ablation == "sample_size":
        run_sample_size_ablation()


if __name__ == "__main__":
    main()
