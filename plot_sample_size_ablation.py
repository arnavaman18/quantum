"""
Plot Sample Size Scaling Ablation.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import LOGS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
})

PALETTE = {
    "global": "#D95F02",
    "local_2qubit": "#1B9E77",
    "multiscale": "#386CB0",
}


def main():
    csv_file = LOGS_DIR / "pneumoniamnist_sample_size_ablation.csv"
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    # 1. Balanced Accuracy vs N
    ax = axes[0]
    for k in ["global", "local_2qubit", "multiscale"]:
        sub = df[df["kernel"] == k]
        ax.plot(sub["sample_size"], sub["balanced_accuracy"], "o-", label=k.replace("_", " ").title(), color=PALETTE[k], linewidth=2.2, markersize=7)
    ax.set_xlabel("Training Sample Size $N_{\\mathrm{train}}$", fontweight="bold")
    ax.set_ylabel("Test Balanced Accuracy", fontweight="bold")
    ax.set_title("Balanced Accuracy vs Sample Size", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Centered KTA vs N
    ax = axes[1]
    for k in ["global", "local_2qubit", "multiscale"]:
        sub = df[df["kernel"] == k]
        ax.plot(sub["sample_size"], sub["kta"], "s-", label=k.replace("_", " ").title(), color=PALETTE[k], linewidth=2.2, markersize=7)
    ax.set_xlabel("Training Sample Size $N_{\\mathrm{train}}$", fontweight="bold")
    ax.set_ylabel("Centered KTA", fontweight="bold")
    ax.set_title("Target Alignment vs Sample Size", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Off-diagonal Variance vs N
    ax = axes[2]
    for k in ["global", "local_2qubit", "multiscale"]:
        sub = df[df["kernel"] == k]
        ax.plot(sub["sample_size"], sub["offdiag_var"], "^-", label=k.replace("_", " ").title(), color=PALETTE[k], linewidth=2.2, markersize=7)
    ax.set_xlabel("Training Sample Size $N_{\\mathrm{train}}$", fontweight="bold")
    ax.set_ylabel("Off-Diagonal Variance $\\sigma^2_{\\mathrm{off}}$", fontweight="bold")
    ax.set_title("Concentration Mitigation vs Sample Size", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Sample Size Scaling Ablation (PneumoniaMNIST, 8 Qubits)", fontsize=14, fontweight="bold")
    out_path = FIGURES_DIR / "sample_size_scaling_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample size ablation plot saved to {out_path}")


if __name__ == "__main__":
    main()
