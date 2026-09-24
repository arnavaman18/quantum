"""
Publication-Quality Visualization Routines (Nature/IEEE Aesthetic):
- Gram matrix comparison heatmaps
- Eigenvalue decay and effective rank profiles
- Off-diagonal concentration histograms
- Multiscale alpha sweep curves
- KTA vs Accuracy correlation scatter plots
- Cross-dataset summary comparison charts
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

PALETTE = {
    "global": "#D95F02",          # Burnt orange
    "local_1qubit": "#7570B3",    # Purple
    "local_2qubit": "#1B9E77",    # Teal green
    "local_4qubit": "#E7298A",    # Magenta
    "multiscale": "#386CB0",      # Deep blue
    "hierarchical": "#FDC086",    # Warm gold
    "global_entangled": "#A6761D",
    "local_2qubit_entangled": "#66A61E",
    "multiscale_entangled": "#E6AB02",
    "classical_rbf": "#666666",   # Neutral dark gray
    "classical_linear": "#999999",# Neutral light gray
}


def plot_gram_matrix_heatmaps(
    gram_dict: Dict[str, np.ndarray],
    target_matrix: np.ndarray,
    dataset_name: str,
    save_path: Optional[Path] = None,
    max_dim: int = 150,
):
    """
    Side-by-side heatmaps of Gram matrices vs Ideal Target Matrix Y.
    """
    keys = ["global", "local_2qubit", "multiscale"]
    titles = [
        "Global Fidelity Kernel",
        "Local (2-Qubit Patch) Kernel",
        "Multiscale Kernel (alpha=0.5)",
        "Ideal Target Matrix Y",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)

    matrices = [gram_dict.get(k) for k in keys] + [target_matrix]

    for ax, mat, title in zip(axes, matrices, titles):
        if mat is None:
            continue
        # Subsample if too large for visualization
        sub_mat = mat[:max_dim, :max_dim] if mat.shape[0] > max_dim else mat
        vmin, vmax = (0.0, 1.0) if title != "Ideal Target Matrix Y" else (np.min(sub_mat), np.max(sub_mat))
        cmap = "mako" if title != "Ideal Target Matrix Y" else "vlag"

        im = ax.imshow(sub_mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"Gram Matrix Structure & Target Alignment: {dataset_name}", fontsize=15, fontweight="bold")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eigenvalue_spectra(
    spectral_dict: Dict[str, Dict[str, Any]],
    dataset_name: str,
    save_path: Optional[Path] = None,
    max_k: int = 100,
):
    """
    Log-scale eigenvalue decay plot comparing Global vs Local vs Multiscale kernels.
    Annotates Effective Rank in legend.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for key, spec in spectral_dict.items():
        if "normalized_eigenvalues" not in spec:
            continue
        p = spec["normalized_eigenvalues"]
        k_eval = min(len(p), max_k)
        eff_r = spec.get("effective_rank", 0.0)
        label = f"{key.replace('_', ' ').title()} (eff-rank = {eff_r:.2f})"
        color = PALETTE.get(key, None)

        ax.plot(
            range(1, k_eval + 1),
            p[:k_eval],
            label=label,
            color=color,
            linewidth=2.2,
            alpha=0.9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue Rank Index $i$")
    ax.set_ylabel("Normalized Eigenvalue $p_i = \\lambda_i / \\mathrm{Tr}(K)$")
    ax.set_title(f"Kernel Eigenvalue Spectrum & Effective Rank: {dataset_name}", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_concentration_distributions(
    gram_dict: Dict[str, np.ndarray],
    dataset_name: str,
    save_path: Optional[Path] = None,
):
    """
    KDE distributions of off-diagonal Gram matrix elements K_ij (i != j)
    demonstrating exponential concentration in Global Fidelity.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))

    for key in ["global", "local_1qubit", "local_2qubit", "multiscale"]:
        if key not in gram_dict:
            continue
        K = gram_dict[key]
        N = K.shape[0]
        mask = ~np.eye(N, dtype=bool)
        off_diags = K[mask]

        var_val = np.var(off_diags)
        mean_val = np.mean(off_diags)
        label = f"{key.replace('_', ' ').title()} ($\\mu={mean_val:.3f}, \\sigma^2={var_val:.2e}$)"
        color = PALETTE.get(key, None)

        sns.kdeplot(
            off_diags,
            label=label,
            color=color,
            linewidth=2.2,
            fill=True,
            alpha=0.18,
            ax=ax,
            clip=(0.0, 1.0),
        )

    ax.set_xlabel("Off-Diagonal Kernel Overlap $K_{ij}$ ($i \\neq j$)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Exponential Concentration Profile: {dataset_name}", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.set_xlim(-0.02, 1.02)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiscale_alpha_sweep(
    alphas: List[float],
    sweep_metrics: Dict[str, List[float]],
    dataset_name: str,
    save_path: Optional[Path] = None,
):
    """
    Parametric sweep of multiscale parameter alpha in [0, 1].
    Shows smooth transition in Accuracy, KTA, Effective Rank, and Off-diagonal Variance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)

    # 1. Balanced Accuracy & Standard Accuracy
    ax = axes[0, 0]
    if "balanced_accuracy" in sweep_metrics:
        ax.plot(alphas, sweep_metrics["balanced_accuracy"], "o-", color="#1B9E77", linewidth=2, label="Balanced Acc")
    if "accuracy" in sweep_metrics:
        ax.plot(alphas, sweep_metrics["accuracy"], "s--", color="#386CB0", linewidth=1.8, label="Standard Acc")
    ax.set_xlabel("$\\alpha$ (0 = Pure Local, 1 = Pure Global)")
    ax.set_ylabel("Classification Score")
    ax.set_title("Classification Performance vs $\\alpha$", fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Centered KTA
    ax = axes[0, 1]
    if "kta" in sweep_metrics:
        ax.plot(alphas, sweep_metrics["kta"], "d-", color="#D95F02", linewidth=2.2)
    ax.set_xlabel("$\\alpha$ (0 = Pure Local, 1 = Pure Global)")
    ax.set_ylabel("Centered KTA")
    ax.set_title("Centered Kernel Target Alignment vs $\\alpha$", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. Effective Rank
    ax = axes[1, 0]
    if "effective_rank" in sweep_metrics:
        ax.plot(alphas, sweep_metrics["effective_rank"], "^-", color="#7570B3", linewidth=2.2)
    ax.set_xlabel("$\\alpha$ (0 = Pure Local, 1 = Pure Global)")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank vs $\\alpha$", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Off-diagonal Variance
    ax = axes[1, 1]
    if "offdiag_var" in sweep_metrics:
        ax.plot(alphas, sweep_metrics["offdiag_var"], "v-", color="#E7298A", linewidth=2.2)
        ax.set_yscale("log")
    ax.set_xlabel("$\\alpha$ (0 = Pure Local, 1 = Pure Global)")
    ax.set_ylabel("Off-Diagonal Variance $\\sigma^2_{\\mathrm{off}}$ (log)")
    ax.set_title("Concentration Mitigation vs $\\alpha$", fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.suptitle(f"Multiscale Convex Combination Analysis: {dataset_name}", fontsize=15, fontweight="bold")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kta_vs_accuracy_scatter(
    kta_acc_records: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
):
    """
    Scatter plot of Centered KTA vs Balanced Accuracy across all datasets and kernel methods.
    Empirically validates the "useful" kernel threshold hypothesis (Incudini et al.).
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))

    ktas = [r["kta"] for r in kta_acc_records]
    accs = [r["balanced_accuracy"] for r in kta_acc_records]
    datasets = [r["dataset"] for r in kta_acc_records]
    kernels = [r["kernel"] for r in kta_acc_records]

    dataset_markers = {"pneumoniamnist": "o", "breastmnist": "s", "retinamnist": "^", "bloodmnist": "D"}

    for d_name, marker in dataset_markers.items():
        idxs = [i for i, d in enumerate(datasets) if d.lower() == d_name]
        if not idxs:
            continue
        sub_kta = [ktas[i] for i in idxs]
        sub_acc = [accs[i] for i in idxs]
        sub_kern = [kernels[i] for i in idxs]
        colors = [PALETTE.get(k, "#333333") for k in sub_kern]

        ax.scatter(
            sub_kta,
            sub_acc,
            label=d_name.replace("mnist", "MNIST").capitalize(),
            marker=marker,
            s=90,
            c=colors,
            edgecolors="black",
            alpha=0.85,
        )

    # Linear trendline if multiple points
    if len(ktas) > 3:
        m, b = np.polyfit(ktas, accs, 1)
        x_trend = np.linspace(min(ktas), max(ktas), 50)
        corr = float(np.corrcoef(ktas, accs)[0, 1])
        ax.plot(x_trend, m * x_trend + b, "--", color="#333333", alpha=0.7, label=f"Trend (r = {corr:.2f})")

    ax.set_xlabel("Centered Kernel Target Alignment (KTA)")
    ax.set_ylabel("Test Balanced Accuracy")
    ax.set_title("Empirical 'Useful' Kernel Hypothesis: KTA vs Classification Generalization", fontweight="bold")
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_comparison(
    summary_data: List[Dict[str, Any]],
    metric_name: str = "balanced_accuracy",
    metric_label: str = "Balanced Accuracy",
    selected_kernels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
):
    """
    Grouped bar chart comparing Global Fidelity vs Local vs Multiscale vs Classical Baselines
    across all four MedMNIST datasets.
    """
    import pandas as pd
    df = pd.DataFrame(summary_data)
    if df.empty or metric_name not in df.columns:
        return

    if selected_kernels is not None:
        df = df[df["kernel"].isin(selected_kernels)]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=df,
        x="dataset",
        y=metric_name,
        hue="kernel",
        palette=PALETTE,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel("Benchmarking Dataset", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.set_title(f"Cross-Dataset Kernel Performance Comparison: {metric_label}", fontweight="bold", pad=12)
    ax.legend(title="Kernel Method", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
