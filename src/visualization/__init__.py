"""
Visualization module for quantum kernel benchmarking.
"""
from src.visualization.plots import (
    plot_gram_matrix_heatmaps,
    plot_eigenvalue_spectra,
    plot_concentration_distributions,
    plot_multiscale_alpha_sweep,
    plot_kta_vs_accuracy_scatter,
    plot_cross_dataset_comparison,
)

__all__ = [
    "plot_gram_matrix_heatmaps",
    "plot_eigenvalue_spectra",
    "plot_concentration_distributions",
    "plot_multiscale_alpha_sweep",
    "plot_kta_vs_accuracy_scatter",
    "plot_cross_dataset_comparison",
]
