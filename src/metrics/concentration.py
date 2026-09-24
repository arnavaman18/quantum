"""
Exponential Concentration Analysis:
- Off-diagonal mean, variance, standard deviation, median, IQR.
Reference: Thanasilp et al. (Nature Communications 2024).
"""
from typing import Dict, Any
import numpy as np


def compute_concentration_statistics(K: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical distribution of off-diagonal elements of Gram matrix K:
    O = { K_ij : i != j }

    Exponential concentration manifests as:
    - var(O) -> 0
    - mean(O) -> 0 (for fidelity kernels) or constant

    Args:
        K: Gram matrix of shape (N, N)

    Returns:
        Dict with mean, variance, std, median, iqr, min, max, concentration_index
    """
    N = K.shape[0]
    if N <= 1:
        return {
            "offdiag_mean": 0.0,
            "offdiag_var": 0.0,
            "offdiag_std": 0.0,
            "offdiag_median": 0.0,
            "offdiag_iqr": 0.0,
            "offdiag_min": 0.0,
            "offdiag_max": 0.0,
            "concentration_index": 0.0,
        }

    mask = ~np.eye(N, dtype=bool)
    off_diags = K[mask]

    mean_val = float(np.mean(off_diags))
    var_val = float(np.var(off_diags))
    std_val = float(np.std(off_diags))
    median_val = float(np.median(off_diags))
    q75, q25 = np.percentile(off_diags, [75, 25])
    iqr_val = float(q75 - q25)
    min_val = float(np.min(off_diags))
    max_val = float(np.max(off_diags))

    # Concentration index: ratio of std to (mean + eps)
    # When kernel concentrates heavily, std vanishes relative to distance from identity
    concentration_index = float(-np.log10(max(var_val, 1e-16)))

    return {
        "offdiag_mean": mean_val,
        "offdiag_var": var_val,
        "offdiag_std": std_val,
        "offdiag_median": median_val,
        "offdiag_iqr": iqr_val,
        "offdiag_min": min_val,
        "offdiag_max": max_val,
        "concentration_index": concentration_index,
    }
