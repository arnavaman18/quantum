"""
Kernel Target Alignment (KTA) Extraction:
- Centered Kernel Target Alignment: Tr(K_c @ Y_c) / ( ||K_c||_F * ||Y_c||_F )
Reference: Cortes et al. (2012), Cristianini et al. (2002), Incudini et al. (2022-2024).
"""
from typing import Dict, Any
import numpy as np


def center_matrix(A: np.ndarray) -> np.ndarray:
    """
    Center an N x N matrix: A_c = C_N @ A @ C_N where C_N = I_N - (1/N) 1 1^T.
    Implemented efficiently using row/column means:
    (A_c)_ij = A_ij - row_mean_i - col_mean_j + total_mean.
    """
    row_means = np.mean(A, axis=1, keepdims=True)
    col_means = np.mean(A, axis=0, keepdims=True)
    total_mean = np.mean(A)
    return A - row_means - col_means + total_mean


def compute_target_matrix(labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """
    Construct the ideal target kernel matrix Y from class labels:
    - Binary (n_classes == 2): Y_ij = y_i * y_j with y in {-1, +1}
    - Multiclass (n_classes > 2): Y_ij = 1.0 if y_i == y_j else -1.0 / (n_classes - 1)
    """
    y = np.asarray(labels).ravel()
    N = len(y)

    if n_classes == 2:
        # Map binary {0, 1} to {-1, +1}
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            y_pm1 = np.where(y == unique_labels[1], 1.0, -1.0)
        else:
            y_pm1 = y.astype(np.float64)
        Y = np.outer(y_pm1, y_pm1)
    else:
        # Multiclass: 1.0 for same class, -1 / (C - 1) for different class
        diff_val = -1.0 / (n_classes - 1.0)
        same_mask = (y[:, None] == y[None, :])
        Y = np.where(same_mask, 1.0, diff_val)

    return Y.astype(np.float64)


def compute_centered_kta(
    K: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 2,
    eps: float = 1e-12,
) -> float:
    """
    Compute Centered Kernel Target Alignment (KTA):
    KTA = <K_c, Y_c>_F / ( ||K_c||_F * ||Y_c||_F )

    Args:
        K: Gram matrix of shape (N, N)
        labels: Class labels of shape (N,)
        n_classes: Number of distinct classes
        eps: Small threshold to prevent zero-norm division

    Returns:
        kta: float in [-1, 1]
    """
    Y = compute_target_matrix(labels, n_classes=n_classes)

    K_c = center_matrix(K)
    Y_c = center_matrix(Y)

    # Frobenius inner product: sum(K_c * Y_c)
    frob_inner = np.sum(K_c * Y_c)

    norm_Kc = np.sqrt(np.sum(K_c ** 2))
    norm_Yc = np.sqrt(np.sum(Y_c ** 2))

    if norm_Kc < eps or norm_Yc < eps:
        return 0.0

    kta = float(frob_inner / (norm_Kc * norm_Yc))
    return min(max(kta, -1.0), 1.0)
