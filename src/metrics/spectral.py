"""
Spectral Metric Extraction:
- Effective Rank: exp(H(p)) where p is the normalized eigenvalue distribution.
- Eigenvalue spectrum decay analysis.
Reference: Roy & Vetterli (2007), Zendejas-Morales et al. (2024-2026).
"""
from typing import Dict, Any, Tuple
import numpy as np


def compute_effective_rank(K: np.ndarray, eps: float = 1e-15) -> Dict[str, Any]:
    """
    Compute the Effective Rank of a Gram matrix K:
    eff_rank(K) = exp( - sum_i p_i * ln(p_i) )
    where p_i = lambda_i / sum_j lambda_j for non-negative eigenvalues lambda_i.

    Args:
        K: Symmetric positive semi-definite Gram matrix of shape (N, N)
        eps: Small threshold to handle zero eigenvalues and numerical stability

    Returns:
        Dict containing:
            - 'effective_rank': float in [1, N]
            - 'effective_rank_ratio': float in (0, 1] (eff_rank / N)
            - 'eigenvalues': sorted descending eigenvalues
            - 'normalized_eigenvalues': p_i distribution
            - 'entropy': Shannon entropy H(p)
            - 'spectral_ratio_90': number of components needed to reach 90% power
    """
    N = K.shape[0]
    # Symmetrize to prevent numerical asymmetry
    K_sym = 0.5 * (K + K.T)

    # Compute eigenvalues for real symmetric matrix
    eigvals = np.linalg.eigvalsh(K_sym)

    # Clamp tiny numerical negatives to 0
    eigvals = np.maximum(eigvals, 0.0)

    # Sort descending
    eigvals = np.sort(eigvals)[::-1]

    total_variance = np.sum(eigvals)
    if total_variance < eps:
        return {
            "effective_rank": 1.0,
            "effective_rank_ratio": 1.0 / N,
            "eigenvalues": eigvals,
            "normalized_eigenvalues": np.zeros(N),
            "entropy": 0.0,
            "spectral_ratio_90": 1,
        }

    p = eigvals / total_variance

    # Shannon entropy of probability distribution p (using natural log)
    # Filter p > eps to prevent log(0)
    p_nonzero = p[p > eps]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))

    effective_rank = float(np.exp(entropy))
    # Bound within [1, N]
    effective_rank = min(max(effective_rank, 1.0), float(N))

    # Calculate 90% cumulative energy threshold
    cum_var = np.cumsum(p)
    n_90 = int(np.searchsorted(cum_var, 0.90) + 1)

    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / N,
        "eigenvalues": eigvals,
        "normalized_eigenvalues": p,
        "entropy": float(entropy),
        "spectral_ratio_90": min(n_90, N),
    }
