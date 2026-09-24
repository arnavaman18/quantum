"""
Unit tests for Metrics:
- Effective Rank mathematical boundaries and behavior
- Centered KTA boundaries and behavior
- Off-diagonal concentration statistics
"""
import numpy as np
import pytest
from src.metrics.spectral import compute_effective_rank
from src.metrics.alignment import compute_centered_kta, compute_target_matrix, center_matrix
from src.metrics.concentration import compute_concentration_statistics


def test_effective_rank_identity():
    # An identity matrix of size N has all equal eigenvalues (1.0).
    # Its effective rank should be exactly N.
    N = 20
    K_id = np.eye(N)
    res = compute_effective_rank(K_id)
    assert np.isclose(res["effective_rank"], float(N), atol=1e-5)
    assert np.isclose(res["effective_rank_ratio"], 1.0, atol=1e-5)


def test_effective_rank_rank_one():
    # An all-ones matrix 1 1^T has rank 1 (one eigenvalue = N, all others = 0).
    # Its effective rank should be exactly 1.0.
    N = 25
    K_ones = np.ones((N, N))
    res = compute_effective_rank(K_ones)
    assert np.isclose(res["effective_rank"], 1.0, atol=1e-5)
    assert np.isclose(res["effective_rank_ratio"], 1.0 / N, atol=1e-5)


def test_centered_kta_ideal():
    # If K is identical to Y, KTA should be 1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    Y = compute_target_matrix(y, n_classes=2)
    kta_perfect = compute_centered_kta(Y, y, n_classes=2)
    assert np.isclose(kta_perfect, 1.0, atol=1e-5)

    # If K is inverted (-Y), KTA should be -1.0
    kta_inverted = compute_centered_kta(-Y, y, n_classes=2)
    assert np.isclose(kta_inverted, -1.0, atol=1e-5)


def test_centered_kta_multiclass():
    y = np.array([0, 1, 2, 0, 1, 2, 3])
    Y = compute_target_matrix(y, n_classes=4)
    kta_perfect = compute_centered_kta(Y, y, n_classes=4)
    assert np.isclose(kta_perfect, 1.0, atol=1e-5)


def test_concentration_statistics():
    # Matrix with known off-diagonals
    N = 4
    K = np.array([
        [1.0, 0.2, 0.4, 0.6],
        [0.2, 1.0, 0.8, 0.1],
        [0.4, 0.8, 1.0, 0.5],
        [0.6, 0.1, 0.5, 1.0],
    ])
    off_diags = [0.2, 0.4, 0.6, 0.2, 0.8, 0.1, 0.4, 0.8, 0.5, 0.6, 0.1, 0.5]
    stats = compute_concentration_statistics(K)

    assert np.isclose(stats["offdiag_mean"], np.mean(off_diags))
    assert np.isclose(stats["offdiag_var"], np.var(off_diags))
    assert np.isclose(stats["offdiag_min"], 0.1)
    assert np.isclose(stats["offdiag_max"], 0.8)
