"""
Quantum Kernel Construction Module:
- Global Fidelity Kernel: K_global(x, x') = |<psi(x)|psi(x')>|^2
- Local Subsystem Kernel: K_local(x, x') = (1/P) sum_p Tr[rho_p(x) rho_p(x')]
- Multiscale Kernel: K_multi(x, x'; alpha) = alpha * K_global + (1 - alpha) * K_local
- Hierarchical Multiscale Kernel: multi-granular convex combination
"""
from typing import Optional, Dict, Tuple, List
import numpy as np
from src.config import PATCH_CONFIGS
from src.quantum.density_matrix import (
    compute_reduced_density_matrices,
    vectorize_density_matrices,
)


def compute_global_kernel(
    states_train: np.ndarray,
    states_test: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Global Fidelity Gram matrix:
    K_global(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2.

    Args:
        states_train: Complex array of shape (N_train, 2^n)
        states_test: Optional complex array of shape (N_test, 2^n).
                     If None, computes train-train (N_train, N_train).

    Returns:
        K: Real array of shape (N_train, N_train) or (N_test, N_train)
    """
    if states_test is None:
        # Train-Train overlap: S = states_train @ states_train.conj().T
        S = np.matmul(states_train, np.conjugate(states_train.T))
        K = np.real(S * np.conjugate(S))
        # Ensure diagonal is identically 1.0
        np.fill_diagonal(K, 1.0)
    else:
        # Test-Train overlap: S = states_test @ states_train.conj().T
        S = np.matmul(states_test, np.conjugate(states_train.T))
        K = np.real(S * np.conjugate(S))

    # Numerical clip to [0, 1]
    return np.clip(K, 0.0, 1.0)


def compute_patch_kernel(
    states_train: np.ndarray,
    states_test: Optional[np.ndarray],
    patch_qubits: Tuple[int, ...],
    normalize_purity: bool = True,
) -> np.ndarray:
    """
    Compute local kernel on a single subsystem patch:
    K_p(x_i, x_j) = Tr[rho_p(x_i) rho_p(x_j)] / ( ||rho_p(x_i)||_F * ||rho_p(x_j)||_F )

    Args:
        states_train: Array of shape (N_train, 256)
        states_test: Optional array of shape (N_test, 256)
        patch_qubits: Tuple of qubit indices
        normalize_purity: Normalizes purity so K_p(x, x) = 1.0

    Returns:
        K_p: Array of shape (N_train, N_train) or (N_test, N_train)
    """
    rdm_train = compute_reduced_density_matrices(states_train, patch_qubits)
    vec_train = vectorize_density_matrices(rdm_train, normalize_purity=normalize_purity)

    if states_test is None:
        # V @ V^dagger
        K_p = np.real(np.matmul(vec_train, np.conjugate(vec_train.T)))
        np.fill_diagonal(K_p, 1.0)
    else:
        rdm_test = compute_reduced_density_matrices(states_test, patch_qubits)
        vec_test = vectorize_density_matrices(rdm_test, normalize_purity=normalize_purity)
        K_p = np.real(np.matmul(vec_test, np.conjugate(vec_train.T)))

    return np.clip(K_p, 0.0, 1.0)


def compute_local_kernel(
    states_train: np.ndarray,
    states_test: Optional[np.ndarray] = None,
    patch_type: str = "2-qubit",
    custom_patches: Optional[List[Tuple[int, ...]]] = None,
    normalize_purity: bool = True,
) -> np.ndarray:
    """
    Compute aggregated Local Subsystem Kernel:
    K_local(x, x') = (1/|P|) sum_{p in P} K_p(x, x')

    Args:
        states_train: (N_train, 256)
        states_test: Optional (N_test, 256)
        patch_type: '1-qubit', '2-qubit', '2-qubit-cyclic', or '4-qubit'
        custom_patches: Explicit list of qubit tuples if patch_type is 'custom'
        normalize_purity: Whether to normalize patch purities

    Returns:
        K_local: (N_train, N_train) or (N_test, N_train)
    """
    if custom_patches is not None:
        patches = custom_patches
    else:
        if patch_type not in PATCH_CONFIGS:
            raise ValueError(f"Unknown patch_type '{patch_type}'. Available: {list(PATCH_CONFIGS.keys())}")
        patches = PATCH_CONFIGS[patch_type]

    n_samples_out = states_train.shape[0] if states_test is None else states_test.shape[0]
    n_train = states_train.shape[0]
    K_acc = np.zeros((n_samples_out, n_train), dtype=np.float64)

    for patch in patches:
        K_p = compute_patch_kernel(
            states_train=states_train,
            states_test=states_test,
            patch_qubits=patch,
            normalize_purity=normalize_purity,
        )
        K_acc += K_p

    K_local = K_acc / len(patches)
    if states_test is None:
        np.fill_diagonal(K_local, 1.0)
    return np.clip(K_local, 0.0, 1.0)


def compute_multiscale_kernel(
    K_global: np.ndarray,
    K_local: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Convex combination of Global and Local kernels:
    K_multi = alpha * K_global + (1 - alpha) * K_local

    Args:
        K_global: Global fidelity Gram matrix
        K_local: Local subsystem Gram matrix
        alpha: Weight in [0, 1] for global fidelity (0 = purely local, 1 = purely global)

    Returns:
        K_multi: Multiscale Gram matrix
    """
    assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0, 1], got {alpha}"
    return alpha * K_global + (1.0 - alpha) * K_local


def compute_hierarchical_multiscale_kernel(
    K_1qubit: np.ndarray,
    K_2qubit: np.ndarray,
    K_4qubit: np.ndarray,
    K_global: np.ndarray,
    weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> np.ndarray:
    """
    Hierarchical multiscale combination across 1-qubit, 2-qubit, 4-qubit, and 8-qubit (global) scales:
    K = w_1*K_1 + w_2*K_2 + w_3*K_4 + w_g*K_global

    Args:
        K_1qubit, K_2qubit, K_4qubit, K_global: Gram matrices
        weights: 4-tuple of non-negative weights summing to 1

    Returns:
        K_hierarchical: Hierarchical multiscale Gram matrix
    """
    norm_w = np.array(weights, dtype=np.float64)
    norm_w = norm_w / np.sum(norm_w)
    return (
        norm_w[0] * K_1qubit
        + norm_w[1] * K_2qubit
        + norm_w[2] * K_4qubit
        + norm_w[3] * K_global
    )
