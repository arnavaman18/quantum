"""
Subsystem Reduced Density Matrix (RDM) Extraction and Partial Trace Engine.
Computes exact reduced density matrices for arbitrary qubit subsystems
using optimized batch tensor contractions.
"""
from typing import Tuple, List
import numpy as np


def compute_reduced_density_matrices(
    statevectors: np.ndarray,
    subsystem_qubits: Tuple[int, ...],
    n_qubits: int = 8,
) -> np.ndarray:
    """
    Compute Reduced Density Matrices (RDMs) for a specific qubit subsystem across N states.

    Args:
        statevectors: Array of shape (N, 2^n_qubits)
        subsystem_qubits: Tuple of qubit indices belonging to the subsystem (0-indexed)
        n_qubits: Total number of qubits (default 8)

    Returns:
        rdms: Complex array of shape (N, d_s, d_s) where d_s = 2^len(subsystem_qubits)
    """
    N = statevectors.shape[0]
    k = len(subsystem_qubits)
    d_s = 1 << k  # 2^k
    d_e = 1 << (n_qubits - k)  # 2^(n-k)

    # Sort subsystem qubits and identify environment qubits
    subsystem = tuple(sorted(subsystem_qubits))
    environment = tuple(q for q in range(n_qubits) if q not in subsystem)

    # Reshape statevector into N + 8 individual qubit axes: (N, 2, 2, 2, 2, 2, 2, 2, 2)
    state_tensor = statevectors.reshape((N,) + (2,) * n_qubits)

    # Permutation: [batch_axis=0] + [subsystem axes (+1)] + [environment axes (+1)]
    perm = [0] + [q + 1 for q in subsystem] + [q + 1 for q in environment]
    permuted = np.transpose(state_tensor, perm)

    # Reshape into matrix form: (N, d_s, d_e)
    # The first index is batch, second is subsystem basis, third is environment basis
    psi_matrix = permuted.reshape(N, d_s, d_e)

    # Partial trace over environment:
    # rho_s(i)_{a, b} = sum_e psi_matrix(i, a, e) * conj(psi_matrix(i, b, e))
    # In batch matrix multiplication: psi_matrix @ psi_matrix.conj().T
    psi_conj_t = np.conjugate(np.transpose(psi_matrix, (0, 2, 1)))
    rdms = np.matmul(psi_matrix, psi_conj_t)

    return rdms


def vectorize_density_matrices(rdms: np.ndarray, normalize_purity: bool = True) -> np.ndarray:
    """
    Flatten (N, d_s, d_s) density matrices into (N, d_s^2) vectors for
    vectorized Hilbert-Schmidt inner product evaluation:
    Tr[rho_i @ rho_j] = <vec(rho_i), vec(rho_j)>.

    Args:
        rdms: Array of shape (N, d_s, d_s)
        normalize_purity: If True, normalizes vectors so Tr[rho_i^2] = 1, ensuring K_ii = 1.0

    Returns:
        vec_rdms: Complex array of shape (N, d_s^2)
    """
    N, d_s, _ = rdms.shape
    vecs = rdms.reshape(N, d_s * d_s)

    if normalize_purity:
        # Purity: Tr[rho_i^2] = sum_ab |(rho_i)_ab|^2 = ||vec(rho_i)||_2^2
        purities = np.real(np.sum(np.abs(vecs) ** 2, axis=1, keepdims=True))
        norms = np.sqrt(np.maximum(purities, 1e-15))
        vecs = vecs / norms

    return vecs
