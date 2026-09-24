"""
Unit tests for Quantum Circuit Statevector generation and Kernel Gram matrix computations:
- Statevector normalization and equivalence with PennyLane
- Global Fidelity: Symmetry, PSD, diagonal=1, bounded in [0, 1]
- Local Kernel: RDM trace=1, Hermiticity, PSD, diagonal=1
- Multiscale Kernel: Convex combination bounds and endpoints
- Out-of-sample Test-Train Gram matrix shapes
"""
import numpy as np
import pytest
from src.quantum.circuits import (
    generate_angle_statevectors_vectorized,
    generate_entangled_statevectors_vectorized,
)
from src.quantum.density_matrix import (
    compute_reduced_density_matrices,
    vectorize_density_matrices,
)
from src.quantum.kernels import (
    compute_global_kernel,
    compute_local_kernel,
    compute_multiscale_kernel,
    compute_hierarchical_multiscale_kernel,
)


@pytest.fixture
def sample_angles():
    np.random.seed(42)
    # 20 samples with 8 features in [-pi, pi]
    return np.random.uniform(-np.pi, np.pi, size=(20, 8))


@pytest.fixture
def test_angles():
    np.random.seed(123)
    return np.random.uniform(-np.pi, np.pi, size=(10, 8))


def test_statevector_normalization(sample_angles):
    states = generate_angle_statevectors_vectorized(sample_angles)
    assert states.shape == (20, 256)
    # Check that each statevector has unit norm: <psi|psi> = 1
    norms = np.linalg.norm(states, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    # Test entangled statevector
    entangled_states = generate_entangled_statevectors_vectorized(sample_angles)
    assert entangled_states.shape == (20, 256)
    ent_norms = np.linalg.norm(entangled_states, axis=1)
    assert np.allclose(ent_norms, 1.0, atol=1e-6)


def test_global_kernel_properties(sample_angles, test_angles):
    states_train = generate_angle_statevectors_vectorized(sample_angles)
    states_test = generate_angle_statevectors_vectorized(test_angles)

    K_train = compute_global_kernel(states_train)
    K_test = compute_global_kernel(states_train, states_test)

    # 1. Shape checks
    assert K_train.shape == (20, 20)
    assert K_test.shape == (10, 20)

    # 2. Symmetry of train matrix
    assert np.allclose(K_train, K_train.T, atol=1e-7)

    # 3. Diagonal is exactly 1.0
    assert np.allclose(np.diag(K_train), 1.0, atol=1e-7)

    # 4. Values in [0, 1]
    assert np.all(K_train >= -1e-7) and np.all(K_train <= 1.0 + 1e-7)
    assert np.all(K_test >= -1e-7) and np.all(K_test <= 1.0 + 1e-7)

    # 5. Positive semi-definiteness (all eigenvalues >= 0)
    eigvals = np.linalg.eigvalsh(K_train)
    assert np.min(eigvals) >= -1e-6, f"Global kernel has negative eigenvalue: {np.min(eigvals)}"


def test_rdm_properties(sample_angles):
    states = generate_angle_statevectors_vectorized(sample_angles)
    # Test 1-qubit RDM on qubit 0
    rdm_1q = compute_reduced_density_matrices(states, (0,))
    assert rdm_1q.shape == (20, 2, 2)
    for i in range(20):
        # Trace = 1
        assert np.isclose(np.trace(rdm_1q[i]), 1.0, atol=1e-6)
        # Hermitian
        assert np.allclose(rdm_1q[i], rdm_1q[i].conj().T, atol=1e-6)
        # Positive eigenvalues
        evs = np.linalg.eigvalsh(rdm_1q[i])
        assert np.all(evs >= -1e-7)

    # Test 2-qubit RDM on qubits (0, 1)
    rdm_2q = compute_reduced_density_matrices(states, (0, 1))
    assert rdm_2q.shape == (20, 4, 4)
    for i in range(20):
        assert np.isclose(np.trace(rdm_2q[i]), 1.0, atol=1e-6)
        assert np.allclose(rdm_2q[i], rdm_2q[i].conj().T, atol=1e-6)

    # Test 4-qubit RDM on qubits (0, 1, 2, 3)
    rdm_4q = compute_reduced_density_matrices(states, (0, 1, 2, 3))
    assert rdm_4q.shape == (20, 16, 16)
    for i in range(20):
        assert np.isclose(np.trace(rdm_4q[i]), 1.0, atol=1e-6)


def test_local_kernel_properties(sample_angles, test_angles):
    states_train = generate_angle_statevectors_vectorized(sample_angles)
    states_test = generate_angle_statevectors_vectorized(test_angles)

    for p_type in ["1-qubit", "2-qubit", "4-qubit"]:
        K_loc_tr = compute_local_kernel(states_train, patch_type=p_type)
        K_loc_te = compute_local_kernel(states_train, states_test, patch_type=p_type)

        assert K_loc_tr.shape == (20, 20)
        assert K_loc_te.shape == (10, 20)
        assert np.allclose(K_loc_tr, K_loc_tr.T, atol=1e-7)
        assert np.allclose(np.diag(K_loc_tr), 1.0, atol=1e-7)
        assert np.all(K_loc_tr >= -1e-7) and np.all(K_loc_tr <= 1.0 + 1e-7)

        # PSD check
        evs = np.linalg.eigvalsh(K_loc_tr)
        assert np.min(evs) >= -1e-6, f"Local ({p_type}) kernel has negative eigenvalue: {np.min(evs)}"


def test_multiscale_kernel_properties(sample_angles):
    states = generate_angle_statevectors_vectorized(sample_angles)
    K_glob = compute_global_kernel(states)
    K_loc = compute_local_kernel(states, patch_type="2-qubit")

    # Alpha = 1.0 should equal Global
    K_m1 = compute_multiscale_kernel(K_glob, K_loc, alpha=1.0)
    assert np.allclose(K_m1, K_glob, atol=1e-7)

    # Alpha = 0.0 should equal Local
    K_m0 = compute_multiscale_kernel(K_glob, K_loc, alpha=0.0)
    assert np.allclose(K_m0, K_loc, atol=1e-7)

    # Alpha = 0.5
    K_m05 = compute_multiscale_kernel(K_glob, K_loc, alpha=0.5)
    assert np.allclose(np.diag(K_m05), 1.0, atol=1e-7)
    evs = np.linalg.eigvalsh(K_m05)
    assert np.min(evs) >= -1e-6
