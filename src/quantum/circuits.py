"""
Quantum circuits for 8-qubit Angle Encoding and Entangled Feature Maps.
Provides both high-performance vectorized NumPy statevector generation
and PennyLane QNode circuit implementations.
"""
from typing import Optional
import numpy as np


def generate_angle_statevectors_vectorized(angles: np.ndarray) -> np.ndarray:
    """
    Generate exact 8-qubit statevectors for N input angle vectors using vectorized tensor product.
    Embedding: |psi(theta)> = (x)_{j=0}^7 [cos(theta_j/2)|0> + sin(theta_j/2)|1>]

    Args:
        angles: Array of shape (N, 8) with values in [-pi, pi]

    Returns:
        states: Complex array of shape (N, 256) where 256 = 2^8
    """
    N, n_qubits = angles.shape
    assert n_qubits == 8, f"Expected 8 qubits, got {n_qubits}"

    half_angles = angles * 0.5
    cos_vals = np.cos(half_angles)  # shape (N, 8)
    sin_vals = np.sin(half_angles)  # shape (N, 8)

    # Construct single-qubit states (N, 8, 2)
    # qubit_states[:, j, 0] = cos, qubit_states[:, j, 1] = sin
    qubit_states = np.stack([cos_vals, sin_vals], axis=-1).astype(np.complex128)

    # Tensor product of 8 qubits along row axis using einsum
    # q0: (N, 2), q1: (N, 2), ..., q7: (N, 2)
    psi = np.einsum(
        "ia,ib,ic,id,ie,if,ig,ih->iabcdefgh",
        qubit_states[:, 0, :],
        qubit_states[:, 1, :],
        qubit_states[:, 2, :],
        qubit_states[:, 3, :],
        qubit_states[:, 4, :],
        qubit_states[:, 5, :],
        qubit_states[:, 6, :],
        qubit_states[:, 7, :],
        optimize=True,
    )

    # Reshape from (N, 2, 2, 2, 2, 2, 2, 2, 2) to (N, 256)
    return psi.reshape(N, 256)


def generate_entangled_statevectors_vectorized(angles: np.ndarray) -> np.ndarray:
    """
    Generate 8-qubit statevectors with an Angle Encoding layer followed by a circular CNOT ring:
    CNOT(0,1), CNOT(1,2), CNOT(2,3), CNOT(3,4), CNOT(4,5), CNOT(5,6), CNOT(6,7), CNOT(7,0).

    Args:
        angles: Array of shape (N, 8)

    Returns:
        states: Complex array of shape (N, 256)
    """
    N, _ = angles.shape
    # Start with unentangled state tensor (N, 2, 2, 2, 2, 2, 2, 2, 2)
    half_angles = angles * 0.5
    cos_vals = np.cos(half_angles)
    sin_vals = np.sin(half_angles)
    qubit_states = np.stack([cos_vals, sin_vals], axis=-1).astype(np.complex128)

    psi = np.einsum(
        "ia,ib,ic,id,ie,if,ig,ih->iabcdefgh",
        qubit_states[:, 0, :],
        qubit_states[:, 1, :],
        qubit_states[:, 2, :],
        qubit_states[:, 3, :],
        qubit_states[:, 4, :],
        qubit_states[:, 5, :],
        qubit_states[:, 6, :],
        qubit_states[:, 7, :],
        optimize=True,
    )

    # Apply circular CNOT ring
    # CNOT on basis states: target index flips if control index == 1
    # Pairs: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)
    cnot_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]

    for ctrl, tgt in cnot_pairs:
        # ctrl and tgt are axes 1..8
        ctrl_axis = ctrl + 1
        tgt_axis = tgt + 1
        # When ctrl == 1, swap tgt=0 and tgt=1
        # Use slicing along axes
        sl_ctrl1 = [slice(None)] * 9
        sl_ctrl1[ctrl_axis] = 1

        sl_ctrl1_tgt0 = list(sl_ctrl1)
        sl_ctrl1_tgt0[tgt_axis] = 0

        sl_ctrl1_tgt1 = list(sl_ctrl1)
        sl_ctrl1_tgt1[tgt_axis] = 1

        # Swap target amplitudes where control is 1
        val0 = psi[tuple(sl_ctrl1_tgt0)].copy()
        val1 = psi[tuple(sl_ctrl1_tgt1)].copy()
        psi[tuple(sl_ctrl1_tgt0)] = val1
        psi[tuple(sl_ctrl1_tgt1)] = val0

    return psi.reshape(N, 256)


def get_pennylane_qnode(n_qubits: int = 8, entangled: bool = False):
    """
    Construct a PennyLane QNode for statevector verification.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        # Angle encoding with RY
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        # Optional circular entanglement
        if entangled:
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.state()

    return circuit
