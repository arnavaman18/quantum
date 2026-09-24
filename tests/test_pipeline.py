"""
End-to-end integration and smoke test for the full pipeline.
"""
import numpy as np
import pytest
from src.data.preprocessor import QuantumPreprocessor
from src.quantum.circuits import generate_angle_statevectors_vectorized
from src.quantum.kernels import (
    compute_global_kernel,
    compute_local_kernel,
    compute_multiscale_kernel,
)
from src.models.qsvm import QSVMClassifier
from src.models.baselines import train_classical_svm_baseline
from src.metrics.spectral import compute_effective_rank
from src.metrics.alignment import compute_centered_kta
from src.metrics.concentration import compute_concentration_statistics
from src.metrics.classification import evaluate_classification_metrics


def test_full_pipeline_smoke():
    # 1. Synthetic classification problem: 60 train, 20 test, 50 features, 2 classes
    np.random.seed(42)
    X_train = np.random.randn(60, 50).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(20, 50).astype(np.float32)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    # 2. Preprocessing: PCA (8 components) + MinMaxScaler to [-pi, pi]
    preprocessor = QuantumPreprocessor(n_components=8)
    angles_tr = preprocessor.fit_transform(X_train)
    angles_te = preprocessor.transform(X_test)

    # 3. Quantum statevector generation
    states_tr = generate_angle_statevectors_vectorized(angles_tr)
    states_te = generate_angle_statevectors_vectorized(angles_te)

    # 4. Kernel computations
    K_glob_tr = compute_global_kernel(states_tr)
    K_glob_te = compute_global_kernel(states_tr, states_te)

    K_loc_tr = compute_local_kernel(states_tr, patch_type="2-qubit")
    K_loc_te = compute_local_kernel(states_tr, states_te, patch_type="2-qubit")

    K_multi_tr = compute_multiscale_kernel(K_glob_tr, K_loc_tr, alpha=0.5)
    K_multi_te = compute_multiscale_kernel(K_glob_te, K_loc_te, alpha=0.5)

    # 5. Deep metric extraction
    eff_rank = compute_effective_rank(K_multi_tr)["effective_rank"]
    kta = compute_centered_kta(K_multi_tr, y_train, n_classes=2)
    conc = compute_concentration_statistics(K_multi_tr)["offdiag_var"]

    assert 1.0 <= eff_rank <= 60.0
    assert -1.0 <= kta <= 1.0
    assert conc >= 0.0

    # 6. QSVM training and evaluation
    qsvm = QSVMClassifier()
    qsvm.fit(K_multi_tr, y_train, tune_C=False)
    preds = qsvm.predict(K_multi_te)
    metrics = evaluate_classification_metrics(y_test, preds, n_classes=2)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # 7. Classical baseline SVM
    classical = train_classical_svm_baseline(angles_tr, y_train, angles_te, kernel="rbf")
    assert len(classical["test_preds"]) == 20
