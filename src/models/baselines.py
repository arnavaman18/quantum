"""
Classical SVM baselines (RBF and Linear) on the 8 PCA features
to provide rigorous benchmarking comparison against quantum kernels.
"""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def train_classical_svm_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel: str = "rbf",
    tune_C: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate classical SVM (RBF or Linear) directly on the 8 PCA features.

    Args:
        X_train: (N_train, 8)
        y_train: (N_train,)
        X_test: (N_test, 8)
        kernel: 'rbf' or 'linear'
        tune_C: whether to cross-validate C
        random_state: random seed

    Returns:
        Dict with 'model', 'best_C', 'train_predictions', 'test_predictions'
    """
    candidates = [0.01, 0.1, 1.0, 10.0, 50.0]
    best_c = 1.0

    if tune_C:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        best_score = -1.0
        for c in candidates:
            fold_scores = []
            for tr_idx, va_idx in skf.split(X_train, y_train):
                clf = SVC(C=c, kernel=kernel, random_state=random_state)
                clf.fit(X_train[tr_idx], y_train[tr_idx])
                preds = clf.predict(X_train[va_idx])
                fold_scores.append(balanced_accuracy_score(y_train[va_idx], preds))
            mean_sc = float(np.mean(fold_scores))
            if mean_sc > best_score:
                best_score = mean_sc
                best_c = c

    final_model = SVC(C=best_c, kernel=kernel, random_state=random_state)
    final_model.fit(X_train, y_train)

    train_preds = final_model.predict(X_train)
    test_preds = final_model.predict(X_test)

    return {
        "kernel": kernel,
        "best_C": best_c,
        "model": final_model,
        "train_preds": train_preds,
        "test_preds": test_preds,
    }
