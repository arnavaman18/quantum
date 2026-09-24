"""
QSVM Classifier with Precomputed Kernels and Cross-Validated Hyperparameter Tuning.
"""
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


class QSVMClassifier:
    """
    Quantum Support Vector Machine using precomputed Gram matrices.
    Supports cross-validated C hyperparameter optimization.
    """

    def __init__(self, C: float = 1.0, probability: bool = False, random_state: int = 42):
        self.C = C
        self.probability = probability
        self.random_state = random_state
        self.model: Optional[SVC] = None
        self.best_C: float = C
        self.cv_results_: Dict[float, float] = {}

    def fit(
        self,
        K_train: np.ndarray,
        y_train: np.ndarray,
        tune_C: bool = True,
        C_candidates: Optional[List[float]] = None,
        cv_folds: int = 3,
    ) -> "QSVMClassifier":
        """
        Fit QSVM on precomputed training Gram matrix K_train (N_train, N_train).

        Args:
            K_train: Symmetric PSD matrix of shape (N, N)
            y_train: Labels of shape (N,)
            tune_C: If True, tunes C via Stratified K-Fold CV
            C_candidates: List of C values to evaluate (default: [0.01, 0.1, 1.0, 10.0, 100.0])
            cv_folds: Number of CV folds
        """
        y = np.asarray(y_train).ravel()

        if tune_C:
            candidates = C_candidates or [0.01, 0.1, 1.0, 10.0, 50.0]
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            best_score = -1.0
            best_c = candidates[0]
            self.cv_results_ = {}

            for c in candidates:
                fold_scores = []
                for train_idx, val_idx in skf.split(K_train, y):
                    # Sub-kernel for training
                    K_tr = K_train[train_idx][:, train_idx]
                    y_tr = y[train_idx]
                    # Sub-kernel for validation: (N_val, N_train_fold)
                    K_va = K_train[val_idx][:, train_idx]
                    y_va = y[val_idx]

                    fold_svc = SVC(C=c, kernel="precomputed", random_state=self.random_state)
                    fold_svc.fit(K_tr, y_tr)
                    preds = fold_svc.predict(K_va)
                    fold_scores.append(balanced_accuracy_score(y_va, preds))

                mean_score = float(np.mean(fold_scores))
                self.cv_results_[c] = mean_score
                if mean_score > best_score:
                    best_score = mean_score
                    best_c = c

            self.best_C = best_c
        else:
            self.best_C = self.C

        self.model = SVC(
            C=self.best_C,
            kernel="precomputed",
            probability=self.probability,
            random_state=self.random_state,
        )
        self.model.fit(K_train, y)
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """
        Predict labels for test samples given test-train Gram matrix (N_test, N_train).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return self.model.predict(K_test)

    def decision_function(self, K_test: np.ndarray) -> np.ndarray:
        """
        Decision function values for test samples.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling decision_function.")
        return self.model.decision_function(K_test)

    @property
    def n_support_(self) -> np.ndarray:
        """Number of support vectors per class."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.n_support_
