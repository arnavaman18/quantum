"""
Classification Evaluation Metrics:
- Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Confusion Matrix.
Tailored for imbalanced medical imaging benchmarks.
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 2,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        n_classes: Number of distinct classes

    Returns:
        Dict containing accuracy, balanced_accuracy, macro_f1, weighted_f1,
        precision, recall, and confusion_matrix.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    precision = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    recall = float(recall_score(y_t, y_p, average="macro", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=list(range(n_classes))).tolist()

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }
