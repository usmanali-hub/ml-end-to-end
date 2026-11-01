"""Evaluation metrics for an imbalanced binary problem (about 1 in 4 customers churns).

Accuracy alone is misleading here: predicting "nobody churns" is already 73% accurate. The primary
metric is therefore PR-AUC (average precision), which a model that cannot rank customers scores at
the churn rate. ROC-AUC is reported alongside because it is the more familiar number.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY = "pr_auc"
METRIC_NAMES = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def evaluate(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, float]:
    """All metrics for one set of predictions. ``scores`` rank customers; ``y_pred`` is the hard decision."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    p = np.asarray(y_pred)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }
