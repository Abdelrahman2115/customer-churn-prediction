"""Metrics, threshold selection and feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import RANDOM_STATE


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-free ranking metrics + threshold-dependent classification metrics."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }


def churners_captured_in_top(y_true, proba, fraction: float = 0.2) -> float:
    """Share of all churners found if we contact only the top `fraction` riskiest customers."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(len(proba) * fraction))
    top_idx = np.argsort(-np.asarray(proba))[:n_top]
    return y_true[top_idx].sum() / y_true.sum()


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximizing F1. Must be called on OUT-OF-FOLD training predictions,
    never on the test set, otherwise the test score is optimistically biased."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def get_confusion_matrix(y_true, proba, threshold: float) -> np.ndarray:
    return confusion_matrix(y_true, (proba >= threshold).astype(int))


def compute_permutation_importance(pipeline, X, y, n_repeats: int = 10) -> pd.DataFrame:
    """Importance of each ORIGINAL column (before one-hot), measured as the drop in
    ROC-AUC when that column is shuffled. Works for any model type."""
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
            index=X.columns,
        )
        .sort_values("importance_mean", ascending=False)
    )
