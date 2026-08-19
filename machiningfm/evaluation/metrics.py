"""Evaluation metrics for machining downstream tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard regression metrics for wear prediction.

    Args:
        y_true: Ground truth values, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dict with 'mae' (mm), 'rmse' (mm), 'r2' (dimensionless).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute classification metrics for wear stage prediction.

    Args:
        y_true: Ground truth class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).
        class_names: Optional list of class names for labeling the confusion matrix.

    Returns:
        Dict with 'accuracy', 'macro_f1', 'confusion_matrix' (list of lists),
        and optionally 'class_names'.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if class_names is not None:
        result["class_names"] = class_names
    return result
