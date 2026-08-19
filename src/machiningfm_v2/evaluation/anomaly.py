from __future__ import annotations

import numpy as np


def standardized_residual_score(actual: np.ndarray, predicted_mean: np.ndarray, predicted_std: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(actual) - np.asarray(predicted_mean)) / np.maximum(np.asarray(predicted_std), 1.0e-6)


def conformal_threshold(calibration_scores: np.ndarray, coverage: float = 0.95) -> float:
    scores = np.asarray(calibration_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return float("inf")
    q = min(1.0, max(0.0, float(coverage)))
    return float(np.quantile(scores, q))
