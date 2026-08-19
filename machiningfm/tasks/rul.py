"""
Task C: Remaining Useful Life (RUL) prediction.

RUL is the remaining machining time until tool wear exceeds the end-of-life threshold.
End-of-life criterion: VB_max = 0.3 mm (ISO 8688-1:1989 for milling).

Two approaches are compared:
  1. Taylor equation: RUL = max(0, T_Taylor - t_elapsed)
  2. MachiningFM-based: learned from sensor features

Evaluation: MAE, RMSE, R²
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from machiningfm.evaluation.metrics import regression_metrics

# ISO 8688-1:1989 end-of-life criterion for milling
ISO_WEAR_LIMIT_MM: float = 0.3


def compute_rul_from_wear(
    wear_vb_mm_series: np.ndarray,
    wear_limit_mm: float = ISO_WEAR_LIMIT_MM,
    time_per_cut_min: float = 1.0,
) -> np.ndarray:
    """
    Compute RUL (minutes) from a sequential VB wear series.

    RUL[i] = time from cut i until VB first exceeds wear_limit_mm.
    If the limit is never reached within the series, RUL is estimated
    as the remaining time after the last cut.

    Args:
        wear_vb_mm_series: Sequential VB measurements, shape (T,).
        wear_limit_mm: End-of-life VB threshold in mm. Default: 0.3 mm (ISO 8688-1).
        time_per_cut_min: Machining duration per cut in minutes.

    Returns:
        RUL array of shape (T,) in minutes.
    """
    n_cuts = len(wear_vb_mm_series)
    rul = np.zeros(n_cuts, dtype=np.float64)
    eol_indices = np.where(wear_vb_mm_series >= wear_limit_mm)[0]

    if len(eol_indices) == 0:
        for i in range(n_cuts):
            rul[i] = (n_cuts - i) * time_per_cut_min
    else:
        eol_cut = eol_indices[0]
        for i in range(n_cuts):
            rul[i] = max(0.0, (eol_cut - i) * time_per_cut_min)

    return rul


def compute_taylor_rul_min(
    elapsed_time_min: float,
    tool_life_taylor_min: float,
) -> float:
    """
    Compute RUL from Taylor equation.

    Equation: RUL = max(0, T_Taylor - t_elapsed)

    Args:
        elapsed_time_min: Elapsed cutting time in minutes. Must be >= 0.
        tool_life_taylor_min: Taylor-predicted total tool life in minutes.

    Returns:
        Remaining useful life in minutes.
    """
    if elapsed_time_min < 0:
        raise ValueError(f"elapsed_time_min must be >= 0, got {elapsed_time_min}")
    return max(0.0, tool_life_taylor_min - elapsed_time_min)


class RULPredictor:
    """
    Predict RUL (minutes) from sensor features using Ridge regression.

    Evaluation: MAE, RMSE, R².
    """

    def __init__(self, ridge_alpha: float = 1.0) -> None:
        self._model = Ridge(alpha=ridge_alpha)
        self._is_fitted = False

    def fit(
        self,
        features_train: np.ndarray,
        rul_train: np.ndarray,
        features_val: np.ndarray | None = None,
        rul_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        self._model.fit(features_train, rul_train)
        self._is_fitted = True
        return regression_metrics(rul_train, self._model.predict(features_train))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted RUL in minutes (clamped to >= 0), shape (N,)."""
        if not self._is_fitted:
            raise RuntimeError("RULPredictor must be fitted before predicting.")
        return np.maximum(0.0, self._model.predict(features))

    def evaluate(self, features: np.ndarray, rul_true: np.ndarray) -> dict[str, float]:
        """Return {'mae', 'rmse', 'r2'} on provided data."""
        return regression_metrics(rul_true, self.predict(features))
