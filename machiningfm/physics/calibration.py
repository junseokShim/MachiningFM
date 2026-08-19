"""
Physics-based Post-hoc Calibration Layer.

Architecture:
    MachiningFM  →  Raw Prediction
                         ↓
    Physics Features  →  Calibration g(.)
                         ↓
    Final Prediction = y_FM + alpha * delta_y_physics

The calibration function g(.) is fitted on TRAIN data only.
Alpha (correction strength) is selected on VALIDATION data only.
Test data is NEVER used for any parameter selection.

Supported methods:
    'linear'       : OLS regression
    'ridge'        : Ridge regression (default — more robust than plain OLS)
    'mlp'          : Multi-layer perceptron
    'residual_mlp' : MLP that directly predicts the residual (y_true - y_FM)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class PhysicsFeatures:
    """
    Container for physics-derived features for a single machining sample.

    Fields set to None indicate quantities that were unavailable for this sample
    (e.g., Archard/Usui terms when the required measurements are absent).
    Only non-None fields are included in the feature vector for calibration.
    """

    tool_life_ratio: float | None = None
    """r_T = t / T_Taylor, in [0, 1]. Physics-derived wear proxy."""

    force_ratio: float | None = None
    """F_measured / F_Kienzle. Ratio > 1 suggests increased wear-related force."""

    force_delta_n: float | None = None
    """F_measured - F_Kienzle in Newtons. Signed deviation from ideal cutting."""

    cumulative_energy_j: float | None = None
    """E_c = integral(F_c * V_c * dt) in Joules. Cumulative work done."""

    specific_cutting_energy_j_per_mm3: float | None = None
    """u = P_c / MRR in J/mm³. Energy efficiency indicator."""

    wear_volume_mm3: float | None = None
    """Archard model wear volume (only if F_N and L are available in dataset)."""

    def to_feature_vector(self) -> np.ndarray:
        """Return all non-None features as a 1D numpy array."""
        ordered = [
            self.tool_life_ratio,
            self.force_ratio,
            self.force_delta_n,
            self.cumulative_energy_j,
            self.specific_cutting_energy_j_per_mm3,
            self.wear_volume_mm3,
        ]
        return np.array([v for v in ordered if v is not None], dtype=np.float64)

    def feature_names(self) -> list[str]:
        """Return names corresponding to non-None features."""
        pairs = [
            ("tool_life_ratio", self.tool_life_ratio),
            ("force_ratio", self.force_ratio),
            ("force_delta_n", self.force_delta_n),
            ("cumulative_energy_j", self.cumulative_energy_j),
            ("specific_cutting_energy_j_per_mm3", self.specific_cutting_energy_j_per_mm3),
            ("wear_volume_mm3", self.wear_volume_mm3),
        ]
        return [name for name, val in pairs if val is not None]


def _stack_physics_features(physics_list: list[PhysicsFeatures]) -> np.ndarray:
    """Stack a list of PhysicsFeatures into a 2D array of shape (N, d_physics)."""
    vectors = [pf.to_feature_vector() for pf in physics_list]
    if not vectors or all(len(v) == 0 for v in vectors):
        return np.zeros((len(physics_list), 0), dtype=np.float64)
    max_len = max(len(v) for v in vectors)
    padded = np.zeros((len(vectors), max_len), dtype=np.float64)
    for i, v in enumerate(vectors):
        padded[i, : len(v)] = v
    return padded


class PhysicsCalibrator:
    """
    Post-hoc calibration layer for MachiningFM predictions.

    Final prediction: y_final = (1-alpha) * y_FM + alpha * g(y_FM, physics_features)
    For 'residual_mlp': y_final = y_FM + alpha * g(y_FM, physics_features)

    Alpha selection MUST use validation data only.
    Never call select_alpha() or fit() with test set targets.
    """

    def __init__(
        self,
        method: str = "ridge",
        alpha_strength: float = 1.0,
        ridge_alpha: float = 1.0,
        mlp_hidden: tuple[int, ...] = (64, 32),
        random_state: int = 42,
    ) -> None:
        """
        Args:
            method: Calibration method — 'linear', 'ridge', 'mlp', 'residual_mlp'.
            alpha_strength: Blending coefficient alpha in [0, 1].
                            0 = pure FM prediction, 1 = full physics correction.
            ridge_alpha: Regularization strength for Ridge (ignored for other methods).
            mlp_hidden: Hidden layer sizes for MLP methods.
            random_state: Random seed for MLP.
        """
        valid_methods = ("linear", "ridge", "mlp", "residual_mlp")
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )
        self.method = method
        self.alpha_strength = alpha_strength
        self.ridge_alpha = ridge_alpha
        self.mlp_hidden = mlp_hidden
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model = self._build_model()
        self._is_fitted = False

    def _build_model(self):
        if self.method == "linear":
            return LinearRegression()
        if self.method == "ridge":
            return Ridge(alpha=self.ridge_alpha)
        if self.method in ("mlp", "residual_mlp"):
            return MLPRegressor(
                hidden_layer_sizes=self.mlp_hidden,
                max_iter=2000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        raise ValueError(f"Unknown method: {self.method}")

    def _build_design_matrix(
        self, fm_predictions: np.ndarray, physics_features: list[PhysicsFeatures]
    ) -> np.ndarray:
        phys = _stack_physics_features(physics_features)
        return np.column_stack([fm_predictions.reshape(-1, 1), phys])

    def fit(
        self,
        fm_predictions: np.ndarray,
        physics_features: list[PhysicsFeatures],
        targets: np.ndarray,
    ) -> "PhysicsCalibrator":
        """
        Fit calibration on training data.

        CRITICAL: targets must come from the TRAINING set only.
        Never pass test set labels to this method.

        Args:
            fm_predictions: Raw FM predictions on training set, shape (N,).
            physics_features: List of PhysicsFeatures, length N.
            targets: Ground truth labels on training set, shape (N,).

        Returns:
            self (for method chaining).
        """
        X = self._build_design_matrix(fm_predictions, physics_features)
        X_scaled = self._scaler.fit_transform(X)

        if self.method == "residual_mlp":
            residuals = targets - fm_predictions
            self._model.fit(X_scaled, residuals)
        else:
            self._model.fit(X_scaled, targets)

        self._is_fitted = True
        return self

    def predict(
        self,
        fm_predictions: np.ndarray,
        physics_features: list[PhysicsFeatures],
    ) -> np.ndarray:
        """
        Apply calibration to produce final predictions.

        For 'residual_mlp': y_final = y_FM + alpha * g(X)
        For others: y_final = alpha * g(X) + (1-alpha) * y_FM

        Args:
            fm_predictions: Raw FM predictions, shape (N,).
            physics_features: List of PhysicsFeatures, length N.

        Returns:
            Calibrated predictions, shape (N,).
        """
        if not self._is_fitted:
            raise RuntimeError("PhysicsCalibrator must be fitted before calling predict().")
        X = self._build_design_matrix(fm_predictions, physics_features)
        X_scaled = self._scaler.transform(X)
        raw_output = self._model.predict(X_scaled)

        if self.method == "residual_mlp":
            return fm_predictions + self.alpha_strength * raw_output
        return (1.0 - self.alpha_strength) * fm_predictions + self.alpha_strength * raw_output

    def select_alpha(
        self,
        fm_predictions: np.ndarray,
        physics_features: list[PhysicsFeatures],
        targets: np.ndarray,
        alpha_grid: list[float] | None = None,
    ) -> float:
        """
        Select optimal alpha_strength on VALIDATION data.

        MUST be called with validation set data only — never test set.
        Selection criterion: minimize MAE on validation predictions.

        Args:
            fm_predictions: FM predictions on validation set.
            physics_features: Physics features for validation set.
            targets: Ground truth labels for validation set (VAL ONLY).
            alpha_grid: Alpha candidates to search. Default: [0.0, 0.1, ..., 1.0].

        Returns:
            Best alpha value (also updates self.alpha_strength).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Fit the calibrator before calling select_alpha()."
            )
        if alpha_grid is None:
            alpha_grid = [round(a * 0.1, 1) for a in range(11)]

        best_alpha = 0.0
        best_mae = float("inf")

        for alpha in alpha_grid:
            self.alpha_strength = alpha
            preds = self.predict(fm_predictions, physics_features)
            mae = float(np.mean(np.abs(preds - targets)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        self.alpha_strength = best_alpha
        return best_alpha
