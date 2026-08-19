"""
Task A: Tool Wear Regression.

Pipeline:
    sensor_features → downstream_head → raw_vb
                                            ↓
                           Physics Calibration (optional)
                                            ↓
                                  final_vb_prediction (mm)

Evaluation: MAE, RMSE, R²
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from machiningfm.evaluation.metrics import regression_metrics

BACKBONE_MODES = ("frozen", "linear_probe", "partial_finetune", "full_finetune")


class ToolWearRegressor:
    """
    Tool Wear Regression: sensor features → VB (mm).

    The backbone_mode parameter controls how much of the foundation model
    is fine-tuned when used with a PyTorch backbone. For the sklearn baseline
    (no backbone), 'frozen' is always implicitly used.

    Physics calibration is applied post-hoc after the downstream head prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        backbone_mode: str = "frozen",
        calibrator=None,
        model_type: str = "ridge",
        ridge_alpha: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            feature_dim: Input feature dimension.
            hidden_dim: Hidden units for MLP model_type.
            backbone_mode: One of 'frozen', 'linear_probe', 'partial_finetune', 'full_finetune'.
            calibrator: PhysicsCalibrator instance (optional). Applied post-hoc.
            model_type: 'ridge' or 'mlp'.
            ridge_alpha: Regularization for Ridge regression.
            random_state: Seed for MLP.
        """
        if backbone_mode not in BACKBONE_MODES:
            raise ValueError(f"backbone_mode must be one of {BACKBONE_MODES}, got '{backbone_mode}'")
        self.feature_dim = feature_dim
        self.backbone_mode = backbone_mode
        self.calibrator = calibrator

        if model_type == "ridge":
            self._model = Ridge(alpha=ridge_alpha)
        elif model_type == "mlp":
            self._model = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
                max_iter=1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"model_type must be 'ridge' or 'mlp', got '{model_type}'")

        self._is_fitted = False

    def fit(
        self,
        features_train: np.ndarray,
        targets_train: np.ndarray,
        physics_features_train: list | None = None,
        features_val: np.ndarray | None = None,
        targets_val: np.ndarray | None = None,
        physics_features_val: list | None = None,
    ) -> dict[str, float]:
        """
        Train the regression head.

        Args:
            features_train: Shape (N_train, d). Training features.
            targets_train: Shape (N_train,). VB labels in mm. TRAIN ONLY.
            physics_features_train: Optional PhysicsFeatures for calibration fitting.
            features_val: Validation features for alpha selection.
            targets_val: Validation labels (VAL ONLY — never test set).
            physics_features_val: Validation physics features.

        Returns:
            Training metrics dict {'mae', 'rmse', 'r2'}.
        """
        self._model.fit(features_train, targets_train)
        self._is_fitted = True

        if self.calibrator is not None and physics_features_train is not None:
            raw_train_preds = self._model.predict(features_train)
            self.calibrator.fit(raw_train_preds, physics_features_train, targets_train)

            if features_val is not None and targets_val is not None and physics_features_val is not None:
                raw_val_preds = self._model.predict(features_val)
                self.calibrator.select_alpha(raw_val_preds, physics_features_val, targets_val)

        train_preds = self.predict(features_train, physics_features_train)
        return regression_metrics(targets_train, train_preds)

    def predict(
        self,
        features: np.ndarray,
        physics_features: list | None = None,
    ) -> np.ndarray:
        """
        Predict VB wear in mm.

        Args:
            features: Shape (N, d).
            physics_features: Optional list of PhysicsFeatures for calibration.

        Returns:
            Predicted VB values in mm, shape (N,).
        """
        if not self._is_fitted:
            raise RuntimeError("ToolWearRegressor must be fitted before calling predict().")
        raw_preds = self._model.predict(features)
        if self.calibrator is not None and physics_features is not None:
            return self.calibrator.predict(raw_preds, physics_features)
        return raw_preds

    def evaluate(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        physics_features: list | None = None,
    ) -> dict[str, float]:
        """Return {'mae', 'rmse', 'r2'} on the provided data."""
        preds = self.predict(features, physics_features)
        return regression_metrics(targets, preds)
