"""
Task B: Tool Wear Stage Classification.

Thresholds follow ISO 8688-1:1989, the standard for tool life testing in milling:
    Healthy  : VB < 0.1 mm
    Moderate : 0.1 mm ≤ VB < 0.2 mm
    Severe   : VB ≥ 0.2 mm

End-of-life criterion (for RUL purposes): VB_max = 0.3 mm

Reference:
    ISO 8688-1:1989, "Tool life testing in milling — Part 1: Face milling",
    International Organization for Standardization.

Evaluation: Accuracy, Macro F1, Confusion Matrix.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from machiningfm.evaluation.metrics import classification_metrics

# ISO 8688-1:1989 inspired thresholds for milling tool wear classification
WEAR_STAGE_THRESHOLDS_MM: dict[str, float] = {
    "healthy_max": 0.1,    # mm — VB below this is Healthy
    "moderate_max": 0.2,   # mm — VB below this is Moderate
}

WEAR_STAGES: list[str] = ["healthy", "moderate", "severe"]


def classify_wear_stage(wear_vb_mm: float) -> int:
    """
    Map flank wear VB (mm) to wear stage index.

    Returns:
        0 = healthy (VB < 0.1 mm)
        1 = moderate (0.1 ≤ VB < 0.2 mm)
        2 = severe (VB ≥ 0.2 mm)

    Thresholds source: ISO 8688-1:1989.
    """
    if wear_vb_mm < WEAR_STAGE_THRESHOLDS_MM["healthy_max"]:
        return 0
    if wear_vb_mm < WEAR_STAGE_THRESHOLDS_MM["moderate_max"]:
        return 1
    return 2


class WearStageClassifier:
    """
    Classify tool wear stage (healthy / moderate / severe) from sensor features.

    Evaluation: Accuracy, Macro F1, Confusion Matrix.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(
        self,
        features_train: np.ndarray,
        labels_train: np.ndarray,
        features_val: np.ndarray | None = None,
        labels_val: np.ndarray | None = None,
    ) -> dict:
        """
        Fit the classifier.

        Args:
            features_train: Shape (N, d).
            labels_train: Integer stage indices (0, 1, 2), shape (N,).

        Returns:
            Training metrics dict {'accuracy', 'macro_f1', 'confusion_matrix'}.
        """
        X_scaled = self._scaler.fit_transform(features_train)
        self._model.fit(X_scaled, labels_train)
        self._is_fitted = True
        train_preds = self.predict(features_train)
        return classification_metrics(labels_train, train_preds, WEAR_STAGES)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted stage indices, shape (N,)."""
        if not self._is_fitted:
            raise RuntimeError("WearStageClassifier must be fitted before predicting.")
        X_scaled = self._scaler.transform(features)
        return self._model.predict(X_scaled)

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Return {'accuracy', 'macro_f1', 'confusion_matrix'}."""
        return classification_metrics(labels, self.predict(features), WEAR_STAGES)
