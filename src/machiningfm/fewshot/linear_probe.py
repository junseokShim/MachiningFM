from __future__ import annotations

from typing import Any

from .base import is_regression


class LinearProbeAdapter:
    def __init__(self) -> None:
        self.model: Any = None
        self.regression = True

    def fit(self, embeddings: list[list[float]], labels: list[Any]) -> "LinearProbeAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        self.regression = is_regression(labels)
        if self.regression:
            from sklearn.linear_model import Ridge

            self.model = Ridge(alpha=1.0).fit(embeddings, labels)
        else:
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(max_iter=200).fit(embeddings, labels)
        return self

    def predict(self, embedding: list[float]) -> dict[str, Any]:
        prediction = self.model.predict([embedding])[0]
        if self.regression:
            prediction = float(prediction)
        result = {"prediction": prediction}
        if not self.regression and hasattr(self.model, "predict_proba"):
            result["class_probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(self.model.classes_, self.model.predict_proba([embedding])[0])
            }
        return result
