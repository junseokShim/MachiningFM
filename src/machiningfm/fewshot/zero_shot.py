from __future__ import annotations

from typing import Any


class ZeroShotPredictor:
    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(self, task: str, query: dict[str, Any]) -> dict[str, Any]:
        return {"mode": "zero-shot", **self.predictor.predict(task, query, include_embedding=True)}
