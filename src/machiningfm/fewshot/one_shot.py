from __future__ import annotations

from typing import Any

from .base import euclidean, support_embeddings


class OneShotPredictor:
    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(self, task: str, support_set: list[dict[str, Any]], query: dict[str, Any]) -> dict[str, Any]:
        embeddings, labels = support_embeddings(self.predictor, support_set[:1], task)
        if not labels:
            raise ValueError("One-shot prediction requires one labeled support sample")
        query_embedding = self.predictor.embed(query)
        distance = euclidean(query_embedding, embeddings[0])
        return {
            "mode": "one-shot",
            "task": task,
            "prediction": labels[0],
            "adaptation_method": "embedding_nearest_neighbor",
            "support_size": 1,
            "support_sample_influence": 1.0 / (distance + 1e-8),
            "uncertainty": None,
            "model_version": self.predictor.model_version,
        }
