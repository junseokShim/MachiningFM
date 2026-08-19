from __future__ import annotations

from collections import defaultdict
from typing import Any

from .base import euclidean, is_regression
from .knn import EmbeddingKNNAdapter


class PrototypicalAdapter:
    def __init__(self) -> None:
        self.prototypes: dict[Any, list[float]] = {}
        self.regression_fallback = EmbeddingKNNAdapter(neighbors=3)
        self.labels: list[Any] = []

    def fit(self, embeddings: list[list[float]], labels: list[Any]) -> "PrototypicalAdapter":
        self.labels = labels
        if is_regression(labels):
            self.regression_fallback.fit(embeddings, labels)
            return self
        groups: dict[Any, list[list[float]]] = defaultdict(list)
        for embedding, label in zip(embeddings, labels):
            groups[label].append(embedding)
        self.prototypes = {
            label: [sum(values) / len(values) for values in zip(*items)] for label, items in groups.items()
        }
        return self

    def predict(self, embedding: list[float]) -> dict[str, Any]:
        if is_regression(self.labels):
            return self.regression_fallback.predict(embedding)
        distances = {label: euclidean(embedding, prototype) for label, prototype in self.prototypes.items()}
        scores = {label: 1.0 / (distance + 1e-8) for label, distance in distances.items()}
        total = sum(scores.values())
        prediction = min(distances, key=distances.__getitem__)
        return {"prediction": prediction, "class_probabilities": {str(key): value / total for key, value in scores.items()}}
