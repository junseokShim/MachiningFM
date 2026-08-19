from __future__ import annotations

from collections import Counter
from typing import Any

from .base import euclidean, is_regression


class EmbeddingKNNAdapter:
    def __init__(self, neighbors: int = 3) -> None:
        self.neighbors = neighbors
        self.embeddings: list[list[float]] = []
        self.labels: list[Any] = []

    def fit(self, embeddings: list[list[float]], labels: list[Any]) -> "EmbeddingKNNAdapter":
        if not embeddings:
            raise ValueError("At least one labeled support sample is required")
        self.embeddings, self.labels = embeddings, labels
        return self

    def predict(self, embedding: list[float]) -> dict[str, Any]:
        ranked = sorted(
            [(euclidean(embedding, support), label) for support, label in zip(self.embeddings, self.labels)],
            key=lambda item: item[0],
        )[: max(1, min(self.neighbors, len(self.labels)))]
        if is_regression(self.labels):
            weights = [1.0 / (distance + 1e-8) for distance, _ in ranked]
            prediction = sum(weight * float(label) for weight, (_, label) in zip(weights, ranked)) / sum(weights)
            return {"prediction": prediction, "uncertainty": _range([float(label) for _, label in ranked])}
        counts = Counter(label for _, label in ranked)
        total = sum(counts.values())
        prediction = counts.most_common(1)[0][0]
        return {"prediction": prediction, "class_probabilities": {str(key): value / total for key, value in counts.items()}}


def _range(values: list[float]) -> float | None:
    return max(values) - min(values) if len(values) > 1 else None
