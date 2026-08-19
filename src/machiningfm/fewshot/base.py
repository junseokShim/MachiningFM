from __future__ import annotations

import math
from typing import Any, Protocol


TASK_LABELS = {
    "toolwear_regression": "tool_wear_vb",
    "rul_prediction": "rul",
    "wear_state_classification": "wear_state",
    "chatter_detection": "chatter",
    "surface_roughness_prediction": "surface_roughness",
    "quality_prediction": "quality_class",
    "anomaly_detection": "anomaly_state",
    "future_forecasting": "future_series",
}
TASK_LABEL_ALIASES = {
    "toolwear_regression": ("tool_wear_vb", "tool_wear", "wear", "vb", "value", "target", "label", "y"),
    "rul_prediction": ("rul", "remaining_useful_life", "value", "target", "label", "y"),
    "wear_state_classification": ("wear_state", "class", "state", "value", "target", "label", "y"),
    "chatter_detection": ("chatter", "chatter_state", "class", "value", "target", "label", "y"),
    "surface_roughness_prediction": ("surface_roughness", "roughness", "ra", "value", "target", "label", "y"),
    "quality_prediction": ("quality_class", "quality", "class", "value", "target", "label", "y"),
    "anomaly_detection": ("anomaly_state", "anomaly", "class", "value", "target", "label", "y"),
    "future_forecasting": ("future_series", "forecast", "target_series", "future", "value", "target", "label", "y"),
}


class Adapter(Protocol):
    def fit(self, embeddings: list[list[float]], labels: list[Any]) -> "Adapter": ...

    def predict(self, embedding: list[float]) -> dict[str, Any]: ...


def support_embeddings(predictor: Any, support_set: list[dict[str, Any]], task: str) -> tuple[list[list[float]], list[Any]]:
    embeddings: list[list[float]] = []
    labels: list[Any] = []
    for sample in support_set:
        value = support_label_value(sample, task)
        if value is None:
            continue
        embeddings.append(predictor.embed(sample))
        labels.append(value)
    return embeddings, labels


def expected_label_keys(task: str) -> tuple[str, ...]:
    return TASK_LABEL_ALIASES.get(task, (TASK_LABELS.get(task, task), "value", "target", "label", "y"))


def support_label_value(sample: dict[str, Any], task: str) -> Any | None:
    label = sample.get("label")
    if label is not None and not isinstance(label, dict):
        return label
    keys = expected_label_keys(task)
    if isinstance(label, dict):
        for key in keys:
            if key in label:
                return label[key]
        if len(label) == 1:
            return next(iter(label.values()))
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def euclidean(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def is_regression(labels: list[Any]) -> bool:
    return bool(labels) and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in labels)
