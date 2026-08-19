from __future__ import annotations


def similarity_calibration(base_prediction: float, support_label: float, similarity: float) -> float:
    weight = min(1.0, max(0.0, similarity))
    return (1.0 - weight) * base_prediction + weight * support_label
