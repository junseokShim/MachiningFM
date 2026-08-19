from __future__ import annotations

import math
from typing import Iterable


def fill_missing(values: Iterable[float | None], fill_value: float = 0.0) -> list[float]:
    return [fill_value if value is None or not math.isfinite(float(value)) else float(value) for value in values]


def zscore(values: Iterable[float], epsilon: float = 1e-8) -> list[float]:
    data = [float(value) for value in values]
    if not data:
        return []
    mean = sum(data) / len(data)
    variance = sum((value - mean) ** 2 for value in data) / len(data)
    scale = math.sqrt(variance) + epsilon
    return [(value - mean) / scale for value in data]


def minmax(values: Iterable[float], epsilon: float = 1e-8) -> list[float]:
    data = [float(value) for value in values]
    if not data:
        return []
    low, high = min(data), max(data)
    return [(value - low) / (high - low + epsilon) for value in data]


def robust_scale(values: Iterable[float], epsilon: float = 1e-8) -> list[float]:
    data = sorted(float(value) for value in values)
    if not data:
        return []
    q1 = data[int(0.25 * (len(data) - 1))]
    median = data[int(0.50 * (len(data) - 1))]
    q3 = data[int(0.75 * (len(data) - 1))]
    return [(value - median) / (q3 - q1 + epsilon) for value in data]


def resample_linear(values: Iterable[float], source_rate: float, target_rate: float) -> list[float]:
    data = [float(value) for value in values]
    if not data or source_rate <= 0 or target_rate <= 0:
        return data
    target_count = max(1, round(len(data) * target_rate / source_rate))
    if target_count == 1:
        return [data[0]]
    output = []
    for index in range(target_count):
        position = index * (len(data) - 1) / (target_count - 1)
        low = int(position)
        high = min(len(data) - 1, low + 1)
        fraction = position - low
        output.append(data[low] * (1 - fraction) + data[high] * fraction)
    return output


def map_variable_alias(name: str, aliases: dict[str, list[str]]) -> str:
    normalized = name.strip().lower()
    for canonical, candidates in aliases.items():
        if normalized == canonical or normalized in {candidate.lower() for candidate in candidates}:
            return canonical
    return normalized
