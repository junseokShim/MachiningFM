from __future__ import annotations

import random
from typing import Any, Iterable, Sequence


def build_sensor_mask(sensor_series: Any, channel_count: int | None = None) -> list[bool]:
    if sensor_series is None:
        return [False] * (channel_count or 0)
    if hasattr(sensor_series, "shape") and len(sensor_series.shape) >= 2:
        return [True] * int(sensor_series.shape[-2])
    return [True] * len(sensor_series)


def build_modality_mask(batch: dict[str, Any]) -> dict[str, bool]:
    return {
        "sensor_series": batch.get("sensor_series") is not None,
        "condition": batch.get("condition") is not None,
        "image": batch.get("image") is not None,
        "frequency": batch.get("frequency") is not None,
    }


def build_condition_mask(condition: Any, names: Sequence[str] | None = None) -> list[bool]:
    if condition is None:
        return [False] * len(names or [])
    if isinstance(condition, dict):
        return [condition.get(name) is not None for name in (names or condition.keys())]
    return [True] * len(condition)


def build_label_mask(labels: dict[str, Any] | None) -> dict[str, bool]:
    return {name: value is not None for name, value in (labels or {}).items()}


def apply_channel_dropout(values: Any, probability: float = 0.1, seed: int | None = None) -> tuple[Any, list[bool]]:
    rng = random.Random(seed)
    if values is None:
        return None, []
    mask = [rng.random() >= probability for _ in range(len(values))]
    output = [channel if keep else _zeros_like(channel) for channel, keep in zip(values, mask)]
    return output, mask


def apply_modality_dropout(batch: dict[str, Any], probability: float = 0.1, seed: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    output = dict(batch)
    for key in ("sensor_series", "condition", "image", "frequency"):
        if output.get(key) is not None and rng.random() < probability:
            output[key] = None
            output[f"{key}_mask"] = False
    return output


def apply_condition_dropout(condition: Any, probability: float = 0.1, seed: int | None = None) -> tuple[Any, list[bool]]:
    if condition is None:
        return None, []
    rng = random.Random(seed)
    if isinstance(condition, dict):
        mask = {key: rng.random() >= probability for key in condition}
        return {key: value if mask[key] else None for key, value in condition.items()}, list(mask.values())
    mask = [rng.random() >= probability for _ in condition]
    return [value if keep else 0 for value, keep in zip(condition, mask)], mask


def validate_available_inputs(batch: dict[str, Any]) -> tuple[bool, list[str]]:
    available = [name for name, present in build_modality_mask(batch).items() if present]
    warnings = []
    if not available:
        warnings.append("No supported input modality is available.")
    if batch.get("sensor_series") is None:
        warnings.append("sensor_series is missing")
    if batch.get("condition") is None and batch.get("process_condition") is None:
        warnings.append("process_condition is missing")
    return bool(available), warnings


def create_missing_variable_report(
    available_variables: Iterable[str] | None,
    expected_variables: Iterable[str] | None,
) -> dict[str, list[str]]:
    available = set(available_variables or [])
    expected = set(expected_variables or [])
    return {"available": sorted(available), "missing": sorted(expected - available)}


def _zeros_like(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
    except ImportError:
        pass
    if isinstance(value, list):
        return [0 for _ in value]
    if isinstance(value, tuple):
        return tuple(0 for _ in value)
    return 0
