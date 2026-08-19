from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))


def time_axis(length: int, sample_rate: float | None, start: float = 0.0) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if sample_rate is None or sample_rate <= 0:
        return np.linspace(start, start + max(length - 1, 0), length, dtype=np.float32)
    return (start + np.arange(length, dtype=np.float32) / float(sample_rate)).astype(np.float32)


def range_mask(times: np.ndarray, window: TimeRange) -> np.ndarray:
    values = np.asarray(times, dtype=np.float32)
    return (values >= float(window.start)) & (values < float(window.end))


def overlapping_time_range(*axes: np.ndarray) -> TimeRange | None:
    valid = [np.asarray(axis, dtype=np.float32) for axis in axes if np.asarray(axis).size]
    if not valid:
        return None
    start = max(float(axis[0]) for axis in valid)
    end = min(float(axis[-1]) for axis in valid)
    if end <= start:
        return None
    return TimeRange(start, end)


def causal_cross_attention_mask(query_times: np.ndarray, key_times: np.ndarray) -> np.ndarray:
    query = np.asarray(query_times, dtype=np.float32).reshape(-1, 1)
    key = np.asarray(key_times, dtype=np.float32).reshape(1, -1)
    return key <= query
