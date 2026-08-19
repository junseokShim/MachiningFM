from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def sliding_windows(values: Sequence[T], window_size: int, stride: int = 1) -> list[Sequence[T]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    return [values[start : start + window_size] for start in range(0, max(0, len(values) - window_size + 1), stride)]
