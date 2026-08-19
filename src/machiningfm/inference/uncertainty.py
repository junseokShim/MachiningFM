from __future__ import annotations

import math
from typing import Iterable


def sample_standard_deviation(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    if len(data) < 2:
        return None
    mean = sum(data) / len(data)
    return math.sqrt(sum((value - mean) ** 2 for value in data) / (len(data) - 1))
