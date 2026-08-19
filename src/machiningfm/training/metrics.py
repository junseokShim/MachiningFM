from __future__ import annotations

import math
from typing import Iterable


def regression_metrics(targets: Iterable[float], predictions: Iterable[float]) -> dict[str, float]:
    y = [float(value) for value in targets]
    p = [float(value) for value in predictions]
    if not y or len(y) != len(p):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "smape": math.nan}
    mae = sum(abs(a - b) for a, b in zip(y, p)) / len(y)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))
    mean = sum(y) / len(y)
    denominator = sum((value - mean) ** 2 for value in y)
    r2 = 1 - sum((a - b) ** 2 for a, b in zip(y, p)) / denominator if denominator else 0.0
    smape = sum(2 * abs(a - b) / (abs(a) + abs(b) + 1e-12) for a, b in zip(y, p)) / len(y)
    return {"mae": mae, "rmse": rmse, "r2": r2, "smape": smape}
