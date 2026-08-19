from __future__ import annotations

import numpy as np


def monotonic_violation_rate(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    if data.size <= 1:
        return 0.0
    return float(np.mean(np.diff(data) < 0.0))
