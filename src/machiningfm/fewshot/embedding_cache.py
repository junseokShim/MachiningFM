from __future__ import annotations

import hashlib
import json
from typing import Any


class EmbeddingCache:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = {}

    def get_or_compute(self, sample: dict[str, Any], compute: Any) -> list[float]:
        key = hashlib.sha1(json.dumps(sample, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if key not in self._values:
            self._values[key] = compute(sample)
        return self._values[key]

    def clear(self) -> None:
        self._values.clear()
