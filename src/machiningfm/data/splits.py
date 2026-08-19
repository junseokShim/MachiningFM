from __future__ import annotations

import random
from typing import Any


def split_records(
    records: list[dict[str, Any]],
    strategy: str,
    holdout_value: Any | None = None,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = {
        "leave-one-tool-out": "tool_id",
        "leave-one-machine-out": "machine_id",
        "leave-one-condition-out": "process_condition",
        "leave-one-dataset-out": "dataset_id",
    }
    if strategy in fields:
        field = fields[strategy]
        values = [record.get(field) for record in records if record.get(field) is not None]
        selected = holdout_value if holdout_value is not None else (values[-1] if values else None)
        return (
            [record for record in records if record.get(field) != selected],
            [record for record in records if record.get(field) == selected],
        )
    if strategy == "chronological":
        ordered = sorted(records, key=lambda record: str(record.get("timestamp_start") or record.get("sample_id") or ""))
        cut = max(1, int(len(ordered) * (1 - validation_ratio)))
        return ordered[:cut], ordered[cut:]
    if strategy == "random":
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        cut = max(1, int(len(shuffled) * (1 - validation_ratio)))
        return shuffled[:cut], shuffled[cut:]
    raise ValueError(f"Unknown split strategy: {strategy}")
