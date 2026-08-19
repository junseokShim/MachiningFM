from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from machiningfm.utils.io import atomic_write_text, read_csv, write_json


def validate_inventory(inventory_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    rows = read_csv(inventory_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    datasets: list[dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        readable_ratio = sum(_truth(item.get("readable")) for item in items) / max(1, len(items))
        errors = sum(bool(item.get("error")) for item in items)
        raw = sum(_truth(item.get("raw_signal")) for item in items)
        labels = _union(items, "label_columns")
        sensors = _union(items, "sensor_channels")
        conditions = _union(items, "process_condition_columns")
        grade = _grade(readable_ratio, raw, sensors, errors, len(items))
        datasets.append(
            {
                "dataset_name": name,
                "grade": grade,
                "files": len(items),
                "readable_ratio": readable_ratio,
                "inspection_errors": errors,
                "raw_signal_files": raw,
                "sensor_candidates": sensors,
                "label_candidates": labels,
                "condition_candidates": conditions,
                "pretraining_candidate": grade in {"A", "B"} and raw > 0,
                "downstream_candidate": bool(labels),
                "manual_review_required": grade in {"D", "E"} or not sensors,
            }
        )
    result = {"datasets": datasets}
    output = Path(output_dir)
    write_json(output / "new_dataset_validation.json", result)
    lines = ["# New Dataset Validation", "", "| Dataset | Grade | Files | Raw files | Sensors | Labels |", "|---|---|---:|---:|---|---|"]
    for item in datasets:
        lines.append(
            f"| {item['dataset_name']} | {item['grade']} | {item['files']} | {item['raw_signal_files']} | "
            f"{', '.join(item['sensor_candidates']) or 'unknown'} | {', '.join(item['label_candidates']) or 'unknown'} |"
        )
    lines.extend(["", "Grades: A=pretraining ready, B=usable after preprocessing, C=auxiliary/downstream only, D=manual review, E=defer."])
    atomic_write_text(output / "new_dataset_validation.md", "\n".join(lines) + "\n")
    return result


def _union(rows: list[dict[str, str]], key: str) -> list[str]:
    values: set[str] = set()
    for row in rows:
        try:
            values.update(json.loads(row.get(key) or "[]"))
        except json.JSONDecodeError:
            continue
    return sorted(values)


def _truth(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _grade(readable_ratio: float, raw: int, sensors: list[str], errors: int, count: int) -> str:
    if readable_ratio < 0.5:
        return "E"
    if errors > count * 0.5:
        return "D"
    if raw and sensors:
        return "A"
    if raw:
        return "B"
    if sensors:
        return "C"
    return "D"
