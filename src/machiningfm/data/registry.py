from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from machiningfm.utils.config import save_config
from machiningfm.utils.io import read_csv


def build_dataset_registry(inventory_path: str | Path) -> dict[str, Any]:
    rows = read_csv(inventory_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset_name"]].append(row)
    datasets = []
    for name, items in sorted(grouped.items()):
        modalities = _union_json(items, "available_modalities")
        tasks = _union_json(items, "downstream_task_candidates")
        known = _known_dataset_metadata(name)
        datasets.append(
            {
                "dataset_id": _slug(name),
                "dataset_name": name,
                "source_path": str(Path(items[0]["file_path"]).parent),
                "process_type": "machining",
                "machine_type": None,
                "tool_type": None,
                "workpiece_material": known.get("material"),
                "workpiece_material_family": known.get("material_family"),
                "material_source": known.get("material_source"),
                "material_confidence": known.get("material_confidence"),
                "known_sampling_rate": known.get("sampling_rate"),
                "license": None,
                "citation": None,
                "raw_available": any(_truth(row.get("raw_signal")) for row in items),
                "feature_only": any(_truth(row.get("feature_only")) for row in items),
                "synthetic_suspected": any(_truth(row.get("synthetic_suspected")) for row in items),
                "available_modalities": modalities,
                "missing_modalities": sorted(
                    set(["sensor_series", "process_condition", "image", "audio", "frequency"]) - set(modalities)
                ),
                "downstream_tasks": tasks,
                "file_count": len(items),
                "notes": "Automatically inferred; verify semantics, license, and split groups.",
            }
        )
    return {"schema_version": "1.0", "datasets": datasets}


def save_dataset_registry(inventory_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    registry = build_dataset_registry(inventory_path)
    save_config(output_path, registry)
    return registry


def _union_json(rows: list[dict[str, str]], key: str) -> list[str]:
    values: set[str] = set()
    for row in rows:
        try:
            value = json.loads(row.get(key) or "[]")
        except json.JSONDecodeError:
            value = []
        values.update(str(item) for item in value)
    return sorted(values)


def _truth(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _slug(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in safe.split("_") if part)


def _known_dataset_metadata(dataset_name: str) -> dict[str, Any]:
    name = dataset_name.lower()
    if "t4000" in name:
        return {
            "sampling_rate": 12800.0,
            "material": "aluminum",
            "material_family": "nonferrous_metal",
            "material_source": "user_defined",
            "material_confidence": "high",
        }
    if "edge_oti" in name or "edge oti" in name:
        return {
            "sampling_rate": 4000.0,
            "material": "S45C",
            "material_family": "carbon_steel",
            "material_source": "user_defined",
            "material_confidence": "high",
        }
    return {}
