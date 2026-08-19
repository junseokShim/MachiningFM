from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from machiningfm.data.channel_schema import CHANNEL_SCHEMA_VERSION
from machiningfm.data.scanner import FileInventoryScanner
from machiningfm.utils.config import load_config
from machiningfm.utils.io import read_csv, write_records


def build_manifest(
    data_root: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    inventory_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    registry = load_config(registry_path)
    by_name = {item["dataset_name"]: item for item in registry.get("datasets", [])}
    if inventory_path and Path(inventory_path).exists():
        inventory = read_csv(inventory_path)
    else:
        inventory = FileInventoryScanner(data_root, inspect_per_group=0).scan()
    rows = []
    for item in inventory:
        dataset_name = str(item["dataset_name"])
        dataset = by_name.get(dataset_name, {})
        available = _json_value(item.get("available_modalities"), [])
        sensors = _json_value(item.get("sensor_channels"), [])
        sensor_quantities = _json_value(item.get("sensor_quantities"), [])
        labels = _json_value(item.get("label_columns"), [])
        relative = str(item["relative_path"])
        known = _known_dataset_metadata(dataset_name, relative)
        sampling_rate = _sampling_rate(item, known)
        material = _material_metadata(item, dataset, known)
        file_type = item.get("file_type")
        modality = available
        split_key = _split_group_key(dataset_name, item)
        subgraphs = _subgraph_candidates(item, material)
        rows.append(
            {
                "sample_id": hashlib.sha1(f"{dataset_name}:{relative}".encode("utf-8")).hexdigest()[:20],
                "dataset_id": dataset.get("dataset_id", dataset_name),
                "dataset_group": dataset_name,
                "machine_id": None,
                "tool_id": None,
                "run_id": None,
                "workpiece_id": None,
                "experiment_id": None,
                "operation_id": None,
                "cut_id": None,
                "cycle_id": None,
                "date_or_batch_id": None,
                "timestamp_start": None,
                "timestamp_end": None,
                "sampling_rate": sampling_rate,
                "estimated_sampling_rate": sampling_rate,
                "file_path": item["file_path"],
                "file_type": file_type,
                "file_size": item.get("file_size"),
                "modality": modality,
                "channel_schema_version": CHANNEL_SCHEMA_VERSION,
                "sensor_type": sensor_quantities,
                "channel_names": sensors,
                "sensor_channels": sensors,
                "cnc_channels": _cnc_channels(sensors, _json_value(item.get("process_condition_columns"), [])),
                "raw_channel_names": _json_value(item.get("raw_channel_names"), []),
                "available_variables": sorted(
                    set(sensor_quantities + _json_value(item.get("process_condition_columns"), []))
                ),
                "missing_variables": [],
                "process_condition": {},
                "label": {name: None for name in labels},
                "split_group": split_key,
                "source_type": "synthetic_suspected" if _truth(item.get("synthetic_suspected")) else "raw",
                "split_group_key_candidate": split_key,
                "material": material["material"],
                "material_family": material["material_family"],
                "material_source": material["material_source"],
                "material_confidence": material["material_confidence"],
                "has_high_sampling_timeseries": _has_high_sampling_timeseries(item, sampling_rate),
                "has_cnc_timeseries": _has_cnc_timeseries(item),
                "has_image": "image" in modality,
                "has_audio": "audio" in modality,
                "has_frequency_view": _has_frequency_view(item),
                "has_time_frequency_view": _has_time_frequency_view(item),
                "subgraph_candidates": subgraphs,
                "dynamic_graph_enabled": bool(sensors or item.get("raw_signal")),
                "read_status": "inspect_error" if item.get("error") else ("readable" if _truth(item.get("readable")) else "not_readable"),
                "read_error_if_any": item.get("error") or "",
                "subgraph_id": None,
                "subgraph_name": None,
                "subgraph_type": None,
                "subgraph_channels": [],
                "subgraph_views": [],
                "subgraph_sampling_rates": [],
                "subgraph_material_context": material,
                "subgraph_construction_method": "manifest_candidate",
                "subgraph_confidence": "low" if not subgraphs else "medium",
                "stemgnn_dynamic_graph_enabled": bool(sensors or item.get("raw_signal")),
                "stemgnn_dynamic_graph_source": "runtime_adaptive_graph_learning",
                "learned_adjacency_available": False,
                "missing_info": {
                    "sensor_mask": None,
                    "modality_mask": None,
                    "condition_mask": None,
                    "label_mask": {name: False for name in labels},
                    "available_variable_mask": None,
                },
            }
        )
    storage = write_records(output_path, rows)
    return rows, storage


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _known_dataset_metadata(dataset_name: str, relative: str) -> dict[str, Any]:
    haystack = f"{dataset_name} {relative}".lower()
    if "t4000" in haystack:
        return {
            "sampling_rate": 12800.0,
            "material": "aluminum",
            "material_family": "nonferrous_metal",
            "material_source": "user_defined",
            "material_confidence": "high",
        }
    if "edge_oti" in haystack or "edge oti" in haystack:
        return {
            "sampling_rate": 4000.0,
            "material": "S45C",
            "material_family": "carbon_steel",
            "material_source": "user_defined",
            "material_confidence": "high",
        }
    return {}


def _sampling_rate(item: dict[str, Any], known: dict[str, Any]) -> float | None:
    if known.get("sampling_rate"):
        return float(known["sampling_rate"])
    for key in ("sampling_rate_estimate", "audio_sampling_rate"):
        value = item.get(key)
        if value in (None, "", "unknown"):
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    return None


def _material_metadata(item: dict[str, Any], dataset: dict[str, Any], known: dict[str, Any]) -> dict[str, str]:
    if known.get("material"):
        return {
            "material": str(known["material"]),
            "material_family": str(known["material_family"]),
            "material_source": str(known["material_source"]),
            "material_confidence": str(known["material_confidence"]),
        }
    configured = dataset.get("workpiece_material")
    if configured:
        return {
            "material": str(configured),
            "material_family": "unknown",
            "material_source": "registry",
            "material_confidence": "medium",
        }
    conditions = " ".join(_json_value(item.get("process_condition_columns"), []))
    if "material" in conditions.lower():
        return {
            "material": "unknown",
            "material_family": "unknown",
            "material_source": "column_metadata",
            "material_confidence": "low",
        }
    return {
        "material": "unknown",
        "material_family": "unknown",
        "material_source": "unavailable",
        "material_confidence": "low",
    }


def _split_group_key(dataset_name: str, item: dict[str, Any]) -> str:
    path = Path(str(item.get("relative_path") or item.get("file_path") or ""))
    pieces = [dataset_name]
    for part in path.parts[:-1][-3:]:
        if part:
            pieces.append(part)
    pieces.append(path.stem)
    return "::".join(pieces)


def _cnc_channels(sensor_channels: list[str], condition_columns: list[str]) -> list[str]:
    cnc_terms = ("rpm", "feed", "position", "load", "servo", "gcode", "tool", "axis", "spindle")
    names = [*sensor_channels, *condition_columns]
    return [name for name in names if any(term in str(name).lower() for term in cnc_terms)]


def _has_high_sampling_timeseries(item: dict[str, Any], sampling_rate: float | None) -> bool:
    if sampling_rate and sampling_rate >= 1000.0:
        return True
    haystack = " ".join(
        [
            str(item.get("relative_path", "")),
            " ".join(_json_value(item.get("sensor_quantities"), [])),
            " ".join(_json_value(item.get("sensor_channels"), [])),
        ]
    ).lower()
    return any(token in haystack for token in ("vibration", "acceler", "acoustic", "waveform", "hf", "high_frequency"))


def _has_cnc_timeseries(item: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            str(item.get("relative_path", "")),
            " ".join(_json_value(item.get("process_condition_columns"), [])),
            " ".join(_json_value(item.get("sensor_channels"), [])),
        ]
    ).lower()
    return any(token in haystack for token in ("cnc", "focas", "gcode", "rpm", "feed", "axis", "spindle", "servo"))


def _has_frequency_view(item: dict[str, Any]) -> bool:
    haystack = str(item.get("relative_path", "")).lower()
    return any(token in haystack for token in ("fft", "spectrum", "spectral", "frequency"))


def _has_time_frequency_view(item: dict[str, Any]) -> bool:
    haystack = str(item.get("relative_path", "")).lower()
    return any(token in haystack for token in ("stft", "scalogram", "wavelet", "spectrogram", "time_frequency"))


def _subgraph_candidates(item: dict[str, Any], material: dict[str, str]) -> list[str]:
    haystack = " ".join(
        [
            str(item.get("relative_path", "")),
            " ".join(_json_value(item.get("sensor_channels"), [])),
            " ".join(_json_value(item.get("process_condition_columns"), [])),
        ]
    ).lower()
    candidates = []
    mapping = {
        "spindle_dynamics": ("spindle", "rpm", "chatter"),
        "feed_drive": ("axis", "feed", "servo", "position", "current", "load"),
        "cutting_condition": ("feedrate", "spindle_speed", "tool", "depth", "width", "material"),
        "cnc_operation_state": ("gcode", "focas", "operation", "tool_num", "tool_call"),
        "high_frequency_vibration": ("vibration", "acceler", "acoustic", "fft", "spectrum"),
    }
    for name, terms in mapping.items():
        if any(term in haystack for term in terms):
            candidates.append(name)
    if material.get("material") != "unknown":
        candidates.append("material_process_context")
    return sorted(set(candidates))


def _truth(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes"}
