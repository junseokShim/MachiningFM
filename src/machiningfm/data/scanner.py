from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machiningfm.data.channel_schema import (
    CHANNEL_SCHEMA_VERSION,
    canonical_signal_names,
    describe_channel,
    find_recognized_header_index,
)
from machiningfm.utils.io import atomic_write_text, write_csv, write_json

LOGGER = logging.getLogger(__name__)

TABULAR = {".csv", ".xls", ".xlsx", ".parquet"}
IMAGE = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
AUDIO = {".wav", ".au", ".flac", ".mp3", ".aup"}
NUMPY = {".npy", ".npz"}
MATLAB = {".mat"}
HDF5 = {".h5", ".hdf5"}
JSON_EXT = {".json", ".jsonl"}
TEXT = {".txt", ".md", ".log"}

LABEL_ALIASES = {
    "tool_wear_vb": ("tool_wear", "wear", "vb"),
    "rul": ("rul", "remaining_useful_life"),
    "wear_state": ("wear_state", "wear_class"),
    "chatter": ("chatter",),
    "surface_roughness": ("roughness", "surface", "ra"),
    "dimension_error": ("dimension", "dim_error"),
    "quality_class": ("quality",),
    "anomaly_state": ("anomaly", "fault"),
    "energy_consumption": ("energy",),
}
CONDITION_ALIASES = {
    "spindle_speed": ("spindle_speed", "rpm", "speed"),
    "feed_rate": ("feed_rate", "feed"),
    "depth_of_cut": ("depth_of_cut", "doc", "ap"),
    "width_of_cut": ("width_of_cut", "woc", "ae"),
    "tool_diameter": ("tool_diameter", "diameter"),
    "material": ("material",),
    "operation_type": ("operation", "process"),
    "machine_id": ("machine_id", "machine"),
    "tool_id": ("tool_id", "tool"),
}


class FileInventoryScanner:
    """Recursively inventories files while only sampling file content."""

    def __init__(
        self,
        data_root: str | Path,
        inspect_per_group: int = 8,
        max_sample_bytes: int = 131_072,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.inspect_per_group = max(0, inspect_per_group)
        self.max_sample_bytes = max_sample_bytes

    def scan(self) -> list[dict[str, Any]]:
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        paths = [path for path in self.data_root.rglob("*") if path.is_file()]
        counts = Counter(self._dataset_name(path) for path in paths)
        sampled: Counter[tuple[str, str]] = Counter()
        records: list[dict[str, Any]] = []
        for index, path in enumerate(paths, start=1):
            dataset = self._dataset_name(path)
            extension = path.suffix.lower()
            key = (dataset, extension)
            should_inspect = sampled[key] < self.inspect_per_group
            if should_inspect:
                sampled[key] += 1
            records.append(self._record(path, counts[dataset], should_inspect))
            if index % 2000 == 0:
                LOGGER.info("Inventoried %s/%s files", index, len(paths))
        return records

    def _record(self, path: Path, file_count: int, inspect: bool) -> dict[str, Any]:
        relative = path.relative_to(self.data_root)
        stat_error = ""
        try:
            stat = path.stat()
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except OSError as exc:
            size, modified, stat_error = 0, "", str(exc)
        columns: list[str] = []
        details: dict[str, Any] = {}
        error = stat_error
        readable = os.access(path, os.R_OK)
        if inspect and readable:
            try:
                details = self._inspect(path, size)
                columns = details.pop("column_names", [])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        raw_channel_names = _channel_candidates(columns, details)
        haystack = " ".join([str(relative), *raw_channel_names]).lower()
        sensors = canonical_signal_names(raw_channel_names)
        sensor_quantities = sorted({describe_channel(name).quantity for name in sensors})
        labels = _matches(haystack, LABEL_ALIASES)
        conditions = _matches(haystack, CONDITION_ALIASES)
        file_type = classify_file(path)
        modalities = _modalities(file_type, sensors)
        tasks = _tasks(labels, haystack)
        raw_signal = bool(
            file_type in {"audio", "hdf5", "numpy"}
            or (file_type in {"tabular", "txt"} and sensors and size >= 100_000)
        )
        feature_only = bool(re.search(r"feature|statistic|summary|extract", haystack))
        synthetic = bool(re.search(r"synthetic|simulation|simulated|simulink", haystack))
        record: dict[str, Any] = {
            "dataset_name": self._dataset_name(path),
            "file_path": str(path),
            "relative_path": str(relative),
            "file_extension": path.suffix.lower(),
            "file_size": size,
            "modified_time": modified,
            "file_count": file_count,
            "directory_depth": max(0, len(relative.parts) - 1),
            "estimated_dataset_group": self._dataset_name(path),
            "readable": readable,
            "file_type": file_type,
            "inspection_mode": "sampled" if inspect else "metadata_only",
            "column_names": columns,
            "raw_channel_names": raw_channel_names,
            "canonical_channel_names": sensors,
            "channel_schema_version": CHANNEL_SCHEMA_VERSION,
            "sensor_channels": sensors,
            "sensor_quantities": sensor_quantities,
            "label_columns": labels,
            "process_condition_columns": conditions,
            "raw_signal": raw_signal,
            "feature_only": feature_only,
            "synthetic_suspected": synthetic,
            "available_modalities": modalities,
            "unavailable_modalities": sorted(
                set(["sensor_series", "process_condition", "image", "audio", "frequency"]) - set(modalities)
            ),
            "downstream_task_candidates": tasks,
            "error": error,
        }
        record.update(details)
        return record

    def _inspect(self, path: Path, size: int) -> dict[str, Any]:
        kind = classify_file(path)
        if kind in {"tabular", "txt"} and path.suffix.lower() in {".csv", ".txt"}:
            return self._inspect_delimited(path, size)
        if kind == "json":
            return self._inspect_json(path)
        if kind == "image":
            return self._inspect_image(path)
        if kind == "audio":
            return self._inspect_audio(path)
        if kind == "numpy":
            return self._inspect_numpy(path)
        if kind == "matlab":
            return self._inspect_matlab(path)
        if kind == "hdf5":
            return self._inspect_hdf5(path)
        if path.suffix.lower() in {".xlsx", ".xls", ".parquet"}:
            return self._inspect_dataframe(path)
        return {}

    def _inspect_delimited(self, path: Path, size: int) -> dict[str, Any]:
        raw = _read_sample(path, self.max_sample_bytes)
        text, encoding = _decode_sample(raw)
        lines = [line for line in text.splitlines() if line.strip()][:100]
        if not lines:
            return {"encoding": encoding, "row_count": 0}
        header_index = find_recognized_header_index(lines) or 0
        lines = lines[header_index:]
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters=",;\t|")
            rows = list(csv.reader(lines, dialect))
        except csv.Error:
            rows = [re.split(r"[,;\t]+", line.strip()) for line in lines]
        columns = [str(value).strip() for value in rows[0]]
        numeric_columns, categorical_columns, missing_ratio = _profile_rows(columns, rows[1:])
        row_count: int | None = None
        if size <= 5_000_000:
            with path.open("rb") as handle:
                row_count = max(0, sum(1 for _ in handle) - header_index - 1)
        sampling_rate = _estimate_sampling_rate(columns, rows[1:])
        return {
            "encoding": encoding,
            "column_names": columns,
            "row_count": row_count,
            "sampling_rate_estimate": sampling_rate,
            "has_time_column": any(_is_time_name(name) for name in columns),
            "missing_ratio": missing_ratio,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        }

    def _inspect_json(self, path: Path) -> dict[str, Any]:
        text, encoding = _decode_sample(_read_sample(path, self.max_sample_bytes))
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            first_line = text.splitlines()[0] if text.splitlines() else ""
            try:
                value = json.loads(first_line)
            except json.JSONDecodeError:
                keys = sorted(set(re.findall(r'"([^"]+)"\s*:', text)))[:100]
                return {
                    "encoding": encoding,
                    "column_names": keys,
                    "inspection_note": "Large or incomplete JSON inspected from a bounded sample",
                }
        if isinstance(value, dict):
            columns = list(value)[:100]
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            columns = list(value[0])[:100]
        else:
            columns = []
        return {"encoding": encoding, "column_names": columns}

    @staticmethod
    def _inspect_image(path: Path) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError:
            return {"image_size": None, "image_channels": None, "inspection_note": "Pillow not installed"}
        with Image.open(path) as image:
            return {"image_size": list(image.size), "image_channels": image.mode}

    @staticmethod
    def _inspect_audio(path: Path) -> dict[str, Any]:
        if path.suffix.lower() != ".wav":
            return {"audio_sampling_rate": None, "inspection_note": "Optional audio reader required"}
        import wave

        with wave.open(str(path), "rb") as audio:
            return {
                "audio_sampling_rate": audio.getframerate(),
                "audio_channels": audio.getnchannels(),
                "audio_frames": audio.getnframes(),
            }

    @staticmethod
    def _inspect_numpy(path: Path) -> dict[str, Any]:
        try:
            import numpy as np
        except ImportError:
            return {"numpy_shape": None, "inspection_note": "NumPy not installed"}
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if hasattr(value, "files"):
            return {"numpy_keys": list(value.files)}
        details = {"numpy_shape": list(value.shape), "numpy_dtype": str(value.dtype)}
        if value.ndim == 2 and min(value.shape) == 3 and re.search(r"(?:^|_)xyz$", path.stem.lower()):
            details["numpy_channel_names"] = [f"{path.stem}/vibration_{axis}" for axis in "xyz"]
        return details

    @staticmethod
    def _inspect_matlab(path: Path) -> dict[str, Any]:
        try:
            from scipy.io import whosmat
        except ImportError:
            return {"matlab_keys": None, "inspection_note": "SciPy not installed"}
        return {"matlab_keys": [name for name, _, _ in whosmat(path)]}

    @staticmethod
    def _inspect_hdf5(path: Path) -> dict[str, Any]:
        try:
            import h5py
        except ImportError:
            return {"hdf5_keys": None, "inspection_note": "h5py not installed"}
        keys: list[str] = []
        with h5py.File(path, "r") as handle:
            handle.visit(lambda name: keys.append(name) if len(keys) < 100 else None)
        return {"hdf5_keys": keys}

    @staticmethod
    def _inspect_dataframe(path: Path) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError:
            return {"inspection_note": "pandas and format reader not installed"}
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path).head(100)
        else:
            frame = pd.read_excel(path, nrows=100)
        return {
            "column_names": [str(column) for column in frame.columns],
            "row_count": None,
            "missing_ratio": float(frame.isna().mean().mean()) if not frame.empty else 0.0,
            "numeric_columns": [str(column) for column in frame.select_dtypes("number").columns],
            "categorical_columns": [str(column) for column in frame.select_dtypes(exclude="number").columns],
        }

    def _dataset_name(self, path: Path) -> str:
        relative = path.relative_to(self.data_root)
        return relative.parts[0] if len(relative.parts) > 1 else self.data_root.name


def classify_file(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in TABULAR:
        return "tabular"
    if extension in IMAGE:
        return "image"
    if extension in AUDIO:
        return "audio"
    if extension in NUMPY:
        return "numpy"
    if extension in MATLAB:
        return "matlab"
    if extension in HDF5:
        return "hdf5"
    if extension in JSON_EXT:
        return "json"
    if extension in TEXT:
        return "txt"
    if extension in {".pt", ".pth", ".bin"}:
        return "binary"
    return "unknown"


def save_inventory(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "data_inventory.csv", records)
    write_json(output / "data_inventory.json", records)
    atomic_write_text(output / "dataset_summary.md", build_summary(records))


def build_summary(records: list[dict[str, Any]]) -> str:
    dataset_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0, "types": Counter()})
    types = Counter()
    extensions = Counter()
    sensors = Counter()
    labels = Counter()
    conditions = Counter()
    errors = 0
    raw_datasets: set[str] = set()
    feature_datasets: set[str] = set()
    synthetic_datasets: set[str] = set()
    task_map: dict[str, set[str]] = defaultdict(set)
    for row in records:
        dataset = str(row["dataset_name"])
        size = int(row.get("file_size") or 0)
        kind = str(row.get("file_type", "unknown"))
        dataset_stats[dataset]["files"] += 1
        dataset_stats[dataset]["bytes"] += size
        dataset_stats[dataset]["types"][kind] += 1
        types[kind] += 1
        extensions[str(row.get("file_extension") or "(none)")] += 1
        sensors.update(row.get("sensor_channels") or [])
        labels.update(row.get("label_columns") or [])
        conditions.update(row.get("process_condition_columns") or [])
        if row.get("raw_signal"):
            raw_datasets.add(dataset)
        if row.get("feature_only"):
            feature_datasets.add(dataset)
        if row.get("synthetic_suspected"):
            synthetic_datasets.add(dataset)
        for task in row.get("downstream_task_candidates") or []:
            task_map[task].add(dataset)
        errors += bool(row.get("error"))
    total_bytes = sum(int(row.get("file_size") or 0) for row in records)
    lines = [
        "# MachiningFM Data Inventory Summary",
        "",
        f"- Files: {len(records):,}",
        f"- Total size: {total_bytes / (1024 ** 3):.2f} GiB",
        f"- Dataset groups: {len(dataset_stats)}",
        f"- Files with inspection errors: {errors}",
        "- Inspection policy: every file is inventoried; content is sampled per dataset/format.",
        "",
        "## Dataset Groups",
        "",
        "| Dataset | Files | GiB | Dominant types |",
        "|---|---:|---:|---|",
    ]
    for name, stats in sorted(dataset_stats.items(), key=lambda item: item[1]["bytes"], reverse=True):
        dominant = ", ".join(f"{key}:{value}" for key, value in stats["types"].most_common(4))
        lines.append(f"| {name} | {stats['files']:,} | {stats['bytes'] / (1024 ** 3):.2f} | {dominant} |")
    lines.extend(["", "## Format Coverage", "", _counter_table(extensions), "", "## Modality Classification", "", _counter_table(types)])
    lines.extend(["", "## Candidate Variables", "", f"- Sensors: {_counter_text(sensors)}", f"- Labels: {_counter_text(labels)}", f"- Process conditions: {_counter_text(conditions)}"])
    lines.extend(["", "## Corpus Assessment", "", f"- Raw-signal candidates: {', '.join(sorted(raw_datasets)) or 'none detected'}", f"- Feature-only candidates: {', '.join(sorted(feature_datasets)) or 'none detected'}", f"- Synthetic-suspected: {', '.join(sorted(synthetic_datasets)) or 'none detected'}"])
    lines.extend(["", "## Downstream Task Candidates", ""])
    for task, datasets in sorted(task_map.items()):
        lines.append(f"- {task}: {', '.join(sorted(datasets))}")
    lines.extend(
        [
            "",
            "## Scratch Pretraining Priority",
            "",
            "1. Large raw-signal HDF5/CSV/audio groups with identifiable sensor channels.",
            "2. Multimodal groups containing aligned images and sensor signals.",
            "3. Label-rich groups for downstream evaluation; do not leak labels into self-supervised objectives.",
            "4. Feature-only and synthetic-suspected groups as auxiliary corpora after manual review.",
            "",
            "## Missing-Variable and API Notes",
            "",
            "- Registry and manifest preserve available modalities and variables per sample.",
            "- API requests should include sensor names and available variables; absent modalities remain valid.",
            "- Sampling rate, unit consistency, label semantics, licenses, and alignment keys require human validation.",
            "",
            "## Known Quality Risks",
            "",
            "- Optional readers may be needed for HDF5, MATLAB, Excel, Parquet, image, and audio metadata.",
            "- Filename-based inference is conservative and must be confirmed before full pretraining.",
            "- Split groups must be reviewed to prevent tool, machine, and temporal leakage.",
        ]
    )
    return "\n".join(lines) + "\n"


def _channel_candidates(columns: list[str], details: dict[str, Any]) -> list[str]:
    numeric_columns = details.get("numeric_columns")
    names = list(numeric_columns) if isinstance(numeric_columns, list) else list(columns)
    for key in ("hdf5_keys", "numpy_keys", "numpy_channel_names", "matlab_keys"):
        value = details.get(key)
        if isinstance(value, list):
            names.extend(str(item) for item in value)
    return list(dict.fromkeys(name for name in names if name))


def _read_sample(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _decode_sample(raw: bytes) -> tuple[str, str]:
    encodings = ["utf-8-sig", "cp949"]
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.append("utf-16")
    encodings.append("latin-1")
    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _matches(haystack: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    found = []
    tokens = set(re.findall(r"[a-z0-9_]+", haystack.lower()))
    for canonical, candidates in aliases.items():
        if any(candidate in tokens or candidate in haystack for candidate in candidates):
            found.append(canonical)
    return found


def _profile_rows(columns: list[str], rows: list[list[str]]) -> tuple[list[str], list[str], float]:
    numeric: list[str] = []
    categorical: list[str] = []
    missing = 0
    cells = 0
    for index, column in enumerate(columns):
        values = [row[index].strip() for row in rows if index < len(row)]
        present = [value for value in values if value and value.lower() not in {"nan", "null", "none"}]
        missing += len(values) - len(present)
        cells += len(values)
        try:
            for value in present[:30]:
                float(value)
        except ValueError:
            categorical.append(column)
        else:
            numeric.append(column)
    return numeric, categorical, (missing / cells if cells else 0.0)


def _estimate_sampling_rate(columns: list[str], rows: list[list[str]]) -> float | None:
    indices = [index for index, name in enumerate(columns) if _is_time_name(name)]
    if not indices:
        return None
    values: list[float] = []
    for row in rows[:100]:
        try:
            values.append(float(row[indices[0]]))
        except (ValueError, IndexError):
            continue
    deltas = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not deltas:
        return None
    median = statistics.median(deltas)
    return (1.0 / median) if median > 0 and math.isfinite(median) else None


def _is_time_name(value: str) -> bool:
    name = value.lower()
    return any(token in name for token in ("time", "timestamp", "second", "sample_time"))


def _modalities(kind: str, sensors: list[str]) -> list[str]:
    result: set[str] = set()
    if sensors or kind in {"hdf5", "numpy", "audio"}:
        result.add("sensor_series")
    if kind == "image":
        result.add("image")
    if kind == "audio":
        result.add("audio")
    if kind in {"tabular", "txt"}:
        result.add("process_condition")
    return sorted(result)


def _tasks(labels: list[str], haystack: str) -> list[str]:
    mapping = {
        "tool_wear_vb": "toolwear_regression",
        "rul": "rul_prediction",
        "wear_state": "wear_state_classification",
        "chatter": "chatter_detection",
        "surface_roughness": "surface_roughness_prediction",
        "dimension_error": "quality_prediction",
        "quality_class": "quality_prediction",
        "anomaly_state": "anomaly_detection",
        "energy_consumption": "energy_prediction",
    }
    tasks = {mapping[label] for label in labels if label in mapping}
    if any(token in haystack for token in ("signal", "sensor", "time", "acc", "force")):
        tasks.add("future_forecasting")
    return sorted(tasks)


def _counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key} ({value})" for key, value in counter.most_common()) or "none detected"


def _counter_table(counter: Counter[str]) -> str:
    lines = ["| Type | Files |", "|---|---:|"]
    lines.extend(f"| {key} | {value:,} |" for key, value in counter.most_common())
    return "\n".join(lines)
