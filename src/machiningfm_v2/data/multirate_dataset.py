from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from machiningfm_v2.tokenizers.cnc_sefc import normalize_cnc, sefc_group
from machiningfm_v2.tokenizers.nc_program import nc_blocks_to_array, parse_nc_program
from machiningfm_v2.tokenizers.spectral import build_spectral_features, resample_1d


HIGH_RATE_HINTS = ("vib", "vibration", "accel", "accraw", "acoustic", "ae", "force", "audio")
CNC_HINTS = ("focas", "rpm", "feed", "load", "pos", "position", "tool", "gcode", "mcode", "spindle", "servo")
WAVEFORM_CHANNEL_NAMES = ("x", "y", "z", "spindle_x", "vibration_x", "vibration_y", "vibration_z")
UNPAIRED_CNC_PATH_MARKERS = (
    "/cnc_data/",
    "/coordinate_data/",
    "/metadata/",
    "/02. machining_data/",
)


def discover_files(
    data_roots: list[str | Path],
    extensions: tuple[str, ...] = (".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5", ".parquet"),
) -> list[Path]:
    files: list[Path] = []
    for root in data_roots:
        root_path = Path(root)
        if root_path.is_file():
            files.append(root_path)
            continue
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    return sorted(dict.fromkeys(files))


def stable_id(path: Path, index: int = 0) -> str:
    return hashlib.sha1(f"{path.resolve()}::{index}".encode("utf-8", errors="ignore")).hexdigest()[:16]


def is_unpaired_cnc_path(path: Path) -> bool:
    normalized = "/" + str(path).replace("\\", "/").lower().strip("/") + "/"
    return any(marker in normalized for marker in UNPAIRED_CNC_PATH_MARKERS)


def load_numeric_table(path: Path, max_rows: int) -> tuple[np.ndarray, list[str]]:
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".dat.csv"):
        return _load_accraw_csv(path, max_rows)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5_table(path, max_rows)
    if suffix == ".parquet":
        frame = pd.read_parquet(path).head(max_rows)
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("parquet has no numeric columns")
        return numeric.to_numpy(dtype=np.float32).T, list(numeric.columns)
    if suffix == ".npy":
        data = np.load(path)
        if data.ndim == 1:
            data = data[None, :]
        return np.asarray(data, dtype=np.float32), [f"npy_{i}" for i in range(data.shape[0])]
    if suffix == ".npz":
        archive = np.load(path)
        rows = []
        names = []
        for key in archive.files:
            value = np.asarray(archive[key])
            if np.issubdtype(value.dtype, np.number):
                rows.append(value.reshape(-1)[:max_rows])
                names.append(key)
        if not rows:
            raise ValueError("npz has no numeric arrays")
        length = max(len(row) for row in rows)
        return np.stack([resample_1d(row, length) for row in rows]).astype(np.float32), names
    errors: list[str] = []
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin1"):
        for delimiter in ("comma", "auto", "whitespace"):
            kwargs: dict[str, Any] = {"nrows": max_rows, "encoding": encoding, "on_bad_lines": "skip"}
            if delimiter == "comma":
                kwargs["low_memory"] = False
            elif delimiter == "auto":
                kwargs.update({"sep": None, "engine": "python"})
            else:
                kwargs.update({"sep": r"\s+", "engine": "python"})
            try:
                frame = pd.read_csv(path, **kwargs)
                numeric = frame.select_dtypes(include=[np.number])
                if not numeric.empty:
                    return numeric.to_numpy(dtype=np.float32).T, list(numeric.columns)
                errors.append(f"{encoding}/{delimiter}: no numeric columns")
            except Exception as error:
                errors.append(f"{encoding}/{delimiter}: {error}")
    raise ValueError("table has no readable numeric columns: " + " | ".join(errors[-3:]))


def _load_accraw_csv(path: Path, max_rows: int) -> tuple[np.ndarray, list[str]]:
    header_index = None
    with path.open("r", encoding="cp949", errors="replace") as handle:
        for index, line in enumerate(handle):
            lowered = line.lower()
            if "time" in lowered and "acc" in lowered and "," in line:
                header_index = index
                break
            if index >= 64:
                break
    if header_index is None:
        raise ValueError("AccRaw header was not found")
    frame = pd.read_csv(
        path,
        encoding="cp949",
        skiprows=header_index,
        nrows=max_rows,
        on_bad_lines="skip",
        low_memory=False,
    )
    numeric = frame.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("AccRaw file has no numeric acceleration columns")
    values = numeric.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return values.to_numpy(dtype=np.float32).T, list(values.columns)


def _load_hdf5_table(path: Path, max_rows: int) -> tuple[np.ndarray, list[str]]:
    import h5py

    rows: list[np.ndarray] = []
    names: list[str] = []
    with h5py.File(path, "r") as handle:
        datasets: list[tuple[str, Any]] = []
        handle.visititems(
            lambda name, obj: datasets.append((name, obj))
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number)
            else None
        )
        for name, dataset in datasets:
            if dataset.ndim == 1:
                rows.append(np.asarray(dataset[:max_rows], dtype=np.float32))
                names.append(name)
            elif dataset.ndim == 2:
                if dataset.shape[0] >= dataset.shape[1]:
                    values = np.asarray(dataset[:max_rows, :], dtype=np.float32).T
                else:
                    values = np.asarray(dataset[:, :max_rows], dtype=np.float32)
                for channel, value in enumerate(values):
                    rows.append(value)
                    names.append(f"{name}_{channel}")
    if not rows:
        raise ValueError("hdf5 has no numeric 1D/2D datasets")
    length = min(max_rows, max(len(row) for row in rows))
    return np.stack([resample_1d(row, length) for row in rows]).astype(np.float32), names


def load_nc_tokens(path: Path, max_blocks: int = 512) -> np.ndarray | None:
    if path.suffix.lower() not in {".nc", ".mpf", ".txt"}:
        return None
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return None
    blocks = parse_nc_program(text, max_blocks=max_blocks)
    if not blocks:
        return None
    return nc_blocks_to_array(blocks, max_blocks=max_blocks)


def split_channels(
    values: np.ndarray,
    names: list[str],
    max_raw_channels: int,
    max_cnc_channels: int,
    source_path: str = "",
) -> tuple[np.ndarray | None, list[str], np.ndarray | None, list[str]]:
    source_lower = source_path.lower()
    raw_indices = [
        i
        for i, name in enumerate(names)
        if any(token in str(name).lower() for token in HIGH_RATE_HINTS) or str(name).lower() in WAVEFORM_CHANNEL_NAMES
    ]
    cnc_indices = [i for i, name in enumerate(names) if i not in raw_indices and any(token in str(name).lower() for token in CNC_HINTS)]
    source_is_high_rate = any(token in source_lower for token in HIGH_RATE_HINTS)
    source_is_cnc_only = any(token in source_lower for token in ("cnc_data", "focas", "coordinate_data", "metadata")) and not source_is_high_rate
    if not raw_indices and values.shape[0] and not source_is_cnc_only:
        raw_indices = list(range(min(3, values.shape[0])))
    if not cnc_indices:
        cnc_indices = [i for i in range(values.shape[0]) if i not in raw_indices]
    raw_indices = raw_indices[:max_raw_channels]
    cnc_indices = cnc_indices[:max_cnc_channels]
    raw = values[raw_indices] if raw_indices else None
    cnc = values[cnc_indices] if cnc_indices else None
    return raw, [names[i] for i in raw_indices], cnc, [names[i] for i in cnc_indices]


class MultiRateMachiningDataset(Dataset):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        roots = self.config.get("data_roots") or []
        if isinstance(roots, (str, Path)):
            roots = [roots]
        manifest = self.config.get("manifest_path")
        if manifest and Path(manifest).exists():
            frame = pd.read_parquet(manifest) if str(manifest).endswith(".parquet") else pd.read_csv(manifest)
            self.files = [Path(value) for value in frame["file_path"].dropna().astype(str)]
        else:
            extensions = tuple(str(value).lower() for value in self.config.get("file_extensions", (".csv", ".txt", ".npy", ".npz", ".h5", ".hdf5", ".parquet")))
            self.files = discover_files(roots, extensions=extensions)
        if not bool(self.config.get("include_non_vibration_parquet", False)):
            self.files = [path for path in self.files if path.suffix.lower() != ".parquet" or any(token in str(path).lower() for token in HIGH_RATE_HINTS)]
        if not bool(self.config.get("include_unpaired_cnc_files", False)):
            self.files = [path for path in self.files if not is_unpaired_cnc_path(path)]
        self.files = self.files[: int(self.config.get("max_files", len(self.files) or 1))]
        self.sequence_length = int(self.config.get("sequence_length", 4096))
        self.frequency_length = int(self.config.get("frequency_length", 512))
        self.forecast_horizons = tuple(int(v) for v in self.config.get("forecast_horizons", (64, 1280, 12800)))
        self.max_raw_channels = int(self.config.get("max_raw_channels", 32))
        self.max_cnc_channels = int(self.config.get("max_cnc_channels", 128))
        self.max_nc_blocks = int(self.config.get("max_nc_blocks", 512))
        self.transforms = tuple(self.config.get("frequency_transforms", ("fft", "stft", "cwt")))
        self.synthetic_if_empty = bool(self.config.get("synthetic_if_empty", True))
        self.synthetic_on_error = bool(self.config.get("synthetic_on_error", False))
        self.max_file_read_attempts = max(1, int(self.config.get("max_file_read_attempts", 64)))
        self.read_errors = 0
        self.invalid_files: set[Path] = set()

    def __len__(self) -> int:
        return max(1, len(self.files)) if self.synthetic_if_empty else len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.files:
            return self._synthetic(index)
        errors: list[str] = []
        attempts = min(self.max_file_read_attempts, len(self.files))
        stride = max(1, math.ceil(len(self.files) / attempts))
        while math.gcd(stride, len(self.files)) != 1:
            stride += 1
        for offset in range(attempts):
            path = self.files[(index + offset * stride) % len(self.files)]
            if path in self.invalid_files:
                continue
            try:
                return self._from_path(path, index)
            except Exception as error:
                self.read_errors += 1
                self.invalid_files.add(path)
                errors.append(f"{path}: {error}")
                if self.synthetic_on_error:
                    return self._synthetic(index, source_path=str(path))
        raise RuntimeError("Failed to load a real sample after retries: " + " | ".join(errors[-3:]))

    def _from_path(self, path: Path, index: int) -> dict[str, Any]:
        values, names = load_numeric_table(path, max_rows=max(self.sequence_length + max(self.forecast_horizons), 12800))
        raw, raw_names, cnc, cnc_names = split_channels(values, names, self.max_raw_channels, self.max_cnc_channels, str(path))
        if raw is None:
            raise ValueError("sample has no high-rate waveform channels")
        sample = self._make_sample(raw, raw_names, cnc, cnc_names, index=index, source_path=str(path), nc_tokens=None)
        return sample

    def _synthetic(self, index: int, source_path: str = "synthetic", nc_tokens: np.ndarray | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(index)
        length = self.sequence_length + max(self.forecast_horizons)
        t = np.linspace(0.0, 1.0, length, dtype=np.float32)
        rpm = 6000.0 + 500.0 * np.sin(2 * np.pi * t)
        raw = np.stack(
            [
                np.sin(2 * np.pi * (80 + i * 30) * t) * (0.2 + 0.8 * t) + 0.02 * rng.standard_normal(length)
                for i in range(3)
            ]
        ).astype(np.float32)
        cnc = np.stack([rpm, np.gradient(rpm), np.sin(2 * np.pi * t), t]).astype(np.float32)
        return self._make_sample(raw, ["vibration_x", "vibration_y", "vibration_z"], cnc, ["spindle_rpm", "rpm_delta", "feed", "position_x"], index=index, source_path=source_path, nc_tokens=nc_tokens)

    def _make_sample(
        self,
        raw: np.ndarray | None,
        raw_names: list[str],
        cnc: np.ndarray | None,
        cnc_names: list[str],
        *,
        index: int,
        source_path: str,
        nc_tokens: np.ndarray | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"sample_id": stable_id(Path(source_path), index), "source_path": source_path}
        if raw is not None:
            selected_raw = np.asarray(raw[: self.max_raw_channels], dtype=np.float32)
            if selected_raw.shape[-1] < self.sequence_length + min(self.forecast_horizons):
                raise ValueError(
                    f"waveform is too short: samples={selected_raw.shape[-1]} required>={self.sequence_length + min(self.forecast_horizons)}"
                )
            raw_context = selected_raw[:, : self.sequence_length]
            centers, scales = robust_channel_stats(
                raw_context,
                absolute_floor=float(self.config.get("target_scale_floor_abs", 1.0e-4)),
            )
            normalization_transform = str(self.config.get("normalization_transform", "asinh")).lower()
            normalized_context = normalize_channels(raw_context, centers, scales, normalization_transform)
            result["raw_waveform"] = normalized_context
            result["spectral"] = build_spectral_features(
                normalized_context,
                output_length=self.frequency_length,
                transforms=self.transforms,
                rpm=_rpm_from_cnc(cnc, cnc_names),
                sample_rate=float(self.config.get("default_high_rate_hz", 12800.0)),
            )
            target_centers, target_scales = pad_output_stats(centers, scales, channels=3)
            result["target_center"] = target_centers[:, None]
            result["target_scale"] = target_scales[:, None]
            result["target_transform_id"] = np.asarray(
                1.0 if normalization_transform == "asinh" else 0.0, dtype=np.float32
            )
            for horizon in self.forecast_horizons:
                if selected_raw.shape[-1] < self.sequence_length + horizon:
                    continue
                future = []
                for channel, row in enumerate(selected_raw[:3]):
                    values = row[self.sequence_length : self.sequence_length + horizon]
                    normalized = normalize_values(
                        values,
                        float(centers[channel]),
                        float(scales[channel]),
                        normalization_transform,
                    )
                    future.append(normalized)
                if future:
                    while len(future) < 3:
                        future.append(future[-1])
                    result[f"forecast_{horizon}"] = np.stack(future[:3]).astype(np.float32)
        if cnc is not None:
            cnc_context = np.stack([resample_1d(row[: self.sequence_length], min(self.sequence_length, 1024)) for row in cnc[: self.max_cnc_channels]]).astype(np.float32)
            result["cnc"] = normalize_cnc(cnc_context)
            group_to_id = {"setpoint": 0, "effort": 1, "feedback": 2, "context": 3}
            result["cnc_group_ids"] = np.asarray([group_to_id[sefc_group(name)] for name in cnc_names[: cnc_context.shape[0]]], dtype=np.int64)
        if nc_tokens is None:
            # Keep an explicit placeholder while avoiding hundreds of empty
            # attention tokens when a sample has no NC program.
            result["nc_tokens"] = np.zeros((1, 14), dtype=np.float32)
        else:
            result["nc_tokens"] = nc_tokens.astype(np.float32)
        metadata = np.zeros(int(self.config.get("metadata_dim", 32)), dtype=np.float32)
        if raw is not None:
            stat_count = min(3, len(centers), metadata.size // 2)
            metadata[:stat_count] = np.sign(centers[:stat_count]) * np.log1p(np.abs(centers[:stat_count]))
            metadata[stat_count : stat_count * 2] = np.log(np.maximum(scales[:stat_count], 1.0e-8))
        result["metadata_vector"] = metadata
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "files": len(self.files),
            "sequence_length": self.sequence_length,
            "frequency_transforms": list(self.transforms),
            "frequency_length": self.frequency_length,
            "forecast_horizons": list(self.forecast_horizons),
            "synthetic_if_empty": self.synthetic_if_empty,
            "synthetic_on_error": self.synthetic_on_error,
            "read_errors": self.read_errors,
            "quarantined_files": len(self.invalid_files),
        }


def robust_channel_stats(values: np.ndarray, absolute_floor: float = 1.0e-4) -> tuple[np.ndarray, np.ndarray]:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    centers = np.median(array, axis=-1).astype(np.float32)
    mad = np.median(np.abs(array - centers[:, None]), axis=-1).astype(np.float32) * 1.4826
    std = array.std(axis=-1).astype(np.float32)
    scales = np.where(mad > 1.0e-6, mad, std)
    scales = np.maximum(scales, max(1.0e-8, float(absolute_floor))).astype(np.float32)
    return centers, scales


def normalize_channels(
    values: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    transform: str = "asinh",
) -> np.ndarray:
    normalized = (np.asarray(values, dtype=np.float32) - centers[:, None]) / scales[:, None]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0e12, neginf=-1.0e12)
    if transform == "asinh":
        normalized = np.arcsinh(normalized)
    elif transform != "linear":
        raise ValueError(f"Unsupported normalization_transform: {transform}")
    return np.clip(normalized, -30.0, 30.0).astype(np.float32)


def normalize_values(values: np.ndarray, center: float, scale: float, transform: str = "asinh") -> np.ndarray:
    normalized = (np.asarray(values, dtype=np.float32) - float(center)) / max(float(scale), 1.0e-8)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0e12, neginf=-1.0e12)
    if transform == "asinh":
        normalized = np.arcsinh(normalized)
    elif transform != "linear":
        raise ValueError(f"Unsupported normalization_transform: {transform}")
    return np.clip(normalized, -30.0, 30.0).astype(np.float32)


def pad_output_stats(centers: np.ndarray, scales: np.ndarray, channels: int) -> tuple[np.ndarray, np.ndarray]:
    if centers.size == 0:
        return np.zeros(channels, dtype=np.float32), np.ones(channels, dtype=np.float32)
    center_values = list(centers[:channels])
    scale_values = list(scales[:channels])
    while len(center_values) < channels:
        center_values.append(center_values[-1])
        scale_values.append(scale_values[-1])
    return np.asarray(center_values, dtype=np.float32), np.asarray(scale_values, dtype=np.float32)


def _rpm_from_cnc(cnc: np.ndarray | None, names: list[str]) -> float | None:
    if cnc is None:
        return None
    for idx, name in enumerate(names):
        if "rpm" in str(name).lower() or "spindle" in str(name).lower():
            value = float(np.nanmedian(cnc[idx]))
            return value if np.isfinite(value) and value > 0 else None
    return None


def _tensor_or_none(samples: list[dict[str, Any]], key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
    values = [sample.get(key) for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    shape = np.asarray(values[0]).shape
    padded = []
    for value in values:
        array = np.asarray(value)
        if array.shape != shape:
            array = np.resize(array, shape)
        padded.append(array)
    return torch.tensor(np.stack(padded), dtype=dtype)


def multirate_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {"batch_size": len(samples), "sample_id": [sample.get("sample_id") for sample in samples]}
    for key in (
        "raw_waveform",
        "spectral",
        "cnc",
        "nc_tokens",
        "metadata_vector",
        "target_center",
        "target_scale",
        "target_transform_id",
    ):
        value = _tensor_or_none(samples, key)
        if value is not None:
            batch[key] = value
    group_ids = _tensor_or_none(samples, "cnc_group_ids", dtype=torch.long)
    if group_ids is not None:
        batch["cnc_group_ids"] = group_ids
    targets = {}
    for sample in samples:
        for key, value in sample.items():
            if key.startswith("forecast_"):
                targets.setdefault(key.replace("forecast_", ""), []).append(value)
    if targets:
        batch["targets"] = {key: torch.tensor(np.stack(values), dtype=torch.float32) for key, values in targets.items()}
    batch["metadata"] = [{"source_path": sample.get("source_path")} for sample in samples]
    return batch


def write_dataset_audit(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    dataset = MultiRateMachiningDataset(config)
    sample = dataset[0]
    report = {
        "dataset": dataset.summary(),
        "first_sample_shapes": {key: list(np.asarray(value).shape) for key, value in sample.items() if isinstance(value, np.ndarray)},
        "data_roots": [str(root) for root in config.get("data_roots", [])],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
