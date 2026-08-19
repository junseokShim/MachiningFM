from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from machiningfm.data.augmentation import apply_frequency_augmentation, apply_physical_noise
from machiningfm.data.channel_schema import (
    CHANNEL_ATTRIBUTE_NAMES,
    CHANNEL_SCHEMA_VERSION,
    describe_channel,
    encode_channel_names,
    find_recognized_header_index,
    is_metadata_channel_name,
    is_pretraining_signal_channel,
)
from machiningfm.data.frequency_views import (
    build_frequency_view,
    finite_vector,
    normalize_frequency_vector,
    resample_vector,
    stft_spectrogram_summary,
    wavelet_scalogram_summary,
)
from machiningfm.data.latent_context import LatentContextModel, heuristic_latent_tokens
from machiningfm.data.nc_code import NC_CONTEXT_DIM, NCContextStore
from machiningfm.data.text_context import (
    DEFAULT_MAX_TEXT_TOKENS,
    DEFAULT_TEXT_VOCAB_SIZE,
    build_pretraining_text_context,
    encode_text_context,
)
from machiningfm.data.virtual_vibration import (
    VirtualVibrationConfig,
    append_virtual_spindle_vibration,
)
from machiningfm.utils.io import read_records

SUPPORTED_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".csv",
    ".txt",
    ".log",
    ".parquet",
    ".npy",
    ".npz",
    ".xls",
    ".xlsx",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
    ".wav",
    ".au",
    ".aup",
    ".flac",
    ".mp3",
    ".mat",
    ".pdf",
    ".sample",
    ".stl",
    ".stp",
    ".ipynb",
    ".mpf",
    ".py",
    ".idx",
    ".pack",
    ".rev",
    "",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
METADATA_ONLY_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
    ".au",
    ".aup",
    ".flac",
    ".mp3",
    ".pdf",
    ".sample",
    ".stl",
    ".stp",
    ".ipynb",
    ".mpf",
    ".py",
    ".idx",
    ".pack",
    ".rev",
    "",
}


class RealPretrainingDataset(Dataset[dict[str, Any]]):
    """Reads bounded numeric windows from heterogeneous files in a manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        sequence_length: int = 512,
        max_channels: int = 16,
        channel_vocab_size: int = 4096,
        windows_per_file: int = 1,
        include_datasets: list[str] | None = None,
        exclude_datasets: list[str] | None = None,
        extensions: list[str] | None = None,
        max_files: int | None = None,
        text_vocab_size: int = DEFAULT_TEXT_VOCAB_SIZE,
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
        latent_context_path: str | Path | None = None,
        nc_context_path: str | Path | None = None,
        nc_context_dim: int = NC_CONTEXT_DIM,
        require_nc_context: bool = False,
        seed: int = 42,
        max_read_attempts: int = 12,
        generate_frequency: bool = False,
        frequency_transforms: list[str] | None = None,
        frequency_length: int = 512,
        frequency_bands: dict[str, Any] | None = None,
        augmentation: dict[str, Any] | None = None,
        image_size: int = 128,
        generate_virtual_vibration: bool = False,
        virtual_vibration_sampling_rate: float = 1600.0,
        virtual_vibration_mode: str = "if_missing",
        virtual_vibration_default_spindle_rpm: float = 6000.0,
    ) -> None:
        records = read_records(manifest_path)
        include = set(include_datasets or [])
        exclude = set(exclude_datasets or [])
        allowed_extensions = {
            ""
            if value in {"", "(none)"}
            else value.lower()
            if value.startswith(".")
            else f".{value.lower()}"
            for value in (extensions or sorted(SUPPORTED_EXTENSIONS))
        }
        self.records = [
            record
            for record in records
            if Path(str(record.get("file_path", ""))).suffix.lower() in allowed_extensions
            and (not include or str(record.get("dataset_id")) in include)
            and str(record.get("dataset_id")) not in exclude
        ]
        if max_files:
            self.records = self.records[:max_files]
        if not self.records:
            raise ValueError("No supported real-data records matched the full-pretraining config")
        self.sequence_length = sequence_length
        self.max_channels = max_channels
        self.channel_vocab_size = channel_vocab_size
        self.text_vocab_size = text_vocab_size
        self.max_text_tokens = max_text_tokens
        self.latent_context_model = LatentContextModel.load(latent_context_path)
        self.nc_context_store = NCContextStore.load(nc_context_path, context_dim=nc_context_dim)
        if require_nc_context and self.nc_context_store is None:
            raise FileNotFoundError(f"NC context cache is required but was not found: {nc_context_path}")
        self.nc_context_dim = nc_context_dim
        self.windows_per_file = max(1, windows_per_file)
        self.seed = seed
        self.epoch = 0
        self.max_read_attempts = max_read_attempts
        self.generate_frequency = generate_frequency
        self.frequency_transforms = tuple(frequency_transforms or ["fft", "stft", "cwt"])
        self.frequency_length = frequency_length
        self.frequency_bands = frequency_bands
        self.augmentation = augmentation or {}
        self.image_size = image_size
        self.generate_virtual_vibration = generate_virtual_vibration
        self.virtual_vibration_config = VirtualVibrationConfig(
            sampling_rate=float(virtual_vibration_sampling_rate),
            output_length=sequence_length,
            default_spindle_rpm=float(virtual_vibration_default_spindle_rpm),
            seed=seed,
        )
        self.virtual_vibration_mode = virtual_vibration_mode

    def __len__(self) -> int:
        return len(self.records) * self.windows_per_file

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        logical_index = index % len(self)
        file_index = logical_index // self.windows_per_file
        errors: list[str] = []
        for attempt, record_index in enumerate(self._candidate_record_indices(file_index, logical_index)):
            record = self.records[record_index]
            seed = self.seed + self.epoch * 1_000_003 + logical_index * 101
            try:
                suffix = Path(str(record.get("file_path", ""))).suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    return self._read_image_sample(record, file_index, record_index, attempt)
                if suffix in METADATA_ONLY_EXTENSIONS:
                    return self._read_metadata_only_sample(record, file_index, record_index, attempt)
                values, channel_names = load_signal_window(
                    record["file_path"], self.sequence_length, self.max_channels, seed
                )
                sampling_rate = _record_sampling_rate(record)
                material = str(record.get("material") or "unknown")
                augmentation_metadata: dict[str, Any] = {"physical_noise": {"applied": []}}
                if self.augmentation.get("enabled", False):
                    rng = np.random.default_rng(seed)
                    values, physical_metadata = apply_physical_noise(values, self.augmentation, rng, material)
                    augmentation_metadata["physical_noise"] = physical_metadata
                virtual_vibration_metadata = None
                if self.generate_virtual_vibration:
                    values, channel_names, virtual_vibration_metadata = append_virtual_spindle_vibration(
                        values,
                        channel_names,
                        max_channels=self.max_channels,
                        config=self.virtual_vibration_config,
                        mode=self.virtual_vibration_mode,
                    )
                channel_encoding = encode_channel_names(channel_names, self.channel_vocab_size)
                frequency_values = None
                frequency_encoding = None
                if self.generate_frequency:
                    frequency_values, frequency_names = build_frequency_features(
                        values,
                        channel_names,
                        transforms=self.frequency_transforms,
                        frequency_length=self.frequency_length,
                        sampling_rate=sampling_rate,
                        frequency_bands=self.frequency_bands,
                    )
                    if self.augmentation.get("enabled", False):
                        rng = np.random.default_rng(seed + 17)
                        frequency_values, frequency_aug = apply_frequency_augmentation(
                            frequency_values, self.augmentation, rng
                        )
                        augmentation_metadata["time_frequency"] = frequency_aug
                    frequency_encoding = encode_channel_names(frequency_names, self.channel_vocab_size)
                latent_tokens = heuristic_latent_tokens(values, channel_names)
                latent_metadata: dict[str, Any] = {"heuristic_tokens": latent_tokens}
                if self.latent_context_model is not None:
                    cluster_tokens, cluster_metadata = self.latent_context_model.tokens_for_signal(values, channel_names)
                    latent_tokens = list(dict.fromkeys([*latent_tokens, *cluster_tokens]))
                    latent_metadata.update(cluster_metadata)
                    latent_metadata["tokens"] = latent_tokens
                text_context = build_pretraining_text_context(record, channel_encoding["descriptors"])
                if latent_tokens:
                    text_context = f"{text_context} {' '.join(latent_tokens)}".strip()
                text_ids, text_mask = encode_text_context(
                    text_context,
                    max_tokens=self.max_text_tokens,
                    vocab_size=self.text_vocab_size,
                )
                nc_context, nc_context_metadata = self._nc_context_for_record(record)
                return {
                    "sensor_series": torch.from_numpy(values),
                    "sensor_mask": torch.ones(values.shape[0], dtype=torch.bool),
                    "sensor_ids": torch.tensor(channel_encoding["sensor_ids"], dtype=torch.long),
                    "sensor_attribute_ids": torch.tensor(channel_encoding["attribute_ids"], dtype=torch.long),
                    "condition": None,
                    "condition_mask": None,
                    "text_ids": torch.tensor(text_ids, dtype=torch.long),
                    "text_mask": torch.tensor(text_mask, dtype=torch.bool),
                    "nc_context": torch.from_numpy(nc_context) if nc_context is not None else None,
                    "nc_context_mask": torch.tensor(nc_context is not None, dtype=torch.bool),
                    "image": None,
                    "image_mask": None,
                    "frequency": torch.from_numpy(frequency_values) if frequency_values is not None else None,
                    "frequency_mask": (
                        torch.ones(frequency_values.shape[0], dtype=torch.bool)
                        if frequency_values is not None
                        else None
                    ),
                    "frequency_ids": (
                        torch.tensor(frequency_encoding["sensor_ids"], dtype=torch.long)
                        if frequency_encoding is not None
                        else None
                    ),
                    "frequency_attribute_ids": (
                        torch.tensor(frequency_encoding["attribute_ids"], dtype=torch.long)
                        if frequency_encoding is not None
                        else None
                    ),
                    "metadata": {
                        "sample_id": record.get("sample_id"),
                        "dataset_id": record.get("dataset_id"),
                        "file_path": record.get("file_path"),
                        "requested_record_index": file_index,
                        "record_index": record_index,
                        "read_attempt": attempt + 1,
                        "channel_schema_version": CHANNEL_SCHEMA_VERSION,
                        "raw_channel_names": channel_names,
                        "channel_names": channel_encoding["canonical_names"],
                        "channel_descriptors": channel_encoding["descriptors"],
                        "frequency_transforms": list(self.frequency_transforms) if frequency_values is not None else [],
                        "sampling_rate": sampling_rate,
                        "estimated_sampling_rate": sampling_rate,
                        "material": material,
                        "material_family": record.get("material_family", "unknown"),
                        "material_source": record.get("material_source", "unavailable"),
                        "material_confidence": record.get("material_confidence", "low"),
                        "source_type": record.get("source_type", "raw"),
                        "split_group_key_candidate": record.get("split_group_key_candidate"),
                        "subgraph_candidates": record.get("subgraph_candidates", []),
                        "dynamic_graph_enabled": record.get("dynamic_graph_enabled", True),
                        "frequency_names": (
                            frequency_encoding["canonical_names"] if frequency_encoding is not None else []
                        ),
                        "frequency_descriptors": (
                            frequency_encoding["descriptors"] if frequency_encoding is not None else []
                        ),
                        "text_context": text_context,
                        "nc_context": nc_context_metadata,
                        "latent_context": latent_metadata,
                        "virtual_vibration": virtual_vibration_metadata,
                        "augmentation": augmentation_metadata,
                    },
                }
            except Exception as exc:
                errors.append(f"{record.get('file_path')}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Could not read a real pretraining sample. " + " | ".join(errors[:3]))

    def _read_image_sample(
        self,
        record: dict[str, Any],
        file_index: int,
        record_index: int,
        attempt: int,
    ) -> dict[str, Any]:
        image = load_image_tensor(record["file_path"], self.image_size)
        text_context = build_pretraining_text_context(record, [])
        text_ids, text_mask = encode_text_context(
            text_context,
            max_tokens=self.max_text_tokens,
            vocab_size=self.text_vocab_size,
        )
        nc_context, nc_context_metadata = self._nc_context_for_record(record)
        return {
            "sensor_series": None,
            "sensor_mask": None,
            "sensor_ids": None,
            "sensor_attribute_ids": None,
            "condition": None,
            "condition_mask": None,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "nc_context": torch.from_numpy(nc_context) if nc_context is not None else None,
            "nc_context_mask": torch.tensor(nc_context is not None, dtype=torch.bool),
            "image": torch.from_numpy(image),
            "image_mask": torch.tensor(True, dtype=torch.bool),
            "frequency": None,
            "frequency_mask": None,
            "frequency_ids": None,
            "frequency_attribute_ids": None,
            "metadata": {
                "sample_id": record.get("sample_id"),
                "dataset_id": record.get("dataset_id"),
                "file_path": record.get("file_path"),
                "requested_record_index": file_index,
                "record_index": record_index,
                "read_attempt": attempt + 1,
                "channel_schema_version": CHANNEL_SCHEMA_VERSION,
                "raw_channel_names": [],
                "channel_names": [],
                "channel_descriptors": [],
                "frequency_transforms": [],
                "frequency_names": [],
                "frequency_descriptors": [],
                "text_context": text_context,
                "nc_context": nc_context_metadata,
                "latent_context": {"heuristic_tokens": []},
                "image_size": self.image_size,
                "sampling_rate": _record_sampling_rate(record),
                "estimated_sampling_rate": _record_sampling_rate(record),
                "material": record.get("material", "unknown"),
                "material_family": record.get("material_family", "unknown"),
                "material_source": record.get("material_source", "unavailable"),
                "material_confidence": record.get("material_confidence", "low"),
                "source_type": record.get("source_type", "raw"),
                "split_group_key_candidate": record.get("split_group_key_candidate"),
                "subgraph_candidates": record.get("subgraph_candidates", []),
                "dynamic_graph_enabled": record.get("dynamic_graph_enabled", True),
            },
        }

    def _read_metadata_only_sample(
        self,
        record: dict[str, Any],
        file_index: int,
        record_index: int,
        attempt: int,
    ) -> dict[str, Any]:
        text_context = build_pretraining_text_context(record, [])
        text_ids, text_mask = encode_text_context(
            text_context,
            max_tokens=self.max_text_tokens,
            vocab_size=self.text_vocab_size,
        )
        nc_context, nc_context_metadata = self._nc_context_for_record(record)
        return {
            "sensor_series": None,
            "sensor_mask": None,
            "sensor_ids": None,
            "sensor_attribute_ids": None,
            "condition": None,
            "condition_mask": None,
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "nc_context": torch.from_numpy(nc_context) if nc_context is not None else None,
            "nc_context_mask": torch.tensor(nc_context is not None, dtype=torch.bool),
            "image": None,
            "image_mask": None,
            "frequency": None,
            "frequency_mask": None,
            "frequency_ids": None,
            "frequency_attribute_ids": None,
            "metadata": {
                "sample_id": record.get("sample_id"),
                "dataset_id": record.get("dataset_id"),
                "file_path": record.get("file_path"),
                "requested_record_index": file_index,
                "record_index": record_index,
                "read_attempt": attempt + 1,
                "channel_schema_version": CHANNEL_SCHEMA_VERSION,
                "raw_channel_names": [],
                "channel_names": [],
                "channel_descriptors": [],
                "frequency_transforms": [],
                "frequency_names": [],
                "frequency_descriptors": [],
                "text_context": text_context,
                "nc_context": nc_context_metadata,
                "latent_context": {"heuristic_tokens": []},
                "sampling_rate": _record_sampling_rate(record),
                "estimated_sampling_rate": _record_sampling_rate(record),
                "material": record.get("material", "unknown"),
                "material_family": record.get("material_family", "unknown"),
                "material_source": record.get("material_source", "unavailable"),
                "material_confidence": record.get("material_confidence", "low"),
                "source_type": record.get("source_type", "metadata"),
                "split_group_key_candidate": record.get("split_group_key_candidate"),
                "subgraph_candidates": record.get("subgraph_candidates", []),
                "dynamic_graph_enabled": record.get("dynamic_graph_enabled", True),
            },
        }

    def _nc_context_for_record(self, record: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        if self.nc_context_store is None:
            return None, None
        vector, metadata = self.nc_context_store.context_for_record(record)
        return vector.astype(np.float32, copy=False), metadata

    def _candidate_record_indices(self, file_index: int, logical_index: int) -> list[int]:
        record_count = len(self.records)
        attempts = min(max(1, self.max_read_attempts), record_count)
        first_index = file_index % record_count
        if attempts == 1:
            return [first_index]

        stride = max(1, record_count // attempts)
        while math.gcd(stride, record_count) != 1:
            stride += 1

        indices = [first_index]
        seen = {first_index}
        for attempt in range(1, attempts):
            candidate = (first_index + attempt * stride) % record_count
            if candidate in seen:
                rng = random.Random(self.seed + self.epoch * 1_000_003 + logical_index * 997 + attempt)
                candidate = rng.randrange(record_count)
                while candidate in seen and len(seen) < record_count:
                    candidate = (candidate + 1) % record_count
            indices.append(candidate)
            seen.add(candidate)
        return indices

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.records:
            dataset_id = str(record.get("dataset_id"))
            counts[dataset_id] = counts.get(dataset_id, 0) + 1
        return {
            "files": len(self.records),
            "logical_samples_per_epoch": len(self),
            "sequence_length": self.sequence_length,
            "max_channels": self.max_channels,
            "channel_schema_version": CHANNEL_SCHEMA_VERSION,
            "text_vocab_size": self.text_vocab_size,
            "max_text_tokens": self.max_text_tokens,
            "latent_context_enabled": self.latent_context_model is not None,
            "frequency_enabled": self.generate_frequency,
            "frequency_transforms": list(self.frequency_transforms),
            "frequency_length": self.frequency_length,
            "image_size": self.image_size,
            "virtual_vibration_enabled": self.generate_virtual_vibration,
            "virtual_vibration_sampling_rate": self.virtual_vibration_config.sampling_rate,
            "virtual_vibration_mode": self.virtual_vibration_mode,
            "nc_context": (
                self.nc_context_store.summary()
                if self.nc_context_store is not None
                else {"enabled": False, "context_dim": self.nc_context_dim}
            ),
            "datasets": counts,
        }


def pretraining_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    sensor_samples = [sample for sample in samples if sample.get("sensor_series") is not None]
    if sensor_samples:
        sequence_length = max(sample["sensor_series"].shape[-1] for sample in sensor_samples)
        channels = max(sample["sensor_series"].shape[0] for sample in sensor_samples)
        series = torch.zeros(batch_size, channels, sequence_length, dtype=torch.float32)
        sensor_mask = torch.zeros(batch_size, channels, dtype=torch.bool)
        sensor_ids = torch.zeros(batch_size, channels, dtype=torch.long)
        sensor_attribute_ids = torch.zeros(batch_size, channels, len(CHANNEL_ATTRIBUTE_NAMES), dtype=torch.long)
    else:
        series = None
        sensor_mask = None
        sensor_ids = None
        sensor_attribute_ids = None
    text_length = samples[0].get("text_ids", torch.zeros(0, dtype=torch.long)).shape[-1]
    text_ids = torch.zeros(batch_size, text_length, dtype=torch.long)
    text_mask = torch.zeros(batch_size, text_length, dtype=torch.bool)
    image_samples = [sample for sample in samples if sample.get("image") is not None]
    if image_samples:
        image_shape = image_samples[0]["image"].shape
        image = torch.zeros(batch_size, *image_shape, dtype=torch.float32)
        image_mask = torch.zeros(batch_size, dtype=torch.bool)
    else:
        image = None
        image_mask = None
    frequency_samples = [sample for sample in samples if sample.get("frequency") is not None]
    if frequency_samples:
        frequency_length = max(sample["frequency"].shape[-1] for sample in frequency_samples)
        frequency_channels = max(sample["frequency"].shape[0] for sample in frequency_samples)
        frequency = torch.zeros(batch_size, frequency_channels, frequency_length, dtype=torch.float32)
        frequency_mask = torch.zeros(batch_size, frequency_channels, dtype=torch.bool)
        frequency_ids = torch.zeros(batch_size, frequency_channels, dtype=torch.long)
        frequency_attribute_ids = torch.zeros(batch_size, frequency_channels, len(CHANNEL_ATTRIBUTE_NAMES), dtype=torch.long)
    else:
        frequency = None
        frequency_mask = None
        frequency_ids = None
        frequency_attribute_ids = None
    nc_samples = [sample for sample in samples if sample.get("nc_context") is not None]
    if nc_samples:
        nc_context_dim = max(sample["nc_context"].shape[-1] for sample in nc_samples)
        nc_context = torch.zeros(batch_size, nc_context_dim, dtype=torch.float32)
        nc_context_mask = torch.zeros(batch_size, dtype=torch.bool)
    else:
        nc_context = None
        nc_context_mask = None
    metadata = []
    for index, sample in enumerate(samples):
        if series is not None and sample.get("sensor_series") is not None:
            count = sample["sensor_series"].shape[0]
            length = sample["sensor_series"].shape[-1]
            series[index, :count, :length] = sample["sensor_series"]
            sensor_mask[index, :count] = sample["sensor_mask"]
            sensor_ids[index, :count] = sample["sensor_ids"]
            sensor_attribute_ids[index, :count] = sample["sensor_attribute_ids"]
        if text_length:
            text_ids[index] = sample["text_ids"]
            text_mask[index] = sample["text_mask"]
        if image is not None and sample.get("image") is not None:
            image[index] = sample["image"]
            image_mask[index] = sample.get("image_mask", torch.tensor(True, dtype=torch.bool))
        if frequency is not None and sample.get("frequency") is not None:
            count = sample["frequency"].shape[0]
            length = sample["frequency"].shape[-1]
            frequency[index, :count, :length] = sample["frequency"]
            frequency_mask[index, :count] = sample["frequency_mask"]
            frequency_ids[index, :count] = sample["frequency_ids"]
            frequency_attribute_ids[index, :count] = sample["frequency_attribute_ids"]
        if nc_context is not None and sample.get("nc_context") is not None:
            length = sample["nc_context"].shape[-1]
            nc_context[index, :length] = sample["nc_context"]
            nc_context_mask[index] = bool(sample.get("nc_context_mask", torch.tensor(True)))
        metadata.append(sample["metadata"])
    return {
        "sensor_series": series,
        "sensor_mask": sensor_mask,
        "sensor_ids": sensor_ids,
        "sensor_attribute_ids": sensor_attribute_ids,
        "condition": None,
        "condition_mask": None,
        "text_ids": text_ids if text_length else None,
        "text_mask": text_mask if text_length else None,
        "image": image,
        "image_mask": image_mask,
        "frequency": frequency,
        "frequency_mask": frequency_mask,
        "frequency_ids": frequency_ids,
        "frequency_attribute_ids": frequency_attribute_ids,
        "nc_context": nc_context,
        "nc_context_mask": nc_context_mask,
        "metadata": metadata,
    }


def load_signal_window(
    path: str | Path,
    sequence_length: int,
    max_channels: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    source = Path(path)
    suffix = source.suffix.lower()
    rng = random.Random(seed)
    candidate_limit = max(max_channels * 8, max_channels)
    if suffix in {".h5", ".hdf5"}:
        values, names = _read_hdf5_window(source, sequence_length, candidate_limit, rng)
    elif suffix in {".csv", ".txt", ".log"}:
        values, names = _read_tabular_window(source, sequence_length, candidate_limit, rng, parquet=False)
    elif suffix == ".parquet":
        values, names = _read_tabular_window(source, sequence_length, candidate_limit, rng, parquet=True)
    elif suffix in {".xls", ".xlsx"}:
        values, names = _read_excel_window(source, sequence_length, candidate_limit, rng)
    elif suffix in {".npy", ".npz"}:
        values, names = _read_numpy_window(source, sequence_length, candidate_limit, rng)
    elif suffix == ".wav":
        values, names = _read_wav_window(source, sequence_length, candidate_limit, rng)
    elif suffix == ".mat":
        values, names = _read_mat_window(source, sequence_length, candidate_limit, rng)
    else:
        raise ValueError(f"Unsupported pretraining format: {suffix}")
    values, names = _deduplicate_standard_channels(values, names, max_channels)
    if not values:
        raise ValueError("No recognized standardized sensor channels found")
    normalized = np.stack([_normalize_channel(value, sequence_length) for value in values])
    return normalized.astype(np.float32, copy=False), names


def load_image_tensor(path: str | Path, image_size: int = 128) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        image = image.resize((image_size, image_size), resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)


def build_frequency_features(
    values: np.ndarray,
    channel_names: list[str],
    transforms: tuple[str, ...] | list[str] = ("fft", "stft", "cwt"),
    frequency_length: int = 512,
    sampling_rate: float | None = None,
    frequency_bands: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    features = []
    names = []
    for value, raw_name in zip(values, channel_names):
        features.append(
            build_frequency_view(
                value,
                sampling_rate=sampling_rate,
                transforms=transforms,
                output_length=frequency_length,
                frequency_bands=frequency_bands,
            )
        )
        names.append(f"{raw_name}_frequency")
    return np.stack(features).astype(np.float32, copy=False), names


def _fft_feature(value: np.ndarray, n_fft: int = 256) -> np.ndarray:
    data = _finite_vector(value)
    windowed = _resample_vector(data, n_fft) * np.hanning(n_fft).astype(np.float32)
    return np.log1p(np.abs(np.fft.rfft(windowed))).astype(np.float32)


def _stft_feature(value: np.ndarray, n_fft: int = 128, hop: int = 64) -> np.ndarray:
    data = _finite_vector(value)
    if len(data) < n_fft:
        data = _resample_vector(data, n_fft)
    starts = list(range(0, max(1, len(data) - n_fft + 1), hop)) or [0]
    window = np.hanning(n_fft).astype(np.float32)
    spectra = []
    for start in starts[:16]:
        segment = data[start : start + n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        spectra.append(np.log1p(np.abs(np.fft.rfft(segment * window))))
    return np.mean(np.stack(spectra), axis=0).astype(np.float32)


def _cwt_feature(value: np.ndarray, scale_count: int = 32) -> np.ndarray:
    data = _finite_vector(value)
    widths = np.linspace(2, min(64, max(3, len(data) // 4)), scale_count).astype(np.float32)
    try:
        from scipy import signal

        if hasattr(signal, "cwt") and hasattr(signal, "ricker"):
            coefficients = signal.cwt(data, signal.ricker, widths)
            return np.log1p(np.mean(np.abs(coefficients), axis=1)).astype(np.float32)
    except Exception:
        pass
    energies = []
    for width in widths.astype(int):
        width = max(2, int(width))
        kernel = np.ones(width, dtype=np.float32) / width
        smooth = np.convolve(data, kernel, mode="same")
        energies.append(float(np.sqrt(np.mean((data - smooth) ** 2))))
    return np.log1p(np.asarray(energies, dtype=np.float32))


def _finite_vector(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(8, dtype=np.float32)
    fill = float(np.median(data[finite]))
    return np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def _resample_vector(value: np.ndarray, length: int) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(data) == length:
        return data
    if len(data) <= 1:
        return np.full(length, float(data[0]) if len(data) else 0.0, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, len(data)),
        data,
    ).astype(np.float32)


def _normalize_frequency_vector(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    mean = float(data.mean())
    std = float(data.std())
    if not math.isfinite(std) or std < 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - mean) / std, -10.0, 10.0).astype(np.float32)


def _read_hdf5_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    import h5py

    values: list[np.ndarray] = []
    names: list[str] = []
    with h5py.File(path, "r") as handle:
        datasets: list[tuple[str, Any]] = []

        def collect(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset) and value.dtype.kind in "iufb" and value.ndim in {1, 2}:
                if value.size >= 8 and is_pretraining_signal_channel(name):
                    datasets.append((name, value))

        handle.visititems(collect)
        for name, dataset in sorted(datasets, key=lambda item: item[0].lower()):
            remaining = max_channels - len(values)
            if remaining <= 0:
                break
            channels = _slice_hdf5_dataset(dataset, sequence_length, remaining, rng)
            for channel_index, channel in enumerate(channels):
                values.append(channel)
                names.append(f"{name}:{channel_index}" if len(channels) > 1 else name)
    return values, names


def _slice_hdf5_dataset(
    dataset: Any,
    sequence_length: int,
    remaining_channels: int,
    rng: random.Random,
) -> list[np.ndarray]:
    rows, columns = dataset.shape if dataset.ndim == 2 else (1, dataset.shape[0])
    if dataset.ndim == 1:
        return [_slice_hdf5_1d(dataset, columns, sequence_length, rng)]
    if rows <= columns and rows <= remaining_channels * 4:
        return [
            _slice_hdf5_row(dataset, index, columns, sequence_length, rng)
            for index in range(min(rows, remaining_channels))
        ]
    if columns < rows and columns <= remaining_channels * 4:
        return [
            _slice_hdf5_column(dataset, index, rows, sequence_length, rng)
            for index in range(min(columns, remaining_channels))
        ]
    return []


def _slice_hdf5_1d(value: Any, length: int, sequence_length: int, rng: random.Random) -> np.ndarray:
    if length > sequence_length:
        start = rng.randint(0, length - sequence_length)
        return np.asarray(value[start : start + sequence_length], dtype=np.float32).reshape(-1)
    return np.asarray(value[:], dtype=np.float32).reshape(-1)


def _slice_hdf5_row(
    dataset: Any,
    row: int,
    length: int,
    sequence_length: int,
    rng: random.Random,
) -> np.ndarray:
    start = rng.randint(0, length - sequence_length) if length > sequence_length else 0
    end = min(length, start + sequence_length)
    return np.asarray(dataset[row, start:end], dtype=np.float32).reshape(-1)


def _slice_hdf5_column(
    dataset: Any,
    column: int,
    length: int,
    sequence_length: int,
    rng: random.Random,
) -> np.ndarray:
    start = rng.randint(0, length - sequence_length) if length > sequence_length else 0
    end = min(length, start + sequence_length)
    return np.asarray(dataset[start:end, column], dtype=np.float32).reshape(-1)


def _read_tabular_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
    parquet: bool,
) -> tuple[list[np.ndarray], list[str]]:
    import pandas as pd

    read_rows = max(sequence_length * 4, sequence_length)
    if parquet:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        row_group = rng.randrange(parquet_file.num_row_groups) if parquet_file.num_row_groups else 0
        frame = parquet_file.read_row_group(row_group).to_pandas().head(read_rows)
    else:
        frame = None
        for encoding in ("utf-8-sig", "cp949", "latin-1"):
            try:
                header_row = _recognized_header_row(path, encoding)
                read_kwargs = {
                    "nrows": read_rows,
                    "encoding": encoding,
                    "on_bad_lines": "skip",
                    "sep": None,
                    "engine": "python",
                }
                if header_row is None:
                    read_kwargs["header"] = None
                else:
                    read_kwargs["skiprows"] = header_row
                frame = pd.read_csv(
                    path,
                    **read_kwargs,
                )
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise ValueError("Could not decode tabular file")
    numeric = frame.select_dtypes(include="number")
    if all(isinstance(column, int) for column in numeric.columns):
        numeric = _assign_headerless_channel_names(path, numeric)
    numeric = _drop_probable_time_column(numeric)
    columns = [str(column) for column in numeric.columns if is_pretraining_signal_channel(str(column))]
    if not columns:
        raise ValueError("No recognized standardized sensor columns found")
    if len(numeric) > sequence_length:
        start = rng.randint(0, len(numeric) - sequence_length)
        numeric = numeric.iloc[start : start + sequence_length]
    columns = columns[:max_channels]
    return [numeric[column].to_numpy(dtype=np.float32, na_value=np.nan) for column in columns], columns


def _read_excel_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    import pandas as pd

    frame = pd.read_excel(path, nrows=max(sequence_length * 4, sequence_length))
    return _numeric_frame_window(path, frame, sequence_length, max_channels, rng)


def _read_wav_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    import wave

    with wave.open(str(path), "rb") as audio:
        source_channels = audio.getnchannels()
        channel_count = min(source_channels, max_channels)
        frame_count = audio.getnframes()
        width = audio.getsampwidth()
        start = rng.randint(0, frame_count - sequence_length) if frame_count > sequence_length else 0
        audio.setpos(start)
        raw = audio.readframes(min(sequence_length, frame_count - start))
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        data = data - 128.0
    if data.size == 0:
        raise ValueError("Empty WAV sample")
    data = data.reshape(-1, source_channels)
    values = [data[:, index] for index in range(channel_count)]
    names = [f"{path.stem}/audio_channel_{index}" for index in range(channel_count)]
    return values, names


def _read_mat_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    from scipy.io import loadmat

    payload = loadmat(path, squeeze_me=True)
    values: list[np.ndarray] = []
    names: list[str] = []
    for name, value in sorted(payload.items()):
        if name.startswith("__") or not np.issubdtype(np.asarray(value).dtype, np.number):
            continue
        array = np.asarray(value)
        if array.ndim == 1:
            values.append(_slice_array_vector(array, sequence_length, rng))
            names.append(name)
        elif array.ndim == 2:
            channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
            count = min(array.shape[channel_axis], max_channels - len(values))
            for index in range(count):
                vector = array[index, :] if channel_axis == 0 else array[:, index]
                values.append(_slice_array_vector(vector, sequence_length, rng))
                names.append(f"{name}:{index}")
        if len(values) >= max_channels:
            break
    return values, names


def _numeric_frame_window(
    path: Path,
    frame: Any,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    numeric = frame.select_dtypes(include="number")
    if all(isinstance(column, int) for column in numeric.columns):
        numeric = _assign_headerless_channel_names(path, numeric)
    numeric = _drop_probable_time_column(numeric)
    columns = [str(column) for column in numeric.columns if is_pretraining_signal_channel(str(column))]
    if not columns:
        raise ValueError("No recognized standardized sensor columns found")
    if len(numeric) > sequence_length:
        start = rng.randint(0, len(numeric) - sequence_length)
        numeric = numeric.iloc[start : start + sequence_length]
    columns = columns[:max_channels]
    return [numeric[column].to_numpy(dtype=np.float32, na_value=np.nan) for column in columns], columns


def _assign_headerless_channel_names(path: Path, numeric: Any) -> Any:
    clue = _normalize_path_clue(path)
    if not clue:
        return numeric
    renamed = numeric.copy()
    count = len(renamed.columns)
    if "vib" in clue or "acc" in clue:
        renamed.columns = [_vector_channel_name(path.stem, "vibration", index) for index in range(count)]
    elif "force" in clue or "torque" in clue:
        renamed.columns = [_force_torque_channel_name(path.stem, index) for index in range(count)]
    elif "current" in clue:
        renamed.columns = [_vector_channel_name(path.stem, "current", index) for index in range(count)]
    return renamed


def _normalize_path_clue(path: Path) -> str:
    parts = [part.lower() for part in (*path.parts[-4:-1], path.stem)]
    return "_".join(re.sub(r"[^a-z0-9]+", "_", part) for part in parts)


def _vector_channel_name(prefix: str, quantity: str, index: int) -> str:
    axes = ("x", "y", "z", "u", "v", "w")
    axis = axes[index] if index < len(axes) else f"channel_{index}"
    return f"{prefix}/{quantity}_{axis}"


def _force_torque_channel_name(prefix: str, index: int) -> str:
    names = ("force_x", "force_y", "force_z", "torque_z", "torque_x", "torque_y")
    name = names[index] if index < len(names) else f"force_channel_{index}"
    return f"{prefix}/{name}"


def _drop_probable_time_column(numeric: Any) -> Any:
    if numeric.empty or len(numeric.columns) < 2:
        return numeric
    first_name = str(numeric.columns[0])
    if not is_metadata_channel_name(first_name):
        values = numeric.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(values)
        if finite.sum() >= 8:
            clean = values[finite]
            diffs = np.diff(clean)
            monotonic = np.all(diffs >= 0.0) or np.all(diffs <= 0.0)
            steady_step = diffs.size > 0 and float(np.std(diffs)) <= max(abs(float(np.mean(diffs))) * 0.01, 1e-12)
            if monotonic and steady_step:
                return numeric.drop(columns=[numeric.columns[0]])
        return numeric
    return numeric.drop(columns=[numeric.columns[0]])


def _read_numpy_window(
    path: Path,
    sequence_length: int,
    max_channels: int,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[str]]:
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    arrays = [(name, value[name]) for name in sorted(value.files)] if isinstance(value, np.lib.npyio.NpzFile) else [(path.stem, value)]
    channels: list[np.ndarray] = []
    names: list[str] = []
    for name, array in arrays:
        if not np.issubdtype(array.dtype, np.number):
            continue
        if array.ndim == 1:
            channels.append(_slice_array_vector(array, sequence_length, rng))
            names.append(name)
        elif array.ndim == 2:
            channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
            count = min(array.shape[channel_axis], max_channels - len(channels))
            for index in range(count):
                vector = array[index, :] if channel_axis == 0 else array[:, index]
                channels.append(_slice_array_vector(vector, sequence_length, rng))
                names.append(_numpy_channel_name(name, index, count))
        if len(channels) >= max_channels:
            break
    return channels, names


def _slice_array_vector(value: Any, sequence_length: int, rng: random.Random) -> np.ndarray:
    length = len(value)
    start = rng.randint(0, length - sequence_length) if length > sequence_length else 0
    return np.asarray(value[start : start + sequence_length], dtype=np.float32).reshape(-1)


def _recognized_header_row(path: Path, encoding: str) -> int | None:
    lines: list[str] = []
    with path.open("r", encoding=encoding) as handle:
        for _ in range(64):
            line = handle.readline()
            if not line:
                break
            lines.append(line)
    return find_recognized_header_index(lines)


def _numpy_channel_name(name: str, index: int, count: int) -> str:
    if count >= 3 and re.search(r"(?:^|_)xyz$", name.lower()):
        return f"{name}/vibration_{'xyz'[index]}" if index < 3 else f"{name}:{index}"
    return f"{name}:{index}"


def _normalize_channel(value: np.ndarray, sequence_length: int) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(data)
    if not finite.any():
        data = np.zeros(max(1, len(data)), dtype=np.float32)
    else:
        fill = float(np.median(data[finite]))
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
    if len(data) != sequence_length:
        if len(data) <= 1:
            data = np.full(sequence_length, float(data[0]) if len(data) else 0.0, dtype=np.float32)
        else:
            data = np.interp(
                np.linspace(0.0, 1.0, sequence_length),
                np.linspace(0.0, 1.0, len(data)),
                data,
            ).astype(np.float32)
    mean = float(data.mean())
    std = float(data.std())
    if not math.isfinite(std) or std < 1e-6:
        return np.zeros(sequence_length, dtype=np.float32)
    return np.clip((data - mean) / std, -10.0, 10.0).astype(np.float32)


def _deduplicate_standard_channels(
    values: list[np.ndarray],
    raw_names: list[str],
    max_channels: int,
) -> tuple[list[np.ndarray], list[str]]:
    selected_values: list[np.ndarray] = []
    selected_names: list[str] = []
    canonical_names: set[str] = set()
    for value, raw_name in zip(values, raw_names):
        if not is_pretraining_signal_channel(raw_name):
            continue
        canonical = describe_channel(raw_name).canonical_name
        if canonical in canonical_names:
            continue
        canonical_names.add(canonical)
        selected_values.append(value)
        selected_names.append(raw_name)
        if len(selected_values) >= max_channels:
            break
    return selected_values, selected_names


def _record_sampling_rate(record: dict[str, Any]) -> float | None:
    for key in ("estimated_sampling_rate", "sampling_rate", "sampling_rate_estimate"):
        value = record.get(key)
        if value in (None, "", "unknown"):
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    return None
