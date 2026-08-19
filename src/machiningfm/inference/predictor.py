from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from machiningfm.data.channel_schema import (
    CHANNEL_SCHEMA_VERSION,
    channel_schema_metadata,
    encode_channel_names,
)
from machiningfm.data.latent_context import LatentContextModel, heuristic_latent_tokens
from machiningfm.data.text_context import (
    DEFAULT_MAX_TEXT_TOKENS,
    DEFAULT_TEXT_VOCAB_SIZE,
    build_request_text_context,
    encode_text_context,
)
from machiningfm.data.virtual_vibration import (
    VirtualVibrationConfig,
    append_virtual_spindle_vibration,
    generate_virtual_spindle_vibration,
    generate_virtual_spindle_vibration_v2,
    virtual_vibration_header_metadata,
)
from machiningfm.data.missing import create_missing_variable_report
from machiningfm.models.machiningfm import MachiningFM
from machiningfm.training.checkpointing import load_checkpoint

DEFAULT_MODEL_CONFIG = {
    "d_model": 64,
    "num_layers": 2,
    "num_heads": 4,
    "patch_size": 8,
    "horizon": 8,
    "max_channels": 128,
    "max_conditions": 64,
    "text_vocab_size": DEFAULT_TEXT_VOCAB_SIZE,
    "max_text_tokens": DEFAULT_MAX_TEXT_TOKENS,
    "image_size": 128,
    "image_patch_size": 16,
    "dropout": 0.0,
}
DEFAULT_EXPECTED_VARIABLES = [
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "force",
    "ae",
    "spindle_speed",
    "feed_rate",
    "depth_of_cut",
]
VIRTUAL_SENSOR_TASKS = {"virtual_sensor", "virtual_vibration"}


class MachiningPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_config: dict[str, Any] | None = None,
        expected_variables: list[str] | None = None,
        latent_context_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        checkpoint: dict[str, Any] | None = None
        if checkpoint_path:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
            checkpoint = load_checkpoint(checkpoint_file, map_location=device)
        config = (checkpoint or {}).get("model_config") or model_config or DEFAULT_MODEL_CONFIG
        checkpoint_schema = config.get("channel_schema_version", CHANNEL_SCHEMA_VERSION if not checkpoint else None)
        if checkpoint_schema != CHANNEL_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint channel schema {checkpoint_schema!r} is incompatible with "
                f"{CHANNEL_SCHEMA_VERSION!r}; train a new checkpoint with standardized CNC channel names"
            )
        self.model = MachiningFM(config).to(self.device)
        if checkpoint and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        self.model.eval()
        self.image_size = int(config.get("image_size", 128))
        self.model_version = (checkpoint or {}).get("model_version", "machiningfm-tiny-untrained-smoke")
        self.expected_variables = expected_variables or DEFAULT_EXPECTED_VARIABLES
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.latent_context_model = LatentContextModel.load(latent_context_path)

    @torch.inference_mode()
    def embed(self, request: dict[str, Any]) -> list[float]:
        batch, _ = self.prepare_batch(request)
        embedding = self.model.encode(batch)["embedding"][0]
        return embedding.detach().cpu().tolist()

    @torch.inference_mode()
    def embed_virtual_sensor_source(self, request: dict[str, Any]) -> list[float]:
        source_values, source_names = _virtual_sensor_source_from_request(request)
        batch, _ = self._prepare_virtual_sensor_source_batch(source_values, source_names, request)
        embedding = self.model.encode(batch)["embedding"][0]
        return embedding.detach().cpu().tolist()

    @torch.inference_mode()
    def predict(self, task: str, request: dict[str, Any], include_embedding: bool = False) -> dict[str, Any]:
        if task in VIRTUAL_SENSOR_TASKS:
            return self.predict_virtual_sensor(task, request)
        batch, metadata = self.prepare_batch(request)
        output = self.model(batch, task=task)
        prediction = _python_value(output["prediction"][0])
        embedding = output["embedding"][0].detach().cpu().tolist() if include_embedding else None
        report = create_missing_variable_report(
            request.get("available_variables") or metadata["available_variables"], self.expected_variables
        )
        return {
            "task": task,
            "prediction": prediction,
            "uncertainty": None,
            "embedding": embedding,
            "used_modalities": metadata["used_modalities"],
            "missing_variables": report["missing"],
            "model_version": self.model_version,
            "channel_schema_version": CHANNEL_SCHEMA_VERSION,
        }

    def predict_virtual_sensor(self, task: str, request: dict[str, Any]) -> dict[str, Any]:
        source_values, source_names = _virtual_sensor_source_from_request(request)
        base_values, names, metadata = generate_virtual_spindle_vibration_v2(
            source_values,
            source_names,
            _virtual_vibration_config_from_request(request),
        )
        values, backbone_metadata = self._condition_virtual_sensor_with_backbone(
            base_values,
            source_values,
            source_names,
            request,
        )
        metadata = {
            **metadata,
            "base_generator": "physics_inspired_virtual_vibration",
            "decoder": "machiningfm_backbone_conditioned_generator",
            "trained_virtual_sensor_decoder": False,
            "backbone_conditioning": backbone_metadata,
        }
        return {
            "task": task,
            "prediction": values.tolist(),
            "sensor_series": values.tolist(),
            "sensor_names": names,
            "canonical_sensor_names": metadata["canonical_names"],
            "virtual_vibration": metadata,
            "used_modalities": ["cnc_series", "machiningfm_backbone", "virtual_vibration"],
            "uncertainty": None,
            "model_version": self.model_version,
            "channel_schema_version": CHANNEL_SCHEMA_VERSION,
        }

    @torch.inference_mode()
    def _condition_virtual_sensor_with_backbone(
        self,
        base_values: np.ndarray,
        source_values: np.ndarray,
        source_names: list[str],
        request: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not bool(request.get("virtual_sensor_use_backbone", True)):
            return base_values.astype(np.float32, copy=False), {"enabled": False}

        batch, batch_metadata = self._prepare_virtual_sensor_source_batch(source_values, source_names, request)
        encoded = self.model.encode(batch)
        embedding = encoded["embedding"]
        forecast = self.model.forecasting_head(embedding, channels=3)[0].detach().cpu().numpy()
        embedding_values = embedding[0].detach().cpu().numpy()

        blend = _clamp_float(request.get("virtual_sensor_backbone_blend", 0.25), 0.0, 1.0)
        conditioned = _apply_backbone_conditioning(base_values, embedding_values, forecast, blend)
        return conditioned.astype(np.float32, copy=False), {
            "enabled": True,
            "source": "MachiningFM.encode",
            "decoder": "latent_conditioned_generator",
            "trained_decoder": False,
            "embedding_dim": int(embedding_values.shape[0]),
            "forecast_horizon": int(forecast.shape[-1]) if forecast.ndim == 2 else 0,
            "backbone_blend": blend,
            "used_modalities": batch_metadata["used_modalities"],
            "canonical_source_names": batch_metadata["canonical_sensor_names"],
        }

    def _prepare_virtual_sensor_source_batch(
        self,
        source_values: np.ndarray,
        source_names: list[str],
        request: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
        backbone_request = dict(request)
        backbone_request["sensor_series"] = source_values.tolist()
        backbone_request["sensor_names"] = source_names
        backbone_request["generate_virtual_vibration"] = False
        backbone_request["available_variables"] = list(
            dict.fromkeys([*backbone_request.get("available_variables", []), *source_names])
        )
        return self.prepare_batch(backbone_request)

    def prepare_batch(self, request: dict[str, Any]) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
        sensor_series = request.get("sensor_series")
        sensor_names = request.get("sensor_names") or []
        cnc_series = _first_present(request, "cnc_series", "cnc_data")
        cnc_names = _first_present(request, "cnc_names", "cnc_channel_names") or []
        frequency_series = request.get("frequency")
        if frequency_series is None:
            frequency_series = _first_present(request, "frequency_series", "stft")
        frequency_names = _first_present(request, "frequency_names", "stft_names") or []
        image_tensor = _request_image_tensor(request, self.image_size, self.device)
        condition = request.get("process_condition") or request.get("condition") or {}
        used_modalities: list[str] = []
        available_variables = list(request.get("available_variables") or [])
        channel_metadata: dict[str, Any] = {
            "channel_schema_version": CHANNEL_SCHEMA_VERSION,
            "canonical_sensor_names": [],
            "canonical_frequency_names": [],
            "channel_descriptors": [],
            "frequency_descriptors": [],
            "text_context": "",
            "virtual_vibration": None,
        }
        virtual_config = _virtual_vibration_config_from_request(request)
        if sensor_series is None and cnc_series is not None:
            cnc_values = np.asarray(_rectangular_series(cnc_series), dtype=np.float32)
            generated, generated_names, virtual_metadata = generate_virtual_spindle_vibration(
                cnc_values,
                cnc_names,
                virtual_config,
            )
            sensor_series = generated.tolist()
            sensor_names = generated_names
            used_modalities.append("virtual_vibration")
            available_variables.extend(generated_names)
            channel_metadata["virtual_vibration"] = virtual_metadata
        if sensor_series is not None:
            series = _rectangular_series(sensor_series)
            if bool(request.get("generate_virtual_vibration", False)):
                source_series = _rectangular_series(cnc_series) if cnc_series is not None else series
                source_names = list(cnc_names) if cnc_names else list(sensor_names)
                generated_values, generated_names, virtual_metadata = append_virtual_spindle_vibration(
                    np.asarray(source_series, dtype=np.float32),
                    source_names,
                    max_channels=self.model.tokenizer.sensor.max_channels,
                    config=virtual_config,
                    mode="always",
                )
                original = np.asarray(series, dtype=np.float32)
                if generated_values.shape[0] > len(source_names):
                    appended = generated_values[len(source_names) :]
                    if appended.size:
                        target_length = max(original.shape[-1], appended.shape[-1])
                        original = np.stack([_resample_request_vector(row, target_length) for row in original])
                        appended = np.stack([_resample_request_vector(row, target_length) for row in appended])
                        series = np.concatenate([original, appended], axis=0).tolist()
                        sensor_names = [*sensor_names, *generated_names[len(source_names) :]]
                        used_modalities.append("virtual_vibration")
                        available_variables.extend(generated_names[len(source_names) :])
                        channel_metadata["virtual_vibration"] = virtual_metadata
            sensor_tensor = torch.tensor(series, dtype=torch.float32, device=self.device).unsqueeze(0)
            sensor_mask = torch.ones((1, sensor_tensor.shape[1]), dtype=torch.bool, device=self.device)
            sensor_ids = None
            sensor_attribute_ids = None
            if sensor_names:
                if len(sensor_names) != sensor_tensor.shape[1]:
                    raise ValueError(
                        f"sensor_names has {len(sensor_names)} entries, "
                        f"but sensor_series has {sensor_tensor.shape[1]} channels"
                    )
                vocabulary_size = self.model.tokenizer.sensor.channel_vocab_size
                channel_encoding = encode_channel_names(sensor_names, vocabulary_size)
                sensor_ids = torch.tensor(
                    [channel_encoding["sensor_ids"]], dtype=torch.long, device=self.device
                )
                sensor_attribute_ids = torch.tensor(
                    [channel_encoding["attribute_ids"]], dtype=torch.long, device=self.device
                )
                channel_metadata["canonical_sensor_names"] = channel_encoding["canonical_names"]
                channel_metadata["channel_descriptors"] = channel_encoding["descriptors"]
                available_variables.extend(
                    descriptor["quantity"] for descriptor in channel_encoding["descriptors"]
                )
                latent_tokens = heuristic_latent_tokens(np.asarray(series, dtype=np.float32), sensor_names)
                latent_metadata: dict[str, Any] = {"heuristic_tokens": latent_tokens}
                if self.latent_context_model is not None:
                    cluster_tokens, cluster_metadata = self.latent_context_model.tokens_for_signal(
                        np.asarray(series, dtype=np.float32),
                        sensor_names,
                    )
                    latent_tokens = list(dict.fromkeys([*latent_tokens, *cluster_tokens]))
                    latent_metadata.update(cluster_metadata)
                    latent_metadata["tokens"] = latent_tokens
                channel_metadata["latent_context"] = latent_metadata
            else:
                latent_tokens = []
            used_modalities.append("sensor_series")
            available_variables.extend(sensor_names)
        else:
            sensor_tensor = None
            sensor_mask = None
            sensor_ids = None
            sensor_attribute_ids = None
            latent_tokens = []
        if frequency_series is not None:
            frequency_values = _rectangular_series(frequency_series)
            frequency_tensor = torch.tensor(frequency_values, dtype=torch.float32, device=self.device).unsqueeze(0)
            frequency_mask = torch.ones((1, frequency_tensor.shape[1]), dtype=torch.bool, device=self.device)
            frequency_ids = None
            frequency_attribute_ids = None
            if frequency_names:
                if len(frequency_names) != frequency_tensor.shape[1]:
                    raise ValueError(
                        f"frequency_names has {len(frequency_names)} entries, "
                        f"but frequency has {frequency_tensor.shape[1]} channels"
                    )
                vocabulary_size = self.model.tokenizer.frequency.channel_vocab_size
                frequency_encoding = encode_channel_names(frequency_names, vocabulary_size)
                frequency_ids = torch.tensor(
                    [frequency_encoding["sensor_ids"]], dtype=torch.long, device=self.device
                )
                frequency_attribute_ids = torch.tensor(
                    [frequency_encoding["attribute_ids"]], dtype=torch.long, device=self.device
                )
                channel_metadata["canonical_frequency_names"] = frequency_encoding["canonical_names"]
                channel_metadata["frequency_descriptors"] = frequency_encoding["descriptors"]
                available_variables.extend(
                    descriptor["quantity"] for descriptor in frequency_encoding["descriptors"]
                )
            used_modalities.append("frequency")
            available_variables.extend(frequency_names)
        else:
            frequency_tensor = None
            frequency_mask = None
            frequency_ids = None
            frequency_attribute_ids = None
        if image_tensor is not None:
            image_mask = torch.ones((1,), dtype=torch.bool, device=self.device)
            used_modalities.append("image")
            available_variables.append("image")
        else:
            image_mask = None
        context_channel_names = [
            *channel_metadata["canonical_sensor_names"],
            *channel_metadata["canonical_frequency_names"],
        ]
        text_context = build_request_text_context(request, context_channel_names)
        if latent_tokens:
            text_context = f"{text_context} {' '.join(latent_tokens)}".strip()
        text_ids = None
        text_mask = None
        if text_context:
            text_ids_value, text_mask_value = encode_text_context(
                text_context,
                max_tokens=self.model.tokenizer.text.max_tokens,
                vocab_size=self.model.tokenizer.text.vocab_size,
            )
            text_ids = torch.tensor([text_ids_value], dtype=torch.long, device=self.device)
            text_mask = torch.tensor([text_mask_value], dtype=torch.bool, device=self.device)
            used_modalities.append("text_context")
            channel_metadata["text_context"] = text_context
        if condition:
            numeric_items = _numeric_condition_items(condition)
            names = [name for name, _ in numeric_items]
            values = [value for _, value in numeric_items]
            available_variables.extend(condition.keys())
        else:
            numeric_items = []
            names = []
            values = []
        if numeric_items:
            condition_tensor = torch.tensor([values], dtype=torch.float32, device=self.device)
            condition_mask = torch.tensor(
                [[True for _ in names]], dtype=torch.bool, device=self.device
            )
            used_modalities.append("process_condition")
        else:
            condition_tensor = None
            condition_mask = None
        return (
            {
                "sensor_series": sensor_tensor,
                "sensor_mask": sensor_mask,
                "sensor_ids": sensor_ids,
                "sensor_attribute_ids": sensor_attribute_ids,
                "condition": condition_tensor,
                "condition_mask": condition_mask,
                "text_ids": text_ids,
                "text_mask": text_mask,
                "image": image_tensor,
                "image_mask": image_mask,
                "frequency": frequency_tensor,
                "frequency_mask": frequency_mask,
                "frequency_ids": frequency_ids,
                "frequency_attribute_ids": frequency_attribute_ids,
            },
            {
                "used_modalities": used_modalities,
                "available_variables": sorted(set(available_variables)),
                **channel_metadata,
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "expected_variables": self.expected_variables,
            "tasks": list(self.model.task_heads) + ["future_forecasting", *sorted(VIRTUAL_SENSOR_TASKS)],
            "channel_schema": channel_schema_metadata(),
            "text_context_schema": {
                "version": "text-context-v1",
                "accepted_fields": [
                    "text_context",
                    "tool_info",
                    "material_info",
                    "machine_info",
                    "operation_info",
                    "process_description",
                ],
                "text_vocab_size": self.model.tokenizer.text.vocab_size,
                "max_text_tokens": self.model.tokenizer.text.max_tokens,
            },
            "latent_context_schema": {
                "version": "latent-context-v1",
                "enabled": self.latent_context_model is not None,
            },
            "virtual_vibration_schema": virtual_vibration_header_metadata(),
            "frequency_schema": {
                "version": "frequency-context-v1",
                "accepted_fields": ["frequency", "frequency_series", "stft"],
                "accepted_name_fields": ["frequency_names", "stft_names"],
                "input_shape": "[channels][frequency_bins_or_features]",
            },
            "image_schema": {
                "version": "image-context-v1",
                "accepted_fields": ["image", "image_base64", "image_path"],
                "input_shape": "numeric image [H][W][C] or [C][H][W]; RGB is resized internally",
                "image_size": self.image_size,
            },
        }


def _rectangular_series(series: Any) -> list[list[float]]:
    if series is None:
        return [[0.0]]
    if isinstance(series, np.ndarray):
        if series.size == 0:
            return [[0.0]]
        if series.ndim == 1:
            series = [series.tolist()]
        elif series.ndim == 2:
            series = series.tolist()
        else:
            raise ValueError(f"Expected 1D or 2D series, got {series.shape}")
    if not series:
        return [[0.0]]
    if not isinstance(series[0], list):
        series = [series]  # type: ignore[list-item]
    channels = [[float(value) if math.isfinite(float(value)) else 0.0 for value in channel] for channel in series]  # type: ignore[union-attr]
    maximum = max(len(channel) for channel in channels)
    return [channel + [0.0] * (maximum - len(channel)) for channel in channels]


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _virtual_sensor_source_from_request(request: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    source_series = _first_present(request, "cnc_series", "cnc_data")
    source_names = _first_present(request, "cnc_names", "cnc_channel_names") or []
    if source_series is None:
        source_series = request.get("sensor_series")
        source_names = request.get("sensor_names") or []
    if source_series is None:
        raise ValueError("virtual_sensor prediction requires cnc_series or sensor_series")
    source_values = np.asarray(_rectangular_series(source_series), dtype=np.float32)
    names = [str(name) for name in source_names]
    if len(names) != source_values.shape[0]:
        names = [f"cnc_channel_{index}" for index in range(source_values.shape[0])]
    return source_values, names


def _apply_backbone_conditioning(
    base_values: np.ndarray,
    embedding: np.ndarray,
    forecast: np.ndarray,
    blend: float,
) -> np.ndarray:
    base = np.asarray(base_values, dtype=np.float32)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    forecast = np.asarray(forecast, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError(f"base virtual vibration must be 2D, got {base.shape}")

    output_length = base.shape[-1]
    latent = _repeat_or_trim(np.tanh(np.nan_to_num(embedding)), 8)
    axis_gain = 1.0 + 0.15 * latent[: base.shape[0]]
    global_gain = 1.0 + 0.08 * float(latent[3])
    conditioned = base * axis_gain[:, None] * global_gain

    if forecast.ndim == 2 and forecast.size and blend > 0.0:
        forecast = _repeat_channels(forecast, base.shape[0])
        modulation = np.stack(
            [_standardize_request_vector(_resample_request_vector(row, output_length)) for row in forecast[: base.shape[0]]]
        )
        conditioned = (1.0 - blend) * conditioned + blend * modulation
    return np.nan_to_num(conditioned, nan=0.0, posinf=0.0, neginf=0.0)


def _repeat_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    if values.size >= length:
        return values[:length]
    repeats = int(math.ceil(length / values.size))
    return np.tile(values, repeats)[:length]


def _repeat_channels(values: np.ndarray, channels: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] >= channels:
        return values
    repeats = int(math.ceil(channels / max(values.shape[0], 1)))
    return np.tile(values, (repeats, 1))[:channels]


def _standardize_request_vector(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return data
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(data.std())
    if std < 1.0e-6:
        return np.zeros_like(data, dtype=np.float32)
    return ((data - float(data.mean())) / std).astype(np.float32)


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    if not math.isfinite(number):
        number = lower
    return max(lower, min(upper, number))


def _virtual_vibration_config_from_request(request: dict[str, Any]) -> VirtualVibrationConfig:
    sampling_rate = request.get("virtual_vibration_sampling_rate") or request.get("sampling_rate") or 1600.0
    output_length = request.get("virtual_vibration_length")
    return VirtualVibrationConfig(
        sampling_rate=float(sampling_rate),
        output_length=int(output_length) if output_length else None,
        input_sampling_rate=_maybe_float(request.get("cnc_sampling_rate") or request.get("input_sampling_rate")),
        default_spindle_rpm=float(request.get("default_spindle_rpm") or 6000.0),
        amplitude=float(request.get("virtual_vibration_amplitude") or 1.0),
        noise_std=float(request.get("virtual_vibration_noise_std") or 0.03),
        seed=int(request.get("virtual_vibration_seed") or 42),
    )


def _resample_request_vector(value: np.ndarray, length: int) -> np.ndarray:
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


def _request_image_tensor(request: dict[str, Any], image_size: int, device: torch.device) -> torch.Tensor | None:
    if request.get("image") is not None:
        return _array_image_tensor(request["image"], image_size, device)
    if request.get("image_base64"):
        return _pil_image_tensor(_decode_base64_image(str(request["image_base64"])), image_size, device)
    if request.get("image_path"):
        from PIL import Image

        with Image.open(request["image_path"]) as image:
            return _pil_image_tensor(image, image_size, device)
    return None


def _decode_base64_image(value: str) -> Any:
    from PIL import Image

    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _pil_image_tensor(image: Any, image_size: int, device: torch.device) -> torch.Tensor:
    from PIL import Image

    resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    image = image.convert("RGB").resize((image_size, image_size), resampling)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0)


def _array_image_tensor(value: Any, image_size: int, device: torch.device) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim != 3:
        raise ValueError(f"image must be a 2D or 3D numeric array, got shape {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"image must have 1, 3, or 4 channels, got shape {array.shape}")
    if float(np.nanmax(array)) > 1.5:
        array = array / 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    tensor = torch.tensor(np.transpose(array, (2, 0, 1)), dtype=torch.float32, device=device).unsqueeze(0)
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.clamp(0.0, 1.0)


def _python_value(value: torch.Tensor) -> Any:
    value = value.detach().cpu()
    if value.numel() == 1:
        return float(value.item())
    return value.tolist()


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_condition_items(condition: dict[str, Any]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for name, raw_value in sorted(condition.items()):
        value = _maybe_float(raw_value)
        if value is not None:
            items.append((name, value))
    return items
