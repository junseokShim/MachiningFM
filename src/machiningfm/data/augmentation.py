from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class GanSyntheticAugmentation:
    """Interface for validated GAN-generated sensor or feature samples."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.synthetic_manifest = self.config.get("synthetic_manifest")

    def load_records(self) -> list[dict[str, Any]]:
        if not self.enabled or not self.synthetic_manifest:
            return []
        path = Path(str(self.synthetic_manifest))
        if not path.exists():
            raise FileNotFoundError(f"GAN synthetic manifest not found: {path}")
        raise NotImplementedError(
            "GAN synthetic augmentation requires a validated synthetic manifest reader."
        )


class PhysicsSyntheticAugmentation:
    """Interface for FEM or physics-provider generated samples."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.provider = self.config.get("provider")

    def generate(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return sample
        if not self.provider:
            raise NotImplementedError(
                "Physics synthetic augmentation is enabled but no provider is configured."
            )
        raise NotImplementedError(
            "FEM/physics synthetic generation requires a real provider implementation."
        )


def apply_physical_noise(
    values: np.ndarray,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
    material: str = "unknown",
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = config or {}
    if not raw.get("enabled", False):
        return values.astype(np.float32, copy=False), {"applied": []}
    physical = raw.get("physical_noise", raw)
    material_cfg = _material_config(raw, material)
    merged = {**physical, **material_cfg}
    result = np.asarray(values, dtype=np.float32).copy()
    applied: list[str] = []

    scale_range = _range(merged.get("amplitude_scale_range"), (1.0, 1.0))
    scale = rng.uniform(scale_range[0], scale_range[1])
    result *= np.float32(scale)
    if scale_range != (1.0, 1.0):
        applied.append("amplitude_scaling")

    jitter_ratio = float(merged.get("sensor_jitter_std_ratio", 0.0) or 0.0)
    if jitter_ratio > 0:
        std = np.std(result, axis=-1, keepdims=True)
        shared = rng.normal(0.0, jitter_ratio, size=(result.shape[0], 1)).astype(np.float32)
        independent = rng.normal(0.0, jitter_ratio * 0.25, size=result.shape).astype(np.float32)
        result += (shared + independent) * np.maximum(std, 1e-6)
        applied.append("sensor_jitter")

    drift_ratio = float(merged.get("bias_drift_ratio", 0.0) or 0.0)
    if drift_ratio > 0 and result.shape[-1] > 1:
        ramp = np.linspace(-0.5, 0.5, result.shape[-1], dtype=np.float32)
        channel_scale = np.std(result, axis=-1, keepdims=True)
        result += drift_ratio * channel_scale * ramp[None, :]
        applied.append("bias_drift")

    noise_ratio = float(merged.get("band_limited_noise_ratio", 0.0) or 0.0)
    if noise_ratio > 0:
        result = _add_band_limited_noise(result, noise_ratio, rng)
        applied.append("band_limited_noise")

    phase_max = float(merged.get("phase_perturbation_max_rad", 0.0) or 0.0)
    frequency_shift = float(merged.get("frequency_shift_ratio", 0.0) or 0.0)
    if phase_max > 0 or frequency_shift > 0:
        result = _spectral_phase_and_shift(result, phase_max, frequency_shift, rng)
        if phase_max > 0:
            applied.append("phase_perturbation")
        if frequency_shift > 0:
            applied.append("small_frequency_shift")

    attenuation_prob = float(merged.get("random_band_attenuation_prob", 0.0) or 0.0)
    if attenuation_prob > 0 and rng.random() < attenuation_prob:
        attenuation_range = _range(merged.get("random_band_attenuation_range"), (0.95, 1.0))
        result = _random_band_attenuation(result, attenuation_range, rng)
        applied.append("random_band_attenuation")

    if bool(merged.get("speed_resampling_enabled", False)):
        ratio = 1.0 + rng.uniform(-abs(frequency_shift), abs(frequency_shift))
        if abs(ratio - 1.0) > 1e-6:
            result = _speed_resample(result, ratio)
            applied.append("speed_normalized_resampling")

    drop_prob = float(merged.get("drop_channel_prob", 0.0) or 0.0)
    if drop_prob > 0 and result.shape[0] > 1:
        keep = rng.random(result.shape[0]) >= drop_prob
        if keep.any():
            result[~keep] = 0.0
            applied.append("drop_channel")

    time_mask_prob = float(merged.get("time_mask_prob", 0.0) or 0.0)
    if time_mask_prob > 0 and rng.random() < time_mask_prob:
        result = _time_mask(result, float(merged.get("time_mask_ratio", 0.05)), rng)
        applied.append("time_mask")

    return result.astype(np.float32, copy=False), {"applied": applied}


def apply_frequency_augmentation(
    frequency: np.ndarray,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = config or {}
    time_frequency = raw.get("time_frequency", raw)
    if not time_frequency.get("enabled", False):
        return frequency.astype(np.float32, copy=False), {"applied": []}
    result = np.asarray(frequency, dtype=np.float32).copy()
    applied: list[str] = []
    gain_range = _range(time_frequency.get("spectral_gain_range"), (1.0, 1.0))
    if gain_range != (1.0, 1.0):
        result *= np.float32(rng.uniform(gain_range[0], gain_range[1]))
        applied.append("spectral_gain_jitter")
    if rng.random() < float(time_frequency.get("frequency_mask_prob", 0.0) or 0.0):
        result = _frequency_mask(result, float(time_frequency.get("frequency_mask_ratio", 0.05)), rng)
        applied.append("frequency_mask")
    if rng.random() < float(time_frequency.get("band_dropout_prob", 0.0) or 0.0):
        result = _frequency_mask(result, float(time_frequency.get("band_dropout_ratio", 0.08)), rng)
        applied.append("band_dropout")
    if bool(time_frequency.get("spectral_smoothing", False)):
        result = _spectral_smooth(result)
        applied.append("spectral_smoothing")
    return result.astype(np.float32, copy=False), {"applied": applied}


def _material_config(config: dict[str, Any], material: str) -> dict[str, Any]:
    material_conditioned = config.get("material_conditioned") or {}
    if not material_conditioned.get("enabled", False):
        return {}
    materials = material_conditioned.get("materials") or {}
    return dict(materials.get(material, materials.get("unknown", {})))


def _range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


def _add_band_limited_noise(values: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=values.shape).astype(np.float32)
    spectrum = np.fft.rfft(noise, axis=-1)
    bins = spectrum.shape[-1]
    low = int(rng.integers(1, max(2, bins // 3)))
    high = int(rng.integers(low + 1, max(low + 2, bins)))
    mask = np.zeros(bins, dtype=np.float32)
    mask[low:high] = 1.0
    filtered = np.fft.irfft(spectrum * mask[None, :], n=values.shape[-1], axis=-1).astype(np.float32)
    scale = np.std(values, axis=-1, keepdims=True)
    return values + filtered * ratio * np.maximum(scale, 1e-6)


def _spectral_phase_and_shift(
    values: np.ndarray,
    phase_max: float,
    shift_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    spectrum = np.fft.rfft(values, axis=-1)
    if phase_max > 0:
        phase = rng.uniform(-phase_max, phase_max, size=spectrum.shape).astype(np.float32)
        spectrum *= np.exp(1j * phase)
    shifted = np.fft.irfft(spectrum, n=values.shape[-1], axis=-1).astype(np.float32)
    if shift_ratio <= 0:
        return shifted
    shift = int(round(rng.uniform(-shift_ratio, shift_ratio) * values.shape[-1]))
    return np.roll(shifted, shift, axis=-1).astype(np.float32)


def _random_band_attenuation(
    values: np.ndarray,
    attenuation_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    spectrum = np.fft.rfft(values, axis=-1)
    bins = spectrum.shape[-1]
    low = int(rng.integers(1, max(2, bins // 2)))
    high = int(rng.integers(low + 1, max(low + 2, bins)))
    gain = rng.uniform(attenuation_range[0], attenuation_range[1])
    spectrum[..., low:high] *= gain
    return np.fft.irfft(spectrum, n=values.shape[-1], axis=-1).astype(np.float32)


def _speed_resample(values: np.ndarray, ratio: float) -> np.ndarray:
    length = values.shape[-1]
    source = np.linspace(0.0, 1.0, length, dtype=np.float32)
    target = np.clip(np.linspace(0.0, 1.0, length, dtype=np.float32) * ratio, 0.0, 1.0)
    return np.stack([np.interp(target, source, channel) for channel in values]).astype(np.float32)


def _time_mask(values: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    length = values.shape[-1]
    width = max(1, int(length * max(0.0, min(0.5, ratio))))
    start = int(rng.integers(0, max(1, length - width + 1)))
    result = values.copy()
    result[:, start : start + width] = 0.0
    return result


def _frequency_mask(values: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    length = values.shape[-1]
    width = max(1, int(length * max(0.0, min(0.5, ratio))))
    start = int(rng.integers(0, max(1, length - width + 1)))
    result = values.copy()
    result[:, start : start + width] = 0.0
    return result


def _spectral_smooth(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] < 3:
        return values
    padded = np.pad(values, ((0, 0), (1, 1)), mode="edge")
    return ((padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0).astype(np.float32)
