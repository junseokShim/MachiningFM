from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def finite_1d(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return np.zeros(8, dtype=np.float32)
    finite = np.isfinite(data)
    fill = float(np.median(data[finite])) if finite.any() else 0.0
    return np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def resample_1d(value: np.ndarray, length: int) -> np.ndarray:
    data = finite_1d(value)
    if data.size == length:
        return data
    if data.size <= 1:
        return np.full(length, float(data[0]) if data.size else 0.0, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, data.size),
        data,
    ).astype(np.float32)


def normalize(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    std = float(data.std())
    if not math.isfinite(std) or std < 1.0e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - float(data.mean())) / std, -10.0, 10.0).astype(np.float32)


def fft_features(value: np.ndarray, length: int = 256) -> np.ndarray:
    data = resample_1d(value, length)
    spectrum = np.fft.rfft(data * np.hanning(length).astype(np.float32))
    mag = np.log1p(np.abs(spectrum)).astype(np.float32)
    phase = np.unwrap(np.angle(spectrum)).astype(np.float32)
    return np.concatenate([mag, normalize(phase)], dtype=np.float32)


def stft_features(value: np.ndarray, n_fft: int = 128, hop: int = 64, max_frames: int = 32) -> np.ndarray:
    data = finite_1d(value)
    if data.size < n_fft:
        data = resample_1d(data, n_fft)
    starts = list(range(0, max(1, data.size - n_fft + 1), hop))[:max_frames] or [0]
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in starts:
        segment = data[start : start + n_fft]
        if segment.size < n_fft:
            segment = np.pad(segment, (0, n_fft - segment.size))
        frames.append(np.log1p(np.abs(np.fft.rfft(segment * window))))
    stacked = np.stack(frames).astype(np.float32)
    return np.concatenate([stacked.mean(axis=0), stacked.std(axis=0)]).astype(np.float32)


def cwt_features(value: np.ndarray, scales: int = 32) -> np.ndarray:
    data = finite_1d(value)
    widths = np.linspace(2, min(96, max(3, data.size // 4)), scales).astype(np.int32)
    energies = []
    for width in widths:
        kernel = np.ones(max(2, int(width)), dtype=np.float32) / max(2, int(width))
        smooth = np.convolve(data, kernel, mode="same")
        energies.append(float(np.sqrt(np.mean((data - smooth) ** 2))))
    return np.log1p(np.asarray(energies, dtype=np.float32))


def order_features(value: np.ndarray, rpm: float | None, sample_rate: float | None, max_order: int = 8) -> np.ndarray:
    if not rpm or rpm <= 0 or not sample_rate or sample_rate <= 0:
        return np.zeros(max_order, dtype=np.float32)
    data = finite_1d(value)
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size).astype(np.float32))).astype(np.float32)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / float(sample_rate))
    base = float(rpm) / 60.0
    amps = []
    for order in range(1, max_order + 1):
        target = base * order
        idx = int(np.argmin(np.abs(freqs - target)))
        amps.append(float(spectrum[idx]) if spectrum.size else 0.0)
    return normalize(np.asarray(amps, dtype=np.float32))


def build_spectral_features(
    values: np.ndarray,
    *,
    output_length: int = 512,
    transforms: tuple[str, ...] | list[str] = ("fft", "stft", "cwt"),
    rpm: float | None = None,
    sample_rate: float | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    enabled = {name.lower() for name in transforms}
    rows = []
    for row in array:
        parts = []
        if "fft" in enabled:
            parts.append(fft_features(row))
        if "stft" in enabled:
            parts.append(stft_features(row))
        if "cwt" in enabled:
            parts.append(cwt_features(row))
        parts.append(order_features(row, rpm, sample_rate))
        rows.append(normalize(resample_1d(np.concatenate(parts), output_length)))
    return np.stack(rows).astype(np.float32)


class SpectralTokenizer(nn.Module):
    def __init__(self, input_length: int = 512, d_model: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_length), nn.Linear(input_length, d_model), nn.GELU())

    def forward(self, spectral: torch.Tensor) -> torch.Tensor:
        if spectral.ndim != 3:
            raise ValueError(f"spectral must be [batch, channels, bins], got {tuple(spectral.shape)}")
        return self.proj(spectral.float())
