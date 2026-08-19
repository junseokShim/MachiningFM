from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_FREQUENCY_BANDS: dict[str, tuple[float, float | None]] = {
    "low": (0.0, 100.0),
    "mid": (100.0, 1000.0),
    "high": (1000.0, None),
}


def build_frequency_view(
    value: np.ndarray,
    sampling_rate: float | None,
    transforms: tuple[str, ...] | list[str] = ("fft", "stft", "cwt"),
    output_length: int = 512,
    frequency_bands: dict[str, Any] | None = None,
) -> np.ndarray:
    enabled = {name.lower() for name in transforms}
    parts: list[np.ndarray] = []
    if "fft" in enabled:
        parts.append(fft_magnitude(value, output_length=min(256, output_length)))
        parts.append(log_power_spectrum(value, output_length=min(256, output_length)))
        parts.append(band_energy(value, sampling_rate, frequency_bands))
    if "stft" in enabled:
        parts.append(stft_spectrogram_summary(value))
    if "cwt" in enabled:
        parts.append(wavelet_scalogram_summary(value))
    if not parts:
        parts.append(fft_magnitude(value, output_length=min(256, output_length)))
    return normalize_frequency_vector(resample_vector(np.concatenate(parts), output_length))


def fft_magnitude(value: np.ndarray, output_length: int = 256) -> np.ndarray:
    data = finite_vector(value)
    windowed = resample_vector(data, output_length) * np.hanning(output_length).astype(np.float32)
    return np.log1p(np.abs(np.fft.rfft(windowed))).astype(np.float32)


def log_power_spectrum(value: np.ndarray, output_length: int = 256) -> np.ndarray:
    magnitude = fft_magnitude(value, output_length=output_length)
    return np.log1p(magnitude * magnitude).astype(np.float32)


def band_energy(
    value: np.ndarray,
    sampling_rate: float | None,
    frequency_bands: dict[str, Any] | None = None,
) -> np.ndarray:
    data = finite_vector(value)
    n_fft = max(64, min(2048, int(2 ** math.ceil(math.log2(max(8, len(data)))))))
    windowed = resample_vector(data, n_fft) * np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(windowed)).astype(np.float32) ** 2
    if sampling_rate and sampling_rate > 0:
        frequencies = np.fft.rfftfreq(n_fft, d=1.0 / float(sampling_rate))
        nyquist = float(sampling_rate) / 2.0
    else:
        frequencies = np.linspace(0.0, 1.0, len(spectrum), dtype=np.float32)
        nyquist = 1.0
    bands = _coerce_bands(frequency_bands)
    total = float(np.sum(spectrum)) + 1e-8
    energies = []
    for lower, upper in bands.values():
        high = nyquist if upper is None else float(upper)
        mask = (frequencies >= float(lower)) & (frequencies <= high)
        energies.append(float(np.sum(spectrum[mask]) / total) if np.any(mask) else 0.0)
    centroid = float(np.sum(frequencies * spectrum) / total)
    peak = float(frequencies[int(np.argmax(spectrum))]) if spectrum.size else 0.0
    high_ratio = energies[-1] if energies else 0.0
    return np.asarray([*energies, centroid, peak, high_ratio], dtype=np.float32)


def stft_spectrogram_summary(value: np.ndarray, n_fft: int = 128, hop: int = 64) -> np.ndarray:
    data = finite_vector(value)
    if len(data) < n_fft:
        data = resample_vector(data, n_fft)
    starts = list(range(0, max(1, len(data) - n_fft + 1), hop)) or [0]
    window = np.hanning(n_fft).astype(np.float32)
    spectra = []
    for start in starts[:32]:
        segment = data[start : start + n_fft]
        if len(segment) < n_fft:
            segment = np.pad(segment, (0, n_fft - len(segment)))
        spectra.append(np.log1p(np.abs(np.fft.rfft(segment * window))))
    stacked = np.stack(spectra).astype(np.float32)
    return np.concatenate([stacked.mean(axis=0), stacked.std(axis=0)]).astype(np.float32)


def wavelet_scalogram_summary(value: np.ndarray, scale_count: int = 32) -> np.ndarray:
    data = finite_vector(value)
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


def finite_vector(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(8, dtype=np.float32)
    fill = float(np.median(data[finite]))
    return np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def resample_vector(value: np.ndarray, length: int) -> np.ndarray:
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


def normalize_frequency_vector(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    mean = float(data.mean())
    std = float(data.std())
    if not math.isfinite(std) or std < 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - mean) / std, -10.0, 10.0).astype(np.float32)


def _coerce_bands(raw: dict[str, Any] | None) -> dict[str, tuple[float, float | None]]:
    if not raw:
        return DEFAULT_FREQUENCY_BANDS
    bands: dict[str, tuple[float, float | None]] = {}
    for name, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            upper = None if value[1] is None else float(value[1])
            bands[str(name)] = (float(value[0]), upper)
    return bands or DEFAULT_FREQUENCY_BANDS
