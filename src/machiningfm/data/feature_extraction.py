from __future__ import annotations

import cmath
import math
from typing import Iterable


def time_domain_features(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {name: 0.0 for name in ("mean", "std", "rms", "peak", "skewness", "kurtosis", "crest_factor")}
    count = len(data)
    mean = sum(data) / count
    centered = [value - mean for value in data]
    variance = sum(value * value for value in centered) / count
    std = math.sqrt(variance)
    rms = math.sqrt(sum(value * value for value in data) / count)
    peak = max(abs(value) for value in data)
    skewness = sum(value**3 for value in centered) / count / (std**3 + 1e-12)
    kurtosis = sum(value**4 for value in centered) / count / (std**4 + 1e-12)
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "crest_factor": peak / (rms + 1e-12),
    }


def frequency_domain_features(values: Iterable[float], sampling_rate: float = 1.0, max_points: int = 512) -> dict[str, float]:
    data = [float(value) for value in values][:max_points]
    if not data:
        return {"band_energy": 0.0, "spectral_centroid": 0.0, "dominant_frequency": 0.0, "spectral_entropy": 0.0}
    spectrum = []
    for frequency_index in range(len(data) // 2 + 1):
        coefficient = sum(
            value * cmath.exp(-2j * math.pi * frequency_index * time_index / len(data))
            for time_index, value in enumerate(data)
        )
        spectrum.append(abs(coefficient) ** 2)
    total = sum(spectrum) + 1e-12
    frequencies = [index * sampling_rate / len(data) for index in range(len(spectrum))]
    probabilities = [value / total for value in spectrum]
    return {
        "band_energy": total,
        "spectral_centroid": sum(freq * probability for freq, probability in zip(frequencies, probabilities)),
        "dominant_frequency": frequencies[max(range(len(spectrum)), key=spectrum.__getitem__)],
        "spectral_entropy": -sum(probability * math.log(probability + 1e-12) for probability in probabilities),
    }
