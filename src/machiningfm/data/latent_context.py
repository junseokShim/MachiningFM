from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from machiningfm.data.channel_schema import describe_channel
from machiningfm.utils.io import read_json, write_json

LATENT_CONTEXT_SCHEMA_VERSION = "latent-context-v1"
DEFAULT_CLUSTER_COUNTS = {
    "latent_tool": 12,
    "latent_material": 12,
    "latent_machine": 8,
    "latent_process": 8,
}
FEATURE_NAMES = [
    "channel_count",
    "external_sensor_ratio",
    "machine_controller_ratio",
    "vibration_ratio",
    "force_ratio",
    "torque_ratio",
    "current_ratio",
    "position_ratio",
    "position_error_ratio",
    "mean_rms",
    "std_rms",
    "max_rms",
    "mean_peak",
    "mean_crest_factor",
    "mean_kurtosis",
    "mean_spectral_centroid",
    "mean_spectral_entropy",
    "mean_high_band_ratio",
    "mean_mid_band_ratio",
    "mean_low_band_ratio",
    "mean_abs_corr",
]


@dataclass
class LatentContextModel:
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    centers: dict[str, list[list[float]]]
    distance_scales: dict[str, float]

    def tokens_for_signal(
        self,
        values: np.ndarray,
        channel_names: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        features = extract_signal_features(values, channel_names)
        standardized = _standardize(features, np.asarray(self.mean), np.asarray(self.std))
        tokens = heuristic_latent_tokens(values, channel_names)
        assignments: dict[str, Any] = {}
        for family, centers_value in self.centers.items():
            centers = np.asarray(centers_value, dtype=np.float32)
            if centers.size == 0:
                continue
            distances = np.linalg.norm(centers - standardized[None, :], axis=1)
            cluster = int(np.argmin(distances))
            confidence = _confidence(float(distances[cluster]), self.distance_scales.get(family, 1.0))
            tokens.append(f"{family}_cluster_{cluster:02d}")
            tokens.append(f"{family}_confidence_{confidence}")
            assignments[family] = {
                "cluster": cluster,
                "distance": float(distances[cluster]),
                "confidence": confidence,
            }
        return tokens, {"features": _feature_dict(features), "assignments": assignments}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LATENT_CONTEXT_SCHEMA_VERSION,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "centers": self.centers,
            "distance_scales": self.distance_scales,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LatentContextModel":
        if value.get("schema_version") != LATENT_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported latent context schema {value.get('schema_version')!r}; "
                f"expected {LATENT_CONTEXT_SCHEMA_VERSION!r}"
            )
        return cls(
            feature_names=list(value["feature_names"]),
            mean=[float(item) for item in value["mean"]],
            std=[float(item) for item in value["std"]],
            centers={
                str(name): [[float(item) for item in row] for row in rows]
                for name, rows in value["centers"].items()
            },
            distance_scales={str(name): float(item) for name, item in value.get("distance_scales", {}).items()},
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "LatentContextModel | None":
        if not path:
            return None
        source = Path(path)
        if not source.exists():
            return None
        return cls.from_dict(read_json(source))

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())


def fit_latent_context_model(
    feature_matrix: np.ndarray,
    cluster_counts: dict[str, int] | None = None,
    seed: int = 42,
    iterations: int = 50,
) -> LatentContextModel:
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
        raise ValueError("feature_matrix must be [N, F] with at least one row")
    counts = cluster_counts or DEFAULT_CLUSTER_COUNTS
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    standardized = _standardize(feature_matrix, mean, std)
    centers: dict[str, list[list[float]]] = {}
    scales: dict[str, float] = {}
    for index, (family, count) in enumerate(counts.items()):
        fitted = _kmeans(standardized, min(max(1, count), len(standardized)), seed + index * 997, iterations)
        centers[family] = fitted.tolist()
        distances = _nearest_distances(standardized, fitted)
        scales[family] = float(np.median(distances) + 1e-6)
    return LatentContextModel(
        feature_names=list(FEATURE_NAMES),
        mean=mean.astype(float).tolist(),
        std=np.maximum(std, 1e-6).astype(float).tolist(),
        centers=centers,
        distance_scales=scales,
    )


def extract_signal_features(values: np.ndarray, channel_names: list[str]) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"Expected signal values [C, T], got {tuple(series.shape)}")
    descriptors = [describe_channel(name) for name in channel_names]
    channel_count = max(1, series.shape[0])
    sources = [descriptor.source for descriptor in descriptors]
    quantities = [descriptor.quantity for descriptor in descriptors]
    rms = np.sqrt(np.mean(series * series, axis=1))
    peaks = np.max(np.abs(series), axis=1)
    centered = series - series.mean(axis=1, keepdims=True)
    std = np.std(series, axis=1) + 1e-6
    kurtosis = np.mean(centered**4, axis=1) / (std**4)
    crest = peaks / (rms + 1e-6)
    spectral = np.asarray([_spectral_features(channel) for channel in series], dtype=np.float32)
    corr = _mean_abs_corr(series)
    features = [
        float(channel_count),
        _ratio(sources, "external_sensor"),
        _ratio(sources, "machine_controller"),
        _ratio(quantities, "vibration"),
        _ratio(quantities, "force"),
        _ratio(quantities, "torque"),
        _ratio(quantities, "current"),
        _ratio(quantities, "position"),
        _ratio(quantities, "position_error"),
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.max(rms)),
        float(np.mean(peaks)),
        float(np.mean(crest)),
        float(np.mean(kurtosis)),
        float(np.mean(spectral[:, 0])),
        float(np.mean(spectral[:, 1])),
        float(np.mean(spectral[:, 2])),
        float(np.mean(spectral[:, 3])),
        float(np.mean(spectral[:, 4])),
        corr,
    ]
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def heuristic_latent_tokens(values: np.ndarray, channel_names: list[str]) -> list[str]:
    features = extract_signal_features(values, channel_names)
    feature_values = _feature_dict(features)
    descriptors = [describe_channel(name) for name in channel_names]
    quantities = sorted({descriptor.quantity for descriptor in descriptors if descriptor.quantity != "unknown"})
    tokens = [f"latent_quantity_{quantity}" for quantity in quantities[:8]]
    tokens.append(f"latent_channel_count_{_bucket(feature_values['channel_count'], (3, 8, 16))}")
    tokens.append(f"latent_dynamic_{_bucket(feature_values['mean_crest_factor'], (1.8, 2.8, 4.0))}")
    tokens.append(f"latent_spectral_entropy_{_bucket(feature_values['mean_spectral_entropy'], (1.0, 2.0, 3.0))}")
    tokens.append(f"latent_high_band_{_bucket(feature_values['mean_high_band_ratio'], (0.15, 0.35, 0.55))}")
    tokens.append(f"latent_cross_channel_{_bucket(feature_values['mean_abs_corr'], (0.15, 0.35, 0.60))}")
    return tokens


def _spectral_features(channel: np.ndarray) -> tuple[float, float, float, float, float]:
    data = np.asarray(channel, dtype=np.float32)
    if len(data) < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    data = data - float(np.mean(data))
    spectrum = np.abs(np.fft.rfft(data, n=min(512, len(data)))) ** 2
    total = float(np.sum(spectrum) + 1e-12)
    probabilities = spectrum / total
    frequencies = np.linspace(0.0, 1.0, len(spectrum), dtype=np.float32)
    centroid = float(np.sum(frequencies * probabilities))
    entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    low = float(np.sum(probabilities[: max(1, len(probabilities) // 4)]))
    mid = float(np.sum(probabilities[len(probabilities) // 4 : max(len(probabilities) // 4 + 1, len(probabilities) // 2)]))
    high = float(np.sum(probabilities[len(probabilities) // 2 :]))
    return centroid, entropy, high, mid, low


def _mean_abs_corr(series: np.ndarray) -> float:
    if series.shape[0] < 2 or series.shape[1] < 2:
        return 0.0
    valid = np.std(series, axis=1) > 1e-6
    if int(np.sum(valid)) < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(series[valid])
    if corr.ndim != 2:
        return 0.0
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    finite = np.isfinite(upper)
    if not upper.size or not finite.any():
        return 0.0
    return float(np.mean(np.abs(upper[finite])))


def _ratio(values: list[str], target: str) -> float:
    if not values:
        return 0.0
    return sum(value == target for value in values) / len(values)


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / np.maximum(std, 1e-6)


def _kmeans(values: np.ndarray, count: int, seed: int, iterations: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(values) <= count:
        return values.astype(np.float32, copy=True)
    centers = values[rng.choice(len(values), size=count, replace=False)].astype(np.float32, copy=True)
    for _ in range(iterations):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for cluster in range(count):
            members = values[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _nearest_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    return np.min(distances, axis=1)


def _confidence(distance: float, scale: float) -> str:
    if distance <= scale:
        return "high"
    if distance <= scale * 2.0:
        return "medium"
    return "low"


def _bucket(value: float, thresholds: tuple[float, float, float]) -> str:
    if value < thresholds[0]:
        return "low"
    if value < thresholds[1]:
        return "medium"
    if value < thresholds[2]:
        return "high"
    return "very_high"


def _feature_dict(features: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
