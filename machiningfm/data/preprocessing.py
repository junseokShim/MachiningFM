"""Signal preprocessing utilities for machining sensor data."""
from __future__ import annotations

import numpy as np
from scipy import stats


def extract_statistical_features(
    signal_array: np.ndarray,
    channel_names: list[str],
) -> dict[str, float]:
    """
    Extract per-channel statistical features from a multi-channel time-series.

    For each channel, computes: mean, std, rms, peak, kurtosis, skewness, crest_factor.
    These 7 features × n_channels become the feature vector for downstream models.

    Args:
        signal_array: Shape (T, n_channels). T is number of time samples.
        channel_names: Channel labels, length n_channels.

    Returns:
        Flat dict: {'channel_name_stat': value, ...}
    """
    if signal_array.ndim == 1:
        signal_array = signal_array.reshape(-1, 1)
    if signal_array.shape[1] != len(channel_names):
        raise ValueError(
            f"signal_array has {signal_array.shape[1]} channels "
            f"but channel_names has {len(channel_names)} entries."
        )

    features: dict[str, float] = {}
    for i, name in enumerate(channel_names):
        col = signal_array[:, i].astype(np.float64)
        rms = float(np.sqrt(np.mean(col ** 2)))
        peak = float(np.max(np.abs(col)))
        features[f"{name}_mean"] = float(np.mean(col))
        features[f"{name}_std"] = float(np.std(col))
        features[f"{name}_rms"] = rms
        features[f"{name}_peak"] = peak
        features[f"{name}_kurtosis"] = float(stats.kurtosis(col))
        features[f"{name}_skewness"] = float(stats.skew(col))
        features[f"{name}_crest_factor"] = peak / (rms + 1e-10)
    return features


def compute_normalization_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training features.

    Must be called on TRAINING data only to prevent leakage into val/test.

    Args:
        features: Shape (N, d). Training feature matrix.

    Returns:
        (mean, std) each of shape (d,).
    """
    return np.mean(features, axis=0), np.std(features, axis=0)


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Apply standard normalization using pre-computed statistics.

    Stats must come from the training set to avoid leakage.

    Args:
        features: Shape (N, d).
        mean: Shape (d,). Training-set mean.
        std: Shape (d,). Training-set standard deviation.
        epsilon: Numerical stability term to avoid division by zero.

    Returns:
        Normalized features, shape (N, d).
    """
    return (features - mean) / (std + epsilon)
