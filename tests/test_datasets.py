"""Unit tests for dataset pipeline and preprocessing.

Uses only synthetic data — no real PHM2010 download required.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from machiningfm.data.preprocessing import (
    compute_normalization_stats,
    extract_statistical_features,
    normalize_features,
)
from machiningfm.tasks.wear_stage import WEAR_STAGE_THRESHOLDS_MM, classify_wear_stage
from machiningfm.tasks.rul import ISO_WEAR_LIMIT_MM, compute_rul_from_wear
from machiningfm.tasks.dimensional_compensation import DimensionalCompensator


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def test_extract_features_shape() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, (500, 3))
    channels = ["fx", "fy", "fz"]
    feat = extract_statistical_features(signal, channels)
    # 3 channels × 7 features = 21
    assert len(feat) == 21


def test_extract_features_keys() -> None:
    signal = np.ones((100, 2))
    feat = extract_statistical_features(signal, ["ch1", "ch2"])
    expected_keys = {"ch1_mean", "ch1_std", "ch1_rms", "ch1_peak",
                     "ch1_kurtosis", "ch1_skewness", "ch1_crest_factor"}
    assert expected_keys.issubset(set(feat.keys()))


def test_extract_features_no_nan() -> None:
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 1, (200, 4))
    feat = extract_statistical_features(signal, ["a", "b", "c", "d"])
    assert all(np.isfinite(v) for v in feat.values())


def test_extract_features_1d_input() -> None:
    signal = np.ones(100)
    feat = extract_statistical_features(signal, ["ch"])
    assert "ch_mean" in feat


def test_extract_features_channel_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="channels"):
        extract_statistical_features(np.ones((100, 3)), ["a", "b"])


def test_normalization_zero_mean() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(5, 2, (100, 10))
    mean, std = compute_normalization_stats(X_train)
    X_norm = normalize_features(X_train, mean, std)
    # Normalized training set should have mean ≈ 0
    np.testing.assert_allclose(np.mean(X_norm, axis=0), 0.0, atol=1e-10)


def test_normalization_does_not_use_test() -> None:
    """Verify that test normalization uses train stats, not test stats."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, (100, 5))
    X_test = rng.normal(10, 1, (20, 5))  # very different distribution
    mean, std = compute_normalization_stats(X_train)
    X_test_norm = normalize_features(X_test, mean, std)
    # Test mean should be ~10 (not 0) because we used train stats
    assert np.mean(X_test_norm) > 5.0


# ──────────────────────────────────────────────
# Wear stage classification
# ──────────────────────────────────────────────

def test_classify_healthy_below_threshold() -> None:
    assert classify_wear_stage(0.0) == 0
    assert classify_wear_stage(0.05) == 0
    assert classify_wear_stage(0.09) == 0


def test_classify_moderate_at_lower_boundary() -> None:
    assert classify_wear_stage(WEAR_STAGE_THRESHOLDS_MM["healthy_max"]) == 1


def test_classify_moderate_mid() -> None:
    assert classify_wear_stage(0.15) == 1


def test_classify_severe_at_lower_boundary() -> None:
    assert classify_wear_stage(WEAR_STAGE_THRESHOLDS_MM["moderate_max"]) == 2


def test_classify_severe_above_threshold() -> None:
    assert classify_wear_stage(0.35) == 2


# ──────────────────────────────────────────────
# RUL computation
# ──────────────────────────────────────────────

def test_rul_decreasing() -> None:
    """RUL should decrease as wear progresses."""
    wear = np.linspace(0.0, 0.35, 20)
    rul = compute_rul_from_wear(wear, wear_limit_mm=0.3, time_per_cut_min=1.0)
    # All values before EOL should be decreasing
    eol = np.where(wear >= 0.3)[0][0]
    assert np.all(np.diff(rul[:eol]) <= 0)


def test_rul_zero_after_eol() -> None:
    """RUL = 0 at or after end-of-life."""
    wear = np.array([0.0, 0.1, 0.3, 0.35])
    rul = compute_rul_from_wear(wear, wear_limit_mm=0.3, time_per_cut_min=1.0)
    assert rul[2] == 0.0
    assert rul[3] == 0.0


def test_rul_never_reaches_limit() -> None:
    """If wear never reaches limit, RUL = time remaining after last cut."""
    wear = np.array([0.0, 0.05, 0.1])
    rul = compute_rul_from_wear(wear, wear_limit_mm=0.3, time_per_cut_min=2.0)
    # Cut 0 has 3 cuts remaining → RUL = 3*2 = 6 min
    assert rul[0] == 6.0


# ──────────────────────────────────────────────
# Dimensional compensator
# ──────────────────────────────────────────────

def test_compensator_with_measurement() -> None:
    comp = DimensionalCompensator()
    result = comp.predict(
        predicted_wear_vb_mm=0.1,
        nominal_dimension_mm=20.0,
        measured_dimension_mm=20.018,
        axis="X",
    )
    # Offset should correct the +0.018 error → offset ≈ -0.018
    assert abs(result.recommended_offset_mm - (-0.018)) < 1e-4
    assert result.confidence > 0.8
    assert "experimentally validated" not in result.reason.lower()


def test_compensator_without_measurement() -> None:
    comp = DimensionalCompensator()
    result = comp.predict(
        predicted_wear_vb_mm=0.2,
        nominal_dimension_mm=50.0,
        measured_dimension_mm=None,
    )
    assert result.recommended_offset_mm <= 0.0  # wear causes positive error → negative offset
    assert 0.0 <= result.confidence <= 1.0
    assert "PHYSICS-DERIVED" in result.reason


def test_compensator_negative_wear_raises() -> None:
    comp = DimensionalCompensator()
    with pytest.raises(ValueError, match="predicted_wear_vb_mm"):
        comp.predict(-0.1, 20.0)


def test_compensator_zero_wear() -> None:
    comp = DimensionalCompensator()
    result = comp.predict(0.0, 20.0)
    assert result.recommended_offset_mm == 0.0


# ──────────────────────────────────────────────
# Synthetic PHM2010 dataset
# ──────────────────────────────────────────────

def _create_synthetic_phm2010(tmp_dir: Path, n_conditions: int = 3, n_cuts: int = 10) -> None:
    """Helper: create minimal synthetic PHM2010 structure for testing."""
    train_dir = tmp_dir / "train"
    conditions = [f"c{i+1}" for i in range(n_conditions)]
    rng = np.random.default_rng(42)

    rows = [["condition", "cut", "flute_1_wear_mm", "flute_2_wear_mm", "flute_3_wear_mm", "average_wear_mm"]]
    header = "time,force_x,force_y,force_z,vibration_x,vibration_y,vibration_z,acoustic_emission_rms"

    for cond in conditions:
        (train_dir / cond).mkdir(parents=True, exist_ok=True)
        for cut in range(1, n_cuts + 1):
            vb = 0.35 * cut / n_cuts
            cut_rows = [header]
            for i in range(50):
                t = i * 0.01
                vals = rng.normal(0, 1, 7)
                cut_rows.append(f"{t:.3f}," + ",".join(f"{v:.4f}" for v in vals))
            (train_dir / cond / f"{cond}_{cut:03d}.csv").write_text("\n".join(cut_rows))
            rows.append([cond, cut, f"{vb:.4f}", f"{vb:.4f}", f"{vb:.4f}", f"{vb:.4f}"])

    with open(tmp_dir / "train_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_phm2010_dataset_loads() -> None:
    from machiningfm.data.datasets import PHM2010Dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_synthetic_phm2010(tmp_path, n_conditions=3, n_cuts=10)

        ds = PHM2010Dataset(
            tmp_path, "train",
            test_conditions=["c3"],
            val_conditions=["c2"],
        )
        assert len(ds) > 0
        sample = ds[0]
        assert sample.wear_vb_mm >= 0
        assert sample.sensor_features.ndim == 1


def test_phm2010_split_non_overlapping() -> None:
    from machiningfm.data.datasets import PHM2010Dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_synthetic_phm2010(tmp_path, n_conditions=3, n_cuts=5)

        train_ds = PHM2010Dataset(
            tmp_path, "train",
            test_conditions=["c3"],
            val_conditions=["c2"],
        )
        test_ds = PHM2010Dataset(
            tmp_path, "test",
            test_conditions=["c3"],
            val_conditions=["c2"],
            feature_stats_from_train=train_ds.get_normalization_stats(),
        )

        train_conds = {s.condition_id for s in train_ds.samples}
        test_conds = {s.condition_id for s in test_ds.samples}
        # Train and test condition sets must be disjoint
        assert len(train_conds & test_conds) == 0


def test_phm2010_val_requires_train_stats() -> None:
    from machiningfm.data.datasets import PHM2010Dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _create_synthetic_phm2010(tmp_path, n_conditions=3, n_cuts=5)

        with pytest.raises(ValueError, match="feature_stats_from_train"):
            PHM2010Dataset(
                tmp_path, "val",
                test_conditions=["c3"],
                val_conditions=["c2"],
                feature_stats_from_train=None,
            )
