"""Unit tests for downstream task implementations."""
from __future__ import annotations

import numpy as np
import pytest

from machiningfm.evaluation.metrics import classification_metrics, regression_metrics
from machiningfm.tasks.dimensional_compensation import CompensationResult, DimensionalCompensator
from machiningfm.tasks.rul import RULPredictor, compute_rul_from_wear, compute_taylor_rul_min
from machiningfm.tasks.tool_wear import ToolWearRegressor
from machiningfm.tasks.wear_stage import WearStageClassifier, classify_wear_stage


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def test_regression_metrics_perfect() -> None:
    y = np.array([0.1, 0.2, 0.3])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_known() -> None:
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 0.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)


def test_classification_metrics_perfect() -> None:
    y = np.array([0, 1, 2, 0])
    m = classification_metrics(y, y, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_classification_metrics_class_names() -> None:
    y = np.array([0, 1])
    m = classification_metrics(y, y, ["healthy", "moderate"])
    assert "class_names" in m
    assert m["class_names"] == ["healthy", "moderate"]


def test_classification_metrics_confusion_matrix_shape() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    m = classification_metrics(y_true, y_pred)
    cm = m["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


# ──────────────────────────────────────────────
# ToolWearRegressor
# ──────────────────────────────────────────────

def _make_wear_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 10))
    y = 0.1 + 0.2 * X[:, 0] + rng.normal(0, 0.01, n)
    y = np.clip(y, 0.0, 0.5)
    return X, y


def test_tool_wear_regressor_fit_predict() -> None:
    X, y = _make_wear_data(60)
    reg = ToolWearRegressor(feature_dim=10)
    metrics = reg.fit(X[:40], y[:40])
    assert "mae" in metrics
    preds = reg.predict(X[40:])
    assert preds.shape == (20,)


def test_tool_wear_regressor_evaluate() -> None:
    X, y = _make_wear_data(60)
    reg = ToolWearRegressor(feature_dim=10)
    reg.fit(X[:40], y[:40])
    m = reg.evaluate(X[40:], y[40:])
    assert all(k in m for k in ("mae", "rmse", "r2"))


def test_tool_wear_predict_before_fit_raises() -> None:
    reg = ToolWearRegressor(feature_dim=10)
    with pytest.raises(RuntimeError, match="fitted"):
        reg.predict(np.ones((5, 10)))


def test_tool_wear_invalid_backbone_mode() -> None:
    with pytest.raises(ValueError, match="backbone_mode"):
        ToolWearRegressor(feature_dim=10, backbone_mode="invalid")


def test_tool_wear_with_physics_calibration() -> None:
    from machiningfm.physics.calibration import PhysicsCalibrator, PhysicsFeatures

    X, y = _make_wear_data(80)
    pf = [PhysicsFeatures(tool_life_ratio=float(i / 80)) for i in range(80)]

    cal = PhysicsCalibrator(method="ridge")
    reg = ToolWearRegressor(feature_dim=10, calibrator=cal)
    reg.fit(
        X[:50], y[:50],
        physics_features_train=pf[:50],
        features_val=X[50:60],
        targets_val=y[50:60],
        physics_features_val=pf[50:60],
    )
    preds = reg.predict(X[60:], pf[60:])
    assert preds.shape == (20,)


# ──────────────────────────────────────────────
# WearStageClassifier
# ──────────────────────────────────────────────

def _make_stage_data(n: int = 90, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 8))
    # Assign stages such that first third is 0, second 1, third 2
    labels = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))
    X += labels.reshape(-1, 1) * 0.5  # add signal
    return X, labels


def test_wear_stage_classifier_fit_predict() -> None:
    X, y = _make_stage_data()
    clf = WearStageClassifier()
    clf.fit(X[:60], y[:60])
    preds = clf.predict(X[60:])
    assert preds.shape == (30,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_wear_stage_classifier_evaluate() -> None:
    X, y = _make_stage_data()
    clf = WearStageClassifier()
    clf.fit(X[:60], y[:60])
    m = clf.evaluate(X[60:], y[60:])
    assert all(k in m for k in ("accuracy", "macro_f1", "confusion_matrix"))
    assert 0.0 <= m["accuracy"] <= 1.0


def test_wear_stage_predict_before_fit_raises() -> None:
    clf = WearStageClassifier()
    with pytest.raises(RuntimeError, match="fitted"):
        clf.predict(np.ones((5, 8)))


# ──────────────────────────────────────────────
# RULPredictor
# ──────────────────────────────────────────────

def _make_rul_data(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 6))
    rul = np.linspace(30.0, 0.0, n) + rng.normal(0, 0.5, n)
    rul = np.maximum(0.0, rul)
    return X, rul


def test_rul_predictor_fit_predict() -> None:
    X, rul = _make_rul_data()
    pred = RULPredictor()
    pred.fit(X[:40], rul[:40])
    preds = pred.predict(X[40:])
    assert preds.shape == (20,)
    assert np.all(preds >= 0)  # RUL must be non-negative


def test_rul_predictor_evaluate() -> None:
    X, rul = _make_rul_data()
    pred = RULPredictor()
    pred.fit(X[:40], rul[:40])
    m = pred.evaluate(X[40:], rul[40:])
    assert all(k in m for k in ("mae", "rmse", "r2"))


def test_rul_predictor_predict_before_fit_raises() -> None:
    pred = RULPredictor()
    with pytest.raises(RuntimeError, match="fitted"):
        pred.predict(np.ones((5, 6)))


def test_compute_taylor_rul() -> None:
    rul = compute_taylor_rul_min(elapsed_time_min=5.0, tool_life_taylor_min=16.0)
    assert abs(rul - 11.0) < 1e-9


def test_compute_taylor_rul_past_eol() -> None:
    rul = compute_taylor_rul_min(elapsed_time_min=20.0, tool_life_taylor_min=16.0)
    assert rul == 0.0


def test_compute_taylor_rul_negative_time_raises() -> None:
    with pytest.raises(ValueError, match="elapsed_time_min"):
        compute_taylor_rul_min(-1.0, 16.0)


# ──────────────────────────────────────────────
# DimensionalCompensator
# ──────────────────────────────────────────────

def test_compensator_result_dataclass() -> None:
    comp = DimensionalCompensator()
    result = comp.predict(0.05, 10.0)
    assert isinstance(result, CompensationResult)
    assert isinstance(result.recommended_offset_mm, float)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.reason, str) and len(result.reason) > 0


def test_compensator_with_measurement_exact() -> None:
    comp = DimensionalCompensator()
    result = comp.predict(0.1, 20.0, measured_dimension_mm=20.018)
    assert abs(result.recommended_offset_mm - (-0.018)) < 1e-4


def test_compensator_high_wear_lower_confidence() -> None:
    comp = DimensionalCompensator()
    low_wear = comp.predict(0.05, 20.0)
    high_wear = comp.predict(0.3, 20.0)
    # Higher wear → lower confidence (empirical model less reliable)
    assert high_wear.confidence < low_wear.confidence
