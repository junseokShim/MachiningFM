"""Unit tests for physics models.

All hand-calculated values are verified against manual calculations.
Tests cover: Taylor, Kienzle, Energy, Archard, Usui, PhysicsFeatures, PhysicsCalibrator.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from machiningfm.physics.archard import ArchardParams, compute_wear_volume_mm3
from machiningfm.physics.calibration import PhysicsCalibrator, PhysicsFeatures
from machiningfm.physics.energy import (
    compute_cumulative_energy_j,
    compute_cutting_power_w,
    compute_mrr_mm3_per_min,
    compute_specific_cutting_energy_j_per_mm3,
)
from machiningfm.physics.kienzle import (
    KienzleParams,
    compute_mean_chip_thickness_milling_mm,
    compute_physics_residual,
    estimate_cutting_force_n,
    estimate_specific_cutting_force_n_per_mm2,
)
from machiningfm.physics.taylor import (
    TaylorParams,
    compute_machining_progress,
    compute_tool_life_ratio,
    estimate_tool_life_min,
)
from machiningfm.physics.usui import UsuiParams, compute_wear_rate_mm_per_s


# ──────────────────────────────────────────────
# Taylor
# ──────────────────────────────────────────────

@pytest.fixture
def taylor_basic() -> TaylorParams:
    """Basic Taylor params: C=200 m/min, n=0.25."""
    return TaylorParams(C=200.0, n=0.25)


def test_taylor_basic_hand_calc(taylor_basic: TaylorParams) -> None:
    """V=100 m/min, C=200, n=0.25 → T = (200/100)^(1/0.25) = 2^4 = 16 min."""
    T = estimate_tool_life_min(
        cutting_speed_m_per_min=100.0,
        feed_mm_per_rev=1.0,
        axial_depth_mm=1.0,
        params=taylor_basic,
    )
    assert abs(T - 16.0) < 1e-9


def test_taylor_zero_speed_returns_inf(taylor_basic: TaylorParams) -> None:
    T = estimate_tool_life_min(0.0, 1.0, 1.0, taylor_basic)
    assert math.isinf(T)


def test_taylor_negative_speed_raises(taylor_basic: TaylorParams) -> None:
    with pytest.raises(ValueError, match="cutting_speed_m_per_min"):
        estimate_tool_life_min(-10.0, 1.0, 1.0, taylor_basic)


def test_taylor_invalid_feed_raises(taylor_basic: TaylorParams) -> None:
    with pytest.raises(ValueError, match="feed_mm_per_rev"):
        estimate_tool_life_min(100.0, 0.0, 1.0, taylor_basic)


def test_taylor_invalid_depth_raises(taylor_basic: TaylorParams) -> None:
    with pytest.raises(ValueError, match="axial_depth_mm"):
        estimate_tool_life_min(100.0, 1.0, -1.0, taylor_basic)


def test_taylor_zero_exponent_raises() -> None:
    with pytest.raises(ValueError, match="exponent n"):
        estimate_tool_life_min(100.0, 1.0, 1.0, TaylorParams(C=200.0, n=0.0))


def test_tool_life_ratio_half(taylor_basic: TaylorParams) -> None:
    """Elapsed 8 min out of 16 min life → ratio = 0.5."""
    ratio = compute_tool_life_ratio(8.0, 100.0, 1.0, 1.0, taylor_basic)
    assert abs(ratio - 0.5) < 1e-9


def test_tool_life_ratio_clamped(taylor_basic: TaylorParams) -> None:
    """Elapsed time > tool life → clamped at 1.0."""
    ratio = compute_tool_life_ratio(100.0, 100.0, 1.0, 1.0, taylor_basic)
    assert ratio == 1.0


def test_tool_life_ratio_zero_speed(taylor_basic: TaylorParams) -> None:
    """Zero speed → infinite life → ratio = 0.0."""
    ratio = compute_tool_life_ratio(10.0, 0.0, 1.0, 1.0, taylor_basic)
    assert ratio == 0.0


def test_machining_progress_alias(taylor_basic: TaylorParams) -> None:
    r1 = compute_tool_life_ratio(5.0, 100.0, 1.0, 1.0, taylor_basic)
    r2 = compute_machining_progress(5.0, 100.0, 1.0, 1.0, taylor_basic)
    assert r1 == r2


def test_taylor_generalized() -> None:
    """Generalized Taylor with m=0.5, p=0.5: hand check."""
    params = TaylorParams(C=200.0, n=0.25, m=0.5, p=0.5)
    # T = (C / (V * f^m * ap^p))^(1/n) = (200 / (100 * 0.25^0.5 * 0.25^0.5))^4
    # = (200 / (100 * 0.5 * 0.5))^4 = (200/25)^4 = 8^4 = 4096
    T = estimate_tool_life_min(100.0, 0.25, 0.25, params)
    assert abs(T - 4096.0) < 1.0


# ──────────────────────────────────────────────
# Kienzle
# ──────────────────────────────────────────────

@pytest.fixture
def kienzle_steel() -> KienzleParams:
    return KienzleParams(kc_1_1_n_per_mm2=1800.0, mc=0.26)


def test_kienzle_specific_force_h1(kienzle_steel: KienzleParams) -> None:
    """At h=1mm, k_c = k_c1.1 * 1^(-mc) = k_c1.1."""
    kc = estimate_specific_cutting_force_n_per_mm2(1.0, kienzle_steel)
    assert abs(kc - 1800.0) < 1e-9


def test_kienzle_specific_force_h01(kienzle_steel: KienzleParams) -> None:
    """h=0.1mm: k_c = 1800 * 0.1^(-0.26) = 1800 * 10^0.26."""
    expected = 1800.0 * (0.1 ** -0.26)
    kc = estimate_specific_cutting_force_n_per_mm2(0.1, kienzle_steel)
    assert abs(kc - expected) < 1e-6


def test_kienzle_cutting_force_h01_b2(kienzle_steel: KienzleParams) -> None:
    """F_c = 1800 * 2 * 0.1^(1-0.26) = 3600 * 0.1^0.74."""
    expected = 1800.0 * 2.0 * (0.1 ** 0.74)
    fc = estimate_cutting_force_n(0.1, 2.0, kienzle_steel)
    assert abs(fc - expected) < 1e-6


def test_kienzle_zero_chip_thickness_raises(kienzle_steel: KienzleParams) -> None:
    with pytest.raises(ValueError, match="chip_thickness_mm"):
        estimate_cutting_force_n(0.0, 2.0, kienzle_steel)


def test_kienzle_negative_chip_thickness_raises(kienzle_steel: KienzleParams) -> None:
    with pytest.raises(ValueError, match="chip_thickness_mm"):
        estimate_specific_cutting_force_n_per_mm2(-0.1, kienzle_steel)


def test_kienzle_zero_width_raises(kienzle_steel: KienzleParams) -> None:
    with pytest.raises(ValueError, match="cutting_width_mm"):
        estimate_cutting_force_n(0.1, 0.0, kienzle_steel)


def test_kienzle_physics_residual(kienzle_steel: KienzleParams) -> None:
    measured = 600.0
    result = compute_physics_residual(measured, 0.1, 2.0, kienzle_steel)
    expected_physics = estimate_cutting_force_n(0.1, 2.0, kienzle_steel)
    assert abs(result["physics_force_n"] - expected_physics) < 1e-6
    assert abs(result["force_delta_n"] - (measured - expected_physics)) < 1e-6
    assert abs(result["force_ratio"] - measured / expected_physics) < 1e-6


def test_kienzle_mean_chip_thickness(kienzle_steel: KienzleParams) -> None:
    """h_m = f_z * sqrt(ae/D) = 0.1 * sqrt(0.25) = 0.05 mm."""
    h = compute_mean_chip_thickness_milling_mm(0.1, 0.25)
    assert abs(h - 0.05) < 1e-9


def test_kienzle_invalid_engagement_ratio() -> None:
    with pytest.raises(ValueError, match="radial_engagement_ratio"):
        compute_mean_chip_thickness_milling_mm(0.1, 0.0)


# ──────────────────────────────────────────────
# Energy
# ──────────────────────────────────────────────

def test_cutting_power_basic() -> None:
    """F=500N, V=100m/min → P = 500 * (100/60) ≈ 833.333 W."""
    P = compute_cutting_power_w(500.0, 100.0)
    assert abs(P - 500.0 * 100.0 / 60.0) < 1e-6


def test_cutting_power_zero_speed() -> None:
    P = compute_cutting_power_w(500.0, 0.0)
    assert P == 0.0


def test_cutting_power_negative_speed_raises() -> None:
    with pytest.raises(ValueError, match="cutting_speed_m_per_min"):
        compute_cutting_power_w(500.0, -10.0)


def test_cumulative_energy_uniform() -> None:
    """F=[100, 200, 300]N, V=60m/min (=1m/s), dt=0.1s → E = 600*1*0.1 = 60 J."""
    forces = np.array([100.0, 200.0, 300.0])
    E = compute_cumulative_energy_j(forces, cutting_speed_m_per_min=60.0, time_step_s=0.1)
    assert abs(E - 60.0) < 1e-9


def test_cumulative_energy_empty() -> None:
    E = compute_cumulative_energy_j(np.array([]), 100.0, 0.01)
    assert E == 0.0


def test_cumulative_energy_zero_timestep_raises() -> None:
    with pytest.raises(ValueError, match="time_step_s"):
        compute_cumulative_energy_j(np.array([100.0]), 100.0, 0.0)


def test_mrr_basic() -> None:
    """V=200m/min, f=0.25mm/rev, ap=2mm, ae=10mm → MRR = 200000*0.25*2*10 = 1e6 mm³/min."""
    mrr = compute_mrr_mm3_per_min(200.0, 0.25, 2.0, 10.0)
    assert abs(mrr - 1_000_000.0) < 1.0


def test_specific_energy_basic() -> None:
    """Verify unit consistency: P = F*V, u = P/MRR."""
    F = 500.0  # N
    V = 120.0  # m/min
    mrr = 600_000.0  # mm³/min
    P_expected = F * V / 60.0  # W
    mrr_per_s = mrr / 60.0     # mm³/s
    u_expected = P_expected / mrr_per_s
    u = compute_specific_cutting_energy_j_per_mm3(F, V, mrr)
    assert abs(u - u_expected) < 1e-9


# ──────────────────────────────────────────────
# Archard
# ──────────────────────────────────────────────

def test_archard_disabled_returns_none() -> None:
    params = ArchardParams(wear_coefficient_k=1e-4, hardness_h_n_per_mm2=2000.0, available=False)
    assert compute_wear_volume_mm3(100.0, 1000.0, params) is None


def test_archard_enabled_hand_calc() -> None:
    """K=1e-4, F_N=100N, L=1000mm, H=2000 N/mm² → V_w = 1e-4*100*1000/2000 = 0.005 mm³."""
    params = ArchardParams(wear_coefficient_k=1e-4, hardness_h_n_per_mm2=2000.0, available=True)
    V_w = compute_wear_volume_mm3(100.0, 1000.0, params)
    assert V_w is not None
    assert abs(V_w - 0.005) < 1e-9


def test_archard_negative_force_raises() -> None:
    params = ArchardParams(wear_coefficient_k=1e-4, hardness_h_n_per_mm2=2000.0, available=True)
    with pytest.raises(ValueError, match="normal_force_n"):
        compute_wear_volume_mm3(-1.0, 100.0, params)


def test_archard_zero_hardness_raises() -> None:
    params = ArchardParams(wear_coefficient_k=1e-4, hardness_h_n_per_mm2=0.0, available=True)
    with pytest.raises(ValueError, match="hardness"):
        compute_wear_volume_mm3(100.0, 100.0, params)


# ──────────────────────────────────────────────
# Usui
# ──────────────────────────────────────────────

def test_usui_disabled_returns_none() -> None:
    params = UsuiParams(B1=1e-5, B2=4000.0, available=False)
    assert compute_wear_rate_mm_per_s(100.0, 500.0, 1000.0, params) is None


def test_usui_zero_temperature_raises() -> None:
    params = UsuiParams(B1=1e-5, B2=4000.0, available=True)
    with pytest.raises(ValueError, match="cutting_temperature_k"):
        compute_wear_rate_mm_per_s(100.0, 500.0, 0.0, params)


def test_usui_negative_velocity_raises() -> None:
    params = UsuiParams(B1=1e-5, B2=4000.0, available=True)
    with pytest.raises(ValueError, match="sliding_velocity_mm_per_s"):
        compute_wear_rate_mm_per_s(100.0, -10.0, 1000.0, params)


def test_usui_enabled_basic() -> None:
    """dW/dt = B1 * sigma * v * exp(-B2/T). Just check no crash and sign."""
    params = UsuiParams(B1=1e-5, B2=4000.0, available=True)
    rate = compute_wear_rate_mm_per_s(100.0, 500.0, 1200.0, params)
    assert rate is not None
    assert rate > 0.0


# ──────────────────────────────────────────────
# PhysicsFeatures
# ──────────────────────────────────────────────

def test_physics_features_all_none() -> None:
    pf = PhysicsFeatures()
    vec = pf.to_feature_vector()
    assert vec.shape == (0,)


def test_physics_features_some_none() -> None:
    pf = PhysicsFeatures(tool_life_ratio=0.5, force_ratio=1.2)
    vec = pf.to_feature_vector()
    assert vec.shape == (2,)
    assert abs(vec[0] - 0.5) < 1e-9
    assert abs(vec[1] - 1.2) < 1e-9


def test_physics_features_all_set() -> None:
    pf = PhysicsFeatures(
        tool_life_ratio=0.3,
        force_ratio=1.1,
        force_delta_n=50.0,
        cumulative_energy_j=200.0,
        specific_cutting_energy_j_per_mm3=2.5,
        wear_volume_mm3=0.01,
    )
    vec = pf.to_feature_vector()
    assert vec.shape == (6,)


def test_physics_features_names_match_non_none() -> None:
    pf = PhysicsFeatures(tool_life_ratio=0.5, cumulative_energy_j=100.0)
    names = pf.feature_names()
    assert "tool_life_ratio" in names
    assert "cumulative_energy_j" in names
    assert "force_ratio" not in names


# ──────────────────────────────────────────────
# PhysicsCalibrator
# ──────────────────────────────────────────────

def _make_synthetic_data(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    fm_preds = rng.uniform(0.05, 0.35, n)
    targets = fm_preds + rng.normal(0, 0.01, n)
    pf_list = [PhysicsFeatures(tool_life_ratio=float(rng.uniform(0, 1))) for _ in range(n)]
    return fm_preds, targets, pf_list


def test_calibrator_predict_before_fit_raises() -> None:
    cal = PhysicsCalibrator()
    with pytest.raises(RuntimeError, match="fitted"):
        cal.predict(np.array([0.1]), [PhysicsFeatures()])


def test_calibrator_select_alpha_before_fit_raises() -> None:
    cal = PhysicsCalibrator()
    with pytest.raises(RuntimeError, match="Fit"):
        cal.select_alpha(np.array([0.1]), [PhysicsFeatures()], np.array([0.1]))


def test_calibrator_fit_predict_ridge() -> None:
    fm, targets, pf = _make_synthetic_data(50)
    cal = PhysicsCalibrator(method="ridge")
    cal.fit(fm[:40], pf[:40], targets[:40])
    preds = cal.predict(fm[40:], pf[40:])
    assert preds.shape == (10,)


def test_calibrator_fit_predict_linear() -> None:
    fm, targets, pf = _make_synthetic_data(50)
    cal = PhysicsCalibrator(method="linear")
    cal.fit(fm[:40], pf[:40], targets[:40])
    preds = cal.predict(fm[40:], pf[40:])
    assert preds.shape == (10,)


def test_calibrator_fit_predict_mlp() -> None:
    fm, targets, pf = _make_synthetic_data(80)
    cal = PhysicsCalibrator(method="mlp")
    cal.fit(fm[:60], pf[:60], targets[:60])
    preds = cal.predict(fm[60:], pf[60:])
    assert preds.shape == (20,)


def test_calibrator_fit_predict_residual_mlp() -> None:
    fm, targets, pf = _make_synthetic_data(80)
    cal = PhysicsCalibrator(method="residual_mlp")
    cal.fit(fm[:60], pf[:60], targets[:60])
    preds = cal.predict(fm[60:], pf[60:])
    assert preds.shape == (20,)


def test_calibrator_select_alpha_returns_float() -> None:
    fm, targets, pf = _make_synthetic_data(60)
    cal = PhysicsCalibrator(method="ridge")
    cal.fit(fm[:40], pf[:40], targets[:40])
    best = cal.select_alpha(fm[40:50], pf[40:50], targets[40:50])
    assert 0.0 <= best <= 1.0


def test_calibrator_alpha_zero_returns_fm() -> None:
    """Alpha=0 → output equals FM predictions."""
    fm, targets, pf = _make_synthetic_data(50)
    cal = PhysicsCalibrator(method="ridge", alpha_strength=0.0)
    cal.fit(fm[:40], pf[:40], targets[:40])
    preds = cal.predict(fm[40:], pf[40:])
    np.testing.assert_allclose(preds, fm[40:], rtol=1e-5)


def test_calibrator_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        PhysicsCalibrator(method="unknown_method")


def test_calibrator_empty_physics_features() -> None:
    """All None PhysicsFeatures should still work (no physics info)."""
    fm, targets, _ = _make_synthetic_data(50)
    pf = [PhysicsFeatures() for _ in range(50)]
    cal = PhysicsCalibrator(method="ridge")
    cal.fit(fm[:40], pf[:40], targets[:40])
    preds = cal.predict(fm[40:], pf[40:])
    assert preds.shape == (10,)
