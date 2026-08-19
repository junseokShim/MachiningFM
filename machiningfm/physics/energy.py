"""
Cutting Power and Specific Cutting Energy calculations.

Units:
    cutting_force_n                     : N
    cutting_speed_m_per_min             : m/min
    cutting_power_w                     : W  (= N·m/s)
    mrr_mm3_per_min                     : mm³/min
    specific_cutting_energy_j_per_mm3   : J/mm³
    cumulative_energy_j                 : J
    time_step_s                         : s
"""
from __future__ import annotations

import numpy as np

_M_PER_MIN_TO_M_PER_S = 1.0 / 60.0
_MM3_PER_MIN_TO_MM3_PER_S = 1.0 / 60.0


def compute_cutting_power_w(
    cutting_force_n: float,
    cutting_speed_m_per_min: float,
) -> float:
    """
    Compute cutting power: P_c = F_c * V_c

    Args:
        cutting_force_n: Cutting force in Newtons.
        cutting_speed_m_per_min: Cutting speed in m/min. Must be >= 0.

    Returns:
        Cutting power in Watts (W = N·m/s).
    """
    if cutting_speed_m_per_min < 0:
        raise ValueError(
            f"cutting_speed_m_per_min must be >= 0, got {cutting_speed_m_per_min}"
        )
    speed_m_per_s = cutting_speed_m_per_min * _M_PER_MIN_TO_M_PER_S
    return cutting_force_n * speed_m_per_s


def compute_mrr_mm3_per_min(
    cutting_speed_m_per_min: float,
    feed_mm_per_rev: float,
    axial_depth_mm: float,
    radial_depth_mm: float,
) -> float:
    """
    Compute material removal rate (turning/milling approximation).

    Equation: MRR = V_c [mm/min] * f [mm/rev] * a_p [mm] * a_e [mm]
    Note: cutting_speed_m_per_min is converted to mm/min internally.

    Args:
        cutting_speed_m_per_min: Cutting speed in m/min. Must be >= 0.
        feed_mm_per_rev: Feed rate in mm/rev. Must be > 0.
        axial_depth_mm: Axial depth of cut in mm. Must be > 0.
        radial_depth_mm: Radial depth of cut in mm. Must be > 0.

    Returns:
        Material removal rate in mm³/min.
    """
    if cutting_speed_m_per_min < 0:
        raise ValueError(
            f"cutting_speed_m_per_min must be >= 0, got {cutting_speed_m_per_min}"
        )
    if feed_mm_per_rev <= 0:
        raise ValueError(f"feed_mm_per_rev must be > 0, got {feed_mm_per_rev}")
    if axial_depth_mm <= 0:
        raise ValueError(f"axial_depth_mm must be > 0, got {axial_depth_mm}")
    if radial_depth_mm <= 0:
        raise ValueError(f"radial_depth_mm must be > 0, got {radial_depth_mm}")

    speed_mm_per_min = cutting_speed_m_per_min * 1000.0
    return speed_mm_per_min * feed_mm_per_rev * axial_depth_mm * radial_depth_mm


def compute_specific_cutting_energy_j_per_mm3(
    cutting_force_n: float,
    cutting_speed_m_per_min: float,
    mrr_mm3_per_min: float,
) -> float:
    """
    Compute specific cutting energy: u = P_c / MRR

    Args:
        cutting_force_n: Cutting force in Newtons.
        cutting_speed_m_per_min: Cutting speed in m/min.
        mrr_mm3_per_min: Material removal rate in mm³/min. Must be > 0.

    Returns:
        Specific cutting energy in J/mm³.
    """
    if mrr_mm3_per_min <= 0:
        raise ValueError(f"mrr_mm3_per_min must be > 0, got {mrr_mm3_per_min}")
    power_w = compute_cutting_power_w(cutting_force_n, cutting_speed_m_per_min)
    # Convert MRR to mm³/s for dimensional consistency with power (W = J/s)
    mrr_mm3_per_s = mrr_mm3_per_min * _MM3_PER_MIN_TO_MM3_PER_S
    return power_w / mrr_mm3_per_s


def compute_cumulative_energy_j(
    cutting_force_n_series: np.ndarray,
    cutting_speed_m_per_min: float,
    time_step_s: float,
) -> float:
    """
    Compute cumulative cutting energy via discrete integration.

    Equation: E_c = sum(F_c * V_c * dt)
    Assumes constant cutting speed throughout the signal.

    Args:
        cutting_force_n_series: Array of cutting forces in Newtons, shape (T,).
        cutting_speed_m_per_min: Cutting speed in m/min (assumed constant).
        time_step_s: Time step between consecutive samples in seconds. Must be > 0.

    Returns:
        Total cutting energy in Joules.
    """
    if time_step_s <= 0:
        raise ValueError(f"time_step_s must be > 0, got {time_step_s}")
    if len(cutting_force_n_series) == 0:
        return 0.0
    speed_m_per_s = cutting_speed_m_per_min * _M_PER_MIN_TO_M_PER_S
    return float(np.sum(cutting_force_n_series)) * speed_m_per_s * time_step_s
