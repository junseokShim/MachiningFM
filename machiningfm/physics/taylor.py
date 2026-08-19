"""
Taylor Tool-Life Equation implementation.

Basic form:       V_c * T^n = C
Generalized form: V_c * T^n * f^m * a_p^p = C

Units:
    cutting_speed_m_per_min : m/min
    feed_mm_per_rev         : mm/rev
    axial_depth_mm          : mm
    tool_life_min           : min
    C                       : m/min (at f=1 mm/rev, a_p=1 mm)
    n, m, p                 : dimensionless exponents

Reference: Taylor (1907), "On the Art of Cutting Metals", ASME Trans. 28:31-350.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TaylorParams:
    """Parameters for the generalized Taylor tool-life equation."""

    C: float        # Taylor constant (m/min)
    n: float        # Speed exponent
    m: float = 0.0  # Feed exponent (0 = basic Taylor equation)
    p: float = 0.0  # Depth exponent (0 = basic Taylor equation)
    source: str = "literature"
    reference: str = ""


def estimate_tool_life_min(
    cutting_speed_m_per_min: float,
    feed_mm_per_rev: float,
    axial_depth_mm: float,
    params: TaylorParams,
) -> float:
    """
    Estimate tool life T (min) using the generalized Taylor equation.

    Equation: V_c * T^n * f^m * a_p^p = C
    Solved for T: T = (C / (V_c * f^m * a_p^p))^(1/n)

    Args:
        cutting_speed_m_per_min: Cutting speed in m/min. Must be >= 0.
        feed_mm_per_rev: Feed rate in mm/rev. Must be > 0.
        axial_depth_mm: Axial depth of cut in mm. Must be > 0.
        params: Taylor equation parameters.

    Returns:
        Estimated tool life in minutes. Returns math.inf when speed is zero.

    Raises:
        ValueError: On invalid inputs or params.n == 0.
    """
    if cutting_speed_m_per_min < 0:
        raise ValueError(
            f"cutting_speed_m_per_min must be >= 0, got {cutting_speed_m_per_min}"
        )
    if feed_mm_per_rev <= 0:
        raise ValueError(f"feed_mm_per_rev must be > 0, got {feed_mm_per_rev}")
    if axial_depth_mm <= 0:
        raise ValueError(f"axial_depth_mm must be > 0, got {axial_depth_mm}")
    if params.n == 0:
        raise ValueError("Taylor exponent n must not be zero.")

    if cutting_speed_m_per_min == 0.0:
        return math.inf

    denominator = cutting_speed_m_per_min
    if params.m != 0.0:
        denominator *= feed_mm_per_rev ** params.m
    if params.p != 0.0:
        denominator *= axial_depth_mm ** params.p

    if denominator <= 0:
        raise ValueError(
            f"Computed denominator C/(V_c*f^m*a_p^p) is non-positive: {denominator}"
        )

    return (params.C / denominator) ** (1.0 / params.n)


def compute_tool_life_ratio(
    elapsed_time_min: float,
    cutting_speed_m_per_min: float,
    feed_mm_per_rev: float,
    axial_depth_mm: float,
    params: TaylorParams,
) -> float:
    """
    Compute normalized tool-life ratio r_T = t / T_Taylor, clamped to [0, 1].

    A value near 1 indicates the tool is approaching end-of-life per Taylor's equation.
    This serves as a physics-informed wear proxy when direct wear measurement is unavailable.

    Args:
        elapsed_time_min: Elapsed machining time in minutes. Must be >= 0.

    Returns:
        Tool-life ratio in [0, 1].
    """
    if elapsed_time_min < 0:
        raise ValueError(f"elapsed_time_min must be >= 0, got {elapsed_time_min}")

    tool_life_min = estimate_tool_life_min(
        cutting_speed_m_per_min, feed_mm_per_rev, axial_depth_mm, params
    )
    if math.isinf(tool_life_min):
        return 0.0

    return min(1.0, elapsed_time_min / tool_life_min)


def compute_machining_progress(
    elapsed_time_min: float,
    cutting_speed_m_per_min: float,
    feed_mm_per_rev: float,
    axial_depth_mm: float,
    params: TaylorParams,
) -> float:
    """Alias for compute_tool_life_ratio. Returns machining progress in [0, 1]."""
    return compute_tool_life_ratio(
        elapsed_time_min, cutting_speed_m_per_min, feed_mm_per_rev, axial_depth_mm, params
    )
