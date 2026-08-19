"""
Kienzle Cutting Force Model.

Basic forms:
    k_c = k_{c1.1} * h^{-m_c}            (specific cutting force, N/mm²)
    F_c = k_{c1.1} * b * h^{1-m_c}       (cutting force, N)

Units:
    chip_thickness_mm (h) : mm
    cutting_width_mm  (b) : mm
    kc_1_1_n_per_mm2      : N/mm²
    cutting_force_n        : N

Reference: Kienzle & Victor (1957), "Spezifische Schnittkräfte bei der Metallbearbeitung",
           Werkstattstechnik und Maschinenbau, 47(5):224-225.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class KienzleParams:
    """Parameters for the Kienzle cutting force model."""

    kc_1_1_n_per_mm2: float  # Specific cutting force at h=1 mm, b=1 mm (N/mm²)
    mc: float                 # Kienzle exponent (dimensionless, typically 0.1–0.4)
    source: str = "literature"
    reference: str = ""


def estimate_specific_cutting_force_n_per_mm2(
    chip_thickness_mm: float,
    params: KienzleParams,
) -> float:
    """
    Estimate specific cutting force using the Kienzle model.

    Equation: k_c = k_{c1.1} * h^{-m_c}

    Args:
        chip_thickness_mm: Chip thickness in mm. Must be > 0.
        params: Kienzle model parameters.

    Returns:
        Specific cutting force in N/mm².

    Raises:
        ValueError: If chip_thickness_mm <= 0.
    """
    if chip_thickness_mm <= 0:
        raise ValueError(f"chip_thickness_mm must be > 0, got {chip_thickness_mm}")
    return params.kc_1_1_n_per_mm2 * (chip_thickness_mm ** (-params.mc))


def estimate_cutting_force_n(
    chip_thickness_mm: float,
    cutting_width_mm: float,
    params: KienzleParams,
) -> float:
    """
    Estimate cutting force using the Kienzle model.

    Equation: F_c = k_{c1.1} * b * h^{1-m_c}

    Args:
        chip_thickness_mm: Chip thickness in mm. Must be > 0.
        cutting_width_mm: Cutting width (chip width) in mm. Must be > 0.
        params: Kienzle model parameters.

    Returns:
        Estimated cutting force in Newtons.

    Raises:
        ValueError: If any dimension is <= 0.
    """
    if chip_thickness_mm <= 0:
        raise ValueError(f"chip_thickness_mm must be > 0, got {chip_thickness_mm}")
    if cutting_width_mm <= 0:
        raise ValueError(f"cutting_width_mm must be > 0, got {cutting_width_mm}")
    return (
        params.kc_1_1_n_per_mm2
        * cutting_width_mm
        * (chip_thickness_mm ** (1.0 - params.mc))
    )


def compute_mean_chip_thickness_milling_mm(
    feed_per_tooth_mm: float,
    radial_engagement_ratio: float,
) -> float:
    """
    Approximate mean chip thickness for milling.

    Equation: h_m ≈ f_z * sqrt(a_e / D)
    Valid for small radial engagement angles (a_e/D << 1).

    Args:
        feed_per_tooth_mm: Feed per tooth in mm. Must be > 0.
        radial_engagement_ratio: a_e/D ratio (radial depth / tool diameter).
                                 Must be in (0, 1].

    Returns:
        Mean chip thickness in mm.
    """
    if feed_per_tooth_mm <= 0:
        raise ValueError(f"feed_per_tooth_mm must be > 0, got {feed_per_tooth_mm}")
    if not (0 < radial_engagement_ratio <= 1.0):
        raise ValueError(
            f"radial_engagement_ratio must be in (0, 1], got {radial_engagement_ratio}"
        )
    return feed_per_tooth_mm * math.sqrt(radial_engagement_ratio)


def compute_physics_residual(
    measured_force_n: float,
    chip_thickness_mm: float,
    cutting_width_mm: float,
    params: KienzleParams,
) -> dict[str, float]:
    """
    Compute physics residual features between measured and Kienzle-predicted force.

    These residuals encode tool wear and other deviations from ideal cutting conditions,
    and are useful as physics-informed features for the calibration layer.
    A force_ratio > 1 typically indicates increased cutting resistance from tool wear.

    Args:
        measured_force_n: Measured cutting force in Newtons.
        chip_thickness_mm: Chip thickness in mm. Must be > 0.
        cutting_width_mm: Cutting width in mm. Must be > 0.
        params: Kienzle model parameters.

    Returns:
        Dict with keys:
            'force_ratio'    : F_measured / F_kienzle
            'force_delta_n'  : F_measured - F_kienzle (N)
            'physics_force_n': Kienzle-predicted force (N)
    """
    physics_force_n = estimate_cutting_force_n(chip_thickness_mm, cutting_width_mm, params)
    return {
        "force_ratio": measured_force_n / physics_force_n,
        "force_delta_n": measured_force_n - physics_force_n,
        "physics_force_n": physics_force_n,
    }
