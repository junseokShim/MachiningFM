"""
Archard Wear Model (Optional).

Equation: V_w = K * F_N * L / H

IMPORTANT: This model requires normal force (F_N), sliding distance (L), and
material hardness (H). These quantities are typically NOT measured in standard
machining sensor datasets (e.g., PHM2010 provides force/vibration/AE, not F_N or L).
The model is disabled by default (available=False). Do NOT generate synthetic values
for missing quantities to enable this model — leave it disabled.

Units:
    normal_force_n         : N
    sliding_distance_mm    : mm
    hardness_h_n_per_mm2   : N/mm²  (= MPa; 1 HV ≈ 9.81 N/mm²)
    wear_volume_mm3        : mm³

Reference: Archard (1953), "Contact and Rubbing of Flat Surfaces",
           Journal of Applied Physics, 24(8):981-988.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArchardParams:
    """Parameters for the Archard wear model."""

    wear_coefficient_k: float       # Dimensionless Archard wear coefficient
    hardness_h_n_per_mm2: float     # Hardness of the softer material (N/mm²)
    source: str = "literature"
    reference: str = ""
    available: bool = False         # Default False: F_N and L rarely available in datasets


def compute_wear_volume_mm3(
    normal_force_n: float,
    sliding_distance_mm: float,
    params: ArchardParams,
) -> float | None:
    """
    Compute wear volume using Archard's equation: V_w = K * F_N * L / H

    Returns None if params.available is False (data not available in dataset).
    Never set available=True and generate synthetic F_N or L values.

    Args:
        normal_force_n: Normal force in Newtons. Must be >= 0.
        sliding_distance_mm: Total sliding distance in mm. Must be >= 0.
        params: Archard model parameters.

    Returns:
        Wear volume in mm³, or None if model is disabled.

    Raises:
        ValueError: If hardness <= 0 or inputs are negative (when available=True).
    """
    if not params.available:
        return None
    if normal_force_n < 0:
        raise ValueError(f"normal_force_n must be >= 0, got {normal_force_n}")
    if sliding_distance_mm < 0:
        raise ValueError(f"sliding_distance_mm must be >= 0, got {sliding_distance_mm}")
    if params.hardness_h_n_per_mm2 <= 0:
        raise ValueError(
            f"hardness_h_n_per_mm2 must be > 0, got {params.hardness_h_n_per_mm2}"
        )
    return (
        params.wear_coefficient_k * normal_force_n * sliding_distance_mm
        / params.hardness_h_n_per_mm2
    )
