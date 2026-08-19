"""
Usui Wear Rate Model (Optional — requires cutting temperature data).

Equation: dW/dt = B1 * sigma_n * v_s * exp(-B2 / T)

IMPORTANT: This model requires:
  - Contact normal stress (sigma_n) in N/mm²
  - Tool-chip sliding velocity (v_s) in mm/s
  - Cutting zone temperature (T) in Kelvin

These quantities are NOT measured in standard machining sensor datasets
(e.g., PHM2010 has force/vibration/AE, not temperature or contact pressure).
The model is disabled by default (available=False).

Do NOT generate synthetic temperature or contact stress values to enable this model.
Leave it as optional infrastructure for datasets that do provide thermal measurements.

Units:
    contact_stress_n_per_mm2   : N/mm²
    sliding_velocity_mm_per_s  : mm/s
    cutting_temperature_k      : K  (absolute temperature, must be > 0)
    wear_rate_mm_per_s         : mm/s

Reference: Usui et al. (1984), "Analytical Prediction of Three Dimensional Cutting
           Process, Part 3: Cutting Temperature and Crater Wear of Carbide Tool",
           ASME J. Eng. for Industry, 100:236-243.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class UsuiParams:
    """Parameters for the Usui wear rate model."""

    B1: float           # Pre-exponential constant
    B2: float           # Thermal activation parameter in Kelvin
    source: str = "literature"
    reference: str = ""
    available: bool = False  # Default False: temperature rarely available in datasets


def compute_wear_rate_mm_per_s(
    contact_stress_n_per_mm2: float,
    sliding_velocity_mm_per_s: float,
    cutting_temperature_k: float,
    params: UsuiParams,
) -> float | None:
    """
    Compute tool wear rate using the Usui model.

    Equation: dW/dt = B1 * sigma_n * v_s * exp(-B2 / T)

    Returns None if params.available is False.

    Args:
        contact_stress_n_per_mm2: Contact normal stress in N/mm².
        sliding_velocity_mm_per_s: Tool-chip sliding velocity in mm/s. Must be >= 0.
        cutting_temperature_k: Cutting zone temperature in Kelvin. Must be > 0.
        params: Usui model parameters.

    Returns:
        Wear rate in mm/s, or None if model is disabled.

    Raises:
        ValueError: If cutting_temperature_k <= 0 or sliding_velocity < 0.
    """
    if not params.available:
        return None
    if cutting_temperature_k <= 0:
        raise ValueError(
            f"cutting_temperature_k must be > 0 K, got {cutting_temperature_k}. "
            "Absolute temperature is required for the Arrhenius-type term."
        )
    if sliding_velocity_mm_per_s < 0:
        raise ValueError(
            f"sliding_velocity_mm_per_s must be >= 0, got {sliding_velocity_mm_per_s}"
        )
    return (
        params.B1
        * contact_stress_n_per_mm2
        * sliding_velocity_mm_per_s
        * math.exp(-params.B2 / cutting_temperature_k)
    )
