"""Physics models for machining: Taylor, Kienzle, Energy, Archard, Usui, Calibration."""

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

__all__ = [
    "TaylorParams",
    "estimate_tool_life_min",
    "compute_tool_life_ratio",
    "compute_machining_progress",
    "KienzleParams",
    "estimate_specific_cutting_force_n_per_mm2",
    "estimate_cutting_force_n",
    "compute_mean_chip_thickness_milling_mm",
    "compute_physics_residual",
    "compute_cutting_power_w",
    "compute_mrr_mm3_per_min",
    "compute_specific_cutting_energy_j_per_mm3",
    "compute_cumulative_energy_j",
    "ArchardParams",
    "compute_wear_volume_mm3",
    "UsuiParams",
    "compute_wear_rate_mm_per_s",
    "PhysicsFeatures",
    "PhysicsCalibrator",
]
