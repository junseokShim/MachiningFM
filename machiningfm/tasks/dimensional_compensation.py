"""
Task D: Dimensional Compensation / Tool Offset Recommendation.

DATA INTEGRITY NOTE:
    This module provides a computational interface for estimating tool offsets
    from predicted tool wear. PHM2010 does NOT contain dimensional accuracy
    measurements (actual vs. nominal workpiece dimensions). Therefore:

    - Output is a PHYSICS-DERIVED ESTIMATE, not experimentally validated on PHM2010.
    - Do not claim experimental validation without real dimensional measurement data.
    - The wear-to-deflection relationship used here is from machining literature,
      not fitted to PHM2010.

    When actual measured_dimension_mm is available, the offset is computed directly
    from the dimensional error (no approximation involved).

Empirical relationship used for estimation:
    estimated_deflection_mm ≈ wear_deflection_factor * VB_mm
    Source: approximate from Salgado et al. (2009), Int. J. Machine Tools & Manufacture.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompensationResult:
    """Output of a dimensional compensation calculation."""

    recommended_offset_mm: float
    """Recommended tool offset adjustment in mm."""

    confidence: float
    """Confidence in the recommendation, in [0, 1].
    Lower when the relationship is extrapolated or empirically approximated."""

    reason: str
    """Human-readable explanation of how the offset was computed."""


class DimensionalCompensator:
    """
    Recommend tool offset based on predicted tool wear.

    When actual dimensional measurements are available (measured_dimension_mm),
    the offset is computed from the direct dimensional error (high confidence).
    Otherwise, an empirical wear-to-deflection approximation is used (lower confidence).

    NOTE: Without actual dimensional measurements, results cannot be experimentally
    validated. PHM2010 does not provide dimensional ground truth data.
    """

    # Empirical proportionality: estimated_error_mm ≈ factor * VB_mm
    # This is a rough approximation and should be calibrated per machine/material.
    # Source: approximate, derived from Salgado et al. (2009) stiffness analysis.
    _WEAR_TO_DEFLECTION_FACTOR: float = 0.4

    def predict(
        self,
        predicted_wear_vb_mm: float,
        nominal_dimension_mm: float,
        measured_dimension_mm: float | None = None,
        axis: str = "X",
    ) -> CompensationResult:
        """
        Compute recommended tool offset.

        Args:
            predicted_wear_vb_mm: Predicted flank wear VB in mm. Must be >= 0.
            nominal_dimension_mm: Target workpiece dimension in mm.
            measured_dimension_mm: Actually measured dimension in mm, if available.
            axis: Machine axis label (informational only).

        Returns:
            CompensationResult with recommended_offset_mm, confidence, reason.
        """
        if predicted_wear_vb_mm < 0:
            raise ValueError(
                f"predicted_wear_vb_mm must be >= 0, got {predicted_wear_vb_mm}"
            )

        if measured_dimension_mm is not None:
            # Direct computation from actual dimensional error
            error_mm = measured_dimension_mm - nominal_dimension_mm
            offset_mm = -error_mm
            confidence = 0.9
            reason = (
                f"Axis {axis}: dimensional error = {error_mm:+.4f} mm "
                f"(measured={measured_dimension_mm:.4f} mm, nominal={nominal_dimension_mm:.4f} mm). "
                f"Offset {offset_mm:+.4f} mm corrects the error."
            )
        else:
            # Empirical estimate from wear
            estimated_error_mm = self._WEAR_TO_DEFLECTION_FACTOR * predicted_wear_vb_mm
            offset_mm = -estimated_error_mm
            # Confidence decreases as wear increases (empirical model less reliable)
            confidence = max(0.05, 0.75 - predicted_wear_vb_mm * 1.5)
            reason = (
                f"Axis {axis}: estimated dimensional error ≈ {estimated_error_mm:.4f} mm "
                f"from VB = {predicted_wear_vb_mm:.4f} mm "
                f"(empirical factor {self._WEAR_TO_DEFLECTION_FACTOR}). "
                "PHYSICS-DERIVED ESTIMATE — not experimentally validated on PHM2010."
            )

        return CompensationResult(
            recommended_offset_mm=round(offset_mm, 4),
            confidence=round(confidence, 3),
            reason=reason,
        )
