from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class Modality(StrEnum):
    NC = "nc"
    CNC = "cnc"
    RAW_WAVEFORM = "raw_waveform"
    SPECTRAL = "spectral"
    TOOL_IMAGE = "tool_image"
    METADATA = "metadata"


@dataclass(frozen=True)
class ChannelMetadata:
    name: str
    unit: str = "unknown"
    axis: str = "none"
    source: str = "unknown"
    sample_rate: float | None = None
    calibration: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeMetadata:
    absolute_start: float | None = None
    relative_start: float = 0.0
    sample_rate: float | None = None
    sync_confidence: float = 1.0
    nc_block: str | None = None
    machining_phase: str = "unknown"


@dataclass
class MultiRateSample:
    raw_waveform: np.ndarray | None = None
    raw_channel_metadata: list[ChannelMetadata] = field(default_factory=list)
    raw_time: TimeMetadata = field(default_factory=TimeMetadata)
    cnc: np.ndarray | None = None
    cnc_channel_metadata: list[ChannelMetadata] = field(default_factory=list)
    cnc_time: TimeMetadata = field(default_factory=TimeMetadata)
    spectral: np.ndarray | None = None
    nc_tokens: np.ndarray | None = None
    image: np.ndarray | None = None
    metadata_vector: np.ndarray | None = None
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    domain: dict[str, str] = field(default_factory=dict)
    masks: dict[str, bool] = field(default_factory=dict)

    def modality_mask(self) -> dict[str, bool]:
        return {
            Modality.RAW_WAVEFORM.value: self.raw_waveform is not None,
            Modality.CNC.value: self.cnc is not None,
            Modality.SPECTRAL.value: self.spectral is not None,
            Modality.NC.value: self.nc_tokens is not None,
            Modality.TOOL_IMAGE.value: self.image is not None,
            Modality.METADATA.value: self.metadata_vector is not None,
            **self.masks,
        }
