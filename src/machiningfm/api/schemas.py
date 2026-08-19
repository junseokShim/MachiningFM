from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SampleInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    sensor_series: list[list[float]] | list[float] | None = None
    sensor_names: list[str] = Field(
        default_factory=list,
        description="One raw alias or cnc-v1 standard name per sensor_series channel.",
    )
    frequency: list[list[float]] | list[float] | None = Field(
        default=None,
        description="Frequency-domain channels such as FFT, PSD, STFT, or CWT features.",
    )
    frequency_names: list[str] = Field(
        default_factory=list,
        description="One raw alias or cnc-v1 standard name per frequency channel.",
    )
    cnc_series: list[list[float]] | list[float] | None = Field(
        default=None,
        description="Optional CNC/controller channels used to synthesize virtual spindle vibration.",
    )
    cnc_names: list[str] = Field(
        default_factory=list,
        description="One CNC/controller channel name per cnc_series channel.",
    )
    generate_virtual_vibration: bool = Field(
        default=False,
        description="Generate virtual spindle-mounted x/y/z vibration from CNC/controller data.",
    )
    virtual_vibration_sampling_rate: float | None = Field(
        default=None,
        description="Sampling rate in Hz for generated virtual spindle vibration.",
    )
    image: Any | None = Field(
        default=None,
        description="Optional image as numeric HWC/CHW array for tool, workpiece, chip, spectrogram, or scalogram input.",
    )
    image_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded image payload.",
    )
    image_path: str | None = Field(
        default=None,
        description="Optional server-local image path.",
    )
    process_condition: dict[str, Any] = Field(default_factory=dict)
    text_context: str | dict[str, Any] | None = None
    tool_info: str | dict[str, Any] | None = None
    material_info: str | dict[str, Any] | None = None
    machine_info: str | dict[str, Any] | None = None
    operation_info: str | dict[str, Any] | None = None
    process_description: str | dict[str, Any] | None = None
    available_variables: list[str] = Field(default_factory=list)
    label: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_sensor_names(self) -> "SampleInput":
        if self.sensor_series is not None and self.sensor_names:
            channel_count = _channel_count(self.sensor_series)
            if len(self.sensor_names) != channel_count:
                raise ValueError(
                    f"sensor_names has {len(self.sensor_names)} entries, "
                    f"but sensor_series has {channel_count} channels"
                )
        if self.frequency is not None and self.frequency_names:
            channel_count = _channel_count(self.frequency)
            if len(self.frequency_names) != channel_count:
                raise ValueError(
                    f"frequency_names has {len(self.frequency_names)} entries, "
                    f"but frequency has {channel_count} channels"
                )
        if self.cnc_series is not None and self.cnc_names:
            channel_count = _channel_count(self.cnc_series)
            if len(self.cnc_names) != channel_count:
                raise ValueError(
                    f"cnc_names has {len(self.cnc_names)} entries, "
                    f"but cnc_series has {channel_count} channels"
                )
        return self


def _channel_count(values: list[list[float]] | list[float]) -> int:
    if not values:
        return 0
    first = values[0]
    return len(values) if isinstance(first, list) else 1


class PredictRequest(SampleInput):
    task: str = "toolwear_regression"
    model_checkpoint_path: str | None = None


class ZeroShotRequest(BaseModel):
    task: str
    query: SampleInput


class FewShotRequest(BaseModel):
    task: str
    support_set: list[SampleInput]
    query: SampleInput
    adaptation_method: str = "prototype"
