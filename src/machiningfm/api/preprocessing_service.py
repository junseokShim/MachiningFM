from __future__ import annotations

from typing import Any


class PreprocessingService:
    def __init__(self, max_sequence_length: int = 8192) -> None:
        self.max_sequence_length = max_sequence_length

    def normalize(self, sample: Any) -> dict[str, Any]:
        if hasattr(sample, "model_dump"):
            value = sample.model_dump(exclude_none=True)
        else:
            value = dict(sample)
        series = value.get("sensor_series")
        if isinstance(series, list) and series:
            if isinstance(series[0], list):
                value["sensor_series"] = [channel[-self.max_sequence_length :] for channel in series]
            else:
                value["sensor_series"] = series[-self.max_sequence_length :]
        frequency = value.get("frequency")
        if isinstance(frequency, list) and frequency:
            if isinstance(frequency[0], list):
                value["frequency"] = [channel[-self.max_sequence_length :] for channel in frequency]
            else:
                value["frequency"] = frequency[-self.max_sequence_length :]
        cnc_series = value.get("cnc_series")
        if isinstance(cnc_series, list) and cnc_series:
            if isinstance(cnc_series[0], list):
                value["cnc_series"] = [channel[-self.max_sequence_length :] for channel in cnc_series]
            else:
                value["cnc_series"] = cnc_series[-self.max_sequence_length :]
        return value
