from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossChannelReconstruction(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        series = batch["sensor_series"]
        if series is None or series.shape[1] < 2:
            return next(self.model.parameters()).sum() * 0
        mask = batch.get("sensor_mask")
        if mask is None:
            mask = series.new_ones(series.shape[:2], dtype=bool)
        valid = mask.sum(dim=1) >= 2
        if not valid.any():
            return next(self.model.parameters()).sum() * 0
        selected = (mask.long().sum(dim=1) - 1).clamp_min(0)
        mask = mask.clone()
        batch_index = torch.arange(series.shape[0], device=series.device)
        mask[batch_index, selected] = False
        forecast = self.model({**batch, "sensor_mask": mask})["forecast"]
        prediction = forecast[batch_index, selected]
        target = series[batch_index, selected, -prediction.shape[-1] :]
        return F.mse_loss(prediction[valid], target[valid])
