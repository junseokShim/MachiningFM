from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class FutureForecasting(nn.Module):
    def __init__(self, model: nn.Module, horizon: int) -> None:
        super().__init__()
        self.model = model
        self.horizon = horizon

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        series = batch["sensor_series"]
        if series is None or series.shape[-1] <= self.horizon:
            return next(self.model.parameters()).sum() * 0
        context = series[..., :-self.horizon]
        target = series[..., -self.horizon :]
        prediction = self.model({**batch, "sensor_series": context})["forecast"]
        loss = F.smooth_l1_loss(prediction, target, reduction="none")
        sensor_mask = batch.get("sensor_mask")
        if sensor_mask is None:
            return loss.mean()
        weights = sensor_mask.to(loss.dtype).unsqueeze(-1)
        return (loss * weights).sum() / (weights.sum() * loss.shape[-1]).clamp_min(1.0)
