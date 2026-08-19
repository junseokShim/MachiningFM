from __future__ import annotations

import torch
from torch import Tensor, nn

from machiningfm.models.losses import masked_mse
from machiningfm.models.patching import patchify


class MaskedSignalModeling(nn.Module):
    def __init__(self, model: nn.Module, patch_size: int, mask_ratio: float = 0.25) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        series = batch["sensor_series"]
        if series is None:
            return next(self.model.parameters()).sum() * 0
        target = patchify(series, self.patch_size).flatten(1, 2)
        masked = series.clone()
        time_mask = torch.rand_like(masked) < self.mask_ratio
        masked[time_mask] = 0
        output = self.model({**batch, "sensor_series": masked})
        sensor_mask = batch.get("sensor_mask")
        patch_mask = None
        if sensor_mask is not None:
            patches_per_channel = target.shape[1] // series.shape[1]
            patch_mask = sensor_mask[:, :, None].expand(-1, -1, patches_per_channel).flatten(1, 2)
        return masked_mse(output["patch_reconstruction"], target, patch_mask)
