from __future__ import annotations

import torch
from torch import Tensor, nn

from machiningfm.models.losses import masked_mse
from machiningfm.models.patching import patchify


class FrequencyReconstruction(nn.Module):
    def __init__(self, model: nn.Module, patch_size: int, mask_ratio: float = 0.25) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        frequency = batch.get("frequency")
        if frequency is None:
            return next(self.model.parameters()).sum() * 0
        target = patchify(frequency, self.patch_size).flatten(1, 2)
        masked = frequency.clone()
        bin_mask = torch.rand_like(masked) < self.mask_ratio
        masked[bin_mask] = 0
        output = self.model({**batch, "frequency": masked})
        frequency_mask = batch.get("frequency_mask")
        patch_mask = None
        if frequency_mask is not None:
            patches_per_channel = target.shape[1] // frequency.shape[1]
            patch_mask = frequency_mask[:, :, None].expand(-1, -1, patches_per_channel).flatten(1, 2)
        return masked_mse(output["frequency_patch_reconstruction"], target, patch_mask)
