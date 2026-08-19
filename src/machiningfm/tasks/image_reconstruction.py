from __future__ import annotations

import torch
from torch import Tensor, nn

from machiningfm.models.losses import masked_mse
from machiningfm.tokenizers.image_tokenizer import mask_image_patches, patchify_image


class ImagePatchReconstruction(nn.Module):
    def __init__(self, model: nn.Module, patch_size: int, mask_ratio: float = 0.4) -> None:
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, batch: dict[str, Tensor | None]) -> Tensor:
        image = batch.get("image")
        if image is None:
            return next(self.model.parameters()).sum() * 0
        target = patchify_image(image, self.patch_size)
        patch_mask = torch.rand(target.shape[:2], device=image.device) < self.mask_ratio
        image_mask = batch.get("image_mask")
        if image_mask is not None:
            patch_mask = patch_mask & image_mask.bool().view(-1, 1)
        masked = mask_image_patches(image, patch_mask, self.patch_size)
        output = self.model({**batch, "image": masked})
        return masked_mse(output["image_patch_reconstruction"], target, patch_mask)
