from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def pad_image_to_patch_grid(image: Tensor, patch_size: int) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected image [B, C, H, W], got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height))
    return image


def patchify_image(image: Tensor, patch_size: int) -> Tensor:
    """Convert [B, C, H, W] images to flattened [B, N, C * P * P] patches."""
    image = pad_image_to_patch_grid(image, patch_size)
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.flatten(1, 2).flatten(2)


def mask_image_patches(image: Tensor, patch_mask: Tensor, patch_size: int) -> Tensor:
    """Zero whole image patches selected by patch_mask [B, N]."""
    padded = pad_image_to_patch_grid(image, patch_size)
    batch, _, height, width = padded.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    expected = grid_h * grid_w
    if tuple(patch_mask.shape) != (batch, expected):
        raise ValueError(f"Expected patch_mask {(batch, expected)}, got {tuple(patch_mask.shape)}")
    pixel_mask = patch_mask.bool().view(batch, grid_h, grid_w)
    pixel_mask = pixel_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return padded.masked_fill(pixel_mask[:, None, :, :], 0.0)


class ImageTokenizer(nn.Module):
    def __init__(self, d_model: int, patch_size: int = 16, input_channels: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.projection = nn.Conv2d(input_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: Tensor, image_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        image = pad_image_to_patch_grid(image, self.patch_size)
        patches = patchify_image(image, self.patch_size)
        tokens = self.projection(image).flatten(2).transpose(1, 2)
        if image_mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=image.device)
        else:
            mask = image_mask.bool().view(image.shape[0], 1).expand(tokens.shape[:2])
        return tokens, mask, patches
