from __future__ import annotations

import torch
from torch import nn


class ToolImageTokenizer(nn.Module):
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must be [batch, channels, height, width], got {tuple(image.shape)}")
        return self.encoder(image.float()).unsqueeze(1)
