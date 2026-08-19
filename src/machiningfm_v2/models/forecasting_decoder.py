from __future__ import annotations

import torch
from torch import nn


class DirectMultiHorizonDecoder(nn.Module):
    def __init__(self, d_model: int, output_channels: int = 3, horizons: tuple[int, ...] = (64, 1280, 12800)) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.horizons = tuple(int(h) for h in horizons)
        self.heads = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, self.output_channels * h * 2),
                )
                for h in self.horizons
            }
        )

    def forward(self, latent: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        if latent.ndim != 2:
            raise ValueError(f"latent must be [batch, d_model], got {tuple(latent.shape)}")
        out: dict[str, dict[str, torch.Tensor]] = {}
        for horizon in self.horizons:
            raw = self.heads[str(horizon)](latent).view(latent.shape[0], self.output_channels, horizon, 2)
            mean = raw[..., 0]
            log_var = raw[..., 1].clamp(-8.0, 6.0)
            out[str(horizon)] = {"mean": mean, "log_var": log_var}
        return out
