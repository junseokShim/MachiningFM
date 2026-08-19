from __future__ import annotations

import torch
from torch import nn


class TaskHeads(nn.Module):
    def __init__(self, d_model: int, wear_stages: int = 4) -> None:
        super().__init__()
        self.energy = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 8))
        self.anomaly = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.health = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1), nn.Sigmoid())
        self.wear_stage = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, wear_stages))
        self.surface_quality = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "energy": self.energy(latent),
            "anomaly_logit": self.anomaly(latent).squeeze(-1),
            "health_index": self.health(latent).squeeze(-1),
            "wear_stage_logits": self.wear_stage(latent),
            "surface_quality": self.surface_quality(latent).squeeze(-1),
        }
