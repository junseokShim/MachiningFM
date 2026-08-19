from __future__ import annotations

import torch
from torch import nn

from .causal_fusion import CausalFusionTransformer
from .forecasting_decoder import DirectMultiHorizonDecoder
from .modality_encoders import ModalityEncoders
from .task_heads import TaskHeads


class MachiningFMV2(nn.Module):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = dict(config or {})
        d_model = int(self.config.get("d_model", 256))
        self.encoders = ModalityEncoders(self.config)
        self.fusion = CausalFusionTransformer(
            d_model=d_model,
            num_layers=int(self.config.get("fusion_layers", 4)),
            num_heads=int(self.config.get("num_heads", 8)),
            dropout=float(self.config.get("dropout", 0.1)),
        )
        self.process_state = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.domain_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.health_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.residual_latent = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.forecasting = DirectMultiHorizonDecoder(
            d_model=d_model,
            output_channels=int(self.config.get("output_channels", 3)),
            horizons=tuple(self.config.get("forecast_horizons", (64, 1280, 12800))),
        )
        self.tasks = TaskHeads(d_model=d_model, wear_stages=int(self.config.get("wear_stages", 4)))

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens, valid = self.encoders(batch)
        hidden = self.fusion(tokens, key_padding_mask=~valid)
        weights = valid.float().unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "tokens": hidden,
            "token_mask": valid,
            "embedding": pooled,
            "process_state_latent": self.process_state(pooled),
            "domain_latent": self.domain_latent(pooled),
            "health_latent": self.health_latent(pooled),
            "residual_latent": self.residual_latent(pooled),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, object]:
        encoded = self.encode(batch)
        latent = encoded["embedding"]
        return {
            **encoded,
            "forecast": self.forecasting(latent),
            "tasks": self.tasks(latent),
        }
