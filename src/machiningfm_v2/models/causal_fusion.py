from __future__ import annotations

import torch
from torch import nn


class CausalFusionTransformer(nn.Module):
    def __init__(self, d_model: int = 256, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.norm(hidden)
