from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import TransformerBlock


class MachiningBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor | None = None) -> Tensor:
        key_padding_mask = None if token_mask is None else ~token_mask.bool()
        causal_mask = None
        if self.causal:
            length = tokens.shape[1]
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
            )
        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask, causal_mask=causal_mask)
        return self.norm(tokens)
