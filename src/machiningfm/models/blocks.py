from __future__ import annotations

from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None, causal_mask: Tensor | None = None) -> Tensor:
        return self.layer(tokens, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)


class SSMBlockInterface(nn.Module):
    """Extension point for a future Mamba/SSM implementation."""

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None, causal_mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError("Install and configure an SSM implementation before using this block.")
