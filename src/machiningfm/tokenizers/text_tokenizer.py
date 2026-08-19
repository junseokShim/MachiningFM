from __future__ import annotations

import torch
from torch import Tensor, nn


class TextTokenizer(nn.Module):
    """Embeds hashed natural-language context tokens."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 8192,
        max_tokens: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor, token_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected text_ids [B, L], got {tuple(token_ids.shape)}")
        batch, length = token_ids.shape
        if length > self.max_tokens:
            token_ids = token_ids[:, : self.max_tokens]
            if token_mask is not None:
                token_mask = token_mask[:, : self.max_tokens]
            length = self.max_tokens
        token_ids = token_ids.long().clamp(0, self.vocab_size - 1)
        positions = torch.arange(length, device=token_ids.device)
        tokens = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        if token_mask is None:
            token_mask = token_ids.ne(0)
        token_mask = token_mask.bool()
        tokens = torch.where(token_mask[..., None], tokens, torch.zeros_like(tokens))
        return self.dropout(tokens), token_mask
