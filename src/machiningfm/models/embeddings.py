from __future__ import annotations

import torch
from torch import Tensor, nn


class ConditionEmbedding(nn.Module):
    """Embeds scalar process conditions independently so missing values stay masked."""

    def __init__(self, d_model: int, max_conditions: int = 64) -> None:
        super().__init__()
        self.max_conditions = max_conditions
        self.value_projection = nn.Linear(1, d_model)
        self.variable_embedding = nn.Embedding(max_conditions, d_model)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, condition: Tensor, condition_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2:
            raise ValueError(f"Expected condition [B, K], got {tuple(condition.shape)}")
        batch, count = condition.shape
        if count > self.max_conditions:
            raise ValueError(f"Received {count} conditions, max_conditions={self.max_conditions}")
        ids = torch.arange(count, device=condition.device)
        tokens = self.value_projection(condition.unsqueeze(-1)) + self.variable_embedding(ids)[None, :, :]
        if condition_mask is None:
            condition_mask = torch.isfinite(condition)
        condition_mask = condition_mask.bool()
        tokens = torch.where(condition_mask[..., None], tokens, self.missing_token.expand(batch, count, -1))
        return tokens, condition_mask


class ModalityEmbedding(nn.Module):
    def __init__(self, d_model: int, modality_count: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(modality_count, d_model)

    def forward(self, tokens: Tensor, modality_id: int) -> Tensor:
        ids = torch.full(tokens.shape[:-1], modality_id, dtype=torch.long, device=tokens.device)
        return tokens + self.embedding(ids)
