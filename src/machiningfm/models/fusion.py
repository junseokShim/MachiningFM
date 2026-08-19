from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskedMeanFusion(nn.Module):
    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def concatenate_modalities(parts: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    if not parts:
        raise ValueError("At least one modality token part is required")
    return torch.cat([part[0] for part in parts], dim=1), torch.cat([part[1] for part in parts], dim=1)
