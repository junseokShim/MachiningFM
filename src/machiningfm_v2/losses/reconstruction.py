from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred.float(), target.float(), reduction="none")
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.device, dtype=loss.dtype)
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)
