from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    residual = (prediction.float() - target.float()).clamp(min=-50.0, max=50.0)
    loss = residual.square()
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype)
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def masked_smooth_l1(prediction: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    loss = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none")
    if mask is None:
        return loss.mean()
    weights = mask.to(loss.dtype)
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(loss)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def masked_task_loss(losses: dict[str, Tensor], label_masks: dict[str, bool | Tensor]) -> Tensor:
    selected = []
    for name, loss in losses.items():
        mask = label_masks.get(name, False)
        if isinstance(mask, Tensor):
            if mask.any():
                selected.append(loss)
        elif mask:
            selected.append(loss)
    return torch.stack(selected).mean() if selected else torch.zeros((), device=next(iter(losses.values())).device)
