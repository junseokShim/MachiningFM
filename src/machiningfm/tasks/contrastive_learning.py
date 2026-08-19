from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def contrastive_loss(first: Tensor, second: Tensor, temperature: float = 0.1) -> Tensor:
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    target = torch.arange(first.shape[0], device=first.device)
    return F.cross_entropy(logits, target)
