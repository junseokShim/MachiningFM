from __future__ import annotations

import torch
from torch.nn import functional as F


def metadata_aware_infonce(anchor: torch.Tensor, positive: torch.Tensor, same_domain_mask: torch.Tensor | None = None, temperature: float = 0.1) -> torch.Tensor:
    a = F.normalize(anchor.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    logits = a @ p.T / max(float(temperature), 1.0e-6)
    if same_domain_mask is not None:
        logits = logits.masked_fill(same_domain_mask.bool() & ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device), -1.0e4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
