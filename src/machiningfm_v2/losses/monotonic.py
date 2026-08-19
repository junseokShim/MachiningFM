from __future__ import annotations

import torch


def monotonic_trend_loss(values: torch.Tensor, direction: str = "increasing") -> torch.Tensor:
    diff = torch.diff(values.float(), dim=-1)
    if direction == "decreasing":
        diff = -diff
    return torch.relu(-diff).mean()
