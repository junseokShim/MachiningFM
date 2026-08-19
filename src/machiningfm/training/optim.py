from __future__ import annotations

from typing import Any

import torch


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    *,
    foreach: bool | None = None,
    fused: bool | None = None,
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": weight_decay}
    if foreach is not None:
        kwargs["foreach"] = foreach
    if fused is not None:
        kwargs["fused"] = fused
    return torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), **kwargs)
