from __future__ import annotations

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
