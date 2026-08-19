from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * grad_output, None


class DomainAdversarialHead(nn.Module):
    def __init__(self, d_model: int, num_domains: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_domains))

    def forward(self, latent: torch.Tensor, labels: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        logits = self.classifier(GradientReverse.apply(latent, strength))
        return F.cross_entropy(logits, labels.long())
