from __future__ import annotations

from torch import Tensor, nn


class ResidualAdapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 32) -> None:
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))

    def forward(self, value: Tensor) -> Tensor:
        return value + self.adapter(value)


def set_backbone_trainable(module: nn.Module, mode: str = "freeze") -> None:
    if mode not in {"freeze", "partial", "full"}:
        raise ValueError(f"Unknown backbone mode: {mode}")
    parameters = list(module.parameters())
    for parameter in parameters:
        parameter.requires_grad = mode == "full"
    if mode == "partial":
        for parameter in parameters[len(parameters) // 2 :]:
            parameter.requires_grad = True
