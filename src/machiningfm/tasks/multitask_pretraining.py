from __future__ import annotations

from torch import Tensor, nn

from .cross_channel_reconstruction import CrossChannelReconstruction
from .frequency_reconstruction import FrequencyReconstruction
from .future_forecasting import FutureForecasting
from .image_reconstruction import ImagePatchReconstruction
from .masked_signal_modeling import MaskedSignalModeling


class MultitaskPretraining(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        horizon: int,
        image_patch_size: int = 16,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.tasks = nn.ModuleDict(
            {
                "masked_signal": MaskedSignalModeling(model, patch_size),
                "forecasting": FutureForecasting(model, horizon),
                "cross_channel": CrossChannelReconstruction(model),
                "frequency_reconstruction": FrequencyReconstruction(model, patch_size),
                "image_reconstruction": ImagePatchReconstruction(model, image_patch_size),
            }
        )
        self.weights = weights or {
            "masked_signal": 1.0,
            "forecasting": 1.0,
            "cross_channel": 0.25,
            "frequency_reconstruction": 0.5,
            "image_reconstruction": 0.25,
        }

    def forward(self, batch: dict[str, Tensor | None]) -> tuple[Tensor, dict[str, Tensor]]:
        losses = {name: task(batch) for name, task in self.tasks.items()}
        total = sum(loss * self.weights.get(name, 1.0) for name, loss in losses.items())
        return total, losses
