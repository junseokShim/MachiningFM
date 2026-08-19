from __future__ import annotations

from torch import Tensor, nn


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, output_dim))

    def forward(self, embedding: Tensor) -> Tensor:
        return self.network(embedding)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, classes))

    def forward(self, embedding: Tensor) -> Tensor:
        return self.network(embedding)


class ForecastingHead(nn.Module):
    '''
    미래 센서값 또는 feature trajectory 예측

    input : embedding [B, d_model]
    output : forecase [B, channels, horizon]

    ex) 
    ForecastingHead(
        d_model=256,
        max_channels=16,
        horizon=32
    )

    를 입력할 경우, 

    batch별로
    최대 16개 채널에 대해
    미래 32 step을 예측
    '''
    def __init__(self, d_model: int, max_channels: int, horizon: int) -> None:
        super().__init__()
        self.max_channels = max_channels
        self.horizon = horizon
        self.projection = nn.Linear(d_model, max_channels * horizon)

    def forward(self, embedding: Tensor, channels: int | None = None) -> Tensor:
        value = self.projection(embedding).view(embedding.shape[0], self.max_channels, self.horizon)
        return value[:, : channels or self.max_channels]


class PatchReconstructionHead(nn.Module):
    '''
    Masked Signal Modeling용 Head, Backbone이 만든 token embedding으로부터 원래 sensor patch를 복원

    input : tokens [B, N, d_model]
    output : reconstructed_patch [B, N, patch_size]
    '''
    def __init__(self, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, patch_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens)


class ImagePatchReconstructionHead(nn.Module):
    def __init__(self, d_model: int, patch_size: int, input_channels: int = 3) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, input_channels * patch_size * patch_size)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.projection(tokens)


class ToolWearRegressionHead(RegressionHead):
    pass


class RULPredictionHead(RegressionHead):
    pass


class WearStateClassificationHead(ClassificationHead):
    pass


class ChatterDetectionHead(ClassificationHead):
    pass


class SurfaceRoughnessRegressionHead(RegressionHead):
    pass


class QualityRegressionHead(RegressionHead):
    pass


class AnomalyDetectionHead(RegressionHead):
    pass


class EmbeddingHead(nn.Identity):
    pass
