from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from machiningfm.data.channel_schema import CHANNEL_SCHEMA_VERSION
from machiningfm.tokenizers.multimodal_tokenizer import MultimodalTokenizer
from .backbone import MachiningBackbone
from .fusion import MaskedMeanFusion
from .heads import (
    AnomalyDetectionHead,
    ChatterDetectionHead,
    ClassificationHead,
    ForecastingHead,
    ImagePatchReconstructionHead,
    PatchReconstructionHead,
    QualityRegressionHead,
    RULPredictionHead,
    RegressionHead,
    SurfaceRoughnessRegressionHead,
    ToolWearRegressionHead,
    WearStateClassificationHead,
)


class MachiningFM(nn.Module):
    """Task-agnostic multimodal foundation backbone with detachable task heads."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.config = cfg
        self.channel_schema_version = str(cfg.setdefault("channel_schema_version", CHANNEL_SCHEMA_VERSION))
        if self.channel_schema_version != CHANNEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported channel schema {self.channel_schema_version!r}; "
                f"expected {CHANNEL_SCHEMA_VERSION!r}"
            )
        d_model = int(cfg.get("d_model", 128))
        patch_size = int(cfg.get("patch_size", 16))
        image_patch_size = int(cfg.get("image_patch_size", 16))
        max_channels = int(cfg.get("max_channels", 128))
        horizon = int(cfg.get("horizon", 16))
        self.tokenizer = MultimodalTokenizer(
            d_model=d_model,
            patch_size=patch_size,
            max_channels=max_channels,
            channel_vocab_size=int(cfg.get("channel_vocab_size", max_channels)),
            max_conditions=int(cfg.get("max_conditions", 64)),
            text_vocab_size=int(cfg.get("text_vocab_size", 8192)),
            max_text_tokens=int(cfg.get("max_text_tokens", 64)),
            image_patch_size=image_patch_size,
            dropout=float(cfg.get("dropout", 0.1)),
        )
        self.backbone = MachiningBackbone(
            d_model=d_model,
            num_layers=int(cfg.get("num_layers", 4)),
            num_heads=int(cfg.get("num_heads", 4)),
            ff_mult=int(cfg.get("ff_mult", 4)),
            dropout=float(cfg.get("dropout", 0.1)),
            causal=bool(cfg.get("causal", False)),
        )
        self.fusion = MaskedMeanFusion()
        self.reconstruction_head = PatchReconstructionHead(d_model, patch_size)
        self.frequency_reconstruction_head = PatchReconstructionHead(d_model, patch_size)
        self.image_reconstruction_head = ImagePatchReconstructionHead(d_model, image_patch_size)
        self.forecasting_head = ForecastingHead(d_model, max_channels, horizon)
        class_count = int(cfg.get("class_count", 3))
        self.task_heads = nn.ModuleDict(
            {
                "toolwear_regression": ToolWearRegressionHead(d_model),
                "rul_prediction": RULPredictionHead(d_model),
                "wear_state_classification": WearStateClassificationHead(d_model, class_count),
                "chatter_detection": ChatterDetectionHead(d_model, 2),
                "surface_roughness_prediction": SurfaceRoughnessRegressionHead(d_model),
                "quality_prediction": QualityRegressionHead(d_model),
                "anomaly_detection": AnomalyDetectionHead(d_model),
                "energy_prediction": RegressionHead(d_model),
                "generic_classification": ClassificationHead(d_model, class_count),
            }
        )

    def encode(self, batch: dict[str, Tensor | None]) -> dict[str, Any]:
        tokens, token_mask, metadata = self.tokenizer(batch)
        hidden = self.backbone(tokens, token_mask)
        embedding = self.fusion(hidden, token_mask)
        return {"tokens": hidden, "token_mask": token_mask, "embedding": embedding, **metadata}

    def forward(self, batch: dict[str, Tensor | None], task: str | None = None) -> dict[str, Any]:
        output = self.encode(batch)
        sensor_tokens = _slice_tokens(output, "sensor")
        frequency_tokens = _slice_tokens(output, "frequency")
        image_tokens = _slice_tokens(output, "image")
        output["patch_reconstruction"] = self.reconstruction_head(sensor_tokens)
        output["frequency_patch_reconstruction"] = self.frequency_reconstruction_head(frequency_tokens)
        output["image_patch_reconstruction"] = self.image_reconstruction_head(image_tokens)
        output["forecast"] = self.forecasting_head(
            output["embedding"], int(output.get("sensor_channels", 1))
        )
        if task:
            if task == "future_forecasting":
                output["prediction"] = output["forecast"]
            elif task in self.task_heads:
                output["prediction"] = self.task_heads[task](output["embedding"])
            else:
                raise KeyError(f"Unknown task head: {task}")
        return output


def _slice_tokens(output: dict[str, Any], name: str) -> Tensor:
    start = int(output.get(f"{name}_token_start", 0))
    count = int(output.get(f"{name}_token_count", 0))
    return output["tokens"][:, start : start + count]
