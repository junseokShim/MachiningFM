from __future__ import annotations

import torch
from torch import Tensor, nn

from machiningfm.models.embeddings import ModalityEmbedding
from machiningfm.models.fusion import concatenate_modalities
from .condition_tokenizer import ConditionTokenizer
from .frequency_tokenizer import FrequencyTokenizer
from .image_tokenizer import ImageTokenizer
from .sensor_tokenizer import SensorTokenizer
from .text_tokenizer import TextTokenizer


class MultimodalTokenizer(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        max_channels: int = 128,
        channel_vocab_size: int | None = None,
        max_conditions: int = 64,
        text_vocab_size: int = 8192,
        max_text_tokens: int = 64,
        image_patch_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sensor = SensorTokenizer(
            patch_size,
            d_model,
            max_channels,
            channel_vocab_size=channel_vocab_size,
            dropout=dropout,
        )
        self.frequency = FrequencyTokenizer(
            patch_size,
            d_model,
            max_channels,
            channel_vocab_size=channel_vocab_size,
            dropout=dropout,
        )
        self.condition = ConditionTokenizer(d_model, max_conditions)
        self.text = TextTokenizer(d_model, text_vocab_size, max_text_tokens, dropout=dropout)
        self.image = ImageTokenizer(d_model, patch_size=image_patch_size)
        self.modality = ModalityEmbedding(d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, batch: dict[str, Tensor | None]) -> tuple[Tensor, Tensor, dict[str, int | Tensor]]:
        parts: list[tuple[Tensor, Tensor]] = []
        metadata: dict[str, int | Tensor] = {}
        batch_size, device = self._batch_context(batch)
        cursor = 0
        sensor = batch.get("sensor_series")
        if sensor is not None:
            tokens, mask, patches = self.sensor(
                sensor,
                batch.get("sensor_mask"),
                batch.get("sensor_ids"),
                batch.get("sensor_attribute_ids"),
            )
            parts.append((self.modality(tokens, 0), mask))
            metadata["sensor_token_start"] = cursor
            metadata["sensor_token_count"] = tokens.shape[1]
            metadata["sensor_channels"] = sensor.shape[1]
            metadata["sensor_patches"] = patches
            cursor += tokens.shape[1]

        condition = batch.get("condition")
        if condition is not None:
            tokens, mask = self.condition(condition, batch.get("condition_mask"))
            parts.append((self.modality(tokens, 1), mask))
            metadata["condition_token_start"] = cursor
            metadata["condition_token_count"] = tokens.shape[1]
            cursor += tokens.shape[1]

        text_ids = batch.get("text_ids")
        if text_ids is not None:
            tokens, mask = self.text(text_ids, batch.get("text_mask"))
            parts.append((self.modality(tokens, 5), mask))
            metadata["text_token_start"] = cursor
            metadata["text_token_count"] = tokens.shape[1]
            cursor += tokens.shape[1]

        image = batch.get("image")
        if image is not None:
            tokens, mask, patches = self.image(image, batch.get("image_mask"))
            parts.append((self.modality(tokens, 2), mask))
            metadata["image_token_start"] = cursor
            metadata["image_token_count"] = tokens.shape[1]
            metadata["image_patches"] = patches
            cursor += tokens.shape[1]
            
        frequency = batch.get("frequency")
        if frequency is not None:
            tokens, mask, _ = self.frequency(
                frequency,
                batch.get("frequency_mask"),
                batch.get("frequency_ids"),
                batch.get("frequency_attribute_ids"),
            )
            parts.append((self.modality(tokens, 3), mask))
            metadata["frequency_token_start"] = cursor
            metadata["frequency_token_count"] = tokens.shape[1]
            metadata["frequency_channels"] = frequency.shape[1]
            cursor += tokens.shape[1]
        global_tokens = self.global_token.expand(batch_size, 1, -1).to(device)
        global_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        parts.append((self.modality(global_tokens, 4), global_mask))
        metadata["global_token_start"] = cursor
        metadata["global_token_count"] = 1
        tokens, mask = concatenate_modalities(parts)
        return tokens, mask, metadata

    def _batch_context(self, batch: dict[str, Tensor | None]) -> tuple[int, torch.device]:
        for value in batch.values():
            if isinstance(value, Tensor):
                return value.shape[0], value.device
        return 1, self.global_token.device
