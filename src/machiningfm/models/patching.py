from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from machiningfm.data.channel_schema import CHANNEL_ATTRIBUTE_NAMES, CHANNEL_ATTRIBUTE_VOCAB_SIZES


def patchify(series: Tensor, patch_size: int, stride: int | None = None) -> Tensor:
    """Convert [B, C, T] series to [B, C, N, P] patches."""
    if series.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got {tuple(series.shape)}")
    stride = stride or patch_size
    length = series.shape[-1]
    patch_count = max(1, math.ceil(max(0, length - patch_size) / stride) + 1)
    required = (patch_count - 1) * stride + patch_size
    if required > length:
        series = F.pad(series, (0, required - length))
    return series.unfold(-1, patch_size, stride)


class SensorPatchEmbedding(nn.Module):
    """Channel-independent patch projection that accepts a variable channel count."""

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        max_channels: int = 128,
        channel_vocab_size: int | None = None,
        channel_attribute_vocab_sizes: tuple[int, ...] | None = None,
        stride: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.max_channels = max_channels
        self.channel_vocab_size = channel_vocab_size or max_channels
        self.projection = nn.Linear(patch_size, d_model)
        self.channel_embedding = nn.Embedding(self.channel_vocab_size, d_model)
        attribute_vocab_sizes = channel_attribute_vocab_sizes or CHANNEL_ATTRIBUTE_VOCAB_SIZES
        if len(attribute_vocab_sizes) != len(CHANNEL_ATTRIBUTE_NAMES):
            raise ValueError(
                f"Expected {len(CHANNEL_ATTRIBUTE_NAMES)} channel attribute vocab sizes, "
                f"got {len(attribute_vocab_sizes)}"
            )
        self.channel_attribute_embeddings = nn.ModuleList(
            [nn.Embedding(size, d_model) for size in attribute_vocab_sizes]
        )
        self.position_embedding = nn.Embedding(8192, d_model)
        self.missing_channel_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        series: Tensor,
        channel_mask: Tensor | None = None,
        channel_ids: Tensor | None = None,
        channel_attribute_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        patches = patchify(series, self.patch_size, self.stride)
        batch, channels, patch_count, _ = patches.shape
        if channels > self.max_channels:
            raise ValueError(f"Received {channels} channels, max_channels={self.max_channels}")
        tokens = self.projection(patches)
        if channel_ids is None:
            channel_ids = torch.arange(channels, device=series.device)[None, :].expand(batch, channels)
        elif channel_ids.ndim == 1:
            channel_ids = channel_ids[None, :].expand(batch, channels)
        channel_ids = channel_ids.long().clamp(0, self.channel_vocab_size - 1)
        position_ids = torch.arange(patch_count, device=series.device)
        tokens = tokens + self.channel_embedding(channel_ids)[:, :, None, :]
        if channel_attribute_ids is not None:
            if channel_attribute_ids.ndim == 2:
                channel_attribute_ids = channel_attribute_ids[None, :, :].expand(batch, -1, -1)
            expected_shape = (batch, channels, len(self.channel_attribute_embeddings))
            if tuple(channel_attribute_ids.shape) != expected_shape:
                raise ValueError(
                    f"Expected channel attributes [B, C, A]={expected_shape}, "
                    f"got {tuple(channel_attribute_ids.shape)}"
                )
            for attribute_index, embedding in enumerate(self.channel_attribute_embeddings):
                attribute_ids = channel_attribute_ids[:, :, attribute_index].long()
                attribute_ids = attribute_ids.clamp(0, embedding.num_embeddings - 1)
                tokens = tokens + embedding(attribute_ids)[:, :, None, :]
        tokens = tokens + self.position_embedding(position_ids)[None, None, :, :]
        if channel_mask is None:
            channel_mask = torch.ones(batch, channels, dtype=torch.bool, device=series.device)
        channel_mask = channel_mask.bool()
        expanded_mask = channel_mask[:, :, None].expand(batch, channels, patch_count)
        missing = self.missing_channel_token.expand(batch, channels, patch_count, -1)
        tokens = torch.where(expanded_mask[..., None], tokens, missing)
        return self.dropout(tokens.flatten(1, 2)), expanded_mask.flatten(1, 2), patches
