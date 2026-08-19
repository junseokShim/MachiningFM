from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleWaveformTokenizer(nn.Module):
    def __init__(self, d_model: int = 256, patch_sizes: tuple[int, ...] = (64, 256, 1280, 12800)) -> None:
        super().__init__()
        self.patch_sizes = tuple(int(size) for size in patch_sizes)
        self.proj = nn.Sequential(nn.Linear(len(self.patch_sizes) * 4, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"waveform must be [batch, channels, samples], got {tuple(waveform.shape)}")
        x = torch.nan_to_num(waveform.float())
        features = []
        for size in self.patch_sizes:
            pooled = F.avg_pool1d(x, kernel_size=min(size, x.shape[-1]), stride=min(size, x.shape[-1]), ceil_mode=True)
            rms = torch.sqrt(F.avg_pool1d(x.square(), kernel_size=min(size, x.shape[-1]), stride=min(size, x.shape[-1]), ceil_mode=True).clamp_min(1.0e-8))
            features.extend([pooled.mean(-1), pooled.std(-1, unbiased=False), rms.mean(-1), rms.amax(-1)])
        stats = torch.stack(features, dim=-1)
        return self.proj(stats)

    @staticmethod
    def reconstruct_rms(tokens: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(tokens.float(), dim=-1)
