from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn


SEFC_GROUPS = {
    "setpoint": ("cmd", "setpoint", "program", "feed", "rpm", "spindle_speed"),
    "effort": ("load", "current", "power", "torque"),
    "feedback": ("actual", "position", "pos", "speed"),
    "context": ("tool", "gcode", "mcode", "material", "machine", "block"),
}


def sefc_group(name: str) -> str:
    lowered = str(name).lower()
    for group, tokens in SEFC_GROUPS.items():
        if any(token in lowered for token in tokens):
            return group
    return "context"


def normalize_cnc(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    finite = np.isfinite(data)
    fill = np.nanmedian(np.where(finite, data, np.nan), axis=1)
    fill = np.nan_to_num(fill, nan=0.0).astype(np.float32)
    data = np.where(finite, data, fill[:, None])
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return np.clip((data - mean) / np.maximum(std, 1.0e-6), -10.0, 10.0).astype(np.float32)


def parse_numeric_nc_words(block: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for letter, value in re.findall(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", str(block).upper()):
        try:
            out[letter] = float(value)
        except ValueError:
            continue
    return out


class CncSEFCTokenizer(nn.Module):
    def __init__(self, d_model: int = 256, max_channels: int = 128) -> None:
        super().__init__()
        self.max_channels = int(max_channels)
        self.channel_proj = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.group_embedding = nn.Embedding(4, d_model)

    def forward(self, cnc: torch.Tensor, group_ids: torch.Tensor | None = None) -> torch.Tensor:
        if cnc.ndim != 3:
            raise ValueError(f"cnc must be [batch, channels, time], got {tuple(cnc.shape)}")
        x = torch.nan_to_num(cnc.float())
        dx = torch.diff(x, dim=-1, prepend=x[..., :1])
        features = torch.stack(
            [x.mean(-1), x.std(-1), x.amin(-1), x.amax(-1), dx.mean(-1), dx.std(-1)],
            dim=-1,
        )
        tokens = self.channel_proj(features)
        if group_ids is not None:
            tokens = tokens + self.group_embedding(group_ids.clamp_min(0).clamp_max(3))
        return tokens
