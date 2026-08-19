from __future__ import annotations

import hashlib
import re

import numpy as np
import torch
from torch import nn


WORD_RE = re.compile(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
BLOCK_RE = re.compile(r"^\s*/?\s*N\s*(\d+)?\s*(.*)$", re.IGNORECASE)


def parse_nc_program(text: str, max_blocks: int = 512) -> list[dict[str, float]]:
    blocks: list[dict[str, float]] = []
    modal: dict[str, float] = {}
    for raw in str(text).splitlines():
        line = raw.split(";", 1)[0].strip().upper()
        if not line or line.startswith("("):
            continue
        words = {letter: float(value) for letter, value in WORD_RE.findall(line)}
        if words:
            modal.update({key: value for key, value in words.items() if key in {"G", "M", "S", "F", "T"}})
            blocks.append({**modal, **words})
        if len(blocks) >= max_blocks:
            break
    return blocks


def nc_blocks_to_array(blocks: list[dict[str, float]], max_blocks: int = 512) -> np.ndarray:
    keys = ["G", "M", "T", "S", "F", "X", "Y", "Z", "I", "J", "K", "R", "Q", "P"]
    out = np.zeros((max_blocks, len(keys)), dtype=np.float32)
    for row, block in enumerate(blocks[:max_blocks]):
        for col, key in enumerate(keys):
            out[row, col] = float(block.get(key, 0.0))
    scale = np.asarray([100, 100, 100, 20000, 20000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float32)
    return np.clip(out / scale[None, :], -10.0, 10.0)


def hash_nc_teacher_summary(text: str, dim: int = 64) -> np.ndarray:
    digest = hashlib.sha256(str(text).encode("utf-8", errors="ignore")).digest()
    values = np.frombuffer((digest * ((dim // len(digest)) + 1))[:dim], dtype=np.uint8).astype(np.float32)
    return (values / 127.5 - 1.0).astype(np.float32)


class NCProgramTokenizer(nn.Module):
    def __init__(self, input_dim: int = 14, d_model: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, nc_tokens: torch.Tensor) -> torch.Tensor:
        if nc_tokens.ndim != 3:
            raise ValueError(f"nc_tokens must be [batch, blocks, features], got {tuple(nc_tokens.shape)}")
        return self.proj(nc_tokens.float())
