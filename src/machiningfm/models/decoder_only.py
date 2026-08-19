from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, token_mask: Tensor | None = None) -> Tensor:
        batch, length, _ = tokens.shape
        q = self.q_proj(tokens).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rotary_cache(length, self.head_dim, self.rope_theta, tokens.device, tokens.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
        if token_mask is not None:
            key_mask = ~token_mask.bool()
            scores = scores.masked_fill(key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v).transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(tokens)) * self.up_proj(tokens)))


class DecoderOnlyBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int | None,
        ffn_dim: int,
        dropout: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, num_kv_heads, dropout, rope_theta)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, ffn_dim, dropout)

    def forward(self, tokens: Tensor, token_mask: Tensor | None = None) -> Tensor:
        tokens = tokens + self.attn(self.attn_norm(tokens), token_mask)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config.get("d_model", 512))
        num_layers = int(config.get("num_layers", 12))
        num_heads = int(config.get("num_heads", 8))
        num_kv_heads = int(config.get("num_kv_heads", num_heads))
        ffn_dim = int(config.get("ffn_dim", d_model * int(config.get("ff_mult", 4))))
        dropout = float(config.get("dropout", 0.1))
        rope_theta = float(config.get("rope_theta", 10000.0))
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", False))
        self.layers = nn.ModuleList(
            [
                DecoderOnlyBlock(d_model, num_heads, num_kv_heads, ffn_dim, dropout, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(layer, tokens, token_mask, use_reentrant=False)
            else:
                tokens = layer(tokens, token_mask)
        return self.norm(tokens)


def estimate_decoder_only_parameters(config: dict[str, Any]) -> int:
    d_model = int(config.get("d_model", 512))
    num_layers = int(config.get("num_layers", 12))
    num_heads = int(config.get("num_heads", 8))
    num_kv_heads = int(config.get("num_kv_heads", num_heads))
    head_dim = d_model // num_heads
    ffn_dim = int(config.get("ffn_dim", d_model * int(config.get("ff_mult", 4))))
    q = d_model * (num_heads * head_dim)
    kv = 2 * d_model * (num_kv_heads * head_dim)
    out = d_model * d_model
    ffn = 3 * d_model * ffn_dim
    norms = 4 * d_model
    final_norm = 2 * d_model
    return int(num_layers * (q + kv + out + ffn + norms) + final_norm)


def _rotary_cache(
    length: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    rotary_dim = dim if dim % 2 == 0 else dim - 1
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(length, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotary_dim = cos.shape[-1] * 2
    rotated = value[..., :rotary_dim]
    rest = value[..., rotary_dim:]
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rope = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((rope, rest), dim=-1) if rest.numel() else rope
