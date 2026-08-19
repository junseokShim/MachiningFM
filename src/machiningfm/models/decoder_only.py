"""
Decoder-only Transformer with GQA, SwiGLU, and RoPE.

Reconstructed from pretrained checkpoint weight shapes:
  d_model=2048, num_heads=32, num_kv_heads=8, ffn_dim=8192, 20 layers
  q_proj: (2048, 2048), k_proj/v_proj: (512, 2048), ffn gate/up/down: (8192, 2048)
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin rotation matrices for RoPE."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    outer = torch.outer(positions, freqs)  # (max_seq_len, half)
    outer = torch.cat([outer, outer], dim=-1)  # (max_seq_len, head_dim)
    return outer.cos(), outer.sin()


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to q and k. Inputs shape: (B, H, T, D)."""
    T = q.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class _GQAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        rope_theta: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # RoPE cache (will be extended if needed)
        cos, sin = _build_rope_cache(max_seq_len, self.head_dim, theta=rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim

        # Extend RoPE cache if sequence is longer than precomputed
        if T > self.rope_cos.shape[0]:
            cos, sin = _build_rope_cache(T, D, device=x.device, dtype=x.dtype)
            self.rope_cos = cos
            self.rope_sin = sin

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)       # (B, H, T, D)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)     # (B, Hkv, T, D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)     # (B, Hkv, T, D)

        cos = self.rope_cos.to(dtype=x.dtype)
        sin = self.rope_sin.to(dtype=x.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        # Expand kv heads to match q heads (GQA)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        # Attention mask from key_padding_mask (True = valid token)
        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            # (B, 1, 1, T) — False for padding positions
            attn_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # prefix-LM: encoder-style over all tokens
        )
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class _SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class _DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_dim: int,
        dropout: float,
        max_seq_len: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = _GQAttention(d_model, num_heads, num_kv_heads, max_seq_len, rope_theta, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = _SwiGLUFFN(d_model, ffn_dim)
        self._dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self._dropout(self.attn(self.attn_norm(x), key_padding_mask))
        x = x + self._dropout(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for MachiningFM.

    Used as the backbone trunk in GraphTokenizedStemGNNDecoderOnlyMachiningFM.
    Attention is bidirectional (prefix-LM style) over the full token sequence.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model      = int(config.get("d_model", 512))
        num_layers   = int(config.get("num_layers", 12))
        num_heads    = int(config.get("num_heads", 8))
        num_kv_heads = int(config.get("num_kv_heads", num_heads))
        ffn_dim      = int(config.get("ffn_dim", d_model * 4))
        dropout      = float(config.get("dropout", 0.0))
        max_seq_len  = int(config.get("max_sequence_length", 4096))
        rope_theta   = float(config.get("rope_theta", 10000.0))

        self.layers = nn.ModuleList([
            _DecoderLayer(d_model, num_heads, num_kv_heads, ffn_dim, dropout, max_seq_len, rope_theta)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            x: Token sequence, shape (B, T, d_model).
            key_padding_mask: Boolean mask, shape (B, T). True = valid token.

        Returns:
            Hidden states, shape (B, T, d_model).
        """
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.norm(x)


def estimate_decoder_only_parameters(config: dict[str, Any]) -> dict[str, int]:
    """Estimate parameter count without instantiating the model."""
    d_model      = int(config.get("d_model", 512))
    num_layers   = int(config.get("num_layers", 12))
    num_heads    = int(config.get("num_heads", 8))
    num_kv_heads = int(config.get("num_kv_heads", num_heads))
    ffn_dim      = int(config.get("ffn_dim", d_model * 4))
    head_dim     = d_model // num_heads

    per_layer = (
        2 * d_model                                          # attn_norm + ffn_norm (w+b each)
        + d_model * (num_heads * head_dim)                   # q_proj
        + d_model * (num_kv_heads * head_dim) * 2           # k_proj + v_proj
        + d_model * d_model                                  # out_proj
        + d_model * ffn_dim * 2                              # gate_proj + up_proj
        + ffn_dim * d_model                                  # down_proj
    )
    norm_params = 2 * d_model  # final norm
    decoder = num_layers * per_layer + norm_params
    return {"decoder": decoder, "total": decoder}
