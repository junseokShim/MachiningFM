"""
Utility for extracting MachiningFM backbone embeddings from raw sensor signals.

Supports both pretrained checkpoints automatically:
  - machiningfm_full_pretrain_best.pt  (7.4GB, d_model=2048, 1.32B params)
  - machiningfm_v2_best.pt             (170MB, d_model=384,  ~17M params)

Usage:
    from machiningfm.models.encoder import load_backbone, extract_embeddings

    # Use the full pretrain checkpoint (recommended)
    backbone = load_backbone("outputs/checkpoints/.../machiningfm_full_pretrain_best.pt")
    embeddings = extract_embeddings(backbone, raw_signals, batch_size=4)
    # embeddings: np.ndarray of shape (N, 2048)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from machiningfm.models.backbone import MachiningFMBackbone

# Config for the small v2 checkpoint (machiningfm_v2_best.pt, ~170MB on HF)
PRETRAINED_CONFIG_V2 = {
    "d_model": 384,
    "fusion_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "forecast_horizons": [64, 1280, 12800],
    "output_channels": 3,
}

# Alias kept for backwards compatibility with tests
PRETRAINED_CONFIG = PRETRAINED_CONFIG_V2

# Default truncation length for raw signals before backbone encoding.
# Keeps the LAST max_len time steps (end of cut = current wear state).
# 512 is practical on CPU (~6 s for 120 samples). Use 4096+ on GPU for more context.
DEFAULT_MAX_LEN = 512


def load_backbone(
    checkpoint_path: str | Path | None = None,
    hf_repo_id: str = "Junseok2/MachiningFM2.0",
    hf_filename: str = "pretrained/machiningfm_v2_base.pt",
    backbone_mode: str = "frozen",
    device: str = "cpu",
) -> "MachiningFMBackbone":
    """
    Load MachiningFMBackbone from a local checkpoint or Hugging Face Hub.

    Architecture is detected automatically from the checkpoint's model_config:
      - machiningfm_graph_tokenized_stemgnn_decoder_only → full pretrain (d=2048)
      - anything else → MachiningFMV2 (d=384)

    Args:
        checkpoint_path: Local .pt file. If None, downloads from HF (v2 only).
        hf_repo_id: HF repo to download from when checkpoint_path is None.
        hf_filename: Filename within the HF repo.
        backbone_mode: 'frozen' | 'linear_probe' | 'partial_finetune' | 'full_finetune'
        device: PyTorch device string ('cpu', 'cuda', 'mps').

    Returns:
        MachiningFMBackbone ready for encode().
    """
    from machiningfm.models.backbone import MachiningFMBackbone

    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the pretrained checkpoint.\n"
                "Install it with: pip install huggingface_hub"
            ) from exc
        print(f"Downloading checkpoint from {hf_repo_id}/{hf_filename} ...")
        checkpoint_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
        print(f"Downloaded to: {checkpoint_path}")

    backbone = MachiningFMBackbone(
        checkpoint_path=checkpoint_path,
        backbone_mode=backbone_mode,
    )
    backbone = backbone.to(device)
    backbone.eval()
    print(f"[encoder] Loaded {backbone.architecture} backbone, d_model={backbone.d_model}")
    return backbone


def extract_embeddings(
    backbone: "MachiningFMBackbone",
    raw_signals: list[np.ndarray],
    batch_size: int = 4,
    device: str = "cpu",
    max_len: int = DEFAULT_MAX_LEN,
) -> np.ndarray:
    """
    Run raw sensor signals through the frozen backbone and return embeddings.

    Each signal is padded or truncated to max_len time steps before encoding.
    Padding uses zeros at the beginning; truncation keeps the LAST max_len steps
    (end of cut carries current wear state).

    Args:
        backbone: Loaded MachiningFMBackbone instance.
        raw_signals: List of (T_i, C) float32 arrays (variable length OK).
        batch_size: Number of samples per forward pass.
        device: PyTorch device string.
        max_len: Fixed sequence length for batching.

    Returns:
        np.ndarray of shape (N, d_model) — one embedding per signal.
    """
    if not raw_signals:
        return np.zeros((0, backbone.d_model), dtype=np.float32)

    backbone.eval()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(raw_signals), batch_size):
        batch_signals = raw_signals[start : start + batch_size]
        padded = _pad_or_truncate_batch(batch_signals, max_len)
        tensor = torch.from_numpy(padded).to(device)  # (B, max_len, C)

        with torch.no_grad():
            out = backbone.encode({"raw_waveform": tensor})

        emb = out["embedding"].cpu().float().numpy()  # (B, d_model)
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def _pad_or_truncate_batch(signals: list[np.ndarray], max_len: int) -> np.ndarray:
    """
    Pad/truncate a list of (T_i, C) arrays to a single (B, max_len, C) array.

    - Truncation: keep the LAST max_len steps.
    - Padding: zeros prepended at the beginning so valid data is at the tail.
    """
    n_channels = signals[0].shape[1] if signals[0].ndim == 2 else 1
    batch = np.zeros((len(signals), max_len, n_channels), dtype=np.float32)
    for i, sig in enumerate(signals):
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
        t = sig.shape[0]
        if t >= max_len:
            batch[i] = sig[-max_len:]
        else:
            batch[i, -t:] = sig
    return batch
