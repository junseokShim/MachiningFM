"""
Utility for extracting MachiningFM backbone embeddings from raw sensor signals.

Usage:
    from machiningfm.models.encoder import load_backbone, extract_embeddings

    backbone = load_backbone("pretrained/machiningfm_v2_base.pt", device="cpu")
    embeddings = extract_embeddings(backbone, raw_signals, batch_size=16)
    # embeddings: np.ndarray of shape (N, 384)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from machiningfm.models.backbone import MachiningFMBackbone

# Config matching the uploaded pretrained checkpoint (base.yaml)
PRETRAINED_CONFIG = {
    "d_model": 384,
    "fusion_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "forecast_horizons": [64, 1280, 12800],
    "output_channels": 3,
}

# Default truncation length — keeps the last N time steps of each cut.
# PHM2010 cuts have ~500 synthetic / ~50,000 real time steps.
# The end of a cut carries the current wear state, so tail-truncation is appropriate.
# 512 is fast on CPU (~6s for 120 samples); use 4096+ on GPU for fuller context.
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

    Args:
        checkpoint_path: Local .pt file. If None, downloads from HF.
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
        config=PRETRAINED_CONFIG,
        backbone_mode=backbone_mode,
    )
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def extract_embeddings(
    backbone: "MachiningFMBackbone",
    raw_signals: list[np.ndarray],
    batch_size: int = 16,
    device: str = "cpu",
    max_len: int = DEFAULT_MAX_LEN,
) -> np.ndarray:
    """
    Run raw sensor signals through the frozen backbone and return embeddings.

    Each signal is padded or truncated to max_len time steps before encoding.
    Padding uses zeros; truncation keeps the last max_len steps (end of cut is
    most informative for wear state).

    Args:
        backbone: Loaded MachiningFMBackbone instance.
        raw_signals: List of (T_i, C) float32 arrays (variable length OK).
        batch_size: Number of samples per forward pass.
        device: PyTorch device string.
        max_len: Fixed sequence length for batching (truncate/pad).

    Returns:
        np.ndarray of shape (N, d_model) — one embedding per signal.
    """
    if not raw_signals:
        d_model = backbone.d_model
        return np.zeros((0, d_model), dtype=np.float32)

    backbone.eval()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(raw_signals), batch_size):
        batch_signals = raw_signals[start : start + batch_size]
        padded = _pad_or_truncate_batch(batch_signals, max_len)
        tensor = torch.from_numpy(padded).to(device)  # (B, max_len, C)

        with torch.no_grad():
            out = backbone.encode({"raw_waveform": tensor})

        emb = out["embedding"].cpu().numpy()  # (B, d_model)
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def _pad_or_truncate_batch(signals: list[np.ndarray], max_len: int) -> np.ndarray:
    """
    Pad/truncate a list of (T_i, C) arrays to a single (B, max_len, C) array.

    Truncation keeps the LAST max_len steps (wear progression is monotonic;
    the end of a cut carries the current wear state).
    Padding adds zeros at the beginning so that valid data is at the end.
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
