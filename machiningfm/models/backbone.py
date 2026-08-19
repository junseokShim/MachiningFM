"""
MachiningFM backbone wrapper for downstream tasks.

Supports two pretrained architectures — detected automatically from the checkpoint:

  1. GraphTokenizedStemGNNDecoderOnlyMachiningFM  (machiningfm_full_pretrain_best.pt)
       d_model=2048, 20 layers, 1.32B params
       Input: batch["sensor_series"] shape (B, C, T)
       Output embedding: (B, 2048)

  2. MachiningFMV2  (machiningfm_v2_best.pt)
       d_model=384, 6 layers, ~17M params
       Input: batch["raw_waveform"] shape (B, T, C)
       Output embedding: (B, 384)

Use from_checkpoint() or load_backbone() in encoder.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

BACKBONE_MODES = ("frozen", "linear_probe", "partial_finetune", "full_finetune")
BackboneMode = Literal["frozen", "linear_probe", "partial_finetune", "full_finetune"]

_ARCH_FULL = "machiningfm_graph_tokenized_stemgnn_decoder_only"
_ARCH_V2   = "machiningfm_v2"


class MachiningFMBackbone(nn.Module):
    """
    Unified backbone wrapper for MachiningFM pretrained models.

    Detects architecture from the checkpoint's model_config and loads the
    correct model class.  Exposes a common encode() → dict interface with
    an "embedding" key regardless of underlying architecture.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: dict | None = None,
        backbone_mode: BackboneMode = "frozen",
    ) -> None:
        super().__init__()
        self.backbone_mode = backbone_mode
        self._arch: str = _ARCH_V2

        ckpt_config: dict[str, Any] = {}
        state_dict: dict | None = None

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_config = ckpt.get("model_config", ckpt.get("config", {}))
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

        merged_config = {**ckpt_config, **(config or {})}
        arch = merged_config.get("architecture", _ARCH_V2)

        if arch == _ARCH_FULL:
            self._arch = _ARCH_FULL
            self.model = self._load_full_pretrain(merged_config, state_dict)
        else:
            self._arch = _ARCH_V2
            self.model = self._load_v2(merged_config, state_dict)

        self._apply_backbone_mode()

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_full_pretrain(self, config: dict, state_dict) -> nn.Module:
        import sys
        from pathlib import Path as _Path

        _root = _Path(__file__).parent.parent.parent
        src = str(_root / "src")

        # Temporarily expose src/machiningfm as the 'machiningfm' package so that
        # its relative imports work.  We save and restore sys.modules and sys.path
        # to avoid polluting the session after loading.
        saved_path = sys.path[:]
        saved_mods = {k: v for k, v in sys.modules.items() if k == "machiningfm" or k.startswith("machiningfm.")}

        try:
            # Remove the downstream machiningfm (root) so src/machiningfm takes over
            for key in list(sys.modules.keys()):
                if key == "machiningfm" or key.startswith("machiningfm."):
                    del sys.modules[key]

            if src not in sys.path:
                sys.path.insert(0, src)

            from machiningfm.models.graph_tokenized_machiningfm import (  # noqa: E402
                GraphTokenizedStemGNNDecoderOnlyMachiningFM,
            )
        finally:
            # Restore sys.path and sys.modules
            sys.path[:] = saved_path
            for key in list(sys.modules.keys()):
                if key == "machiningfm" or key.startswith("machiningfm."):
                    del sys.modules[key]
            sys.modules.update(saved_mods)

        model = GraphTokenizedStemGNNDecoderOnlyMachiningFM(config)
        if state_dict is not None:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[backbone] {len(missing)} missing keys (e.g. {missing[:3]})")
            if unexpected:
                print(f"[backbone] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        return model

    def _load_v2(self, config: dict, state_dict) -> nn.Module:
        import sys
        from pathlib import Path as _Path
        _root = _Path(__file__).parent.parent.parent
        src = str(_root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from machiningfm_v2.models.foundation_model import MachiningFMV2
        except ImportError as e:
            raise ImportError(
                "Cannot import MachiningFMV2. Ensure src/ is in sys.path.\n" + str(e)
            ) from e

        model = MachiningFMV2(config=config)
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
        return model

    # ------------------------------------------------------------------
    # Backbone mode
    # ------------------------------------------------------------------

    def _apply_backbone_mode(self) -> None:
        if self.backbone_mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad = False
        elif self.backbone_mode == "full_finetune":
            for p in self.model.parameters():
                p.requires_grad = True
        # linear_probe / partial_finetune: architecture-specific, default to frozen
        else:
            for p in self.model.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        if self._arch == _ARCH_FULL:
            return int(self.model.d_model)
        return int(self.model.config.get("d_model", 256))

    @property
    def architecture(self) -> str:
        return self._arch

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract latent representations.

        For the full pretrain model, raw_waveform (B, T, C) is automatically
        transposed to sensor_series (B, C, T) before encoding.

        Returns dict with 'embedding' key of shape (B, d_model).
        """
        adapted = self._adapt_batch(batch)
        if self.backbone_mode == "frozen":
            with torch.no_grad():
                return self.model.encode(adapted)
        return self.model.encode(adapted)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.encode(batch)

    def _adapt_batch(self, batch: dict) -> dict:
        """
        Translate the common raw_waveform (B, T, C) key into the format
        expected by the underlying model.
        """
        if self._arch == _ARCH_FULL:
            adapted = {k: v for k, v in batch.items() if k != "raw_waveform"}
            if "raw_waveform" in batch and "sensor_series" not in batch:
                # (B, T, C) → (B, C, T)
                waveform = batch["raw_waveform"]
                adapted["sensor_series"] = waveform.permute(0, 2, 1)
                # Ensure metadata list matches batch size so material_token shape is (B, 1, D)
                if "metadata" not in adapted:
                    B = waveform.shape[0]
                    adapted["metadata"] = [{"material": "unknown"} for _ in range(B)]
            return adapted
        return batch  # MachiningFMV2 already expects raw_waveform

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str | Path,
        backbone_mode: BackboneMode = "frozen",
    ) -> "MachiningFMBackbone":
        return MachiningFMBackbone(
            checkpoint_path=checkpoint_path,
            backbone_mode=backbone_mode,
        )
