"""
MachiningFM backbone wrapper for downstream tasks.

Wraps the existing pretrained MachiningFMV2 model from src/machiningfm_v2/
and exposes a simple encode() interface for extracting latent representations.

The pretrained backbone is frozen by default. Fine-tuning modes are supported
but should be used carefully — freezing generally works well for few-shot adaptation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

BACKBONE_MODES = ("frozen", "linear_probe", "partial_finetune", "full_finetune")
BackboneMode = Literal["frozen", "linear_probe", "partial_finetune", "full_finetune"]


class MachiningFMBackbone(nn.Module):
    """
    Frozen (or optionally fine-tunable) wrapper around MachiningFMV2.

    Provides:
        encode(batch) → embedding of shape (B, d_model)

    The backbone defaults to 'frozen' to preserve the pretrained representations
    and enable few-shot downstream adaptation without catastrophic forgetting.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config: dict | None = None,
        backbone_mode: BackboneMode = "frozen",
    ) -> None:
        """
        Args:
            checkpoint_path: Path to pretrained MachiningFMV2 checkpoint (.pt or .pth).
            config: Model config dict. If None, uses defaults from MachiningFMV2.
            backbone_mode: How much of the backbone to allow gradients through.
        """
        super().__init__()

        # Import from existing src — must run from FOUNDATION root
        try:
            from machiningfm_v2.models.foundation_model import MachiningFMV2
        except ImportError as e:
            raise ImportError(
                "Cannot import MachiningFMV2 from src/machiningfm_v2/. "
                "Ensure you are running from the FOUNDATION root directory "
                "and the package is installed (pip install -e .).\n"
                f"Original error: {e}"
            )

        self.model = MachiningFMV2(config=config)

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state, strict=False)

        self.backbone_mode = backbone_mode
        self._apply_backbone_mode()

    def _apply_backbone_mode(self) -> None:
        """Configure gradient flow according to backbone_mode."""
        if self.backbone_mode == "frozen":
            for param in self.model.parameters():
                param.requires_grad = False

        elif self.backbone_mode == "linear_probe":
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze the final projection heads only
            for module in [self.model.process_state, self.model.health_latent,
                           self.model.domain_latent, self.model.residual_latent]:
                for param in module.parameters():
                    param.requires_grad = True

        elif self.backbone_mode == "partial_finetune":
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze fusion transformer (top layers) and task heads
            for param in self.model.fusion.parameters():
                param.requires_grad = True
            for param in self.model.tasks.parameters():
                param.requires_grad = True

        elif self.backbone_mode == "full_finetune":
            for param in self.model.parameters():
                param.requires_grad = True

    @property
    def d_model(self) -> int:
        return int(self.model.config.get("d_model", 256))

    def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Extract latent representations from the pretrained backbone.

        Args:
            batch: Dict with optional keys: raw_waveform, spectral, cnc,
                   nc_tokens, image, metadata_vector.

        Returns:
            Dict with 'embedding' (B, d_model) and other latents.
        """
        if self.backbone_mode == "frozen":
            with torch.no_grad():
                return self.model.encode(batch)
        return self.model.encode(batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.encode(batch)

    @staticmethod
    def from_pretrained(
        checkpoint_path: str | Path,
        backbone_mode: BackboneMode = "frozen",
    ) -> "MachiningFMBackbone":
        """Load backbone from a local checkpoint file."""
        return MachiningFMBackbone(
            checkpoint_path=checkpoint_path,
            backbone_mode=backbone_mode,
        )
