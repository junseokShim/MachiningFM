"""
Downstream head architectures for MachiningFM.

These heads sit on top of the frozen (or fine-tuned) backbone embedding
and produce task-specific predictions before physics calibration.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WearRegressionHead(nn.Module):
    """
    Linear or MLP head for tool wear regression (→ VB in mm).

    Takes the backbone embedding (shape: B × d_model) and outputs a scalar VB prediction.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        use_mlp: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if use_mlp:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Args: embedding (B, d_model). Returns: wear_vb (B,) in mm (raw, uncalibrated)."""
        return self.head(embedding).squeeze(-1)


class WearStageHead(nn.Module):
    """
    Classification head for tool wear stage (healthy / moderate / severe).

    Takes backbone embedding and outputs logits for each wear stage.
    """

    def __init__(
        self,
        d_model: int,
        n_stages: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_stages),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns: stage_logits (B, n_stages)."""
        return self.head(embedding)


class RULHead(nn.Module):
    """
    Regression head for Remaining Useful Life prediction (→ minutes).

    Outputs are passed through ReLU to enforce non-negative RUL predictions.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns: rul_min (B,) — remaining useful life in minutes (>= 0)."""
        return self.head(embedding).squeeze(-1)
