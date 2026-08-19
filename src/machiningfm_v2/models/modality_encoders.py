from __future__ import annotations

import torch
from torch import nn

from machiningfm_v2.tokenizers.cnc_sefc import CncSEFCTokenizer
from machiningfm_v2.tokenizers.image import ToolImageTokenizer
from machiningfm_v2.tokenizers.nc_program import NCProgramTokenizer
from machiningfm_v2.tokenizers.spectral import SpectralTokenizer
from machiningfm_v2.tokenizers.waveform import MultiScaleWaveformTokenizer


class ModalityEncoders(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        d_model = int(config.get("d_model", 256))
        self.waveform = MultiScaleWaveformTokenizer(d_model=d_model, patch_sizes=tuple(config.get("waveform_patch_sizes", (64, 256, 1280, 12800))))
        self.spectral = SpectralTokenizer(input_length=int(config.get("frequency_length", 512)), d_model=d_model)
        self.cnc = CncSEFCTokenizer(d_model=d_model, max_channels=int(config.get("max_cnc_channels", 128)))
        self.nc = NCProgramTokenizer(input_dim=int(config.get("nc_feature_dim", 14)), d_model=d_model)
        self.image = ToolImageTokenizer(d_model=d_model)
        self.metadata = nn.Sequential(nn.Linear(int(config.get("metadata_dim", 32)), d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.modality_embedding = nn.Embedding(6, d_model)

    def add_modality(self, tokens: torch.Tensor, modality_id: int) -> torch.Tensor:
        return tokens + self.modality_embedding.weight[int(modality_id)].view(1, 1, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        device = next(self.parameters()).device

        def append(tokens: torch.Tensor, modality_id: int, valid: torch.Tensor | None = None) -> None:
            token_values = self.add_modality(tokens, modality_id)
            pieces.append(token_values)
            if valid is None:
                valid = torch.ones(token_values.shape[:2], dtype=torch.bool, device=token_values.device)
            masks.append(valid.bool())

        if (value := batch.get("raw_waveform")) is not None:
            append(self.waveform(value), 0)
        if (value := batch.get("spectral")) is not None:
            append(self.spectral(value), 1)
        if (value := batch.get("cnc")) is not None:
            append(self.cnc(value, batch.get("cnc_group_ids")), 2)
        if (value := batch.get("nc_tokens")) is not None:
            nc_valid = value.float().abs().sum(dim=-1) > 0
            append(self.nc(value), 3, nc_valid)
        if (value := batch.get("image")) is not None:
            append(self.image(value), 4)
        if (value := batch.get("metadata_vector")) is not None:
            append(self.metadata(value.float()).unsqueeze(1), 5)
        if not pieces:
            empty = torch.zeros(batch.get("batch_size", 1), 1, self.modality_embedding.embedding_dim, device=device)
            append(empty, 5, torch.zeros(empty.shape[:2], dtype=torch.bool, device=device))
        tokens = torch.cat(pieces, dim=1)
        valid_mask = torch.cat(masks, dim=1)
        return tokens, valid_mask
