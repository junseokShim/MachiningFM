from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaptiveGraphLearningStemGNN(nn.Module):
    """StemGNN-style adaptive graph learning for heterogeneous variables."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(config.get("d_model", config.get("graph_hidden_dim", 512)))
        graph_dim = int(config.get("graph_projection_dim", d_model))
        adjacency = config.get("adjacency", {})
        spectral = config.get("spectral", {})
        propagation = config.get("graph_propagation", {})
        self.d_model = d_model
        self.temperature = float(adjacency.get("temperature", 0.1))
        self.top_k = int(adjacency.get("top_k", 16))
        self.threshold = float(adjacency.get("threshold", 0.0))
        self.symmetric = bool(adjacency.get("symmetric", True))
        self.normalize = bool(adjacency.get("normalize", True))
        self.mix_metadata_prior = bool(adjacency.get("mix_metadata_prior", True))
        self.metadata_prior_weight = float(adjacency.get("metadata_prior_weight", 0.2))
        self.learned_adjacency_weight = float(adjacency.get("learned_adjacency_weight", 0.8))
        self.use_temporal_fft = bool(spectral.get("use_temporal_fft", True))
        self.spectral_dropout = nn.Dropout(float(spectral.get("spectral_dropout", 0.1)))
        self.residual = bool(propagation.get("residual", True))
        self.variable_projection = nn.Linear(4, d_model)
        self.spectral_projection = nn.Linear(6, d_model)
        self.query_projection = nn.Linear(d_model, graph_dim, bias=False)
        self.key_projection = nn.Linear(d_model, graph_dim, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.propagation_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(float(propagation.get("dropout", 0.1))),
                )
                for _ in range(max(1, int(propagation.get("num_layers", 2))))
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.spectral_reconstruction_head = nn.Linear(d_model, 6)

    def forward(
        self,
        variables: Tensor,
        variable_mask: Tensor | None = None,
        metadata_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if variables.ndim == 3:
            variables = variables.unsqueeze(-1)
        if variables.ndim != 4:
            raise ValueError(f"Expected [B, V, T, F], got {tuple(variables.shape)}")
        batch, variable_count, _, _ = variables.shape
        if variable_mask is None:
            variable_mask = torch.ones(batch, variable_count, dtype=torch.bool, device=variables.device)
        variables = torch.nan_to_num(variables.float())
        temporal_stats = _temporal_stats(variables)
        spectral_stats = _spectral_stats(variables) if self.use_temporal_fft else variables.new_zeros(batch, variable_count, 6)
        hidden = self.variable_projection(temporal_stats) + self.spectral_projection(spectral_stats)
        hidden = self.spectral_dropout(hidden)

        query = F.normalize(self.query_projection(hidden), dim=-1, eps=1e-4)
        key = F.normalize(self.key_projection(hidden), dim=-1, eps=1e-4)
        logits = torch.matmul(query, key.transpose(-2, -1)) / max(self.temperature, 1e-6)
        adjacency = torch.softmax(logits, dim=-1)
        if self.symmetric:
            adjacency = 0.5 * (adjacency + adjacency.transpose(-2, -1))
        if metadata_prior is not None and self.mix_metadata_prior:
            prior = metadata_prior.to(device=adjacency.device, dtype=adjacency.dtype)
            adjacency = self.learned_adjacency_weight * adjacency + self.metadata_prior_weight * prior
        adjacency = _mask_adjacency(adjacency, variable_mask)
        adjacency = _sparsify(adjacency, self.top_k, self.threshold)
        if self.normalize:
            adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        propagated = self.value_projection(hidden)
        for layer in self.propagation_layers:
            message = torch.matmul(adjacency, propagated)
            updated = layer(message)
            propagated = propagated + updated if self.residual else updated
        propagated = self.output_norm(propagated)
        propagated = propagated * variable_mask.to(propagated.dtype).unsqueeze(-1)
        spectral_prediction = self.spectral_reconstruction_head(propagated)
        return {
            "variable_embeddings": propagated,
            "learned_adjacency": adjacency,
            "spectral_prediction": spectral_prediction,
            "spectral_target": spectral_stats,
            "diagnostics": _diagnostics(adjacency, variable_mask),
        }


def _temporal_stats(variables: Tensor) -> Tensor:
    values = variables.mean(dim=-1)
    mean = values.mean(dim=-1)
    std = values.std(dim=-1, unbiased=False)
    first = values[..., 0]
    last = values[..., -1]
    return torch.stack((mean, std, first, last), dim=-1)


def _spectral_stats(variables: Tensor) -> Tensor:
    values = variables.mean(dim=-1)
    spectrum = torch.fft.rfft(values, dim=-1)
    power = spectrum.abs().pow(2)
    total = power.sum(dim=-1).clamp_min(1e-8)
    bins = torch.linspace(0.0, 1.0, power.shape[-1], device=variables.device, dtype=variables.dtype)
    centroid = (power * bins).sum(dim=-1) / total
    spread = torch.sqrt((power * (bins - centroid.unsqueeze(-1)).pow(2)).sum(dim=-1) / total)
    low = power[..., : max(1, power.shape[-1] // 8)].sum(dim=-1) / total
    mid = power[..., power.shape[-1] // 8 : max(power.shape[-1] // 2, power.shape[-1] // 8 + 1)].sum(dim=-1) / total
    high = power[..., max(1, power.shape[-1] // 2) :].sum(dim=-1) / total
    peak = bins[power.argmax(dim=-1)]
    return torch.stack((low, mid, high, centroid, spread, peak), dim=-1)


def _mask_adjacency(adjacency: Tensor, variable_mask: Tensor) -> Tensor:
    mask = variable_mask.bool()
    pair_mask = mask[:, :, None] & mask[:, None, :]
    adjacency = adjacency.masked_fill(~pair_mask, 0.0)
    eye = torch.eye(adjacency.shape[-1], device=adjacency.device, dtype=torch.bool)[None, :, :]
    adjacency = adjacency.masked_fill(eye & mask[:, :, None], 1.0)
    return adjacency


def _sparsify(adjacency: Tensor, top_k: int, threshold: float) -> Tensor:
    if top_k > 0 and top_k < adjacency.shape[-1]:
        values, indices = torch.topk(adjacency, k=top_k, dim=-1)
        sparse = torch.zeros_like(adjacency).scatter(-1, indices, values)
        adjacency = sparse
    if threshold > 0:
        adjacency = adjacency.masked_fill(adjacency < threshold, 0.0)
    return adjacency


def _diagnostics(adjacency: Tensor, variable_mask: Tensor) -> dict[str, Tensor]:
    valid = variable_mask.to(adjacency.dtype)
    edge_count = (adjacency > 0).to(adjacency.dtype).sum(dim=(-2, -1))
    density = edge_count / valid.sum(dim=-1).clamp_min(1.0).pow(2)
    return {
        "edge_count": edge_count,
        "density": density,
        "mean_affinity": adjacency.sum(dim=(-2, -1)) / edge_count.clamp_min(1.0),
    }
