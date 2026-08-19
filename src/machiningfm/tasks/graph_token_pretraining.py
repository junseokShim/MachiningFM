from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from machiningfm.models.losses import masked_mse


class GraphTokenizedDecoderOnlyPretraining(nn.Module):
    """Decoder-only pretraining losses over adaptive graph tokens."""

    def __init__(
        self,
        model: nn.Module,
        weights: dict[str, Any] | None = None,
        mask_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.model = model
        self.weights = weights or {}
        self.mask_ratio = mask_ratio

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        mask = self._graph_mask(batch)
        output = self.model({**batch, "graph_token_reconstruction_mask": mask})
        losses = {
            "next_token_prediction": self._next_token_prediction(output),
            "masked_token_reconstruction": self._masked_graph_token_reconstruction(output, mask),
            "masked_high_rate_raw_reconstruction": self._typed_graph_reconstruction(output, "high_frequency_vibration"),
            "masked_fft_reconstruction": self._typed_graph_reconstruction(output, "high_frequency_vibration"),
            "frequency_band_reconstruction": self._typed_graph_reconstruction(output, "high_frequency_vibration"),
            "time_frequency_patch_reconstruction": self._typed_graph_reconstruction(output, "high_frequency_vibration"),
            "cnc_low_rate_reconstruction": self._typed_graph_reconstruction(output, "cnc_operation_state"),
            "cnc_forecasting": self._forecasting(output, batch),
            "raw_to_fft_generation": self._paired_generation(output, "high_frequency_vibration", "high_frequency_vibration"),
            "fft_to_raw_feature_generation": self._paired_generation(output, "high_frequency_vibration", "spindle_dynamics"),
            "cnc_to_high_rate_feature_generation": self._paired_generation(output, "cnc_operation_state", "high_frequency_vibration"),
            "material_conditioned_generation": self._typed_graph_reconstruction(output, "material_process_context"),
            "high_rate_cnc_alignment": self._alignment(output, "high_frequency_vibration", "cnc_operation_state"),
            "cross_modal_alignment": self._cross_modal_alignment(output),
            "adaptive_graph_sparsity_regularization": self._adjacency_sparsity(output),
            "adaptive_graph_temporal_consistency": self._adjacency_consistency(output),
            "stemgnn_spectral_reconstruction": self._spectral_reconstruction(output),
            "graph_token_next_prediction": self._next_graph_token_prediction(output),
            "masked_graph_token_reconstruction": self._masked_graph_token_reconstruction(output, mask),
            "graph_token_to_graph_token_generation": self._paired_generation(output, None, None),
            "cnc_graph_token_to_high_frequency_graph_token_generation": self._paired_generation(
                output, "cnc_operation_state", "high_frequency_vibration"
            ),
            "material_conditioned_graph_token_generation": self._paired_generation(
                output, "material_process_context", "high_frequency_vibration"
            ),
            "raw_high_frequency_graph_token_to_fft_graph_token_generation": self._paired_generation(
                output, "high_frequency_vibration", "high_frequency_vibration"
            ),
            "fft_graph_token_to_raw_feature_generation": self._paired_generation(
                output, "high_frequency_vibration", "spindle_dynamics"
            ),
            "cross_graph_token_alignment": self._cross_modal_alignment(output),
            "graph_token_contrastive_alignment": self._contrastive_alignment(output),
        }
        total = sum(loss * self._weight(name) for name, loss in losses.items())
        return total, losses

    def _graph_mask(self, batch: dict[str, Any]) -> Tensor:
        batch_size = _batch_size(batch)
        device = _device(batch, self.model)
        count = int(getattr(self.model, "graph_token_count", 1))
        return torch.rand(batch_size, count, device=device) < self.mask_ratio

    def _next_token_prediction(self, output: dict[str, Any]) -> Tensor:
        tokens = output["tokens"]
        prediction = output["next_token_prediction"]
        mask = output["token_mask"]
        if tokens.shape[1] < 2:
            return _zero(self.model)
        return masked_mse(prediction[:, :-1], tokens[:, 1:].detach(), mask[:, 1:])

    def _next_graph_token_prediction(self, output: dict[str, Any]) -> Tensor:
        hidden = output["graph_token_hidden"]
        target = output["graph_token_targets"]
        mask = output["graph_token_mask"]
        if hidden.shape[1] < 2:
            return _zero(self.model)
        prediction = output["graph_token_reconstruction"][:, :-1]
        return masked_mse(prediction, target[:, 1:].detach(), mask[:, 1:])

    def _masked_graph_token_reconstruction(self, output: dict[str, Any], mask: Tensor) -> Tensor:
        valid = output["graph_token_mask"] & mask
        if not valid.any():
            return _zero(self.model)
        return masked_mse(
            output["graph_token_reconstruction"],
            output["graph_token_targets"].detach(),
            valid,
        )

    def _typed_graph_reconstruction(self, output: dict[str, Any], token_type: str) -> Tensor:
        mask = _type_mask(output, token_type)
        if not mask.any():
            return _zero(self.model)
        return masked_mse(output["graph_token_reconstruction"], output["graph_token_targets"].detach(), mask)

    def _paired_generation(self, output: dict[str, Any], source: str | None, target: str | None) -> Tensor:
        source_mask = output["graph_token_mask"] if source is None else _type_mask(output, source)
        target_mask = output["graph_token_mask"] if target is None else _type_mask(output, target)
        if not source_mask.any() or not target_mask.any():
            return _zero(self.model)
        source_token = _masked_mean(output["graph_token_hidden"], source_mask)
        target_token = _masked_mean(output["graph_token_targets"].detach(), target_mask)
        return F.smooth_l1_loss(source_token, target_token)

    def _forecasting(self, output: dict[str, Any], batch: dict[str, Any]) -> Tensor:
        series = batch.get("sensor_series")
        if not isinstance(series, Tensor):
            return _zero(self.model)
        forecast = output["forecast"]
        target = series[..., -forecast.shape[-1] :]
        if target.shape[1] != forecast.shape[1]:
            target = target[:, : forecast.shape[1]]
        loss = F.smooth_l1_loss(forecast, target, reduction="none")
        sensor_mask = batch.get("sensor_mask")
        if isinstance(sensor_mask, Tensor):
            weights = sensor_mask[:, : forecast.shape[1]].to(loss.dtype).unsqueeze(-1)
            return (loss * weights).sum() / (weights.sum() * loss.shape[-1]).clamp_min(1.0)
        return loss.mean()

    def _alignment(self, output: dict[str, Any], left: str, right: str) -> Tensor:
        left_mask = _type_mask(output, left)
        right_mask = _type_mask(output, right)
        if not left_mask.any() or not right_mask.any():
            return _zero(self.model)
        first = F.normalize(_masked_mean(output["graph_token_hidden"], left_mask), dim=-1, eps=1e-4)
        second = F.normalize(_masked_mean(output["graph_token_hidden"], right_mask), dim=-1, eps=1e-4)
        return 1.0 - (first * second).sum(dim=-1).mean()

    def _cross_modal_alignment(self, output: dict[str, Any]) -> Tensor:
        hidden = output["graph_token_hidden"]
        targets = output["graph_token_targets"].detach()
        mask = output["graph_token_mask"]
        if mask.sum() < 2:
            return _zero(self.model)
        return 1.0 - F.cosine_similarity(
            _masked_mean(hidden, mask),
            _masked_mean(targets, mask),
            dim=-1,
            eps=1e-4,
        ).mean()

    def _contrastive_alignment(self, output: dict[str, Any]) -> Tensor:
        embeddings = F.normalize(output["embedding"], dim=-1, eps=1e-4)
        if embeddings.shape[0] < 2:
            return _zero(self.model)
        logits = embeddings @ embeddings.t()
        labels = torch.arange(embeddings.shape[0], device=embeddings.device)
        return F.cross_entropy(logits, labels)

    def _adjacency_sparsity(self, output: dict[str, Any]) -> Tensor:
        adjacency = output["learned_adjacency"]
        return adjacency.abs().mean()

    def _adjacency_consistency(self, output: dict[str, Any]) -> Tensor:
        adjacency = output["learned_adjacency"]
        return (adjacency - adjacency.transpose(-2, -1)).abs().mean()

    def _spectral_reconstruction(self, output: dict[str, Any]) -> Tensor:
        return F.smooth_l1_loss(output["spectral_prediction"], output["spectral_target"].detach())

    def _weight(self, name: str) -> float:
        for group in self.weights.values():
            if isinstance(group, dict) and name in group:
                return float(group[name])
            if isinstance(group, dict) and f"{name}_weight" in group:
                return float(group[f"{name}_weight"])
        return float(self.weights.get(name, 1.0)) if isinstance(self.weights, dict) else 1.0


def _type_mask(output: dict[str, Any], token_type: str) -> Tensor:
    names = list(output.get("graph_token_types") or [])
    indices = [index for index, name in enumerate(names) if str(name) == token_type]
    base = output["graph_token_mask"]
    if not indices:
        return torch.zeros_like(base, dtype=torch.bool)
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:, indices] = base[:, indices]
    return mask


def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.shape[0]
    return 1


def _device(batch: dict[str, Any], model: nn.Module) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return next(model.parameters()).device


def _zero(model: nn.Module) -> Tensor:
    return next(model.parameters()).sum() * 0
