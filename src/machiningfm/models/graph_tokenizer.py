from __future__ import annotations

import re
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_GRAPH_TOKEN_TYPES = {
    "spindle_dynamics": {
        "include_patterns": ["spindle", "vibration", "rpm", "load", "chatter"],
        "views": ["high_rate_raw_timeseries", "high_rate_fft", "high_rate_time_frequency", "cnc_low_rate_timeseries"],
    },
    "feed_drive": {
        "include_patterns": ["axis", "position", "feed", "servo", "current", "load"],
        "views": ["cnc_low_rate_timeseries"],
    },
    "cutting_condition": {
        "include_patterns": ["feedrate", "spindle_speed", "tool", "depth_of_cut", "width_of_cut", "material"],
        "views": ["process_condition", "material_context"],
    },
    "high_frequency_vibration": {
        "include_patterns": ["vibration", "acceler", "acoustic", "fft", "spectrum"],
        "views": ["high_rate_raw_timeseries", "high_rate_fft", "high_rate_time_frequency"],
    },
    "cnc_operation_state": {
        "include_patterns": ["gcode", "focas_current_gcode", "operation_mode", "tool_num", "tool_call_count"],
        "views": ["cnc_low_rate_timeseries", "process_condition"],
    },
    "material_process_context": {
        "include_patterns": ["material", "material_family", "material_confidence", "dataset_group", "machine_id"],
        "views": ["material_context", "global_metadata"],
    },
}


class GraphTokenizationLayer(nn.Module):
    """Pools adaptive-graph variable embeddings into graph spatial tokens."""

    def __init__(self, config: dict[str, Any], d_model: int) -> None:
        super().__init__()
        self.config = config or {}
        raw_subgraphs = self.config.get("subgraphs") or DEFAULT_GRAPH_TOKEN_TYPES
        self.subgraph_specs = {str(name): dict(value or {}) for name, value in raw_subgraphs.items()}
        self.subgraph_names = list(self.subgraph_specs)
        if not self.subgraph_names:
            self.subgraph_specs = dict(DEFAULT_GRAPH_TOKEN_TYPES)
            self.subgraph_names = list(self.subgraph_specs)
        graph_construction = self.config.get("graph_construction", {})
        self.dynamic_threshold = float(graph_construction.get("dynamic_adjacency_threshold", 0.05))
        self.dynamic_top_k = int(graph_construction.get("dynamic_adjacency_top_k", 16))
        self.max_nodes_per_subgraph = int(graph_construction.get("max_nodes_per_subgraph", 32))
        self.min_nodes_per_subgraph = int(graph_construction.get("min_nodes_per_subgraph", 1))
        self.subgraph_embedding = nn.Embedding(len(self.subgraph_names), d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)

    @property
    def graph_token_count(self) -> int:
        return len(self.subgraph_names)

    def forward(
        self,
        variable_embeddings: Tensor,
        adjacency: Tensor,
        variable_mask: Tensor,
        variable_metadata: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        batch, variable_count, d_model = variable_embeddings.shape
        membership = variable_embeddings.new_zeros(batch, len(self.subgraph_names), variable_count)
        for batch_index in range(batch):
            metadata = variable_metadata[batch_index] if batch_index < len(variable_metadata) else []
            for subgraph_index, name in enumerate(self.subgraph_names):
                seeds = _metadata_membership(metadata, self.subgraph_specs[name])
                seeds = seeds[:variable_count]
                if len(seeds) < variable_count:
                    seeds.extend([False] * (variable_count - len(seeds)))
                seed_tensor = torch.tensor(seeds, dtype=torch.bool, device=variable_embeddings.device)
                expanded = _expand_with_adjacency(
                    seed_tensor,
                    adjacency[batch_index],
                    self.dynamic_top_k,
                    self.dynamic_threshold,
                    self.max_nodes_per_subgraph,
                )
                membership[batch_index, subgraph_index] = expanded.to(membership.dtype)
            if not membership[batch_index].bool().any():
                membership[batch_index] = _fallback_membership(
                    metadata,
                    self.subgraph_names,
                    variable_count,
                    variable_embeddings.device,
                    membership.dtype,
                )
        membership = membership * variable_mask[:, None, :].to(membership.dtype)
        token_mask = membership.sum(dim=-1) >= self.min_nodes_per_subgraph
        degree = adjacency.sum(dim=-1)
        attention_scores = self.pool_score(variable_embeddings).squeeze(-1)
        weights = membership * (degree[:, None, :] + 1.0) * torch.softmax(attention_scores, dim=-1)[:, None, :]
        pooled = torch.matmul(weights, variable_embeddings) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        subgraph_ids = torch.arange(len(self.subgraph_names), device=variable_embeddings.device)
        pooled = pooled + self.subgraph_embedding(subgraph_ids)[None, :, :]
        pooled = self.output_norm(pooled)
        pooled = torch.where(token_mask.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return {
            "graph_tokens": pooled,
            "graph_token_mask": token_mask,
            "membership": membership,
            "graph_token_names": self.subgraph_names,
            "graph_token_types": self.subgraph_names,
        }


def _metadata_membership(metadata: list[dict[str, Any]], spec: dict[str, Any]) -> list[bool]:
    patterns = [str(value).lower() for value in spec.get("include_patterns", [])]
    views = {str(value).lower() for value in spec.get("views", [])}
    result = []
    for item in metadata:
        name = str(item.get("name", "")).lower()
        view = str(item.get("view", "")).lower()
        text = " ".join(
            [
                name,
                view,
                str(item.get("quantity", "")).lower(),
                str(item.get("material", "")).lower(),
                str(item.get("dataset_group", "")).lower(),
            ]
        )
        pattern_match = any(pattern in text for pattern in patterns)
        view_match = view in views if views else False
        result.append(pattern_match or view_match)
    return result


def _expand_with_adjacency(
    seeds: Tensor,
    adjacency: Tensor,
    top_k: int,
    threshold: float,
    max_nodes: int,
) -> Tensor:
    if not seeds.any():
        return seeds
    scores = adjacency[seeds].max(dim=0).values
    selected = scores >= threshold
    selected = selected | seeds
    if top_k > 0 and selected.sum() > max_nodes:
        count = min(max_nodes, max(top_k, int(seeds.sum().item())))
        _, indices = torch.topk(scores, k=min(count, scores.numel()))
        limited = torch.zeros_like(selected)
        limited[indices] = True
        selected = limited | seeds
    return selected


def _fallback_membership(
    metadata: list[dict[str, Any]],
    names: list[str],
    variable_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    membership = torch.zeros(len(names), variable_count, device=device, dtype=dtype)
    name_to_index = {name: index for index, name in enumerate(names)}
    for variable_index, item in enumerate(metadata[:variable_count]):
        fallback = _fallback_name(str(item.get("view", "")), str(item.get("name", "")))
        target = name_to_index.get(fallback, 0)
        membership[target, variable_index] = 1.0
    if variable_count and membership.sum() == 0:
        membership[0, :] = 1.0
    return membership


def _fallback_name(view: str, name: str) -> str:
    text = f"{view} {name}".lower()
    if "fft" in text or "frequency" in text or "vibration" in text or "acceler" in text:
        return "high_frequency_vibration"
    if "cnc" in text or re.search(r"\b(feed|rpm|axis|servo|gcode)\b", text):
        return "cnc_operation_state"
    if "material" in text:
        return "material_process_context"
    return "spindle_dynamics"
