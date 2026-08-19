"""
Graph Tokenization Layer.
Reconstructed from checkpoint weights.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor


class GraphTokenizationLayer(nn.Module):
    def __init__(self, config: dict, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        subgraphs = config.get("subgraphs", {})
        self._subgraph_names = list(subgraphs.keys())
        n_types = max(len(self._subgraph_names), 8)
        self.subgraph_embedding = nn.Embedding(n_types, d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.output_norm = nn.LayerNorm(d_model)
        self._patterns = {
            name: (i, cfg.get("include_patterns", []))
            for i, (name, cfg) in enumerate(subgraphs.items())
        }

    def _assign(self, meta: dict) -> int:
        name = str(meta.get("name", "")).lower()
        for sg_name, (idx, pats) in self._patterns.items():
            for pat in pats:
                if pat.lower() in name:
                    return idx
        return 0

    def forward(self, variable_embeddings, learned_adjacency, variable_mask, variable_metadata):
        B, N, D = variable_embeddings.shape
        device = variable_embeddings.device
        batch_tokens, batch_masks, batch_names, batch_types, batch_member = [], [], [], [], []

        for b in range(B):
            meta_b = variable_metadata[b] if variable_metadata else [{}] * N
            valid_b = variable_mask[b].bool()
            assignments = [self._assign(m) for m in meta_b]
            unique_sgs = sorted({a for a, v in zip(assignments, valid_b.tolist()) if v}) or [0]

            tokens_b, types_b, names_b, member_b = [], [], [], []
            for sg_idx in unique_sgs:
                midx = [i for i, (a, v) in enumerate(zip(assignments, valid_b.tolist())) if a == sg_idx and v]
                if not midx:
                    continue
                members = variable_embeddings[b, midx]
                w = torch.softmax(self.pool_score(members), dim=0)
                tok = (members * w).sum(0) + self.subgraph_embedding(torch.tensor(sg_idx, device=device))
                mem_row = torch.zeros(N, device=device)
                for j, idx in enumerate(midx):
                    mem_row[idx] = w[j, 0].detach()
                tokens_b.append(tok); types_b.append(sg_idx)
                names_b.append(self._subgraph_names[sg_idx] if sg_idx < len(self._subgraph_names) else f"subgraph_{sg_idx}")
                member_b.append(mem_row)

            if not tokens_b:
                tok = variable_embeddings[b].mean(0) + self.subgraph_embedding.weight[0]
                tokens_b = [tok]; types_b = [0]; names_b = ["default"]
                member_b = [torch.ones(N, device=device) / max(N, 1)]

            batch_tokens.append(torch.stack(tokens_b))
            batch_masks.append(torch.ones(len(tokens_b), dtype=torch.bool, device=device))
            batch_names.append(names_b); batch_types.append(types_b)
            batch_member.append(torch.stack(member_b))

        max_T = max(t.shape[0] for t in batch_tokens)
        padded_tokens = torch.zeros(B, max_T, D, device=device, dtype=variable_embeddings.dtype)
        padded_masks  = torch.zeros(B, max_T, dtype=torch.bool, device=device)
        padded_member = torch.zeros(B, max_T, N, device=device, dtype=variable_embeddings.dtype)

        for b, (tok, msk, mem) in enumerate(zip(batch_tokens, batch_masks, batch_member)):
            t = tok.shape[0]
            padded_tokens[b, :t] = tok; padded_masks[b, :t] = msk; padded_member[b, :t] = mem

        padded_names = [n + [""] * (max_T - len(n)) for n in batch_names]
        padded_types = [tp + [0] * (max_T - len(tp)) for tp in batch_types]

        return {
            "graph_tokens":      self.output_norm(padded_tokens),
            "graph_token_mask":  padded_masks,
            "graph_token_names": padded_names,
            "graph_token_types": padded_types,
            "membership":        padded_member,
        }
