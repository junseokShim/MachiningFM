from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


V2_FORMAT_VERSION = "machiningfm-v2-checkpoint-v1"


def save_v2_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": V2_FORMAT_VERSION,
            "model_config": config.get("model", config),
            "training_config": config,
            "state_dict": model.state_dict(),
            "metrics": metrics or {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        output,
    )
    return output


def load_v2_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("format") != V2_FORMAT_VERSION:
        raise ValueError(f"Not a MachiningFM v2 checkpoint: {path}")
    return checkpoint


def migrate_v1_to_v2(
    v1_checkpoint_path: str | Path,
    v2_model: torch.nn.Module,
    *,
    output_report_path: str | Path,
    name_map: dict[str, str | dict[str, str]] | None = None,
) -> dict[str, Any]:
    path = Path(v1_checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = v2_model.state_dict()
    loaded: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    unexpected: list[str] = []
    loaded_target_keys: set[str] = set()
    mapped = dict(name_map or {})
    new_state = dict(target_state)
    for source_key, source_value in source_state.items():
        mapping = mapped.get(source_key, source_key)
        if isinstance(mapping, dict):
            target_key = mapping["target"]
            adaptation = mapping.get("adaptation")
        else:
            target_key = mapping
            adaptation = None
        if target_key not in target_state:
            unexpected.append(source_key)
            continue
        source_shape = tuple(source_value.shape)
        target_shape = tuple(target_state[target_key].shape)
        if source_shape != target_shape:
            if adaptation == "subsample":
                adapted, coverage, scale = _subsample_tensor(source_value, target_state[target_key])
                new_state[target_key] = adapted
                loaded_target_keys.add(target_key)
                loaded.append(
                    {
                        "source_key": source_key,
                        "target_key": target_key,
                        "source_shape": list(source_shape),
                        "target_shape": list(target_shape),
                        "method": "explicit_semantic_subsample",
                        "target_coverage": coverage,
                        "fan_in_scale": scale,
                    }
                )
                continue
            mismatched.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_shape": list(source_shape),
                    "target_shape": list(target_shape),
                }
            )
            continue
        new_state[target_key] = source_value
        loaded_target_keys.add(target_key)
        loaded.append({"source_key": source_key, "target_key": target_key, "shape": list(source_shape), "method": "exact"})
    v2_model.load_state_dict(new_state, strict=True)
    missing = sorted(set(target_state) - loaded_target_keys)
    report = {
        "source_checkpoint": str(path),
        "source_model_version": checkpoint.get("model_version"),
        "format": V2_FORMAT_VERSION,
        "loaded_count": len(loaded),
        "adapted_count": sum(item["method"] != "exact" for item in loaded),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "mismatched_count": len(mismatched),
        "loaded": loaded,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "note": "Only parameters with explicit name mapping or identical key and identical shape were copied.",
    }
    output = Path(output_report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def graph_tokenized_v1_semantic_map(v2_model: torch.nn.Module) -> dict[str, dict[str, str]]:
    """Explicitly map semantically equivalent v1 blocks into a narrower v2."""
    mappings: dict[str, dict[str, str]] = {
        "nc_context_projection.1.weight": {"target": "encoders.nc.proj.0.weight", "adaptation": "subsample"},
        "nc_context_projection.1.bias": {"target": "encoders.nc.proj.0.bias", "adaptation": "subsample"},
        "adaptive_graph.variable_projection.weight": {"target": "encoders.cnc.channel_proj.0.weight", "adaptation": "subsample"},
        "adaptive_graph.variable_projection.bias": {"target": "encoders.cnc.channel_proj.0.bias", "adaptation": "subsample"},
        "adaptive_graph.output_norm.weight": {"target": "encoders.cnc.channel_proj.2.weight", "adaptation": "subsample"},
        "adaptive_graph.output_norm.bias": {"target": "encoders.cnc.channel_proj.2.bias", "adaptation": "subsample"},
        "decoder.norm.weight": {"target": "fusion.norm.weight", "adaptation": "subsample"},
        "decoder.norm.bias": {"target": "fusion.norm.bias", "adaptation": "subsample"},
    }
    layer_count = len(v2_model.fusion.encoder.layers)
    for index in range(layer_count):
        source = f"decoder.layers.{index}"
        target = f"fusion.encoder.layers.{index}"
        pairs = {
            f"{source}.attn_norm.weight": f"{target}.norm1.weight",
            f"{source}.attn_norm.bias": f"{target}.norm1.bias",
            f"{source}.attn.out_proj.weight": f"{target}.self_attn.out_proj.weight",
            f"{source}.ffn_norm.weight": f"{target}.norm2.weight",
            f"{source}.ffn_norm.bias": f"{target}.norm2.bias",
            f"{source}.ffn.up_proj.weight": f"{target}.linear1.weight",
            f"{source}.ffn.down_proj.weight": f"{target}.linear2.weight",
        }
        mappings.update({key: {"target": value, "adaptation": "subsample"} for key, value in pairs.items()})
    return mappings


def _subsample_tensor(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    if source.ndim != target.ndim:
        raise ValueError(f"Cannot subsample rank {source.ndim} into rank {target.ndim}")
    selected = source.detach()
    covered_shape = []
    for dimension, (source_size, target_size) in enumerate(zip(source.shape, target.shape)):
        covered = min(source_size, target_size)
        covered_shape.append(covered)
        if source_size > covered:
            indices = torch.linspace(0, source_size - 1, covered, device=source.device).round().long()
            selected = selected.index_select(dimension, indices)
    result = target.detach().clone()
    slices = tuple(slice(0, size) for size in covered_shape)
    scale = 1.0
    if source.ndim >= 2 and source.shape[1] > target.shape[1]:
        scale = math.sqrt(source.shape[1] / target.shape[1])
    result[slices] = selected.to(dtype=result.dtype, device=result.device) * scale
    coverage = math.prod(covered_shape) / max(1, math.prod(target.shape))
    return result, float(coverage), float(scale)
