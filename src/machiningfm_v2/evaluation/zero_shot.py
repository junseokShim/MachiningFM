from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from machiningfm_v2.data.multirate_dataset import MultiRateMachiningDataset, multirate_collate
from machiningfm_v2.models.foundation_model import MachiningFMV2


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred.reshape(-1) - target.reshape(-1)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((target.reshape(-1) - float(np.mean(target))) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 1.0e-12 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def persistence_baseline(batch: dict[str, torch.Tensor], horizon: int) -> torch.Tensor | None:
    raw = batch.get("raw_waveform")
    if raw is None:
        return None
    source = raw[:, :3, :]
    return source[..., -1:].expand(source.shape[0], source.shape[1], horizon)


def evaluate_zero_shot(config: dict[str, Any], checkpoint_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", config.get("model", {}))
    model = MachiningFMV2(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    dataset = MultiRateMachiningDataset(config.get("data", {}))
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 1)), collate_fn=multirate_collate)
    by_horizon: dict[str, list[dict[str, float]]] = {}
    base_by_horizon: dict[str, list[dict[str, float]]] = {}
    max_batches = int(config.get("max_eval_batches", 8))
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_batch(batch, device)
            out = model(batch)
            for horizon, target in batch.get("targets", {}).items():
                pred = out["forecast"][str(horizon)]["mean"][..., : target.shape[-1]]
                physical_pred = denormalize_target(pred, batch)
                physical_target = denormalize_target(target, batch)
                metrics = regression_metrics(physical_pred.detach().cpu().numpy(), physical_target.detach().cpu().numpy())
                by_horizon.setdefault(str(horizon), []).append(metrics)
                baseline = persistence_baseline(batch, int(horizon))
                if baseline is not None:
                    physical_baseline = denormalize_target(baseline, batch)
                    base_by_horizon.setdefault(str(horizon), []).append(
                        regression_metrics(physical_baseline.detach().cpu().numpy(), physical_target.detach().cpu().numpy())
                    )
    report = {"checkpoint": str(checkpoint_path), "horizons": {}, "release_gate": {"passed": False, "reasons": []}}
    passed = True
    for horizon, rows in by_horizon.items():
        current = _mean_metrics(rows)
        baseline = _mean_metrics(base_by_horizon.get(horizon, []))
        improvement = None
        if baseline and baseline.get("rmse", 0.0) > 0:
            improvement = (baseline["rmse"] - current["rmse"]) / baseline["rmse"]
            if improvement < 0.15:
                passed = False
                report["release_gate"]["reasons"].append(f"horizon={horizon} rmse improvement below 15%")
        report["horizons"][horizon] = {"model": current, "persistence": baseline, "rmse_improvement": improvement}
    report["release_gate"]["passed"] = bool(passed and by_horizon)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def denormalize_target(value: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    center = batch.get("target_center")
    scale = batch.get("target_scale")
    if center is None or scale is None:
        return value
    channels = min(value.shape[1], center.shape[1])
    transform_id = batch.get("target_transform_id")
    transformed = value
    if transform_id is not None and bool((transform_id.reshape(-1)[0] > 0.5).item()):
        transformed = torch.sinh(value.clamp(-30.0, 30.0))
    result = transformed.clone()
    result[:, :channels] = transformed[:, :channels] * scale[:, :channels] + center[:, :channels]
    return result


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            out[key] = value
    return out
