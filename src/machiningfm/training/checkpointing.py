from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, float],
    model_version: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": model_version,
            "model_config": config,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        output,
    )
    latest = output.parent / "machiningfm_latest.pt"
    if latest != output:
        shutil.copy2(output, latest)
    return output


def load_checkpoint(path: str | Path, map_location: str = "cpu", *, mmap: bool = True) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False, mmap=mmap)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    metrics: dict[str, Any],
    model_version: str,
    epoch: int,
    global_step: int,
    update_latest: bool = True,
    include_optimizer_state: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": model_version,
        "model_config": config.get("model", config),
        "training_config": config,
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "epoch": epoch,
        "global_step": global_step,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if include_optimizer_state:
        payload.update(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
            }
        )
    torch.save(payload, output)
    latest = output.parent / "machiningfm_full_pretrain_latest.pt"
    if update_latest and latest != output:
        shutil.copy2(output, latest)
    return output
