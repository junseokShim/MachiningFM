from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from machiningfm.models.machiningfm import MachiningFM
from machiningfm.tasks.multitask_pretraining import MultitaskPretraining
from machiningfm.utils.config import load_config
from machiningfm.utils.paths import project_root
from machiningfm.utils.seed import seed_everything
from .checkpointing import save_checkpoint
from .optim import build_optimizer

LOGGER = logging.getLogger(__name__)


def run_pretrain_smoke(config_path: str | Path) -> dict[str, float]:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    model_cfg = cfg.get("model", cfg)
    model = MachiningFM(model_cfg)
    objective = MultitaskPretraining(
        model,
        patch_size=int(model_cfg.get("patch_size", 16)),
        horizon=int(model_cfg.get("horizon", 16)),
        weights=cfg.get("loss_weights"),
    )
    optimizer = build_optimizer(model, float(cfg.get("learning_rate", 1e-3)))
    steps = int(cfg.get("smoke_steps", 2))
    metrics: dict[str, float] = {}
    model.train()
    for step in range(steps):
        batch = synthetic_batch(cfg)
        optimizer.zero_grad(set_to_none=True)
        loss, losses = objective(batch)
        loss.backward()
        optimizer.step()
        metrics = {"loss": float(loss.detach()), **{name: float(value.detach()) for name, value in losses.items()}}
        LOGGER.info("Pretrain smoke step %s: %s", step + 1, metrics)
    checkpoint = project_root() / cfg.get("checkpoint_path", "outputs/checkpoints/machiningfm_small.pt")
    save_checkpoint(checkpoint, model, model_cfg, metrics, cfg.get("model_version", "machiningfm-small-smoke"))
    return metrics


def run_downstream_smoke(config_path: str | Path) -> dict[str, float]:
    cfg = load_config(config_path)
    seed_everything(int(cfg.get("seed", 42)))
    model_cfg = cfg.get("model", cfg)
    task = cfg.get("task", "toolwear_regression")
    model = MachiningFM(model_cfg)
    optimizer = build_optimizer(model, float(cfg.get("learning_rate", 1e-3)))
    steps = int(cfg.get("smoke_steps", 2))
    metrics: dict[str, float] = {}
    model.train()
    for step in range(steps):
        batch = synthetic_batch(cfg)
        label = batch["sensor_series"].abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, task=task)["prediction"]
        loss = F.mse_loss(prediction, label)
        loss.backward()
        optimizer.step()
        metrics = {"loss": float(loss.detach())}
        LOGGER.info("Downstream smoke step %s: %s", step + 1, metrics)
    checkpoint = project_root() / cfg.get("checkpoint_path", "outputs/checkpoints/machiningfm_toolwear_smoke.pt")
    save_checkpoint(checkpoint, model, model_cfg, metrics, cfg.get("model_version", "machiningfm-toolwear-smoke"))
    return metrics


def synthetic_batch(config: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    model_cfg = config.get("model", config)
    batch = int(config.get("batch_size", 2))
    channels = int(config.get("channels", 4))
    length = int(config.get("sequence_length", 128))
    condition_count = int(config.get("condition_count", 5))
    series = torch.randn(batch, channels, length)
    sensor_mask = torch.ones(batch, channels, dtype=torch.bool)
    if channels > 1:
        sensor_mask[0, -1] = False
    condition = torch.randn(batch, condition_count)
    condition_mask = torch.ones(batch, condition_count, dtype=torch.bool)
    condition_mask[0, -1] = False
    return {
        "sensor_series": series,
        "sensor_mask": sensor_mask,
        "condition": condition,
        "condition_mask": condition_mask,
        "image": None,
        "image_mask": None,
        "frequency": None,
        "frequency_mask": None,
    }
