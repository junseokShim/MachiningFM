from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from machiningfm.data.pretraining_dataset import RealPretrainingDataset, pretraining_collate
from machiningfm.models.machiningfm import MachiningFM
from machiningfm.tasks.multitask_pretraining import MultitaskPretraining
from machiningfm.utils.config import load_config
from machiningfm.utils.device import get_device
from machiningfm.utils.io import ensure_parent, write_json
from machiningfm.utils.paths import project_root, resolve_path
from machiningfm.utils.seed import seed_everything
from .checkpointing import load_checkpoint, save_training_checkpoint
from .optim import build_optimizer
from .scheduler import build_scheduler

LOGGER = logging.getLogger(__name__)


def build_real_pretraining_dataset(
    config: dict[str, Any],
    latent_context_path: str | Path | None | object = ...,
) -> RealPretrainingDataset:
    data = config.get("data", {})
    if config.get("model_config_path"):
        model = load_config(resolve_path(config["model_config_path"]))
        model.update(config.get("model", {}))
    else:
        model = config.get("model", {})
    manifest_path = resolve_path(data.get("manifest_path", "reports/manifest.parquet"))
    latent_path_value = data.get("latent_context_path") if latent_context_path is ... else latent_context_path
    latent_path = resolve_path(latent_path_value) if latent_path_value else None
    return RealPretrainingDataset(
        manifest_path=manifest_path,
        sequence_length=int(data.get("sequence_length", 512)),
        max_channels=int(data.get("max_channels", model.get("max_channels", 16))),
        channel_vocab_size=int(model.get("channel_vocab_size", model.get("max_channels", 128))),
        text_vocab_size=int(model.get("text_vocab_size", 8192)),
        max_text_tokens=int(model.get("max_text_tokens", 64)),
        latent_context_path=latent_path,
        nc_context_path=resolve_path(data.get("nc_context_path")) if data.get("nc_context_path") else None,
        nc_context_dim=int(model.get("nc_context_dim", data.get("nc_context_dim", 128))),
        require_nc_context=bool(data.get("require_nc_context", False)),
        windows_per_file=int(data.get("windows_per_file", 1)),
        include_datasets=data.get("include_datasets"),
        exclude_datasets=data.get("exclude_datasets"),
        extensions=data.get("extensions"),
        max_files=data.get("max_files"),
        seed=int(config.get("seed", 42)),
        max_read_attempts=int(data.get("max_read_attempts", 12)),
        generate_frequency=bool(data.get("generate_frequency", False)),
        frequency_transforms=data.get("frequency_transforms"),
        frequency_length=int(data.get("frequency_length", 512)),
        frequency_bands=data.get("frequency_bands"),
        augmentation=config.get("augmentation"),
        image_size=int(data.get("image_size", 128)),
        generate_virtual_vibration=bool(data.get("generate_virtual_vibration", False)),
        virtual_vibration_sampling_rate=float(data.get("virtual_vibration_sampling_rate", 1600.0)),
        virtual_vibration_mode=str(data.get("virtual_vibration_mode", "if_missing")),
        virtual_vibration_default_spindle_rpm=float(data.get("virtual_vibration_default_spindle_rpm", 6000.0)),
    )


def preflight_real_pretraining(config_path: str | Path, samples: int = 4) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = build_real_pretraining_dataset(config)
    inspected = []
    for index in range(min(samples, len(dataset))):
        sample = dataset[index]
        shapes = {
            name: list(value.shape)
            for name in ("sensor_series", "frequency", "image")
            if isinstance((value := sample.get(name)), torch.Tensor)
        }
        inspected.append(
            {
                "shapes": shapes,
                "finite": {
                    name: bool(torch.isfinite(value).all())
                    for name in ("sensor_series", "frequency", "image")
                    if isinstance((value := sample.get(name)), torch.Tensor)
                },
                "metadata": sample["metadata"],
            }
        )
    return {"dataset": dataset.summary(), "samples": inspected}


def run_full_pretrain(
    config_path: str | Path,
    resume_checkpoint: str | Path | None = None,
    max_steps_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device_name = device_override or get_device(str(config.get("device", "auto")))
    device = torch.device(device_name)
    if device.type == "cpu":
        LOGGER.warning("Full pretraining is running on CPU. Use a CUDA-enabled PyTorch environment for long runs.")

    dataset = build_real_pretraining_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=pretraining_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 0)) > 0,
        drop_last=bool(config.get("drop_last", False)),
    )
    model_config = config.get("model", config)
    model = MachiningFM(model_config).to(device)
    objective = MultitaskPretraining(
        model,
        patch_size=int(model_config.get("patch_size", 16)),
        horizon=int(model_config.get("horizon", 32)),
        image_patch_size=int(model_config.get("image_patch_size", 16)),
        weights=config.get("loss_weights"),
    )
    optimizer = build_optimizer(
        model,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        foreach=config.get("optimizer_foreach"),
        fused=config.get("optimizer_fused"),
    )
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    epochs = max(1, int(config.get("epochs", 1)))
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulation))
    configured_max_steps = config.get("max_steps")
    max_steps = max_steps_override if max_steps_override is not None else configured_max_steps
    total_steps = int(max_steps) if max_steps else epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps)
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    resume_batches_to_skip = 0
    if resume_checkpoint:
        checkpoint = load_checkpoint(resume_checkpoint, map_location=device_name)
        _validate_resume_config(checkpoint.get("training_config", {}), config)
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = global_step // steps_per_epoch
        resume_steps_in_epoch = global_step % steps_per_epoch
        resume_batches_to_skip = resume_steps_in_epoch * accumulation
        LOGGER.info(
            "Resumed %s at epoch=%s step=%s; skipping %s completed batches in the resumed epoch",
            resume_checkpoint,
            start_epoch,
            global_step,
            resume_batches_to_skip,
        )

    checkpoint_dir = resolve_path(config.get("checkpoint_dir", "outputs/checkpoints/full_pretrain"))
    log_path = resolve_path(config.get("metrics_log_path", "outputs/training_logs/full_pretrain_metrics.jsonl"))
    ensure_parent(log_path)
    checkpoint_every = max(1, int(config.get("checkpoint_every_steps", 500)))
    log_every = max(1, int(config.get("log_every_steps", 10)))
    console_log_interval = str(config.get("console_log_interval", "step")).lower()
    if console_log_interval not in {"step", "epoch", "none"}:
        raise ValueError("console_log_interval must be one of: step, epoch, none")
    metrics_log_interval = str(config.get("metrics_log_interval", "step")).lower()
    if metrics_log_interval not in {"step", "epoch", "both", "none"}:
        raise ValueError("metrics_log_interval must be one of: step, epoch, both, none")
    show_progress_bar = bool(config.get("show_progress_bar", True))
    clip_norm = float(config.get("gradient_clip_norm", 1.0))
    model_version = str(config.get("model_version", "machiningfm-small-full-pretrain"))
    early_stopping = _build_early_stopping(config.get("early_stopping", {}))
    optimizer.zero_grad(set_to_none=True)
    last_metrics: dict[str, Any] = {}
    last_epoch_metrics: dict[str, Any] = {}
    best_checkpoint: Path | None = None
    started = time.time()
    progress = tqdm(
        total=total_steps,
        initial=min(global_step, total_steps),
        desc="Global steps",
        unit="step",
        dynamic_ncols=True,
        disable=not show_progress_bar,
    )

    try:
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            model.train()
            epoch_started = time.time()
            epoch_start_step = global_step
            epoch_metric_sums: dict[str, float] = {}
            epoch_update_count = 0
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < resume_batches_to_skip:
                    continue
                batch = _move_batch(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, losses = objective(batch)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    LOGGER.warning("Skipping non-finite loss at epoch=%s batch=%s", epoch, batch_index)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(scaled_loss).backward()
                update_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if not update_step:
                    continue
                scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                last_metrics = {
                    "record_type": "step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": gradient_norm,
                    "elapsed_seconds": time.time() - started,
                    **{name: float(value.detach()) for name, value in losses.items()},
                }
                if metrics_log_interval in {"step", "both"}:
                    _append_jsonl(log_path, last_metrics)
                for name in ("loss", "gradient_norm", *losses.keys()):
                    epoch_metric_sums[name] = epoch_metric_sums.get(name, 0.0) + float(last_metrics[name])
                epoch_update_count += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{last_metrics['loss']:.4f}",
                    lr=f"{last_metrics['learning_rate']:.2e}",
                    refresh=False,
                )
                if console_log_interval == "step" and (global_step % log_every == 0 or global_step == 1):
                    progress.write(f"Step {global_step}: {last_metrics}")
                if global_step % checkpoint_every == 0:
                    save_training_checkpoint(
                        checkpoint_dir / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        last_metrics,
                        model_version,
                        epoch,
                        global_step,
                    )
                if global_step >= total_steps:
                    break
            if epoch_update_count:
                last_epoch_metrics = {
                    "record_type": "epoch",
                    "epoch": epoch,
                    "epoch_display": epoch + 1,
                    "epochs_configured": epochs,
                    "global_step": global_step,
                    "optimizer_steps": global_step - epoch_start_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_seconds": time.time() - epoch_started,
                    **{
                        f"mean_{name}": total / epoch_update_count
                        for name, total in epoch_metric_sums.items()
                    },
                }
                if early_stopping["enabled"]:
                    early_status = _update_early_stopping(early_stopping, last_epoch_metrics)
                    last_epoch_metrics["early_stopping"] = early_status
                    if early_status["improved"]:
                        best_checkpoint = save_training_checkpoint(
                            checkpoint_dir / "machiningfm_full_pretrain_best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            config,
                            last_epoch_metrics,
                            model_version,
                            epoch,
                            global_step,
                            update_latest=False,
                        )
                        early_stopping["best_checkpoint"] = str(best_checkpoint)
                if metrics_log_interval in {"epoch", "both"}:
                    _append_jsonl(log_path, last_epoch_metrics)
                if console_log_interval == "epoch":
                    progress.write(_format_epoch_summary(last_epoch_metrics))
                if early_stopping.get("stopped"):
                    progress.write(str(early_stopping["reason"]))
                    break
            resume_batches_to_skip = 0
            if global_step >= total_steps or early_stopping.get("stopped"):
                break
    finally:
        progress.close()

    final_checkpoint = save_training_checkpoint(
        checkpoint_dir / f"step_{global_step:08d}_final.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        last_metrics,
        model_version,
        min(epochs - 1, epoch if "epoch" in locals() else 0),
        global_step,
    )
    summary = {
        "model_version": model_version,
        "device": device_name,
        "dataset": dataset.summary(),
        "global_step": global_step,
        "epochs_configured": epochs,
        "stopped_early": bool(early_stopping.get("stopped", False)),
        "early_stopping": _early_stopping_summary(early_stopping),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else early_stopping.get("best_checkpoint"),
        "final_checkpoint": str(final_checkpoint),
        "last_metrics": last_metrics,
        "last_epoch_metrics": last_epoch_metrics,
    }
    write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _build_early_stopping(raw_config: Any) -> dict[str, Any]:
    if isinstance(raw_config, bool):
        raw_config = {"enabled": raw_config}
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("early_stopping must be a boolean or config object")

    enabled = bool(raw_config.get("enabled", False))
    mode = str(raw_config.get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be 'min' or 'max'")
    patience = max(1, int(raw_config.get("patience", 5)))
    return {
        "enabled": enabled,
        "monitor": str(raw_config.get("monitor", "mean_loss")),
        "mode": mode,
        "patience": patience,
        "min_delta": max(0.0, float(raw_config.get("min_delta", 0.0))),
        "warmup_epochs": max(0, int(raw_config.get("warmup_epochs", 0))),
        "best_value": None,
        "best_epoch": None,
        "best_step": None,
        "best_checkpoint": None,
        "bad_epochs": 0,
        "stopped": False,
        "reason": None,
    }


def _update_early_stopping(state: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    monitor = state["monitor"]
    if monitor not in metrics:
        available = sorted(name for name, value in metrics.items() if isinstance(value, (int, float)))
        raise ValueError(
            f"Early stopping monitor {monitor!r} was not found in epoch metrics. "
            f"Available numeric metrics: {available}"
        )
    value = float(metrics[monitor])
    if not math.isfinite(value):
        improved = False
    else:
        improved = _is_early_stopping_improvement(
            value,
            state.get("best_value"),
            state["mode"],
            state["min_delta"],
        )
    if improved:
        state["best_value"] = value
        state["best_epoch"] = int(metrics["epoch"])
        state["best_step"] = int(metrics["global_step"])
        state["bad_epochs"] = 0
    elif int(metrics["epoch_display"]) > int(state["warmup_epochs"]):
        state["bad_epochs"] += 1

    if state["bad_epochs"] >= state["patience"]:
        state["stopped"] = True
        state["reason"] = (
            "Early stopping triggered: "
            f"{monitor} did not improve for {state['bad_epochs']} epoch(s) "
            f"(best={state['best_value']}, mode={state['mode']}, min_delta={state['min_delta']})."
        )

    return {
        "enabled": True,
        "monitor": monitor,
        "mode": state["mode"],
        "value": value,
        "best_value": state["best_value"],
        "best_epoch": state["best_epoch"],
        "best_step": state["best_step"],
        "bad_epochs": state["bad_epochs"],
        "patience": state["patience"],
        "min_delta": state["min_delta"],
        "warmup_epochs": state["warmup_epochs"],
        "improved": improved,
        "stopped": state["stopped"],
    }


def _is_early_stopping_improvement(
    value: float,
    best_value: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < float(best_value) - min_delta
    return value > float(best_value) + min_delta


def _early_stopping_summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(state.get("enabled", False)),
        "monitor": state.get("monitor"),
        "mode": state.get("mode"),
        "patience": state.get("patience"),
        "min_delta": state.get("min_delta"),
        "warmup_epochs": state.get("warmup_epochs"),
        "best_value": state.get("best_value"),
        "best_epoch": state.get("best_epoch"),
        "best_step": state.get("best_step"),
        "best_checkpoint": state.get("best_checkpoint"),
        "bad_epochs": state.get("bad_epochs", 0),
        "stopped": bool(state.get("stopped", False)),
        "reason": state.get("reason"),
    }


def _validate_resume_config(checkpoint_config: dict[str, Any], current_config: dict[str, Any]) -> None:
    if not checkpoint_config:
        return
    if checkpoint_config.get("model") != current_config.get("model"):
        raise ValueError("Cannot resume: model architecture differs from the checkpoint")
    if checkpoint_config.get("data") != current_config.get("data"):
        raise ValueError("Cannot resume: data configuration differs from the checkpoint")
    old_effective_batch = int(checkpoint_config.get("batch_size", 1)) * int(
        checkpoint_config.get("gradient_accumulation_steps", 1)
    )
    new_effective_batch = int(current_config.get("batch_size", 1)) * int(
        current_config.get("gradient_accumulation_steps", 1)
    )
    if old_effective_batch != new_effective_batch:
        raise ValueError(
            "Cannot resume safely: effective batch size differs "
            f"({old_effective_batch} in checkpoint, {new_effective_batch} in current config)"
        )
    for name in ("epochs", "learning_rate", "weight_decay"):
        if checkpoint_config.get(name) != current_config.get(name):
            raise ValueError(
                f"Cannot resume safely: {name} differs "
                f"({checkpoint_config.get(name)!r} in checkpoint, {current_config.get(name)!r} in current config)"
            )


def _format_epoch_summary(metrics: dict[str, Any]) -> str:
    summary = (
        f"\nEpoch {metrics['epoch_display']}/{metrics['epochs_configured']} complete\n"
        f"  global_step:       {metrics['global_step']}\n"
        f"  optimizer_steps:   {metrics['optimizer_steps']}\n"
        f"  mean_loss:         {metrics.get('mean_loss', float('nan')):.6f}\n"
        f"  mean_masked:       {metrics.get('mean_masked_signal', float('nan')):.6f}\n"
        f"  mean_forecasting:  {metrics.get('mean_forecasting', float('nan')):.6f}\n"
        f"  mean_cross_channel:{metrics.get('mean_cross_channel', float('nan')):.6f}\n"
        f"  mean_frequency:    {metrics.get('mean_frequency_reconstruction', float('nan')):.6f}\n"
        f"  mean_image:        {metrics.get('mean_image_reconstruction', float('nan')):.6f}\n"
        f"  mean_grad_norm:    {metrics.get('mean_gradient_norm', float('nan')):.6f}\n"
        f"  learning_rate:     {metrics['learning_rate']:.3e}\n"
        f"  epoch_seconds:     {metrics['epoch_seconds']:.1f}"
    )
    early = metrics.get("early_stopping")
    if isinstance(early, dict) and early.get("enabled"):
        summary += (
            f"\n  early_stop:        {early['monitor']}={early['value']:.6f}, "
            f"best={early['best_value']:.6f}, "
            f"bad_epochs={early['bad_epochs']}/{early['patience']}"
        )
    return summary
