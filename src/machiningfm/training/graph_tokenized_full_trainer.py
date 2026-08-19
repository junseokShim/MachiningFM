from __future__ import annotations

import json
import logging
import math
import time
import gc
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from machiningfm.data.pretraining_dataset import pretraining_collate
from machiningfm.models.graph_tokenized_machiningfm import (
    GraphTokenizedStemGNNDecoderOnlyMachiningFM,
    estimate_graph_tokenized_decoder_only_parameters,
)
from machiningfm.tasks.graph_token_pretraining import GraphTokenizedDecoderOnlyPretraining
from machiningfm.training.full_trainer import build_real_pretraining_dataset
from machiningfm.utils.config import load_config
from machiningfm.utils.device import get_device
from machiningfm.utils.io import ensure_parent, write_json
from machiningfm.utils.paths import resolve_path
from machiningfm.utils.seed import seed_everything
from .checkpointing import load_checkpoint, save_training_checkpoint
from .optim import build_optimizer
from .scheduler import build_scheduler

LOGGER = logging.getLogger(__name__)


def preflight_graph_tokenized_pretraining(config_path: str | Path, samples: int = 4) -> dict[str, Any]:
    config = _load_training_config(config_path)
    dataset = build_real_pretraining_dataset(config)
    model_config = _model_config(config)
    inspected = []
    for index in range(min(samples, len(dataset))):
        sample = dataset[index]
        shapes = {
            name: list(value.shape)
            for name in ("sensor_series", "frequency", "image")
            if isinstance((value := sample.get(name)), torch.Tensor)
        }
        inspected.append({"shapes": shapes, "metadata": sample.get("metadata", {})})
    return {
        "dataset": dataset.summary(),
        "samples": inspected,
        "parameter_estimate": estimate_graph_tokenized_decoder_only_parameters(model_config),
        "loads_previous_checkpoint": bool(config.get("resume_checkpoint")),
        "architecture": "machiningfm_graph_tokenized_stemgnn_decoder_only",
    }


def run_graph_tokenized_full_pretrain(
    config_path: str | Path,
    max_steps_override: int | None = None,
    device_override: str | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_training_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device_name = device_override or get_device(str(config.get("device", "auto")))
    device = torch.device(device_name)
    model_config = _model_config(config)
    parameter_estimate = estimate_graph_tokenized_decoder_only_parameters(model_config)
    if device.type == "cpu" and parameter_estimate["total"] >= 1_000_000_000:
        raise RuntimeError(
            "The configured graph-tokenized decoder-only model is in the billion-parameter class. "
            "CPU training is not practical; use a CUDA/bf16 environment with enough memory."
        )
    parameter_dtype_name = str(model_config.get("parameter_dtype", config.get("parameter_dtype", "fp32"))).lower()
    if device.type == "cuda":
        required_gib = _estimated_training_memory_gib(
            parameter_estimate["total"],
            parameter_dtype=parameter_dtype_name,
        )
        available_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        cuda_memory_fraction = float(config.get("cuda_memory_safety_fraction", 0.98))
        if required_gib > available_gib * cuda_memory_fraction:
            raise RuntimeError(
                "Insufficient CUDA memory for the configured large graph-tokenized decoder-only model: "
                f"estimated training memory is about {required_gib:.1f} GiB, "
                f"but {torch.cuda.get_device_name(device)} reports {available_gib:.1f} GiB. "
                f"The current preflight limit is {cuda_memory_fraction:.0%} of available CUDA memory. "
                "Use multi-GPU sharding/FSDP/ZeRO or a larger accelerator; the large config was not reduced."
            )

    dataset = build_real_pretraining_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=pretraining_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(config.get("persistent_workers", int(config.get("num_workers", 0)) > 0)),
        drop_last=bool(config.get("drop_last", False)),
    )
    model = GraphTokenizedStemGNNDecoderOnlyMachiningFM(model_config)
    parameter_dtype = _torch_dtype(parameter_dtype_name)
    if parameter_dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=parameter_dtype)
    resume_info = _load_resume_checkpoint(
        model,
        resume_checkpoint or config.get("resume_checkpoint"),
        strict=bool(config.get("strict_checkpoint_load", True)),
        include_training_state=bool(config.get("resume_optimizer", False)),
    )
    objective = GraphTokenizedDecoderOnlyPretraining(
        model,
        weights=config.get("loss"),
        mask_ratio=float(config.get("mask_ratio", 0.25)),
    )
    optimizer = build_optimizer(
        model,
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        foreach=config.get("optimizer_foreach"),
        fused=config.get("optimizer_fused"),
    )
    accumulation = max(1, int(config.get("gradient_accumulation_steps", 1)))
    epochs = max(1, int(config.get("epochs", 1)))
    steps_per_epoch = max(1, math.ceil(len(loader) / accumulation))
    configured_max_steps = config.get("max_steps")
    run_steps = int(max_steps_override or configured_max_steps or (epochs * steps_per_epoch))
    initial_global_step = (
        int(resume_info.get("source_global_step") or 0)
        if resume_info and bool(config.get("continue_global_step", True))
        else 0
    )
    target_global_step = initial_global_step + run_steps
    scheduler = build_scheduler(optimizer, run_steps)
    if resume_info and bool(config.get("resume_optimizer", False)):
        optimizer_state = resume_info.pop("_optimizer_state_dict", None)
        scheduler_state = resume_info.pop("_scheduler_state_dict", None)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
    precision = str(config.get("precision", model_config.get("precision", "bf16"))).lower()
    amp_enabled = bool(config.get("amp", model_config.get("mixed_precision", True))) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype is torch.float16)
    checkpoint_dir = resolve_path(config.get("checkpoint_dir", "outputs/checkpoints/graph_tokenized_decoder_only"))
    log_path = resolve_path(config.get("metrics_log_path", "outputs/training_logs/graph_tokenized_decoder_only.jsonl"))
    ensure_parent(log_path)
    checkpoint_every = int(config.get("checkpoint_every_steps", 1000))
    save_periodic_checkpoints = checkpoint_every > 0
    checkpoint_every = max(1, checkpoint_every)
    save_optimizer_state = bool(config.get("save_optimizer_state", True))
    save_numbered_final_checkpoint = bool(config.get("save_numbered_final_checkpoint", True))
    clip_norm = float(config.get("gradient_clip_norm", 1.0))
    gradient_value_clip = config.get("gradient_value_clip", 1.0)
    gradient_value_clip = None if gradient_value_clip is None else float(gradient_value_clip)
    max_consecutive_nonfinite = max(1, int(config.get("max_consecutive_nonfinite_batches", 4)))
    abort_on_nonfinite_gradients = bool(config.get("abort_on_nonfinite_gradients", False))
    model_version = str(config.get("model_version", "machiningfm-graph-tokenized-stemgnn-decoder-only"))
    console_log_interval = str(config.get("console_log_interval", "step")).lower()
    if console_log_interval not in {"step", "epoch", "none"}:
        raise ValueError("console_log_interval must be one of: step, epoch, none")
    if resume_info:
        LOGGER.info(
            "Loaded graph-tokenized checkpoint for continued pretraining: path=%s, source_global_step=%s, "
            "new_run_steps=%s, target_global_step=%s, mode=%s, missing_keys=%s, unexpected_keys=%s",
            resume_info.get("checkpoint_path"),
            resume_info.get("source_global_step"),
            run_steps,
            target_global_step,
            resume_info.get("mode"),
            resume_info.get("missing_key_count"),
            resume_info.get("unexpected_key_count"),
        )
    show_progress_bar = bool(config.get("show_progress_bar", True))
    heartbeat_interval_seconds = float(config.get("heartbeat_interval_seconds", 0.0) or 0.0)
    early_stopping = _early_stopping_config(config.get("early_stopping"))
    max_train_seconds = float(config.get("max_train_seconds", 0.0) or 0.0)
    checkpoint_config = {**config, "model": model_config}
    global_step = initial_global_step
    consecutive_nonfinite = 0
    last_metrics: dict[str, Any] = {}
    best_early_stop_metric: float | None = None
    best_checkpoint: Path | None = None
    epochs_without_early_stop_improvement = 0
    early_stop_reason: str | None = None
    time_stop_reason: str | None = None
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=run_steps,
        desc="Graph-tokenized pretraining",
        unit="step",
        disable=not show_progress_bar or console_log_interval != "step",
    )
    last_heartbeat_at = 0.0
    if heartbeat_interval_seconds > 0:
        LOGGER.info(
            "Graph-tokenized training loop started: epochs=%s, micro_batch_size=%s, "
            "gradient_accumulation_steps=%s, steps_per_epoch=%s, run_steps=%s, "
            "initial_global_step=%s, target_global_step=%s, heartbeat_interval_seconds=%.1f, max_train_seconds=%.1f",
            epochs,
            int(config.get("batch_size", 1)),
            accumulation,
            steps_per_epoch,
            run_steps,
            initial_global_step,
            target_global_step,
            heartbeat_interval_seconds,
            max_train_seconds,
        )
    try:
        for epoch in range(epochs):
            dataset.set_epoch(epoch)
            model.train()
            epoch_started = time.time()
            epoch_start_step = global_step
            epoch_metric_sums: dict[str, float] = {}
            epoch_update_count = 0
            epoch_nonfinite_count = 0
            epoch_nonfinite_losses: set[str] = set()
            epoch_sanitized_gradients = 0
            for batch_index, batch in enumerate(loader):
                last_heartbeat_at = _maybe_log_training_heartbeat(
                    enabled=heartbeat_interval_seconds > 0,
                    last_logged_at=last_heartbeat_at,
                    interval_seconds=heartbeat_interval_seconds,
                    stage="micro_batch_start",
                    epoch=epoch,
                    epochs=epochs,
                    batch_index=batch_index,
                    batches_per_epoch=len(loader),
                    accumulation=accumulation,
                    global_step=global_step,
                    target_global_step=target_global_step,
                    force=last_heartbeat_at <= 0.0,
                )
                batch = _move_batch(batch, device)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss, losses = objective(batch)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    consecutive_nonfinite += 1
                    epoch_nonfinite_count += 1
                    epoch_nonfinite_losses.update(_nonfinite_loss_names(losses))
                    if console_log_interval == "step":
                        LOGGER.warning(
                            "Skipping non-finite graph-tokenized loss at epoch=%s batch=%s; nonfinite_losses=%s",
                            epoch,
                            batch_index,
                            _nonfinite_loss_names(losses),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_nonfinite >= max_consecutive_nonfinite:
                        raise RuntimeError(
                            "Graph-tokenized training produced consecutive non-finite losses. "
                            f"latest_epoch={epoch}, nonfinite_loss_names={sorted(epoch_nonfinite_losses)}. "
                            "Stop this run, lower learning rate or precision aggressiveness, and restart from a clean checkpoint."
                        )
                    continue
                last_heartbeat_at = _maybe_log_training_heartbeat(
                    enabled=heartbeat_interval_seconds > 0,
                    last_logged_at=last_heartbeat_at,
                    interval_seconds=heartbeat_interval_seconds,
                    stage="backward",
                    epoch=epoch,
                    epochs=epochs,
                    batch_index=batch_index,
                    batches_per_epoch=len(loader),
                    accumulation=accumulation,
                    global_step=global_step,
                    target_global_step=target_global_step,
                )
                scaler.scale(scaled_loss).backward()
                update_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
                if not update_step:
                    continue
                last_heartbeat_at = _maybe_log_training_heartbeat(
                    enabled=heartbeat_interval_seconds > 0,
                    last_logged_at=last_heartbeat_at,
                    interval_seconds=heartbeat_interval_seconds,
                    stage="optimizer_step",
                    epoch=epoch,
                    epochs=epochs,
                    batch_index=batch_index,
                    batches_per_epoch=len(loader),
                    accumulation=accumulation,
                    global_step=global_step,
                    target_global_step=target_global_step,
                )
                scaler.unscale_(optimizer)
                sanitized = _sanitize_nonfinite_gradients(model, max_abs=gradient_value_clip)
                epoch_sanitized_gradients += sanitized
                gradient_norm = _clip_grad_norm_fp32(model, clip_norm)
                if not math.isfinite(gradient_norm):
                    consecutive_nonfinite += 1
                    epoch_nonfinite_count += 1
                    epoch_nonfinite_losses.add("gradient_norm")
                    if console_log_interval == "step":
                        LOGGER.warning(
                            "Skipping optimizer step with non-finite gradient norm at epoch=%s batch=%s",
                            epoch,
                            batch_index,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if abort_on_nonfinite_gradients and consecutive_nonfinite >= max_consecutive_nonfinite:
                        raise RuntimeError(
                            "Graph-tokenized training produced consecutive non-finite gradients. "
                            "The optimizer step was blocked before updating parameters."
                        )
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                consecutive_nonfinite = 0
                global_step += 1
                last_metrics = {
                    "record_type": "step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": gradient_norm,
                    "elapsed_seconds": time.time() - started,
                    "parameter_estimate_total": parameter_estimate["total"],
                    **{name: float(value.detach()) for name, value in losses.items()},
                }
                _append_jsonl(log_path, last_metrics)
                for name in ("loss", "gradient_norm", *losses.keys()):
                    epoch_metric_sums[name] = epoch_metric_sums.get(name, 0.0) + float(last_metrics[name])
                epoch_update_count += 1
                progress.update(1)
                progress.set_postfix(loss=f"{last_metrics['loss']:.4f}", refresh=False)
                if save_periodic_checkpoints and global_step % checkpoint_every == 0:
                    save_training_checkpoint(
                        checkpoint_dir / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        checkpoint_config,
                        last_metrics,
                        model_version,
                        epoch,
                        global_step,
                        include_optimizer_state=save_optimizer_state,
                    )
                if max_train_seconds > 0 and (time.time() - started) >= max_train_seconds:
                    time_stop_reason = f"max_train_seconds={max_train_seconds:.0f} reached"
                    last_metrics["time_limit_stop_reason"] = time_stop_reason
                    break
                if global_step >= target_global_step:
                    break
            if epoch_update_count:
                epoch_metrics = {
                    "record_type": "epoch",
                    "epoch": epoch,
                    "epoch_display": epoch + 1,
                    "epochs_configured": epochs,
                    "global_step": global_step,
                    "optimizer_steps": global_step - epoch_start_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_seconds": time.time() - epoch_started,
                    "nonfinite_batches": epoch_nonfinite_count,
                    "nonfinite_loss_names": sorted(epoch_nonfinite_losses),
                    "sanitized_gradient_tensors": epoch_sanitized_gradients,
                    **{
                        f"mean_{name}": total / epoch_update_count
                        for name, total in epoch_metric_sums.items()
                    },
                }
                early_stop_triggered_this_epoch = False
                if early_stopping["enabled"]:
                    current_metric = epoch_metrics.get(early_stopping["metric"])
                    if current_metric is None or not math.isfinite(float(current_metric)):
                        LOGGER.warning(
                            "Early stopping metric is unavailable or non-finite: metric=%s, value=%s",
                            early_stopping["metric"],
                            current_metric,
                        )
                    elif _is_early_stopping_improvement(
                        float(current_metric),
                        best_early_stop_metric,
                        mode=early_stopping["mode"],
                        min_delta=early_stopping["min_delta"],
                    ):
                        best_early_stop_metric = float(current_metric)
                        epochs_without_early_stop_improvement = 0
                        epoch_metrics["early_stopping_best_metric"] = best_early_stop_metric
                        if early_stopping["save_best_checkpoint"]:
                            best_checkpoint = save_training_checkpoint(
                                checkpoint_dir / "machiningfm_full_pretrain_best.pt",
                                model,
                                optimizer,
                                scheduler,
                                scaler,
                                checkpoint_config,
                                epoch_metrics,
                                model_version,
                                epoch,
                                global_step,
                                update_latest=False,
                                include_optimizer_state=save_optimizer_state,
                            )
                    else:
                        epochs_without_early_stop_improvement += 1
                        epoch_metrics["early_stopping_best_metric"] = best_early_stop_metric
                        epoch_metrics["early_stopping_epochs_without_improvement"] = (
                            epochs_without_early_stop_improvement
                        )
                        if epochs_without_early_stop_improvement >= early_stopping["patience"]:
                            early_stop_reason = (
                                f"metric={early_stopping['metric']} did not improve by "
                                f"min_delta={early_stopping['min_delta']} for "
                                f"{early_stopping['patience']} epochs"
                            )
                            epoch_metrics["early_stopping_stop_reason"] = early_stop_reason
                            early_stop_triggered_this_epoch = True
                _append_jsonl(log_path, epoch_metrics)
                if console_log_interval == "epoch":
                    print(_format_epoch_summary(epoch_metrics), flush=True)
                if early_stop_triggered_this_epoch:
                    print(
                        "\nEarly stopping triggered\n"
                        f"  reason:            {early_stop_reason}\n"
                        f"  best_metric:       {best_early_stop_metric}\n"
                        f"  stopped_epoch:     {epoch + 1}\n"
                        f"  global_step:       {global_step}",
                        flush=True,
                    )
            if global_step >= target_global_step:
                break
            if time_stop_reason:
                print(
                    "\nTime limit reached\n"
                    f"  reason:            {time_stop_reason}\n"
                    f"  stopped_epoch:     {epoch + 1}\n"
                    f"  global_step:       {global_step}",
                    flush=True,
                )
                break
            if early_stop_reason:
                break
    finally:
        progress.close()
    final_checkpoint_path = (
        checkpoint_dir / f"step_{global_step:08d}_final.pt"
        if save_numbered_final_checkpoint
        else checkpoint_dir / "machiningfm_full_pretrain_latest.pt"
    )
    final_checkpoint = save_training_checkpoint(
        final_checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        checkpoint_config,
        last_metrics,
        model_version,
        min(epochs - 1, epoch if "epoch" in locals() else 0),
        global_step,
        update_latest=save_numbered_final_checkpoint,
        include_optimizer_state=save_optimizer_state,
    )
    summary = {
        "model_version": model_version,
        "architecture": "machiningfm_graph_tokenized_stemgnn_decoder_only",
        "loads_previous_checkpoint": bool(resume_info),
        "resume": resume_info,
        "device": device_name,
        "parameter_estimate": parameter_estimate,
        "dataset": dataset.summary(),
        "initial_global_step": initial_global_step,
        "run_steps": run_steps,
        "max_train_seconds": max_train_seconds,
        "time_limit_reached": bool(time_stop_reason),
        "time_stop_reason": time_stop_reason,
        "save_periodic_checkpoints": save_periodic_checkpoints,
        "checkpoint_every_steps": checkpoint_every if save_periodic_checkpoints else 0,
        "save_optimizer_state": save_optimizer_state,
        "save_numbered_final_checkpoint": save_numbered_final_checkpoint,
        "target_global_step": target_global_step,
        "global_step": global_step,
        "final_checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "early_stopping": {
            **early_stopping,
            "best_metric": best_early_stop_metric,
            "epochs_without_improvement": epochs_without_early_stop_improvement,
            "stopped_early": bool(early_stop_reason),
            "stop_reason": early_stop_reason,
        },
        "last_metrics": last_metrics,
    }
    write_json(checkpoint_dir / "training_summary.json", summary)
    return summary


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _early_stopping_config(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("early_stopping must be a mapping")
    enabled = bool(value.get("enabled", False))
    metric = str(value.get("metric", "mean_loss"))
    mode = str(value.get("mode", "min")).lower()
    if mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be one of: min, max")
    patience = max(1, int(value.get("patience", 20)))
    min_delta = max(0.0, float(value.get("min_delta", 0.0)))
    save_best_checkpoint = bool(value.get("save_best_checkpoint", True))
    return {
        "enabled": enabled,
        "metric": metric,
        "mode": mode,
        "patience": patience,
        "min_delta": min_delta,
        "save_best_checkpoint": save_best_checkpoint,
    }


def _is_early_stopping_improvement(
    current: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError("early_stopping.mode must be one of: min, max")


def _maybe_log_training_heartbeat(
    *,
    enabled: bool,
    last_logged_at: float,
    interval_seconds: float,
    stage: str,
    epoch: int,
    epochs: int,
    batch_index: int,
    batches_per_epoch: int,
    accumulation: int,
    global_step: int,
    target_global_step: int,
    force: bool = False,
) -> float:
    if not enabled:
        return last_logged_at
    now = time.time()
    if not force and last_logged_at > 0.0 and now - last_logged_at < interval_seconds:
        return last_logged_at
    LOGGER.info(
        "Training heartbeat: stage=%s, epoch=%s/%s, micro_batch=%s/%s, "
        "accumulation_slot=%s/%s, global_step=%s/%s",
        stage,
        epoch + 1,
        epochs,
        batch_index + 1,
        batches_per_epoch,
        (batch_index % accumulation) + 1,
        accumulation,
        global_step,
        target_global_step,
    )
    return now


def _load_training_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    base_config_path = config.get("base_config_path")
    if not base_config_path:
        return config
    base_config = _load_training_config(resolve_path(base_config_path))
    override = {key: value for key, value in config.items() if key != "base_config_path"}
    return _deep_merge(base_config, override)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_resume_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path | None,
    *,
    strict: bool = True,
    include_training_state: bool = False,
) -> dict[str, Any]:
    if not checkpoint_path:
        return {}
    path = resolve_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = load_checkpoint(path, map_location="cpu")
    if not include_training_state:
        checkpoint.pop("optimizer_state_dict", None)
        checkpoint.pop("scheduler_state_dict", None)
        checkpoint.pop("scaler_state_dict", None)
    state_dict = checkpoint["state_dict"]
    skipped_shape_mismatch: list[str] = []
    if not strict:
        current_state = model.state_dict()
        filtered_state = {}
        for name, value in state_dict.items():
            current_value = current_state.get(name)
            if current_value is not None and tuple(current_value.shape) != tuple(value.shape):
                skipped_shape_mismatch.append(name)
                continue
            filtered_state[name] = value
        state_dict = filtered_state
    load_result = model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    resume_info: dict[str, Any] = {
        "checkpoint_path": str(path),
        "mode": "optimizer_and_scheduler" if include_training_state else "model_only",
        "source_model_version": checkpoint.get("model_version"),
        "source_epoch": checkpoint.get("epoch"),
        "source_global_step": checkpoint.get("global_step"),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "skipped_shape_mismatch_count": len(skipped_shape_mismatch),
        "missing_keys_sample": missing_keys[:20],
        "unexpected_keys_sample": unexpected_keys[:20],
        "skipped_shape_mismatch_sample": skipped_shape_mismatch[:20],
    }
    if include_training_state:
        resume_info["_optimizer_state_dict"] = checkpoint.get("optimizer_state_dict")
        resume_info["_scheduler_state_dict"] = checkpoint.get("scheduler_state_dict")
    if not include_training_state:
        del checkpoint
        del state_dict
        gc.collect()
    return resume_info


def _nonfinite_loss_names(losses: dict[str, torch.Tensor]) -> list[str]:
    names = []
    for name, value in losses.items():
        if not torch.isfinite(value.detach()):
            names.append(name)
    return names


def _format_epoch_summary(metrics: dict[str, Any]) -> str:
    return (
        f"\nEpoch {metrics['epoch_display']}/{metrics['epochs_configured']} complete\n"
        f"  global_step:       {metrics['global_step']}\n"
        f"  optimizer_steps:   {metrics['optimizer_steps']}\n"
        f"  mean_loss:         {metrics.get('mean_loss', float('nan')):.6f}\n"
        f"  mean_grad_norm:    {metrics.get('mean_gradient_norm', float('nan')):.6f}\n"
        f"  nonfinite_batches: {metrics.get('nonfinite_batches', 0)}\n"
        f"  nonfinite_losses:  {metrics.get('nonfinite_loss_names', [])}\n"
        f"  sanitized_grads:   {metrics.get('sanitized_gradient_tensors', 0)}\n"
        f"  learning_rate:     {metrics['learning_rate']:.3e}\n"
        f"  epoch_seconds:     {metrics['epoch_seconds']:.1f}"
    )


def _sanitize_nonfinite_gradients(model: torch.nn.Module, max_abs: float | None = None) -> int:
    sanitized = 0
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized += 1
        if max_abs is not None and max_abs > 0:
            grad.data.clamp_(min=-max_abs, max=max_abs)
    return sanitized


def _clip_grad_norm_fp32(model: torch.nn.Module, max_norm: float) -> float:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    norms = [torch.linalg.vector_norm(grad.detach().float(), ord=2) for grad in gradients]
    total_norm_tensor = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    total_norm = float(total_norm_tensor.detach().cpu())
    if not math.isfinite(total_norm):
        return total_norm
    if max_norm > 0 and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for grad in gradients:
            grad.mul_(scale)
    return total_norm


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("model_config_path"):
        model_config = load_config(resolve_path(config["model_config_path"]))
        model_config.update(config.get("model", {}))
        return model_config
    return config.get("model", config)


def _estimated_training_memory_gib(parameter_count: int, parameter_dtype: str = "fp32") -> float:
    low_precision = parameter_dtype.lower() in {"bf16", "bfloat16", "fp16", "float16"}
    parameter_bytes = 2 if low_precision else 4
    gradient_bytes = parameter_bytes
    adam_state_bytes = 4 if low_precision else 8
    activation_overhead_bytes = 1 if low_precision else 2
    total_bytes = parameter_count * (parameter_bytes + gradient_bytes + adam_state_bytes + activation_overhead_bytes)
    return total_bytes / (1024**3)


def _torch_dtype(value: str) -> torch.dtype | None:
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32", "none"}:
        return None
    raise ValueError(f"Unsupported parameter_dtype: {value}")
