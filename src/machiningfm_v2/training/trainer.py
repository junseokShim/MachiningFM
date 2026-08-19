from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from machiningfm.utils.config import load_config
from machiningfm_v2.data.multirate_dataset import MultiRateMachiningDataset, multirate_collate
from machiningfm_v2.losses.forecasting import multi_horizon_forecasting_loss
from machiningfm_v2.models.foundation_model import MachiningFMV2
from machiningfm_v2.training.checkpoint import graph_tokenized_v1_semantic_map, migrate_v1_to_v2, save_v2_checkpoint


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_v2_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    if config.get("base_config_path"):
        base = load_v2_config(config["base_config_path"])
        config = deep_merge(base, {k: v for k, v in config.items() if k != "base_config_path"})
    return config


def train_v2(
    config_path: str | Path,
    *,
    device_override: str | None = None,
    max_steps: int | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config = load_v2_config(config_path)
    cpu_threads = max(1, int(config.get("cpu_threads", 2)))
    torch.set_num_threads(cpu_threads)
    device = torch.device(device_override or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MultiRateMachiningDataset(config.get("data", {}))
    train_dataset, validation_dataset = split_train_validation(dataset, float(config.get("validation_ratio", 0.05)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=multirate_collate,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=int(config.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(config.get("num_workers", 0)),
            collate_fn=multirate_collate,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )
    if len(train_loader) == 0:
        raise RuntimeError("MachiningFM v2 training dataset is empty.")
    model = MachiningFMV2(config.get("model", {})).to(device)
    migration_report = None
    source_global_step = 0
    source_elapsed_seconds = 0.0
    source_stage_epoch = 0
    resumed_best_loss = math.inf
    resume_path = resume_checkpoint or config.get("resume_v2_checkpoint")
    if resume_path:
        # v2 checkpoints are small enough to load eagerly. On Windows, mmap
        # would keep the resumed latest.pt locked when it is overwritten.
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        checkpoint_metrics = checkpoint.get("metrics", {})
        source_global_step = int(checkpoint_metrics.get("step", 0))
        source_elapsed_seconds = float(
            checkpoint_metrics.get("total_elapsed_seconds", checkpoint_metrics.get("elapsed_seconds", 0.0))
        )
        source_stage_epoch = int(checkpoint_metrics.get("stage_epoch", 0))
        resumed_best_loss = float(checkpoint_metrics.get("selection_metric", math.inf))
        print(f"[machiningfm_v2] resumed: path={resume_path} source_global_step={source_global_step} mode=model_only")
    elif config.get("init_from_v1_checkpoint"):
        name_map = dict(config.get("checkpoint_name_map") or {})
        if config.get("use_graph_tokenized_v1_semantic_init", False):
            name_map = {**graph_tokenized_v1_semantic_map(model), **name_map}
        migration_report = migrate_v1_to_v2(
            config["init_from_v1_checkpoint"],
            model,
            output_report_path=config.get("migration_report_path", "outputs/reports/machiningfm_v2_migration_report.json"),
            name_map=name_map,
        )
        print(
            "[machiningfm_v2] v1 semantic init: "
            f"loaded={migration_report['loaded_count']} adapted={migration_report['adapted_count']} "
            f"missing={migration_report['missing_count']} report={config.get('migration_report_path')}"
        )
    started = time.time()
    learning_rate = float(config.get("learning_rate", 3.0e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(config.get("weight_decay", 1.0e-4)))
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    if max_steps is not None:
        planned_total_steps = source_global_step + int(max_steps)
        run_step_limit = int(max_steps)
    else:
        configured_steps = config.get("max_steps")
        if configured_steps:
            planned_total_steps = int(configured_steps)
        else:
            remaining_epochs = max(0, int(config.get("epochs", 1)) - source_stage_epoch)
            planned_total_steps = source_global_step + optimizer_steps_per_epoch * remaining_epochs
        run_step_limit = max(0, planned_total_steps - source_global_step)
    warmup_steps = max(0, int(config.get("warmup_steps", 500)))
    min_lr_ratio = min(1.0, max(0.0, float(config.get("min_lr_ratio", 0.05))))
    scheduler_total_seconds = float(config.get("scheduler_total_seconds", 0.0) or 0.0)

    def lr_multiplier(local_step: int) -> float:
        absolute_step = source_global_step + int(local_step)
        if warmup_steps > 0 and absolute_step < warmup_steps:
            return max(1.0e-4, (absolute_step + 1) / warmup_steps)
        if scheduler_total_seconds > 0:
            progress = (source_elapsed_seconds + time.time() - started) / scheduler_total_seconds
            progress = min(1.0, max(0.0, progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        progress = (absolute_step - warmup_steps) / max(1, planned_total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(config.get("precision", "bf16")).lower() == "bf16" else torch.float16
    max_train_seconds = float(config.get("max_train_seconds", 0.0) or 0.0)
    if scheduler_total_seconds > 0:
        remaining_total_seconds = max(0.0, scheduler_total_seconds - source_elapsed_seconds)
        max_train_seconds = remaining_total_seconds if max_train_seconds <= 0 else min(max_train_seconds, remaining_total_seconds)
    checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints/machiningfm_v2"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.get("metrics_log_path", "outputs/training_logs/machiningfm_v2.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    global_step = source_global_step
    run_steps = 0
    micro_step = 0
    epoch = source_stage_epoch
    configured_epochs = int(config.get("epochs", 1))
    best_loss = resumed_best_loss
    optimizer.zero_grad(set_to_none=True)
    print(
        "[machiningfm_v2] training started: "
        f"device={device} train_files={len(train_dataset.files)} val_files={len(validation_dataset.files) if validation_dataset else 0} "
        f"params={sum(p.numel() for p in model.parameters()):,} source_step={source_global_step} "
        f"planned_total_steps={planned_total_steps} remaining_steps={run_step_limit} "
        f"source_elapsed_seconds={source_elapsed_seconds:.1f} max_train_seconds={max_train_seconds:g}"
    )
    interrupted = False
    last_epoch_loss = math.inf
    last_selection_metric = math.inf
    nonfinite_batches = 0
    try:
        while run_steps < run_step_limit:
            epoch += 1
            epoch_started = time.time()
            epoch_loss_sum = 0.0
            epoch_component_sums: dict[str, float] = {}
            epoch_micro_batches = 0
            group_loss_sum = 0.0
            group_component_sums: dict[str, float] = {}
            group_micro_batches = 0
            group_worst_sample: dict[str, Any] | None = None
            for batch_index, batch in enumerate(train_loader):
                batch = move_batch(batch, device)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                    out = model(batch)
                    raw_loss, components = multi_horizon_forecasting_loss(
                        out["forecast"],
                        batch.get("targets", {}),
                        config.get("loss", {}).get("forecasting"),
                        return_components=True,
                    )
                if not torch.isfinite(raw_loss):
                    nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    group_loss_sum = 0.0
                    group_component_sums.clear()
                    group_micro_batches = 0
                    continue
                (raw_loss / grad_accum).backward()
                micro_step += 1
                raw_value = float(raw_loss.detach())
                component_values = {name: float(value.detach()) for name, value in components.items()}
                epoch_loss_sum += raw_value
                epoch_micro_batches += 1
                group_loss_sum += raw_value
                group_micro_batches += 1
                if group_worst_sample is None or raw_value > float(group_worst_sample["loss"]):
                    group_worst_sample = batch_diagnostics(batch, raw_value)
                add_component_values(epoch_component_sums, component_values)
                add_component_values(group_component_sums, component_values)
                is_last_batch = batch_index + 1 == len(train_loader)
                if group_micro_batches < grad_accum and not is_last_batch:
                    continue
                if group_micro_batches < grad_accum:
                    gradient_scale = grad_accum / max(1, group_micro_batches)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_scale)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                run_steps += 1
                step_components = {name: value / group_micro_batches for name, value in group_component_sums.items()}
                record = {
                    "step": global_step,
                    "loss": group_loss_sum / group_micro_batches,
                    "selection_metric": step_components.get("selection"),
                    "components": step_components,
                    "grad_norm": float(grad_norm),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "micro_batches": group_micro_batches,
                    "elapsed_seconds": time.time() - started,
                    "total_elapsed_seconds": source_elapsed_seconds + time.time() - started,
                }
                outlier_threshold = float(config.get("outlier_log_threshold", 100.0))
                if record["loss"] >= outlier_threshold and group_worst_sample is not None:
                    record["worst_sample"] = group_worst_sample
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                group_loss_sum = 0.0
                group_component_sums.clear()
                group_micro_batches = 0
                group_worst_sample = None
                if run_steps >= run_step_limit:
                    break
                if max_train_seconds > 0 and time.time() - started >= max_train_seconds:
                    break
            if epoch_micro_batches == 0:
                raise RuntimeError("No finite training batches were produced in this epoch.")
            train_components = {name: value / epoch_micro_batches for name, value in epoch_component_sums.items()}
            last_epoch_loss = epoch_loss_sum / epoch_micro_batches
            validation = evaluate_validation(
                model,
                validation_loader,
                device=device,
                amp=amp,
                amp_dtype=amp_dtype,
                loss_config=config.get("loss", {}).get("forecasting"),
                max_batches=int(config.get("max_validation_batches", 256)),
            )
            last_selection_metric = float(
                validation.get("selection", train_components.get("selection", last_epoch_loss))
            )
            epoch_record = {
                "step": global_step,
                "epoch": epoch,
                "stage_epoch": epoch,
                "loss": last_epoch_loss,
                "selection_metric": last_selection_metric,
                "train_components": train_components,
                "validation": validation,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
                "total_elapsed_seconds": source_elapsed_seconds + time.time() - started,
            }
            print(
                f"Epoch {epoch}/{configured_epochs} complete | global_step={global_step} | train_loss={last_epoch_loss:.6f} | "
                f"train_selection={train_components.get('selection', float('nan')):.6f} | "
                f"val_selection={last_selection_metric:.6f} | lr={optimizer.param_groups[0]['lr']:.3e} | "
                f"nonfinite={nonfinite_batches} | epoch_seconds={time.time() - epoch_started:.1f}"
            )
            if last_selection_metric < best_loss:
                best_loss = last_selection_metric
                save_v2_checkpoint(checkpoint_dir / "machiningfm_v2_best.pt", model, config, epoch_record, overwrite=True)
            if max_train_seconds > 0 and time.time() - started >= max_train_seconds:
                break
    except KeyboardInterrupt:
        interrupted = True
        print(f"[machiningfm_v2] interrupted: saving latest checkpoint at global_step={global_step}")
    latest_metrics = {
        "step": global_step,
        "loss": last_epoch_loss,
        "selection_metric": last_selection_metric,
        "best_selection_metric": best_loss,
        "stage_epoch": epoch,
        "interrupted": interrupted,
        "elapsed_seconds": time.time() - started,
        "total_elapsed_seconds": source_elapsed_seconds + time.time() - started,
    }
    latest = save_v2_checkpoint(checkpoint_dir / "machiningfm_v2_latest.pt", model, config, latest_metrics, overwrite=True)
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "latest_checkpoint": str(latest),
        "best_checkpoint": str(checkpoint_dir / "machiningfm_v2_best.pt"),
        "steps": global_step,
        "run_steps": run_steps,
        "best_selection_metric": best_loss,
        "interrupted": interrupted,
        "planned_total_steps": planned_total_steps,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "stage_epoch": epoch,
        "dataset": dataset.summary(),
        "migration_report": migration_report,
    }
    (checkpoint_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def split_train_validation(
    dataset: MultiRateMachiningDataset,
    validation_ratio: float,
) -> tuple[MultiRateMachiningDataset, MultiRateMachiningDataset | None]:
    if not dataset.files or validation_ratio <= 0.0 or len(dataset.files) < 2:
        return dataset, None
    validation_ratio = min(0.5, max(0.0, validation_ratio))
    train_files: list[Path] = []
    validation_files: list[Path] = []
    for path in dataset.files:
        group = str(path.parent).lower()
        fraction = int(hashlib.sha1(group.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (validation_files if fraction < validation_ratio else train_files).append(path)
    if not train_files or not validation_files:
        ordered = sorted(dataset.files, key=lambda path: hashlib.sha1(str(path).encode("utf-8", errors="ignore")).hexdigest())
        count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_ratio)))
        validation_files = ordered[:count]
        train_files = ordered[count:]
    train_dataset = copy.copy(dataset)
    validation_dataset = copy.copy(dataset)
    train_dataset.files = train_files
    validation_dataset.files = validation_files
    train_dataset.synthetic_if_empty = False
    validation_dataset.synthetic_if_empty = False
    train_dataset.read_errors = 0
    validation_dataset.read_errors = 0
    train_dataset.invalid_files = set()
    validation_dataset.invalid_files = set()
    return train_dataset, validation_dataset


def add_component_values(destination: dict[str, float], values: dict[str, float]) -> None:
    for name, value in values.items():
        destination[name] = destination.get(name, 0.0) + value


def batch_diagnostics(batch: dict[str, Any], loss: float) -> dict[str, Any]:
    metadata = batch.get("metadata") or []
    source_path = metadata[0].get("source_path") if metadata else None
    target_abs_max = 0.0
    for target in batch.get("targets", {}).values():
        if isinstance(target, torch.Tensor) and target.numel():
            target_abs_max = max(target_abs_max, float(target.detach().abs().amax()))
    scale = batch.get("target_scale")
    scale_min = float(scale.detach().amin()) if isinstance(scale, torch.Tensor) and scale.numel() else None
    return {
        "loss": float(loss),
        "source_path": source_path,
        "target_abs_max": target_abs_max,
        "target_scale_min": scale_min,
    }


def evaluate_validation(
    model: torch.nn.Module,
    loader: DataLoader | None,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    loss_config: dict[str, Any] | None,
    max_batches: int,
) -> dict[str, float]:
    if loader is None:
        return {}
    sums: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(batch)
                loss, components = multi_horizon_forecasting_loss(
                    output["forecast"], batch.get("targets", {}), loss_config, return_components=True
                )
            if not torch.isfinite(loss):
                continue
            add_component_values(sums, {name: float(value.detach()) for name, value in components.items()})
            count += 1
    model.train()
    if count == 0:
        raise RuntimeError("Validation produced no finite batches.")
    return {name: value / count for name, value in sums.items()}


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}
        else:
            out[key] = value
    return out
