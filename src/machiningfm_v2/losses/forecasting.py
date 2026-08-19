from __future__ import annotations

import torch
from torch.nn import functional as F


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Robust Student-t NLL parameterized by log variance.

    The old quadratic Gaussian residual let a single scale-mismatched sample
    dominate an entire accumulated optimizer step.
    """
    var = log_var.exp().clamp_min(1.0e-6)
    degrees_of_freedom = 3.0
    standardized_sq = (target - mean).square() / var
    return (
        0.5 * log_var
        + 0.5 * (degrees_of_freedom + 1.0) * torch.log1p(standardized_sq / degrees_of_freedom)
    ).mean()


def stft_magnitude_loss(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 256) -> torch.Tensor:
    pred_flat = pred.reshape(-1, pred.shape[-1]).float()
    target_flat = target.reshape(-1, target.shape[-1]).float()
    fft_size = min(n_fft, pred.shape[-1])
    window = torch.hann_window(fft_size, device=pred.device, dtype=pred_flat.dtype)
    pred_spec = torch.stft(pred_flat, n_fft=fft_size, window=window, return_complex=True)
    target_spec = torch.stft(target_flat, n_fft=fft_size, window=window, return_complex=True)
    return F.l1_loss(torch.log1p(pred_spec.abs()), torch.log1p(target_spec.abs()))


def rms_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rms = torch.sqrt(pred.square().mean(dim=-1).clamp_min(1.0e-8))
    target_rms = torch.sqrt(target.square().mean(dim=-1).clamp_min(1.0e-8))
    return F.l1_loss(pred_rms, target_rms)


def derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))


def multi_horizon_forecasting_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, torch.Tensor],
    weights: dict | None = None,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = weights or {}
    horizon_losses: list[torch.Tensor] = []
    component_values: dict[str, list[torch.Tensor]] = {
        "nll": [],
        "huber": [],
        "rms": [],
        "derivative": [],
        "stft": [],
    }
    for horizon, prediction in outputs.items():
        target = targets.get(horizon)
        if target is None:
            target = targets.get(f"forecast_{horizon}")
        if target is None:
            continue
        target = target.to(prediction["mean"].device).float()
        mean = prediction["mean"][..., : target.shape[-1]]
        log_var = prediction["log_var"][..., : target.shape[-1]]
        terms = {
            "nll": float(cfg.get("nll", 0.2)) * gaussian_nll(mean, log_var, target),
            "huber": float(cfg.get("huber", 1.0)) * F.smooth_l1_loss(mean, target),
            "rms": float(cfg.get("rms", 0.5)) * rms_loss(mean, target),
        }
        if target.shape[-1] > 2:
            terms["derivative"] = float(cfg.get("derivative", 0.2)) * derivative_loss(mean, target)
        if target.shape[-1] >= 64:
            terms["stft"] = float(cfg.get("stft", 0.2)) * stft_magnitude_loss(mean, target)
        horizon_losses.append(torch.stack(list(terms.values())).sum())
        for name, value in terms.items():
            component_values[name].append(value)
    if not horizon_losses:
        device = next(iter(outputs.values()))["mean"].device
        total = torch.zeros((), device=device)
        components = {name: total for name in (*component_values, "selection", "total")}
        return (total, components) if return_components else total
    total = torch.stack(horizon_losses).mean()
    components = {
        name: torch.stack(values).mean() if values else total.new_zeros(())
        for name, values in component_values.items()
    }
    components["selection"] = sum(value for name, value in components.items() if name != "nll")
    components["total"] = total
    return (total, components) if return_components else total
