from __future__ import annotations

import torch


def spindle_order_consistency_loss(predicted_spectrum: torch.Tensor, target_order_amplitudes: torch.Tensor | None) -> torch.Tensor:
    if target_order_amplitudes is None:
        return torch.zeros((), device=predicted_spectrum.device)
    channels = min(predicted_spectrum.shape[-1], target_order_amplitudes.shape[-1])
    return torch.nn.functional.smooth_l1_loss(predicted_spectrum[..., :channels], target_order_amplitudes[..., :channels])


def energy_consistency_loss(waveform: torch.Tensor, expected_rms: torch.Tensor | None) -> torch.Tensor:
    if expected_rms is None:
        return torch.zeros((), device=waveform.device)
    rms = torch.sqrt(waveform.float().square().mean(dim=-1).clamp_min(1.0e-8))
    return torch.nn.functional.smooth_l1_loss(rms, expected_rms.to(waveform.device).float())
