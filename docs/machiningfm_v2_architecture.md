# MachiningFM v2 Architecture

MachiningFM v2 is implemented as a separate package under `src/machiningfm_v2` so existing v1 inference code and checkpoints are not overwritten.

## Current v1 Failure Notes

- The v1 forecasting head has a native 64-sample horizon.
- One-second 12,800-sample predictions require repeated autoregressive rollout, which accumulates error.
- FFT channels were injected as ordinary time-domain channels during inference, without matching pretraining.
- The v2 path therefore separates raw waveform and spectral representations and uses direct multi-horizon decoders.

## Input Modalities

- NC program tokens from structured G/M/S/F/T/axis parsing.
- CNC low-rate Setpoint/Effort/Feedback/Context channels.
- Raw high-rate waveform channels.
- Spectral tokens from FFT, STFT, CWT, phase/order-preserving summaries.
- Tool image tokens.
- Metadata tokens for machine, tool, material, run, calibration, and missing modality state.

## Tokenization

High-rate signals are not resampled into CNC rate. The raw branch uses anti-aliased multi-scale patch summaries at 64, 256, 1280, and 12800 samples. The spectral branch builds FFT, STFT, CWT, and spindle-order features as separate spectral tokens.

Low-rate CNC channels use a Setpoint/Effort/Feedback/Context grouping. Timestamp alignment is represented by metadata and causal masks instead of forcing all streams to the same sample count.

## Model

The v2 model contains modality-specific encoders, a cross-modal Transformer fusion block, separated latents for process state, domain identity, health invariants, and stochastic residuals, and a direct multi-horizon forecasting decoder for 64, 1280, and 12800 samples.

## Checkpoint Policy

V2 checkpoints use `format = machiningfm-v2-checkpoint-v1`. V1 checkpoints are never overwritten. Migration copies only tensors with explicit name mapping or identical key and identical shape, and writes a migration report listing loaded, missing, unexpected, and mismatched parameters.

## Release Gate

Best checkpoints should only be promoted after held-out zero-shot evaluation beats persistence and the previous checkpoint. The initial evaluator reports RMSE/MAE/R2 by horizon and marks the release gate failed if RMSE improvement is below 15%.
