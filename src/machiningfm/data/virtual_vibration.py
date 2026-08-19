from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

VIRTUAL_VIBRATION_SCHEMA_VERSION = "virtual-vibration-v1"
VIRTUAL_VIBRATION_ATTACHMENT = "spindle"
VIRTUAL_SPINDLE_VIBRATION_HEADERS = (
    "virtual_spindle_vibration_x",
    "virtual_spindle_vibration_y",
    "virtual_spindle_vibration_z",
)
VIRTUAL_SPINDLE_VIBRATION_CANONICAL_NAMES = (
    "external_sensor.spindle.vibration.x.raw",
    "external_sensor.spindle.vibration.y.raw",
    "external_sensor.spindle.vibration.z.raw",
)


@dataclass(frozen=True)
class VirtualVibrationConfig:
    sampling_rate: float = 1600.0
    output_length: int | None = None
    input_sampling_rate: float | None = None
    default_spindle_rpm: float = 6000.0
    amplitude: float = 1.0
    noise_std: float = 0.03
    seed: int = 42


def virtual_vibration_header_metadata() -> dict[str, Any]:
    return {
        "schema_version": VIRTUAL_VIBRATION_SCHEMA_VERSION,
        "attachment": VIRTUAL_VIBRATION_ATTACHMENT,
        "headers": [
            {
                "raw_name": raw_name,
                "canonical_name": canonical_name,
                "mount_component": "spindle",
                "quantity": "vibration",
                "axis": axis,
                "source": "external_sensor",
                "representation": "raw",
            }
            for raw_name, canonical_name, axis in zip(
                VIRTUAL_SPINDLE_VIBRATION_HEADERS,
                VIRTUAL_SPINDLE_VIBRATION_CANONICAL_NAMES,
                ("x", "y", "z"),
            )
        ],
        "api_input_fields": ["cnc_series", "cnc_names", "virtual_vibration_sampling_rate"],
    }


def generate_virtual_spindle_vibration(
    cnc_values: np.ndarray,
    cnc_channel_names: Iterable[str] | None = None,
    config: VirtualVibrationConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Generate spindle-mounted x/y/z virtual vibration from CNC/controller signals.

    The output is deterministic for the same input and seed. It is intended as a
    virtual sensor representation, not a physically exact digital twin.
    """
    cfg = _coerce_config(config)
    values = _as_2d_float(cnc_values)
    names = [str(name) for name in (cnc_channel_names or [f"cnc_channel_{i}" for i in range(values.shape[0])])]
    if len(names) != values.shape[0]:
        names = [f"cnc_channel_{i}" for i in range(values.shape[0])]
    output_length = cfg.output_length or _infer_output_length(values.shape[-1], cfg)
    output_length = max(8, int(output_length))
    source = np.stack([_resample(_finite(row), output_length) for row in values])
    source = np.stack([_standardize(row) for row in source])
    t = np.arange(output_length, dtype=np.float32) / max(float(cfg.sampling_rate), 1.0)

    envelope = _build_envelope(source, names)
    rpm = _estimate_rpm(source, names, cfg.default_spindle_rpm)
    tooth_frequency = max(rpm / 60.0, 1.0)
    harmonics = (1.0, 2.0, 3.0, 4.0)
    axis_phase = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    axis_gain = (1.0, 0.82, 0.65)
    rng = np.random.default_rng(cfg.seed + _stable_signal_hash(source))
    virtual = []
    drive = _principal_drive(source)
    for axis_index, phase in enumerate(axis_phase):
        carrier = np.zeros(output_length, dtype=np.float32)
        for harmonic_index, harmonic in enumerate(harmonics):
            gain = 1.0 / (harmonic_index + 1)
            carrier += gain * np.sin(2.0 * math.pi * tooth_frequency * harmonic * t + phase)
        modulation = 0.35 * np.roll(drive, axis_index * max(1, output_length // 17))
        noise = rng.normal(0.0, cfg.noise_std, size=output_length).astype(np.float32)
        signal = cfg.amplitude * axis_gain[axis_index] * envelope * (carrier + modulation) + noise
        virtual.append(_standardize(signal))
    metadata = {
        "schema_version": VIRTUAL_VIBRATION_SCHEMA_VERSION,
        "attachment": VIRTUAL_VIBRATION_ATTACHMENT,
        "sampling_rate": float(cfg.sampling_rate),
        "input_sampling_rate": cfg.input_sampling_rate,
        "output_length": output_length,
        "default_spindle_rpm": float(cfg.default_spindle_rpm),
        "headers": list(VIRTUAL_SPINDLE_VIBRATION_HEADERS),
        "canonical_names": list(VIRTUAL_SPINDLE_VIBRATION_CANONICAL_NAMES),
        "config": asdict(cfg),
    }
    return np.stack(virtual).astype(np.float32, copy=False), list(VIRTUAL_SPINDLE_VIBRATION_HEADERS), metadata


def generate_virtual_spindle_vibration_v2(
    cnc_values: np.ndarray,
    cnc_channel_names: Iterable[str] | None = None,
    config: VirtualVibrationConfig | dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Generate spindle-mounted x/y/z virtual vibration from CNC/controller signals.

    This version uses a physics-structured vibration proxy:

        force proxy -> modal acceleration response -> normalized virtual vibration

    The generated signal is still a virtual sensor representation. Since the
    excitation force is inferred from CNC/controller signals, it should not be
    interpreted as a calibrated physical acceleration signal unless modal
    parameters and gains are identified from real vibration measurements.
    """
    cfg = _coerce_config(config)
    values = _as_2d_float(cnc_values)

    names = [
        str(name)
        for name in (
            cnc_channel_names
            or [f"cnc_channel_{i}" for i in range(values.shape[0])]
        )
    ]

    if len(names) != values.shape[0]:
        names = [f"cnc_channel_{i}" for i in range(values.shape[0])]

    output_length = cfg.output_length or _infer_output_length(values.shape[-1], cfg)
    output_length = max(8, int(output_length))

    # Keep raw-resampled source for RPM estimation.
    source_raw = np.stack([
        _resample(_finite(row), output_length)
        for row in values
    ]).astype(np.float32, copy=False)

    # Standardized source is used for force proxy terms.
    source = np.stack([
        _standardize(row)
        for row in source_raw
    ]).astype(np.float32, copy=False)

    fs = max(float(cfg.sampling_rate), 1.0)
    t = np.arange(output_length, dtype=np.float32) / fs

    # Positive CNC-derived amplitude envelope.
    envelope = _positive_envelope(_build_envelope(source, names))

    # RPM should be estimated before standardization.
    rpm_series = _estimate_rpm_series(
        source_raw=source_raw,
        names=names,
        default_rpm=float(cfg.default_spindle_rpm),
        output_length=output_length,
    )

    # Angular position:
    # theta[n] = integral 2*pi*RPM(t)/60 dt
    omega = 2.0 * math.pi * rpm_series / 60.0
    theta = np.zeros(output_length, dtype=np.float32)
    if output_length > 1:
        theta[1:] = np.cumsum(omega[:-1] / fs).astype(np.float32)

    rng = np.random.default_rng(cfg.seed + _stable_signal_hash(source))

    axis_names = ("x", "y", "z")
    axis_phase = (
        0.0,
        2.0 * math.pi / 3.0,
        4.0 * math.pi / 3.0,
    )
    axis_gain = (1.0, 0.82, 0.65)

    drive = _principal_drive(source)

    # Build physics-structured force proxy:
    # F_a = F_cut,a + F_rot,a + F_drive,a
    cutting_force = _build_cutting_force_proxy(
        theta=theta,
        envelope=envelope,
        cfg=cfg,
    )

    rotational_force = _build_rotational_force_proxy(
        theta=theta,
        rpm_series=rpm_series,
        envelope=envelope,
        axis_phase=axis_phase,
        cfg=cfg,
    )

    drive_force = _build_drive_force_proxy(
        source=source,
        names=names,
        drive=drive,
        cfg=cfg,
    )

    virtual = []
    used_modal_params: dict[str, list[dict[str, float]]] = {}

    for axis_index, axis_name in enumerate(axis_names):
        force_axis = (
            cutting_force[axis_index]
            + rotational_force[axis_index]
            + drive_force[axis_index]
        ).astype(np.float32, copy=False)

        # Normalize force proxy because it is not calibrated in Newtons.
        force_axis = (
            float(cfg.amplitude)
            * axis_gain[axis_index]
            * _standardize(force_axis)
        ).astype(np.float32, copy=False)

        modes = _get_axis_modal_params(
            cfg=cfg,
            axis_name=axis_name,
            axis_index=axis_index,
            fs=fs,
        )

        used_modal_params[axis_name] = [
            {
                "natural_frequency_hz": float(mode["fn"]),
                "damping_ratio": float(mode["zeta"]),
                "modal_gain": float(mode["gain"]),
            }
            for mode in modes
        ]

        # Modal acceleration response:
        # H_acc(s) = G*s^2 / (s^2 + 2*zeta*wn*s + wn^2)
        signal = _modal_acceleration_response(
            force=force_axis,
            fs=fs,
            modes=modes,
        )

        noise = rng.normal(
            0.0,
            cfg.noise_std,
            size=output_length,
        ).astype(np.float32)

        signal = signal + noise

        # Final output remains normalized representation.
        virtual.append(_standardize(signal))

    metadata = {
        "schema_version": VIRTUAL_VIBRATION_SCHEMA_VERSION,
        "attachment": VIRTUAL_VIBRATION_ATTACHMENT,
        "generation_model": "physics_structured_modal_acceleration_proxy",
        "sampling_rate": float(cfg.sampling_rate),
        "input_sampling_rate": cfg.input_sampling_rate,
        "output_length": output_length,
        "default_spindle_rpm": float(cfg.default_spindle_rpm),
        "headers": list(VIRTUAL_SPINDLE_VIBRATION_HEADERS),
        "canonical_names": list(VIRTUAL_SPINDLE_VIBRATION_CANONICAL_NAMES),
        "modal_params": used_modal_params,
        "force_model": {
            "total_force": "F_axis = F_cut + F_rot + F_drive",
            "response": "V_axis = Std(sum_r h_acc,r * F_axis + noise)",
            "note": "Force is a CNC-derived proxy, not calibrated Newton-scale force.",
        },
        "config": asdict(cfg),
    }

    return (
        np.stack(virtual).astype(np.float32, copy=False),
        list(VIRTUAL_SPINDLE_VIBRATION_HEADERS),
        metadata,
    )

def append_virtual_spindle_vibration(
    values: np.ndarray,
    channel_names: list[str],
    max_channels: int,
    config: VirtualVibrationConfig | dict[str, Any] | None = None,
    mode: str = "if_missing",
) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    if max_channels <= 0:
        return values, channel_names, None
    lower_names = " ".join(name.lower() for name in channel_names)
    has_real_vibration = "vibration" in lower_names or "vib" in lower_names or "acc" in lower_names
    if mode not in {"always", "if_missing"}:
        raise ValueError("virtual vibration mode must be 'always' or 'if_missing'")
    if mode == "if_missing" and has_real_vibration:
        return values, channel_names, None
    available_slots = max_channels - int(values.shape[0])
    if available_slots <= 0:
        return values, channel_names, None
    cfg = _coerce_config(config)
    cfg = VirtualVibrationConfig(**{**asdict(cfg), "output_length": int(values.shape[-1])})
    virtual, names, metadata = generate_virtual_spindle_vibration(values, channel_names, cfg)
    count = min(available_slots, virtual.shape[0])
    combined_values = np.concatenate([values, virtual[:count]], axis=0)
    combined_names = [*channel_names, *names[:count]]
    metadata["appended_channels"] = count
    metadata["mode"] = mode
    return combined_values.astype(np.float32, copy=False), combined_names, metadata


def _coerce_config(config: VirtualVibrationConfig | dict[str, Any] | None) -> VirtualVibrationConfig:
    if isinstance(config, VirtualVibrationConfig):
        return config
    return VirtualVibrationConfig(**(config or {}))


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected CNC values [channels, time], got {tuple(array.shape)}")
    return array


def _infer_output_length(input_length: int, cfg: VirtualVibrationConfig) -> int:
    if cfg.input_sampling_rate and cfg.input_sampling_rate > 0:
        duration = input_length / float(cfg.input_sampling_rate)
        return max(8, int(round(duration * float(cfg.sampling_rate))))
    return input_length


def _finite(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(max(8, len(data)), dtype=np.float32)
    fill = float(np.median(data[finite]))
    return np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def _resample(value: np.ndarray, length: int) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(data) == length:
        return data
    if len(data) <= 1:
        return np.full(length, float(data[0]) if len(data) else 0.0, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, length),
        np.linspace(0.0, 1.0, len(data)),
        data,
    ).astype(np.float32)


def _standardize(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32).reshape(-1)
    std = float(data.std())
    if not math.isfinite(std) or std < 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - float(data.mean())) / std, -10.0, 10.0).astype(np.float32)


def _build_envelope(source: np.ndarray, names: list[str]) -> np.ndarray:
    weights = np.ones(source.shape[0], dtype=np.float32)
    for index, name in enumerate(names):
        lower = name.lower()
        if any(token in lower for token in ("load", "torque", "current", "power", "force")):
            weights[index] = 1.8
        elif any(token in lower for token in ("rpm", "speed", "spindle")):
            weights[index] = 1.2
        elif any(token in lower for token in ("feed", "position", "axis")):
            weights[index] = 1.1
    energy = np.average(np.abs(source), axis=0, weights=weights)
    smoothed = _moving_average(energy, max(3, min(33, len(energy) // 16 or 3)))
    return (0.35 + 0.65 * _normalize_unit(smoothed)).astype(np.float32)


def _estimate_rpm(source: np.ndarray, names: list[str], default_rpm: float) -> float:
    candidates = []
    for row, name in zip(source, names):
        lower = name.lower()
        if any(token in lower for token in ("rpm", "spindle_speed", "speed")):
            value = float(np.median(np.abs(row)))
            if math.isfinite(value) and value > 0:
                candidates.append(value)
    if not candidates:
        return float(default_rpm)
    # Standardized controller values no longer preserve physical RPM, so map
    # their relative level to a conservative machining speed band.
    level = float(np.median(candidates))
    return float(np.clip(default_rpm * (0.75 + 0.1 * level), 500.0, 24000.0))


def _principal_drive(source: np.ndarray) -> np.ndarray:
    if source.size == 0:
        return np.zeros(8, dtype=np.float32)
    return _standardize(np.mean(source, axis=0))


def _moving_average(value: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return value.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(value, kernel, mode="same").astype(np.float32)


def _normalize_unit(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32)
    minimum = float(data.min())
    maximum = float(data.max())
    if not math.isfinite(maximum - minimum) or maximum <= minimum:
        return np.ones_like(data, dtype=np.float32) * 0.5
    return ((data - minimum) / (maximum - minimum)).astype(np.float32)


def _stable_signal_hash(value: np.ndarray) -> int:
    data = np.asarray(value, dtype=np.float32)
    if data.size == 0:
        return 0
    sample = data.reshape(-1)[:: max(1, data.size // 128)]
    return int(abs(float(sample.sum() * 1009.0))) % 1_000_003

def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    return getattr(cfg, key, default)


def _positive_envelope(envelope: np.ndarray) -> np.ndarray:
    e = _finite(np.asarray(envelope, dtype=np.float32))

    if e.size == 0:
        return np.ones(8, dtype=np.float32)

    e = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
    e = e - float(np.min(e))

    scale = float(np.percentile(e, 95))
    if scale <= 1e-8:
        return np.ones_like(e, dtype=np.float32)

    e = e / scale
    e = np.clip(e, 0.0, 2.0)

    # Avoid exact zero excitation.
    return (0.15 + 0.85 * e).astype(np.float32, copy=False)


def _find_channel(
    source: np.ndarray,
    names: Iterable[str],
    keywords: tuple[str, ...],
) -> np.ndarray | None:
    lowered = [str(name).lower().replace(" ", "_") for name in names]

    for idx, name in enumerate(lowered):
        if any(key in name for key in keywords):
            if idx < source.shape[0]:
                return source[idx]

    return None


def _estimate_rpm_series(
    source_raw: np.ndarray,
    names: Iterable[str],
    default_rpm: float,
    output_length: int,
) -> np.ndarray:
    rpm_keywords = (
        "rpm",
        "spindle_rpm",
        "spindle_speed",
        "spindle_actual_speed",
        "spindle_act",
        "sact",
    )

    rpm_row = _find_channel(source_raw, names, rpm_keywords)

    if rpm_row is None:
        try:
            rpm_scalar = float(_estimate_rpm(source_raw, names, default_rpm))
        except Exception:
            rpm_scalar = float(default_rpm)

        if not np.isfinite(rpm_scalar) or rpm_scalar <= 10.0:
            rpm_scalar = float(default_rpm)

        return np.full(output_length, rpm_scalar, dtype=np.float32)

    rpm = _finite(np.asarray(rpm_row, dtype=np.float32))
    rpm = np.nan_to_num(rpm, nan=default_rpm, posinf=default_rpm, neginf=default_rpm)

    positive = rpm[rpm > 10.0]
    if positive.size == 0:
        fill_value = float(default_rpm)
    else:
        fill_value = float(np.median(positive))

    rpm = np.where(rpm > 10.0, rpm, fill_value)
    rpm = np.clip(rpm, 1.0, 100000.0)

    return rpm.astype(np.float32, copy=False)


def _build_cutting_force_proxy(
    theta: np.ndarray,
    envelope: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    """Mechanistic milling-inspired cutting force proxy.

    This is not calibrated cutting force. It provides physically structured
    periodic excitation using tooth engagement and tangential/radial components.
    """
    tool_teeth = int(_cfg_get(cfg, "tool_teeth", 2))
    tool_teeth = max(1, tool_teeth)

    radial_force_ratio = float(_cfg_get(cfg, "radial_force_ratio", 0.35))
    axial_force_ratio = float(_cfg_get(cfg, "axial_force_ratio", 0.20))
    cutting_gain = float(_cfg_get(cfg, "cutting_gain", 1.0))

    force = np.zeros((3, theta.size), dtype=np.float32)

    for tooth_idx in range(tool_teeth):
        theta_j = theta + 2.0 * math.pi * tooth_idx / tool_teeth

        # Simplified chip thickness proxy.
        chip = np.maximum(np.sin(theta_j), 0.0).astype(np.float32)
        chip = chip * envelope

        # Tangential / radial force proxy.
        f_t = chip
        f_r = radial_force_ratio * f_t

        # Tool-frame to machine-frame simplified projection.
        f_x = -f_t * np.cos(theta_j) - f_r * np.sin(theta_j)
        f_y = f_t * np.sin(theta_j) - f_r * np.cos(theta_j)
        f_z = axial_force_ratio * f_t

        force[0] += f_x.astype(np.float32)
        force[1] += f_y.astype(np.float32)
        force[2] += f_z.astype(np.float32)

    force /= float(tool_teeth)
    force *= cutting_gain

    return force.astype(np.float32, copy=False)


def _build_rotational_force_proxy(
    theta: np.ndarray,
    rpm_series: np.ndarray,
    envelope: np.ndarray,
    axis_phase: tuple[float, float, float],
    cfg: Any,
) -> np.ndarray:
    """Spindle rotation / imbalance-inspired harmonic force proxy."""
    harmonics = _cfg_get(cfg, "force_harmonics", (1.0, 2.0, 3.0, 4.0))
    rotational_gain = float(_cfg_get(cfg, "rotational_gain", 0.35))

    rpm_positive = rpm_series[rpm_series > 10.0]
    if rpm_positive.size == 0:
        rpm_ref = 1.0
    else:
        rpm_ref = max(float(np.median(rpm_positive)), 1.0)

    # Unbalance force is proportional to omega^2.
    rpm_ratio_sq = (rpm_series / rpm_ref) ** 2
    rpm_ratio_sq = np.clip(rpm_ratio_sq, 0.0, 9.0).astype(np.float32)

    amp = envelope * rpm_ratio_sq

    force = np.zeros((3, theta.size), dtype=np.float32)

    for axis_index, phase in enumerate(axis_phase):
        axis_force = np.zeros(theta.size, dtype=np.float32)

        for harmonic_index, harmonic in enumerate(harmonics):
            k = float(harmonic)
            gain = 1.0 / float(harmonic_index + 1)

            axis_force += (
                gain
                * np.sin(k * theta + phase)
            ).astype(np.float32)

        force[axis_index] = rotational_gain * amp * axis_force

    return force.astype(np.float32, copy=False)


def _axis_load_channel(
    source: np.ndarray,
    names: Iterable[str],
    axis_name: str,
) -> np.ndarray | None:
    axis_name = axis_name.lower()

    keywords = (
        f"{axis_name}_load",
        f"{axis_name}load",
        f"{axis_name}_axis_load",
        f"axis_{axis_name}_load",
        f"{axis_name}_current",
        f"{axis_name}current",
        f"{axis_name}_torque",
        f"{axis_name}torque",
        f"servo_{axis_name}",
    )

    return _find_channel(source, names, keywords)


def _build_drive_force_proxy(
    source: np.ndarray,
    names: Iterable[str],
    drive: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    """CNC drive/load-based force proxy."""
    drive_gain = float(_cfg_get(cfg, "drive_gain", 0.55))

    spindle_load = _find_channel(
        source,
        names,
        (
            "spindle_load",
            "spindleload",
            "sp_load",
            "spload",
            "spindle_current",
            "spindle_torque",
        ),
    )

    feed = _find_channel(
        source,
        names,
        (
            "feed",
            "feedrate",
            "feed_rate",
            "actual_feed",
            "cmd_feed",
        ),
    )

    if spindle_load is None:
        spindle_load = np.zeros(source.shape[-1], dtype=np.float32)
    else:
        spindle_load = _standardize(spindle_load)

    if feed is None:
        feed = np.zeros(source.shape[-1], dtype=np.float32)
        d_feed = np.zeros(source.shape[-1], dtype=np.float32)
    else:
        feed = _standardize(feed)
        d_feed = np.diff(feed, prepend=feed[0])
        d_feed = _standardize(d_feed)

    drive = _standardize(drive)

    force = np.zeros((3, source.shape[-1]), dtype=np.float32)

    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axis_load = _axis_load_channel(source, names, axis_name)

        if axis_load is None:
            axis_load = np.zeros(source.shape[-1], dtype=np.float32)
        else:
            axis_load = _standardize(axis_load)

        shifted_drive = np.roll(
            drive,
            axis_index * max(1, source.shape[-1] // 17),
        )

        force[axis_index] = drive_gain * (
            0.40 * axis_load
            + 0.25 * spindle_load
            + 0.15 * feed
            + 0.10 * d_feed
            + 0.30 * shifted_drive
        )

    return force.astype(np.float32, copy=False)


def _get_axis_modal_params(
    cfg: Any,
    axis_name: str,
    axis_index: int,
    fs: float,
) -> list[dict[str, float]]:
    """Get modal parameters.

    Optional config example:
        modal_params = {
            "x": [
                {"fn": 180.0, "zeta": 0.055, "gain": 1.0},
                {"fn": 520.0, "zeta": 0.040, "gain": 0.35},
            ],
            "y": [...],
            "z": [...],
        }
    """
    modal_params = _cfg_get(cfg, "modal_params", None)

    raw_modes = None

    if isinstance(modal_params, dict):
        raw_modes = (
            modal_params.get(axis_name)
            or modal_params.get(axis_index)
            or modal_params.get(str(axis_index))
        )

    if raw_modes is None:
        # Safe fallback: modal frequencies below Nyquist.
        # These are not identified machine modal parameters.
        axis_factor = (1.00, 1.12, 0.85)[axis_index]

        raw_modes = [
            {
                "fn": 0.12 * fs * axis_factor,
                "zeta": 0.055,
                "gain": 1.00,
            },
            {
                "fn": 0.28 * fs * axis_factor,
                "zeta": 0.040,
                "gain": 0.35,
            },
        ]

    modes: list[dict[str, float]] = []

    for mode in raw_modes:
        if isinstance(mode, dict):
            fn = float(
                mode.get(
                    "fn",
                    mode.get(
                        "natural_frequency_hz",
                        mode.get("frequency_hz", 0.0),
                    ),
                )
            )
            zeta = float(
                mode.get(
                    "zeta",
                    mode.get("damping_ratio", 0.05),
                )
            )
            gain = float(
                mode.get(
                    "gain",
                    mode.get("modal_gain", 1.0),
                )
            )
        else:
            fn, zeta, gain = mode
            fn = float(fn)
            zeta = float(zeta)
            gain = float(gain)

        # Keep filter stable and below Nyquist.
        fn = max(1.0, min(fn, 0.45 * fs))
        zeta = max(1e-4, min(zeta, 1.5))

        modes.append(
            {
                "fn": fn,
                "zeta": zeta,
                "gain": gain,
            }
        )

    return modes


def _modal_acceleration_response(
    force: np.ndarray,
    fs: float,
    modes: list[dict[str, float]],
) -> np.ndarray:
    """Apply multi-modal acceleration response.

    Continuous model per mode:

        H_acc(s) = G*s^2 / (s^2 + 2*zeta*wn*s + wn^2)

    Discretized by bilinear transform without scipy.
    """
    x = np.asarray(force, dtype=np.float64)
    y_total = np.zeros_like(x, dtype=np.float64)

    c = 2.0 * fs

    for mode in modes:
        fn = float(mode["fn"])
        zeta = float(mode["zeta"])
        gain = float(mode["gain"])

        wn = 2.0 * math.pi * fn

        # Bilinear transform:
        # s = c * (1 - z^-1) / (1 + z^-1)
        #
        # H(s) = G*s^2 / (s^2 + 2*zeta*wn*s + wn^2)
        a0 = c * c + 2.0 * zeta * wn * c + wn * wn
        a1 = -2.0 * c * c + 2.0 * wn * wn
        a2 = c * c - 2.0 * zeta * wn * c + wn * wn

        b0 = gain * c * c
        b1 = -2.0 * gain * c * c
        b2 = gain * c * c

        if abs(a0) < 1e-12:
            continue

        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        y = np.zeros_like(x, dtype=np.float64)

        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0

        for n, x0 in enumerate(x):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

            y[n] = y0

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0

        y_total += y

    return y_total.astype(np.float32, copy=False)