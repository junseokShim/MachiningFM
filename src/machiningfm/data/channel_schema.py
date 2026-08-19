from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable

CHANNEL_SCHEMA_VERSION = "cnc-v1"
CHANNEL_ATTRIBUTE_NAMES = ("quantity", "axis", "source", "component", "representation")

QUANTITIES = (
    "unknown",
    "vibration",
    "acoustic_emission",
    "force",
    "torque",
    "current",
    "voltage",
    "power",
    "energy",
    "load",
    "sound",
    "temperature",
    "pressure",
    "flow",
    "position",
    "position_error",
    "velocity",
    "speed",
    "feed_rate",
    "depth_of_cut",
    "width_of_cut",
    "tool_diameter",
    "tool_wear",
    "surface_roughness",
)
AXES = (
    "unknown",
    "none",
    "x",
    "y",
    "z",
    "u",
    "v",
    "w",
    "a",
    "b",
    "c",
    "radial",
    "axial",
    "tangential",
    *(f"channel_{index}" for index in range(32)),
)
SOURCES = (
    "unknown",
    "external_sensor",
    "machine_controller",
    "process_condition",
    "derived_feature",
)
COMPONENTS = (
    "unknown",
    "machine",
    "spindle",
    "tool",
    "workpiece",
    "table",
    "axis_drive",
    "motor",
    "fixture",
    "coolant",
    "environment",
)
REPRESENTATIONS = (
    "unknown",
    "raw",
    "actual",
    "command",
    "setpoint",
    "rms",
    "mean",
    "std",
    "min",
    "max",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
    "spectrum",
    "energy",
)

CHANNEL_ATTRIBUTE_VOCABULARIES = {
    "quantity": QUANTITIES,
    "axis": AXES,
    "source": SOURCES,
    "component": COMPONENTS,
    "representation": REPRESENTATIONS,
}
CHANNEL_ATTRIBUTE_VOCAB_SIZES = tuple(
    len(CHANNEL_ATTRIBUTE_VOCABULARIES[name]) for name in CHANNEL_ATTRIBUTE_NAMES
)
_ATTRIBUTE_LOOKUPS = {
    name: {value: index for index, value in enumerate(values)}
    for name, values in CHANNEL_ATTRIBUTE_VOCABULARIES.items()
}

_QUANTITY_ALIASES = (
    ("position_error", ("position_control_deviation", "position_deviation", "position_error", "following_error")),
    ("surface_roughness", ("surface_roughness", "roughness", "ra")),
    ("acoustic_emission", ("acoustic_emission", "acoustic", "emission", "ae")),
    ("depth_of_cut", ("depth_of_cut", "doc", "ap")),
    ("width_of_cut", ("width_of_cut", "woc", "ae_width")),
    ("tool_diameter", ("tool_diameter", "diameter")),
    ("tool_wear", ("tool_wear", "wear", "vb")),
    ("feed_rate", ("feed_rate", "feedrate", "feed")),
    ("vibration", ("vibration", "vib", "accelerometer", "acceleration", "acc")),
    ("force", ("force", "fx", "fy", "fz")),
    ("torque", ("torque", "moment", "mx", "my", "mz")),
    ("current", ("current", "ampere", "amps")),
    ("voltage", ("voltage", "volt")),
    ("power", ("power", "watt")),
    ("energy", ("energy", "joule")),
    ("load", ("load",)),
    ("sound", ("sound", "audio", "microphone", "mic")),
    ("temperature", ("temperature", "temp")),
    ("pressure", ("pressure",)),
    ("flow", ("flow",)),
    ("position", ("position", "displacement")),
    ("velocity", ("velocity",)),
    ("speed", ("spindle_speed", "rotational_speed", "rpm", "speed")),
)
_REPRESENTATION_ALIASES = (
    ("peak_to_peak", ("peak_to_peak", "peak2peak", "p2p", "ptp")),
    ("crest_factor", ("crest_factor", "crestfactor")),
    ("kurtosis", ("kurtosis", "kurt")),
    ("skewness", ("skewness", "skew")),
    ("spectrum", ("spectrogram", "spectrum", "fft", "stft", "cwt", "frequency")),
    ("setpoint", ("setpoint", "set_point")),
    ("command", ("command", "cmd", "reference")),
    ("actual", ("actual", "measured")),
    ("rms", ("rms",)),
    ("std", ("std", "standard_deviation")),
    ("mean", ("mean", "average", "avg")),
    ("min", ("minimum", "min")),
    ("max", ("maximum", "max")),
    ("peak", ("peak",)),
    ("energy", ("energy",)),
)
_DERIVED_REPRESENTATIONS = {
    "rms",
    "mean",
    "std",
    "min",
    "max",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
    "spectrum",
    "energy",
}
_PROCESS_CONDITION_QUANTITIES = {
    "feed_rate",
    "depth_of_cut",
    "width_of_cut",
    "tool_diameter",
}
_EXTERNAL_SENSOR_QUANTITIES = {
    "vibration",
    "acoustic_emission",
    "force",
    "sound",
    "temperature",
    "pressure",
    "flow",
}
_VECTOR_QUANTITIES = {
    "vibration",
    "force",
    "torque",
    "position",
    "position_error",
    "velocity",
}
_PRETRAINING_EXCLUDED_QUANTITIES = {
    "depth_of_cut",
    "surface_roughness",
    "tool_diameter",
    "tool_wear",
    "width_of_cut",
}
_METADATA_NAME_PATTERNS = (
    r"(?:^|_)(?:row|sample|block)_?index(?:_|$)",
    r"(?:^|_)(?:time|timestamp|received_at|created_at|start_time|end_time)(?:_|$)",
    r"(?:^|_)(?:gcode|mcode|filename|source_file|path)(?:_|$)",
    r"(?:^|_)(?:tool|part|machine)_(?:id|number|db_number)(?:_|$)",
    r"(?:^|_)(?:label|labels|target|class|dummy|unnamed)(?:_|$)",
)
_EXACT_CANONICAL_NAMES = {
    "vx": "external_sensor.unknown.vibration.x.raw",
    "vy": "external_sensor.unknown.vibration.y.raw",
    "vz": "external_sensor.unknown.vibration.z.raw",
    "acc1_value": "external_sensor.unknown.vibration.x.raw",
    "acc2_value": "external_sensor.unknown.vibration.y.raw",
    "acc3_value": "external_sensor.unknown.vibration.z.raw",
    "ai1_01": "external_sensor.unknown.vibration.x.raw",
    "ai1_02": "external_sensor.unknown.vibration.y.raw",
    "ai1_03": "external_sensor.unknown.vibration.z.raw",
    "ai1_07": "external_sensor.unknown.pressure.none.raw",
    "iu": "machine_controller.motor.current.u.actual",
    "iv": "machine_controller.motor.current.v.actual",
    "iw": "machine_controller.motor.current.w.actual",
    "current_u": "machine_controller.motor.current.u.actual",
    "current_v": "machine_controller.motor.current.v.actual",
    "current_w": "machine_controller.motor.current.w.actual",
    "rpm": "machine_controller.spindle.speed.none.actual",
    "sn": "machine_controller.spindle.speed.none.actual",
    "spindle_load": "machine_controller.spindle.load.none.actual",
    "actload": "machine_controller.spindle.load.none.actual",
    "actual_rpm": "machine_controller.spindle.speed.none.actual",
    "command_rpm": "machine_controller.spindle.speed.none.command",
    "actual_feedrate": "machine_controller.machine.feed_rate.none.actual",
    "command_feedrate": "machine_controller.machine.feed_rate.none.command",
    "abs_x": "machine_controller.axis_drive.position.x.actual",
    "abs_y": "machine_controller.axis_drive.position.y.actual",
    "abs_z": "machine_controller.axis_drive.position.z.actual",
    "mechanical_x": "machine_controller.axis_drive.position.x.actual",
    "mechanical_y": "machine_controller.axis_drive.position.y.actual",
    "mechanical_z": "machine_controller.axis_drive.position.z.actual",
    "spindle_x": "external_sensor.spindle.vibration.x.raw",
    "virtual_spindle_vibration_x": "external_sensor.spindle.vibration.x.raw",
    "virtual_spindle_vibration_y": "external_sensor.spindle.vibration.y.raw",
    "virtual_spindle_vibration_z": "external_sensor.spindle.vibration.z.raw",
}


@dataclass(frozen=True)
class ChannelDescriptor:
    raw_name: str
    canonical_name: str
    quantity: str
    axis: str
    source: str
    component: str
    representation: str

    def attribute_ids(self) -> list[int]:
        return [
            _ATTRIBUTE_LOOKUPS[name][getattr(self, name)]
            for name in CHANNEL_ATTRIBUTE_NAMES
        ]

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def describe_channel(name: str) -> ChannelDescriptor:
    raw_name = str(name)
    canonical = _parse_canonical_name(raw_name)
    if canonical is not None:
        return canonical
    normalized = _normalize_name(raw_name)
    exact = _EXACT_CANONICAL_NAMES.get(normalized)
    if exact is not None:
        return _descriptor_from_canonical(raw_name, exact)
    tokens = set(normalized.split("_")) if normalized else set()
    quantity = _infer_quantity(normalized, tokens)
    axis = _infer_axis(raw_name, normalized, tokens, quantity)
    component = _infer_component(normalized, tokens)
    explicit_representation = _infer_representation(normalized, tokens)
    source = _infer_source(normalized, tokens, quantity, explicit_representation)
    representation = explicit_representation or ("actual" if source == "machine_controller" else "raw")

    parts = [source, component, quantity, axis, representation]
    if quantity == "unknown":
        parts.append(normalized[-64:] or _short_hash(raw_name))
    canonical_name = ".".join(parts)
    return ChannelDescriptor(
        raw_name=raw_name,
        canonical_name=canonical_name,
        quantity=quantity,
        axis=axis,
        source=source,
        component=component,
        representation=representation,
    )


def stable_channel_id(name: str, vocabulary_size: int) -> int:
    """Return a stable ID for the CNC-standardized channel identity."""
    canonical_name = describe_channel(name).canonical_name
    return _stable_text_id(canonical_name, vocabulary_size)


def is_pretraining_signal_channel(name: str) -> bool:
    """Return whether a numeric channel is a recognized non-label time series."""
    if is_metadata_channel_name(name):
        return False
    descriptor = describe_channel(name)
    return bool(
        descriptor.quantity != "unknown"
        and descriptor.quantity not in _PRETRAINING_EXCLUDED_QUANTITIES
        and descriptor.source != "derived_feature"
        and descriptor.representation not in _DERIVED_REPRESENTATIONS
    )


def is_metadata_channel_name(name: str) -> bool:
    normalized = _normalize_name(name)
    return any(re.search(pattern, normalized) for pattern in _METADATA_NAME_PATTERNS)


def canonical_signal_names(names: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(
            describe_channel(name).canonical_name
            for name in names
            if is_pretraining_signal_channel(name)
        )
    )


def find_recognized_header_index(lines: Iterable[str]) -> int | None:
    for index, line in enumerate(lines):
        fields = [field.strip() for field in re.split(r"[,;\t]", line)]
        if len(fields) >= 2 and any(is_pretraining_signal_channel(field) for field in fields):
            return index
    return None


def encode_channel_names(names: Iterable[str], vocabulary_size: int) -> dict[str, Any]:
    descriptors = [describe_channel(name) for name in names]
    return {
        "schema_version": CHANNEL_SCHEMA_VERSION,
        "canonical_names": [descriptor.canonical_name for descriptor in descriptors],
        "sensor_ids": [
            _stable_text_id(descriptor.canonical_name, vocabulary_size)
            for descriptor in descriptors
        ],
        "attribute_ids": [descriptor.attribute_ids() for descriptor in descriptors],
        "descriptors": [descriptor.to_dict() for descriptor in descriptors],
    }


def channel_schema_metadata() -> dict[str, Any]:
    return {
        "version": CHANNEL_SCHEMA_VERSION,
        "canonical_name_format": "source.component.quantity.axis.representation",
        "attribute_names": list(CHANNEL_ATTRIBUTE_NAMES),
        "pretraining_policy": {
            "unknown_channels": "excluded",
            "derived_features": "registered_but_excluded_from_raw_signal_pretraining",
            "excluded_quantities": sorted(_PRETRAINING_EXCLUDED_QUANTITIES),
        },
        "attribute_vocabularies": {
            name: list(values) for name, values in CHANNEL_ATTRIBUTE_VOCABULARIES.items()
        },
    }


def _normalize_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", name)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = value.lower().replace("\\", "/")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _parse_canonical_name(name: str) -> ChannelDescriptor | None:
    parts = name.split(".")
    if len(parts) < 5:
        return None
    source, component, quantity, axis, representation = parts[:5]
    if (
        source not in SOURCES
        or component not in COMPONENTS
        or quantity not in QUANTITIES
        or axis not in AXES
        or representation not in REPRESENTATIONS
    ):
        return None
    return ChannelDescriptor(
        raw_name=name,
        canonical_name=name,
        quantity=quantity,
        axis=axis,
        source=source,
        component=component,
        representation=representation,
    )


def _descriptor_from_canonical(raw_name: str, canonical_name: str) -> ChannelDescriptor:
    descriptor = _parse_canonical_name(canonical_name)
    if descriptor is None:
        raise ValueError(f"Invalid canonical channel override: {canonical_name}")
    return ChannelDescriptor(
        raw_name=raw_name,
        canonical_name=canonical_name,
        quantity=descriptor.quantity,
        axis=descriptor.axis,
        source=descriptor.source,
        component=descriptor.component,
        representation=descriptor.representation,
    )


def _infer_quantity(normalized: str, tokens: set[str]) -> str:
    for quantity, aliases in _QUANTITY_ALIASES:
        if _contains_any(normalized, tokens, aliases):
            return quantity
    return "unknown"


def _infer_axis(raw_name: str, normalized: str, tokens: set[str], quantity: str) -> str:
    for axis in ("radial", "axial", "tangential"):
        if axis in tokens:
            return axis
    for axis in ("x", "y", "z", "u", "v", "w", "a", "b", "c"):
        if axis in tokens or re.search(rf"(?:axis|sensor|force|torque|vibration|position|velocity)_{axis}(?:_|$)", normalized):
            return axis
    compact_axes = {
        "fx": "x",
        "fy": "y",
        "fz": "z",
        "mx": "x",
        "my": "y",
        "mz": "z",
        "ax": "x",
        "ay": "y",
        "az": "z",
        "iu": "u",
        "iv": "v",
        "iw": "w",
    }
    for token, axis in compact_axes.items():
        if token in tokens:
            return axis
    match = re.search(r"(?::|[_-])(\d+)$", raw_name) or re.search(r"(?:^|_)(\d+)$", normalized)
    if match:
        index = int(match.group(1))
        if quantity in _VECTOR_QUANTITIES and index < 3:
            return ("x", "y", "z")[index]
        if index < 32:
            return f"channel_{index}"
    return "none"


def _infer_component(normalized: str, tokens: set[str]) -> str:
    if "spindle" in tokens:
        return "spindle"
    if _contains_any(normalized, tokens, ("workpiece", "work_piece", "part")):
        return "workpiece"
    if "tool" in tokens:
        return "tool"
    if _contains_any(normalized, tokens, ("axis_drive", "feed_drive", "servo", "axis")):
        return "axis_drive"
    for component in ("table", "motor", "fixture", "coolant", "environment", "machine"):
        if component in tokens:
            return component
    return "unknown"


def _infer_representation(normalized: str, tokens: set[str]) -> str | None:
    for representation, aliases in _REPRESENTATION_ALIASES:
        if _contains_any(normalized, tokens, aliases):
            return representation
    return None


def _infer_source(
    normalized: str,
    tokens: set[str],
    quantity: str,
    representation: str | None,
) -> str:
    if representation in _DERIVED_REPRESENTATIONS:
        return "derived_feature"
    if quantity in _PROCESS_CONDITION_QUANTITIES:
        return "process_condition"
    if _contains_any(normalized, tokens, ("signals_machine", "machine_signal", "controller", "cnc", "plc")):
        return "machine_controller"
    if _contains_any(
        normalized,
        tokens,
        ("signals_sensor", "sensor", "accelerometer", "microphone", "dynamometer"),
    ):
        return "external_sensor"
    if quantity in _EXTERNAL_SENSOR_QUANTITIES:
        return "external_sensor"
    if quantity != "unknown":
        return "machine_controller"
    return "unknown"


def _contains_any(normalized: str, tokens: set[str], aliases: Iterable[str]) -> bool:
    return any(alias in normalized if "_" in alias else alias in tokens for alias in aliases)


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:12]


def _stable_text_id(value: str, vocabulary_size: int) -> int:
    if vocabulary_size <= 1:
        return 0
    digest = hashlib.sha1(value.encode("utf-8", errors="replace")).digest()
    return 1 + int.from_bytes(digest[:4], "little") % (vocabulary_size - 1)
