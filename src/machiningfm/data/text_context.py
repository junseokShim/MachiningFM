from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

TEXT_CONTEXT_SCHEMA_VERSION = "text-context-v1"
DEFAULT_TEXT_VOCAB_SIZE = 8192
DEFAULT_MAX_TEXT_TOKENS = 64

_TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z가-힣]+")
_TRAINING_BLOCKLIST = {
    "anomaly",
    "bad",
    "chatter",
    "fault",
    "good",
    "label",
    "labels",
    "roughness",
    "surface",
    "target",
    "vb",
    "wear",
}
_SAFE_OPERATION_WORDS = {
    "cutting",
    "drilling",
    "face",
    "finishing",
    "milling",
    "slot",
    "slotting",
    "turning",
}
_MATERIAL_PATTERNS = (
    re.compile(r"\baisi[_ -]?\d{3,4}\b", re.IGNORECASE),
    re.compile(r"\b(?:al|aa)[_ -]?(?:6061|7075|2024|5083)\b", re.IGNORECASE),
    re.compile(r"\b(?:aluminum|aluminium|steel|titanium|inconel|brass|copper|sus\d*|skd\d*|scm\d*)\b", re.IGNORECASE),
)


def encode_text_context(
    text: str | None,
    max_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
    vocab_size: int = DEFAULT_TEXT_VOCAB_SIZE,
) -> tuple[list[int], list[bool]]:
    tokens = tokenize_text(text or "")[:max_tokens]
    ids = [_stable_token_id(token, vocab_size) for token in tokens]
    mask = [True] * len(ids)
    pad = max(0, max_tokens - len(ids))
    return ids + [0] * pad, mask + [False] * pad


def tokenize_text(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_PATTERN.findall(normalized)


def build_pretraining_text_context(
    record: dict[str, Any],
    channel_descriptors: Iterable[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = ["cnc machining sample"]
    dataset_id = _safe_text(str(record.get("dataset_id") or ""))
    if dataset_id:
        parts.append(f"dataset {dataset_id}")
    for key in ("machine_id", "tool_id", "operation_id", "cut_id", "cycle_id"):
        value = record.get(key)
        if value not in (None, ""):
            parts.append(f"{key.replace('_', ' ')} {value}")
    path_context = _safe_path_context(record.get("file_path"))
    if path_context:
        parts.append(path_context)
    condition_context = _condition_text(record.get("process_condition") or {}, sanitize=True)
    if condition_context:
        parts.append(condition_context)
    descriptors = list(channel_descriptors or [])
    if descriptors:
        quantities = sorted({str(item.get("quantity")) for item in descriptors if item.get("quantity")})
        sources = sorted({str(item.get("source")) for item in descriptors if item.get("source")})
        if quantities:
            parts.append("sensor quantities " + " ".join(quantities))
        if sources:
            parts.append("sensor sources " + " ".join(sources))
    return _safe_text(" ".join(parts))


def build_request_text_context(request: dict[str, Any], canonical_sensor_names: Iterable[str] | None = None) -> str:
    parts: list[str] = []
    for key, label in (
        ("text_context", "context"),
        ("tool_info", "tool"),
        ("material_info", "material"),
        ("machine_info", "machine"),
        ("operation_info", "operation"),
        ("process_description", "process"),
    ):
        value = request.get(key)
        if value not in (None, "", {}):
            parts.append(f"{label}: {_stringify(value)}")
    condition = request.get("process_condition") or request.get("condition") or {}
    condition_text = _condition_text(condition, sanitize=False)
    if condition_text:
        parts.append(condition_text)
    sensor_names = list(canonical_sensor_names or [])
    if sensor_names:
        parts.append("sensors: " + " ".join(sensor_names))
    return " ".join(parts).strip()


def _stable_token_id(token: str, vocab_size: int) -> int:
    if vocab_size <= 1:
        return 0
    digest = hashlib.sha1(token.encode("utf-8", errors="replace")).digest()
    return 1 + int.from_bytes(digest[:4], "little") % (vocab_size - 1)


def _condition_text(condition: dict[str, Any], sanitize: bool) -> str:
    items = []
    for key in sorted(condition):
        value = condition[key]
        if value in (None, ""):
            continue
        items.append(f"{key} {value}")
    text = "process conditions " + " ".join(items) if items else ""
    return _safe_text(text) if sanitize else text


def _safe_path_context(path_value: Any) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    candidates: list[str] = []
    for part in path.parts:
        normalized = _normalize(part)
        if not normalized:
            continue
        if normalized in _SAFE_OPERATION_WORDS:
            candidates.append(normalized)
        if re.fullmatch(r"m\d+", normalized):
            candidates.append(f"machine {normalized}")
        if re.fullmatch(r"op\d+", normalized):
            candidates.append(f"operation {normalized}")
        if re.fullmatch(r"t\d+", normalized):
            candidates.append(f"tool {normalized}")
        for pattern in _MATERIAL_PATTERNS:
            candidates.extend(pattern.findall(normalized))
        match = re.search(r"\bm(\d+)t(\d+)r(\d+)", normalized)
        if match:
            machine, tool, run = match.groups()
            candidates.append(f"machine m{machine}")
            candidates.append(f"tool t{tool}")
            candidates.append(f"run r{run}")
    return _safe_text(" ".join(candidates))


def _safe_text(text: str) -> str:
    tokens = [token for token in tokenize_text(text) if token not in _TRAINING_BLOCKLIST]
    return " ".join(tokens)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "_", normalized).strip("_")


def _stringify(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in sorted(value.items()) if item not in (None, ""))
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item not in (None, ""))
    return str(value)
