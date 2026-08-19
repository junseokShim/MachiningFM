from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from machiningfm.utils.io import write_json

NC_CONTEXT_DIM = 128
NC_EXTENSIONS = {".nc", ".txt", ".mpf", ".tap", ".cnc", ".ncc", ".min"}
WORD_RE = re.compile(r"([A-Z])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
PROGRAM_RE = re.compile(r"\bO\s*(\d{3,6})\b", re.IGNORECASE)


def build_nc_context_cache(
    nc_root: str | Path,
    output_path: str | Path,
    *,
    context_dim: int = NC_CONTEXT_DIM,
) -> dict[str, Any]:
    root = Path(nc_root)
    if not root.exists():
        raise FileNotFoundError(f"NC code root does not exist: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in NC_EXTENSIONS:
            records.append(parse_nc_program(path, root=root, context_dim=context_dim))
    if not records:
        raise ValueError(f"No NC code files found under: {root}")
    vectors = np.asarray([record["context_vector"] for record in records], dtype=np.float32)
    global_vector = vectors.mean(axis=0)
    cache = {
        "schema_version": "nc-context-v1",
        "source_root": str(root.resolve()),
        "context_dim": int(context_dim),
        "builder": "deterministic_nc_parser_v1",
        "program_count": len(records),
        "global_context_vector": _normalize_vector(global_vector).tolist(),
        "programs": records,
    }
    write_json(output_path, cache)
    return cache


def parse_nc_program(path: str | Path, *, root: str | Path | None = None, context_dim: int = NC_CONTEXT_DIM) -> dict[str, Any]:
    source = Path(path)
    text = _read_text_lossy(source)
    blocks = _clean_blocks(text)
    stats = _program_stats(blocks)
    vector = _stats_to_vector(stats, context_dim=context_dim)
    program_ids = sorted(set(PROGRAM_RE.findall(text)))
    relative_path = str(source.relative_to(root)) if root else str(source)
    return {
        "file_path": str(source.resolve()),
        "relative_path": relative_path,
        "file_name": source.name,
        "stem": source.stem,
        "program_ids": program_ids,
        "matching_keys": _matching_keys(source, program_ids),
        "stats": stats,
        "context_vector": vector.tolist(),
        "teacher": {"model": None, "summary": None, "embedding": None},
    }


class NCContextStore:
    def __init__(self, cache: dict[str, Any], context_dim: int | None = None) -> None:
        self.cache = cache
        self.context_dim = int(context_dim or cache.get("context_dim") or NC_CONTEXT_DIM)
        global_vector = cache.get("global_context_vector") or [0.0] * self.context_dim
        self.global_context = _coerce_vector(global_vector, self.context_dim)
        self.by_key: dict[str, dict[str, Any]] = {}
        for program in cache.get("programs", []):
            for key in program.get("matching_keys", []):
                self.by_key[_normalize_key(key)] = program

    @classmethod
    def load(cls, path: str | Path | None, context_dim: int | None = None) -> "NCContextStore | None":
        if not path:
            return None
        source = Path(path)
        if not source.exists():
            return None
        return cls(json.loads(source.read_text(encoding="utf-8-sig")), context_dim=context_dim)

    def context_for_record(self, record: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        keys = _record_keys(record)
        for key in keys:
            program = self.by_key.get(_normalize_key(key))
            if program:
                vector = _coerce_vector(program.get("context_vector"), self.context_dim)
                return vector, {
                    "source": "matched_program",
                    "program_file": program.get("relative_path") or program.get("file_name"),
                    "matching_key": key,
                    "program_ids": program.get("program_ids", []),
                    "stats": program.get("stats", {}),
                }
        return self.global_context.copy(), {
            "source": "global_nc_corpus",
            "program_count": self.cache.get("program_count", 0),
            "source_root": self.cache.get("source_root"),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "context_dim": self.context_dim,
            "program_count": self.cache.get("program_count", 0),
            "source_root": self.cache.get("source_root"),
            "builder": self.cache.get("builder"),
        }


def _program_stats(blocks: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "block_count": len(blocks),
        "word_count": 0,
        "motion_counts": {"rapid": 0, "linear": 0, "cw_arc": 0, "ccw_arc": 0},
        "cycle_count": 0,
        "tool_change_count": 0,
        "coolant_on_count": 0,
        "coolant_off_count": 0,
        "spindle_on_count": 0,
        "spindle_stop_count": 0,
        "optional_stop_count": 0,
        "absolute_count": 0,
        "incremental_count": 0,
        "work_offset_count": 0,
        "feed_values": [],
        "spindle_values": [],
        "tool_values": [],
        "axis_min": {},
        "axis_max": {},
        "g_codes": {},
        "m_codes": {},
        "word_buckets": [0.0] * 32,
    }
    current_motion = None
    for block in blocks:
        words = [(letter.upper(), float(value)) for letter, value in WORD_RE.findall(block)]
        stats["word_count"] += len(words)
        seen_motion = False
        for letter, value in words:
            code = int(round(value))
            token = f"{letter}{code:g}"
            _bucket(stats["word_buckets"], token)
            if letter == "G":
                stats["g_codes"][str(code)] = stats["g_codes"].get(str(code), 0) + 1
                if code in {0, 1, 2, 3}:
                    current_motion = code
                    seen_motion = True
                if 80 <= code <= 89:
                    stats["cycle_count"] += 1
                if code == 90:
                    stats["absolute_count"] += 1
                if code == 91:
                    stats["incremental_count"] += 1
                if 54 <= code <= 59:
                    stats["work_offset_count"] += 1
            elif letter == "M":
                stats["m_codes"][str(code)] = stats["m_codes"].get(str(code), 0) + 1
                if code in {3, 4}:
                    stats["spindle_on_count"] += 1
                elif code == 5:
                    stats["spindle_stop_count"] += 1
                elif code in {6}:
                    stats["tool_change_count"] += 1
                elif code in {8, 7}:
                    stats["coolant_on_count"] += 1
                elif code == 9:
                    stats["coolant_off_count"] += 1
                elif code == 1:
                    stats["optional_stop_count"] += 1
            elif letter == "F":
                stats["feed_values"].append(float(value))
            elif letter == "S":
                stats["spindle_values"].append(float(value))
            elif letter == "T":
                stats["tool_values"].append(int(round(value)))
            elif letter in {"X", "Y", "Z", "A", "B", "C", "I", "J", "K", "R"}:
                name = letter.lower()
                stats["axis_min"][name] = min(float(value), stats["axis_min"].get(name, float(value)))
                stats["axis_max"][name] = max(float(value), stats["axis_max"].get(name, float(value)))
        if seen_motion or any(letter in {"X", "Y", "Z", "A", "B", "C"} for letter, _ in words):
            if current_motion == 0:
                stats["motion_counts"]["rapid"] += 1
            elif current_motion == 1:
                stats["motion_counts"]["linear"] += 1
            elif current_motion == 2:
                stats["motion_counts"]["cw_arc"] += 1
            elif current_motion == 3:
                stats["motion_counts"]["ccw_arc"] += 1
    stats["feed"] = _numeric_summary(stats.pop("feed_values"))
    stats["spindle"] = _numeric_summary(stats.pop("spindle_values"))
    tools = sorted(set(int(value) for value in stats.pop("tool_values")))
    stats["tools"] = {"count": len(tools), "values": tools[:64]}
    stats["axis_span"] = {
        axis: float(stats["axis_max"].get(axis, 0.0) - stats["axis_min"].get(axis, 0.0))
        for axis in sorted(set(stats["axis_min"]) | set(stats["axis_max"]))
    }
    return stats


def _stats_to_vector(stats: dict[str, Any], *, context_dim: int) -> np.ndarray:
    vector = np.zeros(context_dim, dtype=np.float32)
    motion = stats.get("motion_counts", {})
    motion_total = max(1.0, float(sum(motion.values())))
    scalars = [
        _log_scale(stats.get("block_count", 0), 11.0),
        _log_scale(stats.get("word_count", 0), 12.0),
        motion.get("rapid", 0) / motion_total,
        motion.get("linear", 0) / motion_total,
        motion.get("cw_arc", 0) / motion_total,
        motion.get("ccw_arc", 0) / motion_total,
        _log_scale(stats.get("cycle_count", 0), 5.0),
        _log_scale(stats.get("tool_change_count", 0), 5.0),
        _log_scale(stats.get("coolant_on_count", 0), 5.0),
        _log_scale(stats.get("spindle_on_count", 0), 5.0),
        _log_scale(stats.get("spindle_stop_count", 0), 5.0),
        _log_scale(stats.get("optional_stop_count", 0), 5.0),
        _log_scale(stats.get("absolute_count", 0), 5.0),
        _log_scale(stats.get("incremental_count", 0), 5.0),
        _log_scale(stats.get("work_offset_count", 0), 5.0),
        _log_scale(stats.get("tools", {}).get("count", 0), 4.0),
    ]
    feed = stats.get("feed", {})
    spindle = stats.get("spindle", {})
    scalars.extend(
        [
            _log_scale(feed.get("min", 0.0), 9.0),
            _log_scale(feed.get("max", 0.0), 9.0),
            _log_scale(feed.get("mean", 0.0), 9.0),
            _log_scale(spindle.get("min", 0.0), 11.0),
            _log_scale(spindle.get("max", 0.0), 11.0),
            _log_scale(spindle.get("mean", 0.0), 11.0),
        ]
    )
    for axis in ("x", "y", "z", "a", "b", "c", "i", "j", "k", "r"):
        scalars.append(math.tanh(float(stats.get("axis_span", {}).get(axis, 0.0)) / 500.0))
    count = min(len(scalars), context_dim)
    vector[:count] = np.asarray(scalars[:count], dtype=np.float32)
    offset = count
    for bucket_value in stats.get("word_buckets", []):
        if offset >= context_dim:
            break
        vector[offset] = _log_scale(bucket_value, 8.0)
        offset += 1
    for group_name in ("g_codes", "m_codes"):
        for code, value in sorted((stats.get(group_name) or {}).items(), key=lambda item: int(item[0])):
            _hash_add(vector, f"{group_name}:{code}", _log_scale(value, 7.0), start=offset)
    return _normalize_vector(vector)


def _clean_blocks(text: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", text)
    blocks = []
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip().upper()
        if not line:
            continue
        if line.startswith("%"):
            continue
        blocks.append(line)
    return blocks


def _read_text_lossy(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean())}


def _matching_keys(path: Path, program_ids: list[str]) -> list[str]:
    values = {path.name, path.stem}
    values.update(f"O{program_id}" for program_id in program_ids)
    values.update(program_ids)
    return sorted(_normalize_key(value) for value in values if value)


def _record_keys(record: dict[str, Any]) -> list[str]:
    values = []
    for name in ("file_path", "sample_id", "dataset_id", "split_group_key_candidate"):
        value = record.get(name)
        if value:
            values.extend([str(value), Path(str(value)).name, Path(str(value)).stem])
    return [_normalize_key(value) for value in values if value]


def _normalize_key(value: Any) -> str:
    text = str(value or "").lower().replace("\\", "/")
    text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\.[a-z0-9]+$", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _bucket(buckets: list[float], token: str) -> None:
    index = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % len(buckets)
    buckets[index] += 1.0


def _hash_add(vector: np.ndarray, key: str, value: float, *, start: int) -> None:
    if start >= vector.size:
        return
    index = start + int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % (vector.size - start)
    vector[index] += float(value)


def _log_scale(value: Any, denominator: float) -> float:
    try:
        numeric = abs(float(value))
    except (TypeError, ValueError):
        numeric = 0.0
    return min(1.0, math.log1p(numeric) / denominator)


def _coerce_vector(value: Any, context_dim: int) -> np.ndarray:
    vector = np.asarray(value if value is not None else [], dtype=np.float32).reshape(-1)
    if vector.size < context_dim:
        vector = np.pad(vector, (0, context_dim - vector.size))
    elif vector.size > context_dim:
        vector = vector[:context_dim]
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(data))
    if not math.isfinite(norm) or norm < 1e-6:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip(data / norm, -1.0, 1.0).astype(np.float32, copy=False)
