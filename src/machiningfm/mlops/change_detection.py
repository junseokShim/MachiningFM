from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machiningfm.utils.io import atomic_write_text, read_json, write_json


def detect_dataset_changes(
    data_root: str | Path,
    mlops_dir: str | Path,
    strict: bool = False,
    hash_large_files: bool = False,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    datasets_dir = Path(mlops_dir) / "datasets"
    snapshot_path = datasets_dir / "latest_file_snapshot.json"
    previous = read_json(snapshot_path, {"files": {}})
    current_files: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        relative = str(path.relative_to(root))
        item = {"size": stat.st_size, "modified_time_ns": stat.st_mtime_ns}
        if strict and (hash_large_files or stat.st_size <= 256 * 1024 * 1024):
            item["sha256"] = _hash_file(path)
        current_files[relative] = item
    previous_files = previous.get("files", {})
    added = sorted(set(current_files) - set(previous_files))
    deleted = sorted(set(previous_files) - set(current_files))
    modified = sorted(
        path for path in set(current_files) & set(previous_files) if current_files[path] != previous_files[path]
    )
    unchanged_count = len(current_files) - len(added) - len(modified)
    now = datetime.now(timezone.utc).isoformat()
    result = {
        "created_at": now,
        "data_root": str(root),
        "strict": strict,
        "summary": {
            "added": len(added),
            "modified": len(modified),
            "deleted": len(deleted),
            "unchanged": unchanged_count,
        },
        "added": added,
        "modified": modified,
        "deleted": deleted,
    }
    write_json(snapshot_path, {"created_at": now, "data_root": str(root), "files": current_files})
    atomic_write_text(datasets_dir / "data_change_log.md", _markdown(result))
    return result


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Data Change Log",
        "",
        f"- Created: {result['created_at']}",
        f"- Root: `{result['data_root']}`",
        f"- Strict hashing: {result['strict']}",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in result["summary"].items())
    for category in ("added", "modified", "deleted"):
        lines.extend(["", f"## {category.title()}", ""])
        values = result[category]
        lines.extend(f"- `{value}`" for value in values[:100])
        if len(values) > 100:
            lines.append(f"- ... {len(values) - 100} more")
    return "\n".join(lines) + "\n"
