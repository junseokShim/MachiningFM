from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machiningfm.utils.io import read_json, write_json


def register_dataset_version(
    dataset_id: str,
    snapshot: dict[str, Any],
    versions_path: str | Path,
) -> dict[str, Any]:
    path = Path(versions_path)
    registry = read_json(path, {"datasets": {}})
    versions = registry["datasets"].setdefault(dataset_id, [])
    digest = hashlib.sha256(str(snapshot).encode("utf-8")).hexdigest()
    if versions and versions[-1]["snapshot_hash"] == digest:
        return versions[-1]
    entry = {
        "dataset_id": dataset_id,
        "dataset_version": f"v{len(versions) + 1}",
        "snapshot_hash": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": snapshot.get("summary", snapshot),
    }
    versions.append(entry)
    write_json(path, registry)
    return entry


def register_manifest_version(
    manifest_path: str | Path,
    dataset_versions: dict[str, str],
    versions_path: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    digest = _hash_file(manifest)
    path = Path(versions_path)
    registry = read_json(path, {"manifests": []})
    for entry in registry["manifests"]:
        if entry["manifest_hash"] == digest:
            return entry
    entry = {
        "manifest_version": f"v{len(registry['manifests']) + 1}",
        "manifest_path": str(manifest),
        "manifest_hash": digest,
        "dataset_versions": dataset_versions,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["manifests"].append(entry)
    write_json(path, registry)
    return entry


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
