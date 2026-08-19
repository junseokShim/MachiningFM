from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


GROUP_COLUMNS = ("run_id", "tool_id", "machine_id", "material", "condition_id", "site", "dataset_id")


def split_by_complete_group(
    manifest: pd.DataFrame,
    *,
    holdout_column: str = "run_id",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> pd.DataFrame:
    if holdout_column not in manifest.columns:
        available = [column for column in GROUP_COLUMNS if column in manifest.columns]
        holdout_column = available[0] if available else "file_path"
    frame = manifest.copy()
    groups = frame[holdout_column].fillna(frame.get("file_path", "")).astype(str)
    splits = []
    for group in groups:
        digest = int(hashlib.sha1(group.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if digest < test_ratio:
            splits.append("test")
        elif digest < test_ratio + val_ratio:
            splits.append("val")
        else:
            splits.append("train")
    frame["split"] = splits
    frame["split_group_column"] = holdout_column
    return frame


def write_split_protocol(manifest_path: str | Path, output_path: str | Path, holdout_column: str = "run_id") -> Path:
    manifest = pd.read_parquet(manifest_path) if str(manifest_path).endswith(".parquet") else pd.read_csv(manifest_path)
    split = split_by_complete_group(manifest, holdout_column=holdout_column)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        split.to_parquet(output, index=False)
    else:
        split.to_csv(output, index=False)
    return output
