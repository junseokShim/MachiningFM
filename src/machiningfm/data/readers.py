from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_delimited_sample(path: str | Path, max_rows: int = 256) -> list[dict[str, Any]]:
    source = Path(path)
    for encoding in ("utf-8-sig", "cp949", "latin-1"):
        try:
            sample = source.read_bytes()[:8192].decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return []
    with source.open("r", encoding=encoding, newline="", errors="replace") as handle:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [row for _, row in zip(range(max_rows), reader)]


def read_sample(path: str | Path, max_rows: int = 256) -> Any:
    source = Path(path)
    if source.suffix.lower() in {".csv", ".txt"}:
        return read_delimited_sample(source, max_rows)
    if source.suffix.lower() in {".json", ".jsonl"}:
        with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
            values = []
            for _, line in zip(range(max_rows), handle):
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return values
    if source.suffix.lower() in {".npy", ".npz"}:
        import numpy as np

        return np.load(source, mmap_mode="r", allow_pickle=False)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        return h5py.File(source, "r")
    raise ValueError(f"No dependency-free reader for {source.suffix}")
