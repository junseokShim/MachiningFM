from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def atomic_write_text(path: str | Path, text: str) -> Path:
    output = ensure_parent(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=output.parent
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, output)
    return output


def write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str))


def read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.exists():
        return default
    return json.loads(source.read_text(encoding="utf-8-sig"))


def write_csv(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    rows = list(records)
    output = ensure_parent(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(value) for key, value in row.items()})
    return output


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    output = ensure_parent(path)
    if output.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            parquet_rows = [
                {
                    key: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (list, dict, tuple, set))
                    else value
                    for key, value in row.items()
                }
                for row in rows
            ]
            pd.DataFrame(parquet_rows).to_parquet(output, index=False)
            return "parquet"
        except Exception:
            text = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
            atomic_write_text(output, text + ("\n" if text else ""))
            write_json(output.with_suffix(".format.json"), {"format": "jsonl_fallback"})
            return "jsonl_fallback"
    if output.suffix.lower() == ".csv":
        write_csv(output, rows)
        return "csv"
    write_json(output, rows)
    return "json"


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return read_csv(source)
    if source.suffix.lower() == ".parquet":
        try:
            import pandas as pd

            return [_decode_structured_cells(row) for row in pd.read_parquet(source).to_dict(orient="records")]
        except Exception:
            return list(iter_jsonl(source))
    value = read_json(source, [])
    return value if isinstance(value, list) else []


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _decode_structured_cells(row: dict[str, Any]) -> dict[str, Any]:
    decoded = {}
    for key, value in row.items():
        if isinstance(value, str) and value[:1] in {"[", "{"}:
            try:
                decoded[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        decoded[key] = value
    return decoded
