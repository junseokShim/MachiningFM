from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import atomic_write_text


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{config_path} is YAML, but PyYAML is not installed. "
                "Install requirements.txt or use JSON-compatible YAML."
            ) from exc
        value = yaml.safe_load(text)
        return value or {}


def save_config(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    try:
        import yaml
    except ImportError:
        atomic_write_text(output, json.dumps(value, indent=2, ensure_ascii=False))
    else:
        atomic_write_text(output, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
