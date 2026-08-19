from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_registry import ModelRegistry


def promote_model(
    registry_path: str | Path,
    model_version: str,
    target_stage: str,
    allow_production: bool = False,
) -> dict[str, Any]:
    return ModelRegistry(registry_path).promote(model_version, target_stage, allow_production)
