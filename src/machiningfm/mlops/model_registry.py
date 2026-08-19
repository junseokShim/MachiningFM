from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machiningfm.utils.io import read_json, write_json

ALLOWED_STAGES = {"candidate", "validated", "staging", "production", "archived", "rejected"}


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def register(
        self,
        model_name: str,
        model_version: str,
        checkpoint_path: str | Path,
        metrics: dict[str, Any],
        **metadata: Any,
    ) -> dict[str, Any]:
        registry = read_json(self.path, {"models": []})
        existing = self.get(model_version, registry)
        promotion_status = metadata.pop(
            "promotion_status", existing.get("promotion_status", "candidate") if existing else "candidate"
        )
        api_serving = existing.get("api_serving", False) if existing else False
        entry = {
            "model_name": model_name,
            "model_version": model_version,
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "metrics": metrics,
            "promotion_status": promotion_status,
            "api_serving": api_serving,
            "registered_at": existing.get("registered_at") if existing else datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        if existing:
            existing.update(entry)
        else:
            registry["models"].append(entry)
        write_json(self.path, registry)
        return entry

    def promote(self, model_version: str, stage: str, allow_production: bool = False) -> dict[str, Any]:
        if stage not in ALLOWED_STAGES:
            raise ValueError(f"Unknown promotion stage: {stage}")
        if stage == "production" and not allow_production:
            raise PermissionError("Production promotion requires --allow_production")
        registry = read_json(self.path, {"models": []})
        entry = self.get(model_version, registry)
        if not entry:
            raise KeyError(f"Model version not found: {model_version}")
        if stage in {"validated", "staging", "production"} and not _benchmark_passed(entry):
            raise ValueError("Model cannot be promoted because regression benchmark did not pass")
        if stage in {"staging", "production"}:
            for model in registry["models"]:
                if model.get("promotion_status") == stage:
                    model["api_serving"] = False
        entry["promotion_status"] = stage
        entry["api_serving"] = stage in {"staging", "production"}
        entry["promoted_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.path, registry)
        return entry

    def serving_model(self, stage: str = "staging") -> dict[str, Any] | None:
        registry = read_json(self.path, {"models": []})
        values = [
            model for model in registry["models"] if model.get("promotion_status") == stage and model.get("api_serving")
        ]
        return values[-1] if values else None

    @staticmethod
    def get(model_version: str, registry: dict[str, Any]) -> dict[str, Any] | None:
        return next((model for model in registry.get("models", []) if model.get("model_version") == model_version), None)


def _benchmark_passed(entry: dict[str, Any]) -> bool:
    value = entry.get("metrics", {}).get("regression_benchmark_passed")
    return value is True or str(value).lower() == "true"
