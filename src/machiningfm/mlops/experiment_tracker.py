from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from machiningfm.utils.io import read_json, write_json


class ExperimentTracker:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def start(self, experiment_name: str, config: dict[str, Any]) -> str:
        registry = read_json(self.path, {"runs": []})
        run_id = f"{experiment_name}-{len(registry['runs']) + 1:04d}"
        registry["runs"].append(
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "config": config,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": None,
                "status": "running",
                "metrics": {},
                "artifacts": [],
                "error_message": None,
            }
        )
        write_json(self.path, registry)
        return run_id

    def finish(self, run_id: str, metrics: dict[str, Any], artifacts: list[str], error: str | None = None) -> None:
        registry = read_json(self.path, {"runs": []})
        for run in registry["runs"]:
            if run["run_id"] == run_id:
                run.update(
                    {
                        "end_time": datetime.now(timezone.utc).isoformat(),
                        "status": "failed" if error else "completed",
                        "metrics": metrics,
                        "artifacts": artifacts,
                        "error_message": error,
                    }
                )
        write_json(self.path, registry)
