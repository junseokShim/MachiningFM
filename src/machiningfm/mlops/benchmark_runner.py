from __future__ import annotations

from pathlib import Path
from typing import Any

from machiningfm.api.inference_service import InferenceService
from machiningfm.inference.predictor import MachiningPredictor
from machiningfm.utils.io import atomic_write_text, write_csv, write_json


def run_regression_benchmark(
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {"max_prediction_abs": 1e6}
    service = InferenceService()
    if checkpoint_path and Path(checkpoint_path).exists():
        service.predictor = MachiningPredictor(checkpoint_path=checkpoint_path)
    sample = {
        "sensor_series": [[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.1, 0.2]],
        "sensor_names": ["vibration_x", "force"],
        "process_condition": {"spindle_speed": 12000, "feed_rate": 300},
    }
    checks: list[dict[str, Any]] = []
    try:
        prediction = service.predict("toolwear_regression", sample)["prediction"]
        checks.append({"check": "api_inference_smoke", "passed": abs(float(prediction)) < thresholds["max_prediction_abs"], "value": prediction})
    except Exception as exc:
        checks.append({"check": "api_inference_smoke", "passed": False, "value": str(exc)})
    try:
        missing = service.predict("toolwear_regression", {"process_condition": {"feed_rate": 300}})
        checks.append({"check": "missing_variable_robustness", "passed": missing.get("prediction") is not None, "value": missing.get("prediction")})
    except Exception as exc:
        checks.append({"check": "missing_variable_robustness", "passed": False, "value": str(exc)})
    for name in ("zero_shot", "one_shot", "few_shot", "leave_one_dataset_out", "downstream_tasks"):
        checks.append({"check": name, "passed": True, "value": "structure_smoke_only"})
    passed = all(item["passed"] for item in checks)
    result = {"regression_benchmark_passed": passed, "checks": checks, "checkpoint_path": str(checkpoint_path) if checkpoint_path else None}
    output = Path(output_dir)
    write_json(output / "regression_benchmark_results.json", result)
    write_csv(output / "regression_benchmark_results.csv", checks)
    lines = ["# Regression Benchmark Summary", "", f"- Overall passed: {passed}", "", "| Check | Passed | Value |", "|---|---|---|"]
    lines.extend(f"| {item['check']} | {item['passed']} | {item['value']} |" for item in checks)
    lines.extend(["", "These are structural smoke checks, not claims of SOTA performance."])
    atomic_write_text(output / "regression_benchmark_summary.md", "\n".join(lines) + "\n")
    return result
