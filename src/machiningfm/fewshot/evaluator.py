from __future__ import annotations

from typing import Any

from machiningfm.training.metrics import regression_metrics


class FewShotEvaluator:
    def evaluate(self, targets: list[Any], predictions: list[Any]) -> dict[str, float]:
        if targets and all(isinstance(value, (int, float)) for value in targets):
            return regression_metrics(targets, predictions)
        if not targets:
            return {"accuracy": 0.0}
        return {"accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets)}

    def performance_gap(self, result: float, baseline: float | None, higher_is_better: bool = False) -> float | None:
        if baseline is None:
            return None
        return (baseline - result) if higher_is_better else (result - baseline)
