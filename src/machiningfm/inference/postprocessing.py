from __future__ import annotations

from typing import Any


def postprocess_prediction(task: str, prediction: Any) -> Any:
    if task.endswith("classification") or task == "chatter_detection":
        if isinstance(prediction, list):
            return max(range(len(prediction)), key=prediction.__getitem__)
    return prediction
