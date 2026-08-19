from __future__ import annotations


class Callback:
    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        return None
