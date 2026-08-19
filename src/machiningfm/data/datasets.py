from __future__ import annotations

from typing import Any, Sequence


class ManifestDataset:
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]
