from __future__ import annotations

from typing import Any

from .missing import build_modality_mask


def multimodal_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set().union(*(sample.keys() for sample in samples))
    batch = {key: [sample.get(key) for sample in samples] for key in keys}
    batch["modality_masks"] = [build_modality_mask(sample) for sample in samples]
    return batch
