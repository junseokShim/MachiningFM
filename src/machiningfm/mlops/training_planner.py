from __future__ import annotations

from pathlib import Path
from typing import Any

from machiningfm.utils.config import load_config
from machiningfm.utils.io import atomic_write_text, read_records


def build_training_plan(registry_path: str | Path, manifest_path: str | Path, output_path: str | Path) -> str:
    registry = load_config(registry_path)
    manifest = read_records(manifest_path)
    sample_counts: dict[str, int] = {}
    for row in manifest:
        dataset_id = str(row.get("dataset_id", "unknown"))
        sample_counts[dataset_id] = sample_counts.get(dataset_id, 0) + 1
    lines = [
        "# MachiningFM Training Plan",
        "",
        "This plan is generated before training. It does not start a large training run.",
        "",
        "| Dataset | Samples | Modalities | Tasks | Recommendation | Risks |",
        "|---|---:|---|---|---|---|",
    ]
    for dataset in registry.get("datasets", []):
        recommendation = _recommendation(dataset)
        risks = []
        if dataset.get("synthetic_suspected"):
            risks.append("synthetic suspected")
        if not dataset.get("available_modalities"):
            risks.append("modality unknown")
        if not dataset.get("downstream_tasks"):
            risks.append("labels/tasks unknown")
        risks.append("verify units, sampling rate, split leakage")
        lines.append(
            f"| {dataset['dataset_name']} | {sample_counts.get(dataset['dataset_id'], 0)} | "
            f"{', '.join(dataset.get('available_modalities', [])) or 'unknown'} | "
            f"{', '.join(dataset.get('downstream_tasks', [])) or 'self-supervised only'} | "
            f"{recommendation} | {', '.join(risks)} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Sequence",
            "",
            "1. Validate newly added HDF5/CSV schemas and split groups.",
            "2. Run smoke_test_only with representative windows.",
            "3. Use incremental_pretraining with replay after regression benchmarks exist.",
            "4. Use full_retraining only for major modality/schema changes or scheduled releases.",
        ]
    )
    text = "\n".join(lines) + "\n"
    atomic_write_text(output_path, text)
    return text


def _recommendation(dataset: dict[str, Any]) -> str:
    if dataset.get("raw_available") and dataset.get("available_modalities"):
        return "smoke_test_only, then incremental_pretraining"
    if dataset.get("downstream_tasks"):
        return "downstream_only"
    return "no_train pending manual review"
