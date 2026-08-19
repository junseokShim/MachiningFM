#!/usr/bin/env python3
"""
Download and validate public machining datasets.

Usage:
    # Show download instructions for PHM2010:
    python scripts/download_dataset.py --dataset phm2010 --output data/raw/phm2010

    # Validate existing PHM2010 download:
    python scripts/download_dataset.py --dataset phm2010 --validate-only --output data/raw/phm2010

    # Create synthetic PHM2010-like data for code testing:
    python scripts/download_dataset.py --dataset phm2010 --create-synthetic --output data/raw/phm2010

PHM2010 requires manual registration at https://www.phmsociety.org/competition/phm/10.
The synthetic option creates structurally compatible data for testing the pipeline
without the real dataset — it does NOT represent real machining behavior.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PHM2010_INFO = """
PHM Society Data Challenge 2010 — Milling Tool Wear Dataset
============================================================
License  : PHM Society competition data. Verify PHM Society terms before redistribution.
Citation :
    PHM Society (2010). PHM Data Challenge.
    Prognostics and Health Management Society.
    https://www.phmsociety.org/competition/phm/10

Manual download instructions:
  1. Visit: https://www.phmsociety.org/competition/phm/10
  2. Register or log in to PHM Society.
  3. Download the dataset archive.
  4. Extract to: {output_dir}/

Expected structure after extraction:
  {output_dir}/
  ├── train/
  │   ├── c1/   (condition 1 — one CSV per cut, e.g., c1_001.csv ... c1_315.csv)
  │   ├── c2/ ... c6/
  ├── test/
  │   ├── c1/ ... c6/
  ├── train_labels.csv
  └── (optional) train_condition_params.csv

Each cut CSV columns:
  time, force_x, force_y, force_z,
  vibration_x, vibration_y, vibration_z, acoustic_emission_rms

train_labels.csv columns:
  condition, cut, flute_1_wear_mm, flute_2_wear_mm, flute_3_wear_mm, average_wear_mm
"""

EXPECTED_PATHS = [
    ("train_labels.csv", "file"),
    ("train", "dir"),
    ("train/c1", "dir"),
    ("train/c2", "dir"),
]


def validate_phm2010(output_dir: Path) -> bool:
    """Check whether PHM2010 data is in the expected location and structure."""
    print(f"\nValidating PHM2010 in: {output_dir}")
    all_ok = True

    for rel_path, kind in EXPECTED_PATHS:
        target = output_dir / rel_path
        if kind == "file" and not target.is_file():
            print(f"  MISSING  {target}")
            all_ok = False
        elif kind == "dir" and not target.is_dir():
            print(f"  MISSING  {target}")
            all_ok = False
        else:
            print(f"  OK       {target}")

    if all_ok:
        n_cuts = sum(1 for _ in (output_dir / "train").rglob("*.csv"))
        print(f"\n  Found {n_cuts} cut CSV files in train/")
        print("PHM2010 validation PASSED.")
    else:
        print("\nPHM2010 validation FAILED. Follow the download instructions above.")

    return all_ok


def create_synthetic_phm2010(
    output_dir: Path,
    n_conditions: int = 3,
    n_cuts: int = 30,
) -> None:
    """
    Create synthetic PHM2010-compatible data for pipeline testing.

    WARNING: This is synthetic data for code testing ONLY.
    It does NOT represent real machining behavior and must NOT be used
    to claim experimental results or benchmark performance.
    """
    print(f"\nCreating SYNTHETIC PHM2010-like data in: {output_dir}")
    print("WARNING: Synthetic data for testing only — NOT real PHM2010 data.")

    import numpy as np

    train_dir = output_dir / "train"
    conditions = [f"c{i+1}" for i in range(n_conditions)]
    rng = np.random.default_rng(42)
    n_time_steps = 500
    sensor_header = (
        "time,force_x,force_y,force_z,"
        "vibration_x,vibration_y,vibration_z,acoustic_emission_rms"
    )

    labels_rows: list[list] = [
        ["condition", "cut", "flute_1_wear_mm", "flute_2_wear_mm", "flute_3_wear_mm", "average_wear_mm"]
    ]

    for cond in conditions:
        (train_dir / cond).mkdir(parents=True, exist_ok=True)

        for cut_idx in range(1, n_cuts + 1):
            # Wear increases monotonically with cut number (realistic behavior)
            progress = cut_idx / n_cuts
            base_vb = 0.35 * progress  # peaks near 0.35 mm
            vb = max(0.0, base_vb + rng.normal(0, 0.008))

            t = np.linspace(0, 0.5, n_time_steps)
            f_mag = 80.0 + 60.0 * progress  # force increases with wear
            fx = f_mag * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 5, n_time_steps)
            fy = f_mag * 0.6 * np.sin(2 * np.pi * 10 * t + 0.5) + rng.normal(0, 3, n_time_steps)
            fz = f_mag * 0.3 * np.ones(n_time_steps) + rng.normal(0, 2, n_time_steps)
            vx = 0.1 * (1 + progress) * rng.normal(0, 1, n_time_steps)
            vy = 0.08 * (1 + progress) * rng.normal(0, 1, n_time_steps)
            vz = 0.05 * rng.normal(0, 1, n_time_steps)
            ae = 0.5 * (1 + 2 * progress) * np.abs(rng.normal(0, 1, n_time_steps))

            rows = [sensor_header]
            for i in range(n_time_steps):
                rows.append(
                    f"{t[i]:.4f},{fx[i]:.4f},{fy[i]:.4f},{fz[i]:.4f},"
                    f"{vx[i]:.6f},{vy[i]:.6f},{vz[i]:.6f},{ae[i]:.6f}"
                )
            cut_path = train_dir / cond / f"{cond}_{cut_idx:03d}.csv"
            cut_path.write_text("\n".join(rows))

            flute_noise = rng.normal(0, 0.004, 3)
            f1, f2, f3 = vb + flute_noise[0], vb + flute_noise[1], vb + flute_noise[2]
            labels_rows.append([cond, cut_idx, f"{f1:.4f}", f"{f2:.4f}", f"{f3:.4f}", f"{vb:.4f}"])

    labels_path = output_dir / "train_labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels_rows)

    print(f"Created {n_conditions} conditions × {n_cuts} cuts = {n_conditions * n_cuts} files.")
    print(f"Labels: {labels_path}")
    print("\nTo test the pipeline with this data:")
    print(f"  python scripts/run_tool_wear.py --data-dir {output_dir} --no-backbone")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate machining datasets.")
    parser.add_argument(
        "--dataset", choices=["phm2010"], required=True,
        help="Dataset to download/validate.",
    )
    parser.add_argument(
        "--output", default="data/raw/phm2010",
        help="Output directory (default: data/raw/phm2010).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate an existing download.",
    )
    parser.add_argument(
        "--create-synthetic", action="store_true",
        help="Create synthetic data for testing (NOT real PHM2010).",
    )
    parser.add_argument("--n-conditions", type=int, default=3)
    parser.add_argument("--n-cuts", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.dataset == "phm2010":
        print(PHM2010_INFO.format(output_dir=output_dir))

        if args.create_synthetic:
            output_dir.mkdir(parents=True, exist_ok=True)
            create_synthetic_phm2010(output_dir, args.n_conditions, args.n_cuts)
            validate_phm2010(output_dir)
        elif args.validate_only:
            sys.exit(0 if validate_phm2010(output_dir) else 1)
        else:
            print("PHM2010 requires manual registration and download.")
            print("Options:")
            print("  --validate-only        : Check if existing download is correct")
            print("  --create-synthetic     : Generate synthetic test data")
            sys.exit(0)


if __name__ == "__main__":
    main()
