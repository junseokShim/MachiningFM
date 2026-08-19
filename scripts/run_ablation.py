#!/usr/bin/env python3
"""
Full ablation study: physics modules × few-shot sample counts.

Tests Hypothesis 3: Physics calibration is more beneficial in few-shot settings.

Usage:
    python scripts/run_ablation.py --dataset phm2010 --data-dir data/raw/phm2010
    python scripts/run_ablation.py --dataset phm2010 --physics-config configs/physics/default.yaml

Results saved to results/phm2010/ablation/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "src"))

from machiningfm.data.datasets import PHM2010Dataset
from machiningfm.evaluation.ablation import DEFAULT_ABLATION_CONFIGS, run_ablation_study
from machiningfm.physics.calibration import PhysicsFeatures
from machiningfm.physics.taylor import TaylorParams, compute_tool_life_ratio


def build_physics_features_for_dataset(
    dataset: PHM2010Dataset,
    taylor_params: TaylorParams | None,
) -> list[PhysicsFeatures]:
    features = []
    for sample in dataset.samples:
        pf = PhysicsFeatures()
        if taylor_params is not None:
            try:
                pf.tool_life_ratio = compute_tool_life_ratio(
                    elapsed_time_min=sample.elapsed_time_min,
                    cutting_speed_m_per_min=sample.cutting_speed_m_per_min,
                    feed_mm_per_rev=sample.feed_mm_per_tooth,
                    axial_depth_mm=sample.axial_depth_mm,
                    params=taylor_params,
                )
            except Exception:
                pf.tool_life_ratio = None
        features.append(pf)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for physics-guided MachiningFM.")
    parser.add_argument("--dataset", default="phm2010")
    parser.add_argument("--data-dir", default="data/raw/phm2010")
    parser.add_argument("--physics-config", default="configs/physics/default.yaml")
    parser.add_argument("--test-conditions", nargs="*", default=["c6"])
    parser.add_argument("--val-conditions", nargs="*", default=["c5"])
    parser.add_argument("--output-dir", default="results/phm2010/ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Ablation Study ===")
    print(f"Dataset    : {args.dataset}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Test conds : {args.test_conditions}")
    print(f"Val conds  : {args.val_conditions}")

    # Load data
    try:
        train_ds = PHM2010Dataset(
            args.data_dir, "train",
            test_conditions=args.test_conditions,
            val_conditions=args.val_conditions,
            seed=args.seed,
        )
        stats = train_ds.get_normalization_stats()
        val_ds = PHM2010Dataset(
            args.data_dir, "val",
            test_conditions=args.test_conditions,
            val_conditions=args.val_conditions,
            feature_stats_from_train=stats,
            seed=args.seed,
        )
        test_ds = PHM2010Dataset(
            args.data_dir, "test",
            test_conditions=args.test_conditions,
            val_conditions=args.val_conditions,
            feature_stats_from_train=stats,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Create synthetic data:")
        print(f"  python scripts/download_dataset.py --dataset phm2010 --create-synthetic --output {args.data_dir}")
        sys.exit(1)

    X_train, y_train = train_ds.get_features_and_targets()
    X_val, y_val = val_ds.get_features_and_targets()
    X_test, y_test = test_ds.get_features_and_targets()

    print(f"\nSamples  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Physics config
    physics_config = {}
    if Path(args.physics_config).exists():
        with open(args.physics_config) as f:
            physics_config = yaml.safe_load(f)

    taylor_params = None
    if "taylor" in physics_config:
        tc = physics_config["taylor"]
        taylor_params = TaylorParams(
            C=tc.get("C", 200.0),
            n=tc.get("n", 0.25),
            m=tc.get("m", 0.0),
            p=tc.get("p", 0.0),
        )

    pf_train = build_physics_features_for_dataset(train_ds, taylor_params)
    pf_val = build_physics_features_for_dataset(val_ds, taylor_params)
    pf_test = build_physics_features_for_dataset(test_ds, taylor_params)

    # Run ablation
    print("\nRunning ablation study...")
    results_df = run_ablation_study(
        X_train, y_train, X_val, y_val, X_test, y_test,
        physics_features_train=pf_train,
        physics_features_val=pf_val,
        physics_features_test=pf_test,
        ablation_configs=DEFAULT_ABLATION_CONFIGS,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n=== Ablation Results (test MAE, mm) ===")
    pivot = results_df.pivot_table(
        values="test_mae",
        index="name",
        columns="few_shot_n",
        aggfunc="first",
    )
    print(pivot.to_string())
    print(f"\nFull results saved to: {Path(args.output_dir) / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
