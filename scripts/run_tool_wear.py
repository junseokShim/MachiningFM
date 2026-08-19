#!/usr/bin/env python3
"""
Task A: Tool Wear Regression benchmark on PHM2010.

Usage:
    # Baseline (no backbone, no physics):
    python scripts/run_tool_wear.py --data-dir data/raw/phm2010 --no-backbone

    # MachiningFM features only:
    python scripts/run_tool_wear.py \\
        --data-dir data/raw/phm2010 \\
        --checkpoint outputs/checkpoints/machiningfm_v2_base/best.pt

    # MachiningFM + physics calibration:
    python scripts/run_tool_wear.py \\
        --data-dir data/raw/phm2010 \\
        --checkpoint outputs/checkpoints/machiningfm_v2_base/best.pt \\
        --physics taylor kienzle energy \\
        --physics-config configs/physics/default.yaml

Results are saved to results/phm2010/tool_wear/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Put FOUNDATION root first so new machiningfm/ package takes priority over src/machiningfm/
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "src"))

from machiningfm.data.datasets import PHM2010Dataset
from machiningfm.evaluation.metrics import regression_metrics
from machiningfm.physics.calibration import PhysicsCalibrator, PhysicsFeatures
from machiningfm.physics.taylor import TaylorParams, compute_tool_life_ratio
from machiningfm.tasks.tool_wear import ToolWearRegressor


def load_physics_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_physics_features(
    dataset: PHM2010Dataset,
    physics_modules: list[str],
    physics_config: dict,
) -> list[PhysicsFeatures]:
    """Compute PhysicsFeatures for each sample in the dataset."""
    features = []
    taylor_params = None
    if "taylor" in physics_modules:
        tc = physics_config.get("taylor", {})
        taylor_params = TaylorParams(
            C=tc.get("C", 200.0),
            n=tc.get("n", 0.25),
            m=tc.get("m", 0.0),
            p=tc.get("p", 0.0),
            source=tc.get("source", "literature"),
        )

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
    parser = argparse.ArgumentParser(description="Tool Wear Regression on PHM2010.")
    parser.add_argument("--data-dir", default="data/raw/phm2010")
    parser.add_argument("--checkpoint", default=None, help="Path to MachiningFM checkpoint.")
    parser.add_argument("--no-backbone", action="store_true", help="Use sensor stat features only.")
    parser.add_argument(
        "--physics", nargs="*", default=[],
        choices=["taylor", "kienzle", "energy"],
        help="Physics modules to use in calibration.",
    )
    parser.add_argument(
        "--physics-config", default="configs/physics/default.yaml",
        help="Path to physics parameter YAML.",
    )
    parser.add_argument("--calibration-method", default="ridge",
                        choices=["linear", "ridge", "mlp", "residual_mlp"])
    parser.add_argument("--test-conditions", nargs="*", default=["c6"])
    parser.add_argument("--val-conditions", nargs="*", default=["c5"])
    parser.add_argument("--output-dir", default="results/phm2010/tool_wear")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Tool Wear Regression — PHM2010 ===")
    print(f"Physics modules : {args.physics or 'none'}")
    print(f"Calibration     : {args.calibration_method if args.physics else 'N/A'}")

    # Load data
    print("\nLoading dataset...")
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
        print("Create synthetic data with:")
        print(f"  python scripts/download_dataset.py --dataset phm2010 --create-synthetic --output {args.data_dir}")
        sys.exit(1)

    X_train, y_train = train_ds.get_features_and_targets()
    X_val, y_val = val_ds.get_features_and_targets()
    X_test, y_test = test_ds.get_features_and_targets()

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val  : {len(X_val)} samples")
    print(f"  Test : {len(X_test)} samples")

    if len(X_train) == 0:
        print("ERROR: No training samples found. Check data directory and split conditions.")
        sys.exit(1)

    # Physics features
    physics_config = {}
    if args.physics and Path(args.physics_config).exists():
        physics_config = load_physics_config(args.physics_config)

    pf_train = build_physics_features(train_ds, args.physics, physics_config) if args.physics else None
    pf_val = build_physics_features(val_ds, args.physics, physics_config) if args.physics else None
    pf_test = build_physics_features(test_ds, args.physics, physics_config) if args.physics else None

    # Build calibrator
    calibrator = None
    if args.physics:
        calibrator = PhysicsCalibrator(method=args.calibration_method)

    # Train
    regressor = ToolWearRegressor(
        feature_dim=X_train.shape[1],
        calibrator=calibrator,
    )
    print("\nTraining...")
    train_metrics = regressor.fit(
        X_train, y_train,
        physics_features_train=pf_train,
        features_val=X_val,
        targets_val=y_val,
        physics_features_val=pf_val,
    )

    # Evaluate
    val_metrics = regressor.evaluate(X_val, y_val, pf_val)
    test_metrics = regressor.evaluate(X_test, y_test, pf_test)

    print("\n--- Results ---")
    print(f"  Val  MAE={val_metrics['mae']:.4f}mm  RMSE={val_metrics['rmse']:.4f}mm  R²={val_metrics['r2']:.4f}")
    print(f"  Test MAE={test_metrics['mae']:.4f}mm  RMSE={test_metrics['rmse']:.4f}mm  R²={test_metrics['r2']:.4f}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": "phm2010",
        "split": {
            "test_conditions": args.test_conditions,
            "val_conditions": args.val_conditions,
            "method": "leave-one-condition-out",
        },
        "physics_modules": args.physics,
        "calibration_method": args.calibration_method if args.physics else None,
        "use_backbone": args.checkpoint is not None and not args.no_backbone,
        "checkpoint": args.checkpoint,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "seed": args.seed,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    suffix = "_".join(["machiningfm"] + args.physics) if args.physics else "baseline"
    out_path = out_dir / f"{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
