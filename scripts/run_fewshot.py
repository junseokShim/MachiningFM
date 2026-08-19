#!/usr/bin/env python3
"""
Few-shot evaluation: compare MachiningFM vs physics-calibrated predictions
across different numbers of training samples.

Tests Hypotheses 1, 2, 3:
  H1: MachiningFM provides useful representations even with few target-domain samples.
  H2: Physics calibration reduces downstream prediction error.
  H3: Physics calibration benefit is larger in few-shot settings than full-data.

Usage (statistical features — no backbone):
    python scripts/run_fewshot.py --data-dir data/raw/phm2010

Usage (real backbone embeddings — recommended):
    python scripts/run_fewshot.py --data-dir data/raw/phm2010 \\
        --backbone-checkpoint pretrained/machiningfm_v2_base.pt

Usage (auto-download backbone from HF):
    python scripts/run_fewshot.py --data-dir data/raw/phm2010 --backbone-hf

Usage (backbone + physics):
    python scripts/run_fewshot.py --data-dir data/raw/phm2010 \\
        --backbone-checkpoint pretrained/machiningfm_v2_base.pt \\
        --physics taylor kienzle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "src"))

from machiningfm.data.datasets import PHM2010Dataset
from machiningfm.evaluation.metrics import regression_metrics
from machiningfm.physics.calibration import PhysicsCalibrator, PhysicsFeatures
from machiningfm.physics.taylor import TaylorParams, compute_tool_life_ratio
from machiningfm.tasks.tool_wear import ToolWearRegressor

FEW_SHOT_NS = [5, 10, 25, 50, None]  # None = full data


def _load_features(
    ds: PHM2010Dataset,
    backbone=None,
    batch_size: int = 16,
    device: str = "cpu",
    max_len: int = 4096,
) -> np.ndarray:
    """Return feature matrix: backbone embeddings if backbone given, else statistical features."""
    if backbone is not None:
        from machiningfm.models.encoder import extract_embeddings
        raw_signals = ds.get_raw_signals()
        if len(raw_signals) != len(ds):
            raise RuntimeError(
                f"Only {len(raw_signals)} raw signals for {len(ds)} samples. "
                "Ensure PHM2010 CSV files are present."
            )
        return extract_embeddings(backbone, raw_signals, batch_size=batch_size,
                                  device=device, max_len=max_len)
    X, _ = ds.get_features_and_targets()
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot tool wear evaluation.")
    parser.add_argument("--data-dir", default="data/raw/phm2010")
    parser.add_argument("--physics", nargs="*", default=[], choices=["taylor", "kienzle", "energy"])
    parser.add_argument("--physics-config", default="configs/physics/default.yaml")
    parser.add_argument("--calibration-method", default="ridge")
    parser.add_argument("--test-conditions", nargs="*", default=["c6"])
    parser.add_argument("--val-conditions", nargs="*", default=["c5"])
    parser.add_argument("--output-dir", default="results/phm2010/fewshot")
    parser.add_argument("--seed", type=int, default=42)
    # Backbone args
    parser.add_argument(
        "--backbone-checkpoint", default=None,
        help="Path to local pretrained checkpoint (.pt). "
             "Enables backbone embeddings instead of statistical features.",
    )
    parser.add_argument(
        "--backbone-hf", action="store_true",
        help="Download backbone checkpoint from Hugging Face Hub.",
    )
    parser.add_argument("--hf-repo-id", default="Junseok2/MachiningFM2.0")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda/mps).")
    parser.add_argument("--batch-size", type=int, default=16, help="Backbone encoding batch size.")
    parser.add_argument("--max-len", type=int, default=512, help="Signal truncation length for backbone (tail-truncation). Use 4096+ on GPU.")
    args = parser.parse_args()

    use_backbone = args.backbone_checkpoint is not None or args.backbone_hf

    print("=== Few-Shot Evaluation ===")
    if use_backbone:
        src = args.backbone_checkpoint if args.backbone_checkpoint else f"HF:{args.hf_repo_id}"
        print(f"Feature source : MachiningFM backbone embeddings (d=384) [{src}]")
    else:
        print("Feature source : Statistical features (7 stats × 7 channels = 49-dim)")
        print("                 Pass --backbone-checkpoint or --backbone-hf to use real backbone.")

    # Load dataset splits
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
        print(f"ERROR: {e}")
        print(f"python scripts/download_dataset.py --dataset phm2010 --create-synthetic --output {args.data_dir}")
        sys.exit(1)

    # Load backbone (once, shared across all splits)
    backbone = None
    if use_backbone:
        from machiningfm.models.encoder import load_backbone
        ckpt = None if args.backbone_hf else args.backbone_checkpoint
        backbone = load_backbone(
            checkpoint_path=ckpt,
            hf_repo_id=args.hf_repo_id,
            backbone_mode="frozen",
            device=args.device,
        )
        print(f"Backbone loaded. d_model={backbone.d_model}, device={args.device}")

    # Extract features
    print("\nExtracting features ...")
    X_train = _load_features(train_ds, backbone, args.batch_size, args.device, args.max_len)
    X_val   = _load_features(val_ds,   backbone, args.batch_size, args.device, args.max_len)
    X_test  = _load_features(test_ds,  backbone, args.batch_size, args.device, args.max_len)
    _, y_train = train_ds.get_features_and_targets()
    _, y_val   = val_ds.get_features_and_targets()
    _, y_test  = test_ds.get_features_and_targets()

    print(f"Feature dim    : {X_train.shape[1]}")
    print(f"Train / Val / Test samples: {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # Physics setup
    physics_config: dict = {}
    if args.physics and Path(args.physics_config).exists():
        with open(args.physics_config) as f:
            physics_config = yaml.safe_load(f)

    taylor_params = None
    if "taylor" in args.physics and "taylor" in physics_config:
        tc = physics_config["taylor"]
        taylor_params = TaylorParams(C=tc["C"], n=tc["n"], m=tc.get("m", 0), p=tc.get("p", 0))

    def build_pf(dataset: PHM2010Dataset) -> list[PhysicsFeatures] | None:
        if not args.physics:
            return None
        features = []
        for sample in dataset.samples:
            pf = PhysicsFeatures()
            if taylor_params is not None:
                try:
                    pf.tool_life_ratio = compute_tool_life_ratio(
                        sample.elapsed_time_min, sample.cutting_speed_m_per_min,
                        sample.feed_mm_per_tooth, sample.axial_depth_mm, taylor_params,
                    )
                except Exception:
                    pf.tool_life_ratio = None
            features.append(pf)
        return features

    pf_train = build_pf(train_ds)
    pf_val   = build_pf(val_ds)
    pf_test  = build_pf(test_ds)

    rng = np.random.default_rng(args.seed)
    records = []

    print(f"\n{'N':>6}  {'without_physics_MAE':>20}  {'with_physics_MAE':>18}  {'improvement_pct':>15}")
    print("-" * 66)

    for n_shots in FEW_SHOT_NS:
        if n_shots is None:
            idx = np.arange(len(X_train))
            label = "full"
        else:
            if n_shots > len(X_train):
                continue
            idx = rng.choice(len(X_train), size=n_shots, replace=False)
            label = str(n_shots)

        X_tr = X_train[idx]
        y_tr = y_train[idx]
        pf_tr = [pf_train[i] for i in idx] if pf_train else None

        # Without physics
        reg_base = ToolWearRegressor(feature_dim=X_tr.shape[1])
        reg_base.fit(X_tr, y_tr)
        m_base = regression_metrics(y_test, reg_base.predict(X_test))

        # With physics calibration
        m_phys = m_base.copy()
        if args.physics and pf_tr:
            cal = PhysicsCalibrator(method=args.calibration_method)
            reg_phys = ToolWearRegressor(
                feature_dim=X_tr.shape[1], calibrator=cal
            )
            reg_phys.fit(X_tr, y_tr, pf_tr, X_val, y_val, pf_val)
            m_phys = regression_metrics(y_test, reg_phys.predict(X_test, pf_test))

        improvement = (m_base["mae"] - m_phys["mae"]) / (m_base["mae"] + 1e-9) * 100
        print(f"{label:>6}  {m_base['mae']:>20.4f}  {m_phys['mae']:>18.4f}  {improvement:>+14.1f}%")

        records.append({
            "n_shots": n_shots if n_shots is not None else len(X_train),
            "is_full_data": n_shots is None,
            "feature_source": "backbone" if use_backbone else "statistical",
            "feature_dim": int(X_train.shape[1]),
            "without_physics_mae": m_base["mae"],
            "without_physics_rmse": m_base["rmse"],
            "without_physics_r2": m_base["r2"],
            "with_physics_mae": m_phys["mae"],
            "with_physics_rmse": m_phys["rmse"],
            "with_physics_r2": m_phys["r2"],
            "mae_improvement_pct": improvement,
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fewshot_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "feature_source": "backbone" if use_backbone else "statistical",
            "backbone_checkpoint": args.backbone_checkpoint,
            "physics": args.physics,
            "calibration_method": args.calibration_method,
            "results": records,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
