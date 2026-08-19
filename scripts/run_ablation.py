#!/usr/bin/env python3
"""
Full ablation study: physics modules × few-shot sample counts.

Tests Hypothesis 3: Physics calibration is more beneficial in few-shot settings.

Usage (statistical features — no backbone):
    python scripts/run_ablation.py --dataset phm2010 --data-dir data/raw/phm2010

Usage (real backbone embeddings — recommended):
    python scripts/run_ablation.py --dataset phm2010 --data-dir data/raw/phm2010 \\
        --backbone-checkpoint pretrained/machiningfm_v2_base.pt

Usage (auto-download backbone from HF):
    python scripts/run_ablation.py --dataset phm2010 --data-dir data/raw/phm2010 \\
        --backbone-hf

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

    print("=== Ablation Study ===")
    print(f"Dataset    : {args.dataset}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Test conds : {args.test_conditions}")
    print(f"Val conds  : {args.val_conditions}")
    if use_backbone:
        src = args.backbone_checkpoint if args.backbone_checkpoint else f"HF:{args.hf_repo_id}"
        print(f"Backbone   : {src} (d=384)")
    else:
        print("Backbone   : None — using statistical features (49-dim)")
        print("             Pass --backbone-checkpoint or --backbone-hf to use real backbone.")

    # Load datasets
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

    print(f"Feature dim: {X_train.shape[1]}  "
          f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    # Physics config
    physics_config: dict = {}
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
    pf_val   = build_physics_features_for_dataset(val_ds, taylor_params)
    pf_test  = build_physics_features_for_dataset(test_ds, taylor_params)

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
