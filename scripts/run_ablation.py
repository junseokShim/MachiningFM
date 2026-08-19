#!/usr/bin/env python3
"""
Ablation study: physics modules × few-shot sample counts.

Tests H3: Physics calibration is more beneficial in few-shot settings.

Usage:
    python scripts/run_ablation.py --data-dir data/raw/phm2010
    python scripts/run_ablation.py --data-dir data/raw/phm2010 --device mps --max-len 4096

Results saved to results/phm2010/ablation/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.append(str(_ROOT / "src"))

from machiningfm.data.datasets import PHM2010Dataset
from machiningfm.evaluation.ablation import DEFAULT_ABLATION_CONFIGS, run_ablation_study
from machiningfm.models.encoder import extract_embeddings, load_backbone
from machiningfm.physics.calibration import PhysicsFeatures
from machiningfm.physics.taylor import TaylorParams, compute_tool_life_ratio

_DEFAULT_CKPT = (
    "outputs/checkpoints/"
    "full_pretrain_graph_tokenized_stemgnn_decoder_only_5070_nc_e4b_zeroshot_boost_oomsafe/"
    "machiningfm_full_pretrain_best.pt"
)


def build_physics_features(
    dataset: PHM2010Dataset,
    taylor_params: TaylorParams | None,
) -> list[PhysicsFeatures]:
    out = []
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
                pass
        out.append(pf)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for physics-guided MachiningFM.")
    parser.add_argument("--data-dir", default="data/raw/phm2010")
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                        help="Path to pretrained backbone checkpoint (.pt).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512,
                        help="Signal tail-truncation length. Use 4096+ on GPU.")
    parser.add_argument("--physics-config", default="configs/physics/default.yaml")
    parser.add_argument("--test-conditions", nargs="*", default=["c6"])
    parser.add_argument("--val-conditions", nargs="*", default=["c5"])
    parser.add_argument("--output-dir", default="results/phm2010/ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Ablation Study ===")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {args.device}  |  max_len={args.max_len}")
    print(f"Test conds : {args.test_conditions}  |  Val conds : {args.val_conditions}")

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
        sys.exit(1)

    # Load backbone
    backbone = load_backbone(
        checkpoint_path=args.checkpoint,
        backbone_mode="frozen",
        device=args.device,
    )
    print(f"Backbone   : {backbone.architecture}  d_model={backbone.d_model}\n")

    # Encode all splits
    print("Encoding signals ...")
    X_train = extract_embeddings(backbone, train_ds.get_raw_signals(),
                                 batch_size=args.batch_size, device=args.device, max_len=args.max_len)
    X_val   = extract_embeddings(backbone, val_ds.get_raw_signals(),
                                 batch_size=args.batch_size, device=args.device, max_len=args.max_len)
    X_test  = extract_embeddings(backbone, test_ds.get_raw_signals(),
                                 batch_size=args.batch_size, device=args.device, max_len=args.max_len)

    _, y_train = train_ds.get_features_and_targets()
    _, y_val   = val_ds.get_features_and_targets()
    _, y_test  = test_ds.get_features_and_targets()

    print(f"Feature dim : {X_train.shape[1]}")
    print(f"Train / Val / Test : {len(X_train)} / {len(X_val)} / {len(X_test)}")

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

    pf_train = build_physics_features(train_ds, taylor_params)
    pf_val   = build_physics_features(val_ds, taylor_params)
    pf_test  = build_physics_features(test_ds, taylor_params)

    # Run ablation
    print("\nRunning ablation ...")
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
