#!/usr/bin/env python3
"""
Few-shot evaluation: MachiningFM backbone embeddings → downstream regression.

Tests Hypotheses 1-3:
  H1: Backbone representations are useful with few labeled samples.
  H2: Physics calibration reduces prediction error.
  H3: Physics benefit is larger in few-shot settings.

Usage:
    python scripts/run_fewshot.py --data-dir data/raw/phm2010
    python scripts/run_fewshot.py --data-dir data/raw/phm2010 --physics taylor kienzle
    python scripts/run_fewshot.py --data-dir data/raw/phm2010 --device mps --max-len 4096
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
from machiningfm.models.encoder import extract_embeddings, load_backbone
from machiningfm.physics.calibration import PhysicsCalibrator, PhysicsFeatures
from machiningfm.physics.taylor import TaylorParams, compute_tool_life_ratio
from machiningfm.tasks.tool_wear import ToolWearRegressor

_DEFAULT_CKPT = (
    "outputs/checkpoints/"
    "full_pretrain_graph_tokenized_stemgnn_decoder_only_5070_nc_e4b_zeroshot_boost_oomsafe/"
    "machiningfm_full_pretrain_best.pt"
)

FEW_SHOT_NS = [5, 10, 25, 50, None]  # None = full data


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot tool wear evaluation.")
    parser.add_argument("--data-dir", default="data/raw/phm2010")
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                        help="Path to pretrained backbone checkpoint (.pt).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512,
                        help="Signal tail-truncation length. Use 4096+ on GPU.")
    parser.add_argument("--physics", nargs="*", default=[],
                        choices=["taylor", "kienzle", "energy"])
    parser.add_argument("--physics-config", default="configs/physics/default.yaml")
    parser.add_argument("--calibration-method", default="ridge")
    parser.add_argument("--test-conditions", nargs="*", default=["c6"])
    parser.add_argument("--val-conditions", nargs="*", default=["c5"])
    parser.add_argument("--output-dir", default="results/phm2010/fewshot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Few-Shot Evaluation ===")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {args.device}  |  max_len={args.max_len}")

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
        print(f"ERROR: {e}")
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

    # Physics features
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
        out = []
        for sample in dataset.samples:
            pf = PhysicsFeatures()
            if taylor_params is not None:
                try:
                    pf.tool_life_ratio = compute_tool_life_ratio(
                        sample.elapsed_time_min, sample.cutting_speed_m_per_min,
                        sample.feed_mm_per_tooth, sample.axial_depth_mm, taylor_params,
                    )
                except Exception:
                    pass
            out.append(pf)
        return out

    pf_train = build_pf(train_ds)
    pf_val   = build_pf(val_ds)
    pf_test  = build_pf(test_ds)

    rng = np.random.default_rng(args.seed)
    records = []

    print(f"\n{'N':>6}  {'no_physics_MAE':>16}  {'with_physics_MAE':>18}  {'improvement':>12}")
    print("-" * 58)

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

        reg_base = ToolWearRegressor(feature_dim=X_tr.shape[1])
        reg_base.fit(X_tr, y_tr)
        m_base = regression_metrics(y_test, reg_base.predict(X_test))

        m_phys = m_base.copy()
        if args.physics and pf_tr:
            cal = PhysicsCalibrator(method=args.calibration_method)
            reg_phys = ToolWearRegressor(feature_dim=X_tr.shape[1], calibrator=cal)
            reg_phys.fit(X_tr, y_tr, pf_tr, X_val, y_val, pf_val)
            m_phys = regression_metrics(y_test, reg_phys.predict(X_test, pf_test))

        improvement = (m_base["mae"] - m_phys["mae"]) / (m_base["mae"] + 1e-9) * 100
        print(f"{label:>6}  {m_base['mae']:>16.4f}  {m_phys['mae']:>18.4f}  {improvement:>+11.1f}%")

        records.append({
            "n_shots": n_shots if n_shots is not None else len(X_train),
            "is_full_data": n_shots is None,
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
            "checkpoint": args.checkpoint,
            "backbone": backbone.architecture,
            "d_model": backbone.d_model,
            "physics": args.physics,
            "calibration_method": args.calibration_method,
            "results": records,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
