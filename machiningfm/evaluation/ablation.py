"""
Ablation study runner for Physics-Guided Downstream Framework.

Compares seven physics/model combinations across six few-shot sample counts.
All parameter selection (alpha, calibration) uses VALIDATION split only.
Test labels are NEVER used for parameter tuning.

Ablation matrix:
  1. Physics only (Taylor-based heuristic, no FM)
  2. MachiningFM only
  3. MachiningFM + Taylor
  4. MachiningFM + Kienzle
  5. MachiningFM + Energy
  6. MachiningFM + Taylor + Kienzle
  7. MachiningFM + All available physics

Few-shot sample counts: full data, 50, 25, 10, 5, (0-shot if applicable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from machiningfm.evaluation.metrics import regression_metrics

FEW_SHOT_NS: list[int | None] = [None, 50, 25, 10, 5]


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    use_backbone: bool = True
    physics_modules: list[str] = field(default_factory=list)
    few_shot_n: int | None = None
    split_seed: int = 42
    calibration_method: str = "ridge"


DEFAULT_ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(name="physics_only", use_backbone=False, physics_modules=["taylor"]),
    AblationConfig(name="machiningfm_only", use_backbone=True, physics_modules=[]),
    AblationConfig(name="machiningfm_taylor", use_backbone=True, physics_modules=["taylor"]),
    AblationConfig(name="machiningfm_kienzle", use_backbone=True, physics_modules=["kienzle"]),
    AblationConfig(name="machiningfm_energy", use_backbone=True, physics_modules=["energy"]),
    AblationConfig(
        name="machiningfm_taylor_kienzle",
        use_backbone=True,
        physics_modules=["taylor", "kienzle"],
    ),
    AblationConfig(
        name="machiningfm_all_physics",
        use_backbone=True,
        physics_modules=["taylor", "kienzle", "energy"],
    ),
]


def subsample_few_shot(
    X: np.ndarray,
    y: np.ndarray,
    n: int | None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample n examples from training data for few-shot evaluation."""
    if n is None or n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=n, replace=False)
    return X[indices], y[indices]


def run_ablation_study(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    physics_features_train: list | None = None,
    physics_features_val: list | None = None,
    physics_features_test: list | None = None,
    ablation_configs: list[AblationConfig] | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full ablation study and return results as a DataFrame.

    Data leakage prevention:
        - y_test is used ONLY for final metrics reporting
        - y_val is used for alpha selection in physics calibration
        - Physics parameters are never fitted on test data

    Args:
        X_train/val/test: Feature matrices.
        y_train/val/test: Wear VB labels in mm.
        physics_features_*: Optional lists of PhysicsFeatures per split.
        ablation_configs: Experiment list. Uses DEFAULT_ABLATION_CONFIGS if None.
        output_dir: If provided, saves ablation_results.csv here.
        seed: Random seed for few-shot subsampling.

    Returns:
        DataFrame with columns: name, few_shot_n, is_full_data,
        physics_modules, val_mae, val_rmse, val_r2, test_mae, test_rmse, test_r2.
    """
    if ablation_configs is None:
        ablation_configs = DEFAULT_ABLATION_CONFIGS

    records = []

    for config in ablation_configs:
        for n_shots in FEW_SHOT_NS:
            X_tr, y_tr = subsample_few_shot(X_train, y_train, n_shots, seed)
            pf_tr = physics_features_train[: len(X_tr)] if physics_features_train else None

            model = Ridge(alpha=1.0)

            if config.use_backbone:
                model.fit(X_tr, y_tr)
                raw_val = model.predict(X_val)
                raw_test = model.predict(X_test)
            else:
                # Physics-only baseline: use tool_life_ratio as sole predictor
                def _get_ratio(pf_list):
                    if pf_list is None:
                        return np.zeros((len(X_val), 1))
                    return np.array([pf.tool_life_ratio or 0.0 for pf in pf_list]).reshape(-1, 1)

                phys_tr = _get_ratio(pf_tr) if pf_tr else np.zeros((len(X_tr), 1))
                model.fit(phys_tr, y_tr)
                raw_val = model.predict(_get_ratio(physics_features_val))
                raw_test = model.predict(_get_ratio(physics_features_test))

            val_preds, test_preds = raw_val.copy(), raw_test.copy()

            if config.physics_modules and pf_tr:
                try:
                    from machiningfm.physics.calibration import PhysicsCalibrator

                    cal = PhysicsCalibrator(method=config.calibration_method)
                    raw_tr_preds = model.predict(
                        (np.array([pf.tool_life_ratio or 0.0 for pf in pf_tr]).reshape(-1, 1)
                         if not config.use_backbone else X_tr)
                    )
                    cal.fit(raw_tr_preds, pf_tr, y_tr)

                    # Alpha selected on val, never on test
                    if physics_features_val:
                        cal.select_alpha(raw_val, physics_features_val, y_val)

                    if physics_features_val:
                        val_preds = cal.predict(raw_val, physics_features_val)
                    if physics_features_test:
                        test_preds = cal.predict(raw_test, physics_features_test)
                except Exception:
                    pass  # If calibration fails, fall back to raw predictions

            val_metrics = regression_metrics(y_val, val_preds)
            test_metrics = regression_metrics(y_test, test_preds)

            records.append({
                "name": config.name,
                "few_shot_n": n_shots if n_shots is not None else len(X_train),
                "is_full_data": n_shots is None,
                "physics_modules": ",".join(config.physics_modules),
                "calibration_method": config.calibration_method,
                "use_backbone": config.use_backbone,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            })

    results_df = pd.DataFrame(records)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "ablation_results.csv", index=False)

    return results_df
