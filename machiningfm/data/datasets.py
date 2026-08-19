"""
PHM2010 Milling Tool Wear Dataset loader.

Dataset: PHM Society Data Challenge 2010
URL: https://www.phmsociety.org/competition/phm/10
License: PHM Society competition data — verify PHM Society terms before redistribution.
Citation:
    PHM Society (2010). PHM Data Challenge.
    Prognostics and Health Management Society.
    https://www.phmsociety.org/competition/phm/10

Expected directory structure after download:
    data/raw/phm2010/
    ├── train/
    │   ├── c1/    # condition 1 — one CSV file per cut (e.g., c1_001.csv)
    │   ├── c2/ ... c6/
    ├── test/
    │   ├── c1/ ... c6/
    ├── train_labels.csv
    └── (optional) train_condition_params.csv

Sensor columns in each cut CSV:
    time, force_x, force_y, force_z,
    vibration_x, vibration_y, vibration_z, acoustic_emission_rms

train_labels.csv columns:
    condition, cut, flute_1_wear_mm, flute_2_wear_mm, flute_3_wear_mm, average_wear_mm

PHM2010 Machining Conditions (milling, Ti6Al4V-like workpiece, 4-flute end mill):
    Nominal values used for physics calculations (actual values from dataset docs):
        cutting_speed ≈ 250 m/min
        feed_per_tooth ≈ 0.25 mm/tooth
        axial_depth ≈ 0.125 mm
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from machiningfm.data.preprocessing import (
    compute_normalization_stats,
    extract_statistical_features,
    normalize_features,
)

# PHM2010 sensor channels — present in cut CSV files
SENSOR_CHANNELS = [
    "force_x",
    "force_y",
    "force_z",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "acoustic_emission_rms",
]

# PHM2010 nominal machining conditions (same for all conditions in the challenge)
# Source: PHM Society 2010 Data Challenge description
PHM2010_NOMINAL_CONDITIONS: dict[str, float] = {
    "cutting_speed_m_per_min": 250.0,
    "feed_mm_per_tooth": 0.25,
    "axial_depth_mm": 0.125,
    "radial_depth_mm": 0.2,  # approximate; not always explicitly stated
}


@dataclass
class PHM2010Sample:
    """Single sample from PHM2010 (one cut within one condition)."""

    condition_id: str           # e.g., "c1", "c2", ..., "c6"
    cut_number: int             # 1-based cut index within this condition's tool run
    wear_vb_mm: float           # Average flank wear VB in mm (ground truth label)
    sensor_features: np.ndarray  # Statistical features, shape (n_features,)
    cutting_speed_m_per_min: float
    feed_mm_per_tooth: float
    axial_depth_mm: float
    elapsed_time_min: float     # Approximate cumulative cutting time for this cut
    raw_signal: np.ndarray | None = None  # Raw multi-channel signal (T, C) — for backbone encoding


def _assign_split(
    condition_id: str,
    test_conditions: list[str],
    val_conditions: list[str],
) -> str:
    if condition_id in test_conditions:
        return "test"
    if condition_id in val_conditions:
        return "val"
    return "train"


def _parse_cut_number(filename: str) -> int | None:
    """Parse cut index from filenames like 'c1_001.csv' → 1."""
    match = re.search(r"_(\d+)", filename)
    return int(match.group(1)) if match else None


def _extract_wear_vb(row: pd.Series) -> float:
    """Extract average flank wear from a label row."""
    cols = row.index.tolist()
    if "average_wear_mm" in cols:
        return float(row["average_wear_mm"])
    flute_cols = [c for c in cols if "flute" in c and "wear" in c]
    if flute_cols:
        return float(row[flute_cols].mean())
    raise ValueError(f"Cannot find wear column in labels: {cols}")


def _load_cut_data(cut_file: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load one cut CSV. Returns (statistical_features, raw_signal) or (None, None)."""
    try:
        df = pd.read_csv(cut_file)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        available = [ch for ch in SENSOR_CHANNELS if ch in df.columns]
        if not available:
            return None, None
        signal = df[available].values.astype(np.float64)
        feat_dict = extract_statistical_features(signal, available)
        return np.array(list(feat_dict.values()), dtype=np.float32), signal.astype(np.float32)
    except Exception:
        return None, None


class PHM2010Dataset:
    """
    PHM2010 Milling Tool Wear Dataset with condition-wise (leave-one-condition-out) splits.

    Split strategy: entire conditions are assigned to train/val/test to prevent
    temporal data leakage (same tool's time-series cannot appear in both train and test).

    Normalization: statistics are computed on the training split only, and must be
    explicitly passed to val/test splits via feature_stats_from_train.

    Citation:
        PHM Society (2010). PHM Data Challenge.
        https://www.phmsociety.org/competition/phm/10
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
        test_conditions: list[str] | None = None,
        val_conditions: list[str] | None = None,
        feature_stats_from_train: dict[str, np.ndarray] | None = None,
        time_per_cut_min: float = 1.5,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Root directory of PHM2010 data (must contain a 'train/' subdirectory).
            split: Dataset split to load.
            test_conditions: Condition IDs held out for testing. Default: ['c6'].
            val_conditions: Condition IDs used for validation. Default: ['c5'].
            feature_stats_from_train: {'mean': ndarray, 'std': ndarray} from the train split.
                                      Required for val/test to prevent normalization leakage.
            time_per_cut_min: Approximate machining time per cut in minutes.
            seed: Kept for API consistency; splits are deterministic by condition ID.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.test_conditions = test_conditions or ["c6"]
        self.val_conditions = val_conditions or ["c5"]
        self.time_per_cut_min = time_per_cut_min

        self.samples: list[PHM2010Sample] = []
        self._load_data()

        raw = np.stack([s.sensor_features for s in self.samples]) if self.samples else np.zeros((0, 1))

        if split == "train":
            mean, std = compute_normalization_stats(raw) if len(raw) > 0 else (np.zeros(raw.shape[1:]), np.ones(raw.shape[1:]))
            self._feature_mean = mean
            self._feature_std = std
        else:
            if feature_stats_from_train is None:
                raise ValueError(
                    "feature_stats_from_train must be provided for val/test splits "
                    "to prevent normalization leakage."
                )
            self._feature_mean = feature_stats_from_train["mean"]
            self._feature_std = feature_stats_from_train["std"]

        for sample in self.samples:
            sample.sensor_features = normalize_features(
                sample.sensor_features.reshape(1, -1),
                self._feature_mean,
                self._feature_std,
            ).reshape(-1).astype(np.float32)

    def _load_data(self) -> None:
        train_dir = self.data_dir / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"PHM2010 train directory not found: {train_dir}\n"
                "Run: python scripts/download_dataset.py --dataset phm2010 --output data/raw/phm2010"
            )

        labels_path = self.data_dir / "train_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"PHM2010 labels file not found: {labels_path}")

        labels_df = pd.read_csv(labels_path)
        labels_df.columns = [c.strip().lower().replace(" ", "_") for c in labels_df.columns]

        for condition_dir in sorted(train_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition_id = condition_dir.name
            if _assign_split(condition_id, self.test_conditions, self.val_conditions) != self.split:
                continue

            cond_labels = labels_df[labels_df["condition"] == condition_id].sort_values("cut")

            for cut_file in sorted(condition_dir.glob("*.csv")):
                cut_number = _parse_cut_number(cut_file.name)
                if cut_number is None:
                    continue

                label_row = cond_labels[cond_labels["cut"] == cut_number]
                if label_row.empty:
                    continue

                wear_vb_mm = _extract_wear_vb(label_row.iloc[0])
                sensor_features, raw_signal = _load_cut_data(cut_file)
                if sensor_features is None:
                    continue

                self.samples.append(PHM2010Sample(
                    condition_id=condition_id,
                    cut_number=cut_number,
                    wear_vb_mm=wear_vb_mm,
                    sensor_features=sensor_features,
                    cutting_speed_m_per_min=PHM2010_NOMINAL_CONDITIONS["cutting_speed_m_per_min"],
                    feed_mm_per_tooth=PHM2010_NOMINAL_CONDITIONS["feed_mm_per_tooth"],
                    axial_depth_mm=PHM2010_NOMINAL_CONDITIONS["axial_depth_mm"],
                    elapsed_time_min=cut_number * self.time_per_cut_min,
                    raw_signal=raw_signal,
                ))

    def get_normalization_stats(self) -> dict[str, np.ndarray]:
        """Return normalization stats (only meaningful when called on the train split)."""
        return {"mean": self._feature_mean, "std": self._feature_std}

    def get_raw_signals(self) -> list[np.ndarray]:
        """Return list of raw sensor signals, each shape (T_i, C). Used for backbone encoding."""
        return [s.raw_signal for s in self.samples if s.raw_signal is not None]

    def get_features_and_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) as numpy arrays for sklearn-style downstream training."""
        if not self.samples:
            return np.zeros((0, 0)), np.zeros(0)
        X = np.stack([s.sensor_features for s in self.samples])
        y = np.array([s.wear_vb_mm for s in self.samples], dtype=np.float32)
        return X, y

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PHM2010Sample:
        return self.samples[idx]
