# MachiningFM — Physics-Guided Downstream Framework

> **Machining Foundation Model + Classical Machining Physics + Few-shot Downstream Adaptation**

This framework extends the pretrained MachiningFM foundation model with physics-based post-hoc calibration using classical machining equations (Taylor, Kienzle, Energy) to improve downstream task predictions, especially in few-shot settings.

```
Raw Machining Signal
        │
        ▼
    MachiningFM (pretrained, frozen by default)
        │
        ▼
  Latent Representation
        │
        ▼
  Downstream Head (Ridge/MLP)
        │
        ├── Taylor Tool-Life (r_T = t / T_Taylor)
        ├── Kienzle Force (F_measured / F_Kienzle)
        └── Cutting Energy (E_c = ∫ F_c · V_c · dt)
        │
        ▼
  Physics Calibration (y_final = y_FM + α · g(physics_features))
        │
        ▼
   Final Prediction
```

## Quick Start

```bash
git clone https://github.com/junseokShim/MachiningFM.git
cd MachiningFM
pip install -e .
```

**Create synthetic test data** (real PHM2010 requires registration at phmsociety.org):

```bash
python scripts/download_dataset.py --dataset phm2010 --create-synthetic --output data/raw/phm2010
```

**Tool wear benchmark:**

```bash
python scripts/run_tool_wear.py --data-dir data/raw/phm2010 --no-backbone

python scripts/run_tool_wear.py \
    --data-dir data/raw/phm2010 \
    --physics taylor kienzle energy \
    --physics-config configs/physics/default.yaml
```

**Physics calibration:**

```bash
python scripts/run_fewshot.py --data-dir data/raw/phm2010 --physics taylor kienzle
```

**Ablation study:**

```bash
python scripts/run_ablation.py --data-dir data/raw/phm2010
```

---

## Project Structure

```
MachiningFM/
│
├── machiningfm/                         # Physics-guided downstream framework
│   ├── physics/
│   │   ├── taylor.py                    # Taylor tool-life equation
│   │   ├── kienzle.py                   # Kienzle cutting force model
│   │   ├── energy.py                    # Cutting power & specific energy
│   │   ├── archard.py                   # Archard wear model (optional)
│   │   ├── usui.py                      # Usui wear rate model (optional)
│   │   └── calibration.py               # Post-hoc physics calibration layer
│   │
│   ├── data/
│   │   ├── datasets.py                  # PHM2010 dataset loader
│   │   └── preprocessing.py             # Statistical feature extraction
│   │
│   ├── tasks/
│   │   ├── tool_wear.py                 # Task A: VB regression
│   │   ├── wear_stage.py                # Task B: Stage classification
│   │   ├── rul.py                       # Task C: RUL prediction
│   │   └── dimensional_compensation.py  # Task D: Tool offset interface
│   │
│   ├── evaluation/
│   │   ├── metrics.py                   # MAE, RMSE, R², Acc, F1
│   │   └── ablation.py                  # Ablation study runner
│   │
│   └── models/
│       ├── backbone.py                  # MachiningFMV2 wrapper
│       └── downstream.py                # Downstream head architectures
│
├── src/machiningfm_v2/                  # Pretrained foundation model (existing)
│
├── configs/
│   └── physics/
│       ├── default.yaml                 # Generic steel / carbide parameters
│       └── ti6al4v_carbide.yaml         # Ti-6Al-4V aerospace parameters
│
├── scripts/
│   ├── download_dataset.py              # Download / validate / create synthetic data
│   ├── run_tool_wear.py                 # Task A benchmark
│   ├── run_fewshot.py                   # Few-shot comparison
│   └── run_ablation.py                  # Full ablation study
│
└── tests/                               # 94 unit tests
    ├── test_physics.py                  # 50 physics model tests
    ├── test_datasets.py                 # Dataset pipeline tests
    └── test_tasks.py                    # Downstream task tests
```

---

## Physics Models

| Model | Equation | Default | Required Data |
|-------|----------|---------|--------------|
| Taylor Tool-Life | V_c · T^n · f^m · a_p^p = C | **Enabled** | speed, feed, depth |
| Kienzle Force | F_c = k_{c1.1} · b · h^{1-mc} | **Enabled** | chip thickness, width |
| Cutting Energy | E_c = ∫ F_c · V_c · dt | **Enabled** | force series, speed |
| Archard Wear | V_w = K · F_N · L / H | Disabled | F_N, L (not in PHM2010) |
| Usui Wear Rate | dW/dt = B1 · σ_n · v_s · exp(−B2/T) | Disabled | temperature (not in PHM2010) |

Archard and Usui are disabled by default. Do not enable them with synthetic data.

---

## Downstream Tasks

| Task | Input | Output | Evaluation |
|------|-------|--------|-----------|
| A. Wear Regression | sensor features | VB (mm) | MAE, RMSE, R² |
| B. Stage Classification | sensor features | healthy/moderate/severe | Acc, Macro F1 |
| C. RUL Prediction | sensor features | remaining time (min) | MAE, RMSE, R² |
| D. Dimensional Compensation | wear + condition | offset (mm) | interface only* |

\* PHM2010 has no dimensional accuracy measurements. Task D outputs are physics-derived
estimates, **not experimentally validated results**.

### Wear Stage Thresholds (ISO 8688-1:1989)
- Healthy:  VB < 0.1 mm
- Moderate: 0.1 ≤ VB < 0.2 mm
- Severe:   VB ≥ 0.2 mm
- End-of-life: VB ≥ 0.3 mm

---

## Data Split Policy

All experiments use **leave-one-condition-out** splits to prevent temporal leakage.
Random window splits would allow the same tool's time-series in both train and test.

```
Conditions c1..c4  →  train
Condition  c5      →  val    (alpha selection only — never evaluation)
Condition  c6      →  test   (final evaluation only — never tuning)
```

Normalization statistics always computed on training set and passed to val/test.

---

## Dataset Citation

**PHM Society Data Challenge 2010:**
- URL: https://www.phmsociety.org/competition/phm/10
- License: PHM Society competition data — verify terms before redistribution
- Citation: PHM Society (2010). PHM Data Challenge. Prognostics and Health Management Society.

**Model weights:** [Hugging Face — Junseok2/MachiningFM2.0](https://huggingface.co/Junseok2/MachiningFM2.0)

---

## Tests

```bash
python -m pytest tests/ -v
# 94 tests passing
```

---

## Requirements

```
Python >= 3.11
torch >= 2.2
numpy >= 1.26
pandas >= 2.1
scikit-learn >= 1.4
scipy >= 1.11
PyYAML >= 6.0
```

Install: `pip install -e .`
