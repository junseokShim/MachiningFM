#!/usr/bin/env python3
"""
Generate a PDF report from few-shot and ablation results.

Usage:
    python scripts/make_report.py
    python scripts/make_report.py --fewshot results/phm2010/fewshot/fewshot_results.json \
                                   --ablation results/phm2010/ablation/ablation_results.csv \
                                   --output report.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

BLUE   = "#2563EB"
ORANGE = "#F97316"
GREEN  = "#16A34A"
GRAY   = "#6B7280"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cover_page(pdf: PdfPages, meta: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("#1E293B")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.5, 0.72, "MachiningFM 2.0", ha="center", va="center",
            fontsize=32, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.64, "Physics-Guided Downstream Evaluation Report",
            ha="center", va="center", fontsize=16, color="#94A3B8", transform=ax.transAxes)
    ax.axhline(0.60, xmin=0.1, xmax=0.9, color="#3B82F6", linewidth=1.5,
               transform=ax.transAxes)

    info_lines = [
        f"Backbone  :  {meta.get('backbone', 'machiningfm_graph_tokenized_stemgnn_decoder_only')}",
        f"d_model   :  {meta.get('d_model', 2048)}",
        f"Dataset   :  PHM Society Data Challenge 2010",
        f"Split     :  Leave-one-condition-out (test=c6, val=c5)",
        f"Physics   :  Taylor Tool-Life Calibration",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.5, 0.52 - i * 0.05, line, ha="center", va="center",
                fontsize=11, color="#CBD5E1", transform=ax.transAxes,
                fontfamily="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _fewshot_page(pdf: PdfPages, records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Few-Shot Tool Wear Regression  (Task A)", fontsize=14, fontweight="bold", y=1.01)

    ns      = [r["n_shots"] for r in records]
    labels  = [str(r["n_shots"]) if not r["is_full_data"] else "full" for r in records]
    mae_b   = [r["without_physics_mae"]  for r in records]
    mae_p   = [r["with_physics_mae"]     for r in records]
    r2_b    = [r["without_physics_r2"]   for r in records]
    imp     = [r["mae_improvement_pct"]  for r in records]

    x = np.arange(len(records))
    w = 0.35

    # MAE comparison
    ax = axes[0]
    ax.bar(x - w/2, mae_b, w, label="No physics", color=BLUE,   alpha=0.85)
    ax.bar(x + w/2, mae_p, w, label="+ Physics",  color=ORANGE, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("# training samples")
    ax.set_ylabel("MAE (mm)")
    ax.set_title("MAE vs Sample Count")
    ax.legend(fontsize=9)

    # R² vs sample count
    ax = axes[1]
    ax.plot(labels, r2_b, "o-", color=BLUE, linewidth=2, markersize=6, label="No physics")
    ax.set_xlabel("# training samples")
    ax.set_ylabel("R²")
    ax.set_title("R² vs Sample Count")
    ax.set_ylim([-0.1, 1.05])
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.legend(fontsize=9)

    # Physics improvement %
    ax = axes[2]
    colors = [GREEN if v > 0 else "#EF4444" for v in imp]
    ax.bar(labels, imp, color=colors, alpha=0.85)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("# training samples")
    ax.set_ylabel("MAE improvement (%)")
    ax.set_title("Physics Calibration Benefit")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _ablation_page(pdf: PdfPages, df) -> None:
    if not _HAS_PANDAS or df is None:
        return

    pivot = df.pivot_table(values="test_mae", index="name", columns="few_shot_n", aggfunc="first")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ablation Study — Test MAE (mm) by Physics Config × Few-Shot N",
                 fontsize=13, fontweight="bold", y=1.01)

    # Heatmap-style table
    ax = axes[0]
    ax.set_axis_off()
    table_data = pivot.round(4).reset_index()
    col_labels = ["Config"] + [str(c) for c in pivot.columns]
    rows = table_data.values.tolist()
    t = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1.1, 1.6)
    # Header color
    for j in range(len(col_labels)):
        t[0, j].set_facecolor("#1E293B")
        t[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("MAE Table (mm)", fontsize=11, pad=12)

    # Line plot per config
    ax = axes[1]
    cols = sorted(pivot.columns)
    for name, row in pivot.iterrows():
        vals = [row[c] for c in cols if c in row.index and not np.isnan(row[c])]
        x_vals = [c for c in cols if c in row.index and not np.isnan(row[c])]
        if not vals:
            continue
        style = "--" if "physics_only" in name else "-"
        ax.plot(x_vals, vals, marker="o", linestyle=style, linewidth=1.8,
                markersize=5, label=name)
    ax.set_xlabel("# training samples (few-shot N)")
    ax.set_ylabel("Test MAE (mm)")
    ax.set_title("MAE by Config")
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _summary_page(pdf: PdfPages, records: list[dict], meta: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.1, 0.84, 0.82])
    ax.set_axis_off()

    full = next((r for r in records if r["is_full_data"]), records[-1])
    shot5 = next((r for r in records if r["n_shots"] == 5), None)

    lines = [
        ("MachiningFM 2.0 — Experiment Summary", True),
        ("", False),
        ("Model", True),
        (f"  Architecture : {meta.get('backbone', 'N/A')}", False),
        (f"  d_model      : {meta.get('d_model', 'N/A')}", False),
        (f"  Backbone     : frozen (no fine-tuning)", False),
        ("", False),
        ("Dataset", True),
        ("  PHM Society Data Challenge 2010", False),
        ("  Sensor channels : force x/y/z, vibration x/y/z, AE_RMS", False),
        ("  Split strategy  : leave-one-condition-out", False),
        ("  Test condition  : c6  |  Val condition : c5", False),
        ("", False),
        ("Task A — Tool Wear Regression (full data)", True),
        (f"  MAE  = {full['without_physics_mae']:.4f} mm  (no physics)", False),
        (f"  RMSE = {full['without_physics_rmse']:.4f} mm", False),
        (f"  R²   = {full['without_physics_r2']:.4f}", False),
    ]

    if shot5:
        lines += [
            ("", False),
            ("Task A — 5-shot performance", True),
            (f"  MAE (no physics) = {shot5['without_physics_mae']:.4f} mm", False),
            (f"  MAE (+ physics)  = {shot5['with_physics_mae']:.4f} mm", False),
            (f"  Physics benefit  = {shot5['mae_improvement_pct']:+.1f}%", False),
        ]

    lines += [
        ("", False),
        ("Physics Calibration", True),
        (f"  Method : {meta.get('calibration_method', 'ridge')}", False),
        (f"  Models : {', '.join(meta.get('physics', [])) or 'none'}", False),
        ("", False),
        ("Wear Stage Thresholds (ISO 8688-1:1989)", True),
        ("  Healthy  : VB < 0.10 mm", False),
        ("  Moderate : 0.10 ≤ VB < 0.20 mm", False),
        ("  Severe   : VB ≥ 0.20 mm", False),
    ]

    y = 0.97
    for text, bold in lines:
        if not text:
            y -= 0.022
            continue
        ax.text(0.0, y, text, transform=ax.transAxes,
                fontsize=11 if bold else 10,
                fontweight="bold" if bold else "normal",
                color="#1E293B" if bold else "#374151",
                va="top")
        y -= 0.038 if bold else 0.030

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDF report from evaluation results.")
    parser.add_argument("--fewshot",  default="results/phm2010/fewshot/fewshot_results.json")
    parser.add_argument("--ablation", default="results/phm2010/ablation/ablation_results.csv")
    parser.add_argument("--output",   default="report.pdf")
    args = parser.parse_args()

    # Load few-shot results
    fewshot_path = Path(args.fewshot)
    if not fewshot_path.exists():
        print(f"ERROR: {fewshot_path} not found. Run run_fewshot.py first.")
        sys.exit(1)
    with open(fewshot_path) as f:
        fewshot_data = json.load(f)

    records = fewshot_data.get("results", [])
    meta = {k: v for k, v in fewshot_data.items() if k != "results"}

    # Load ablation results (optional)
    ablation_df = None
    if _HAS_PANDAS:
        import pandas as pd
        ablation_path = Path(args.ablation)
        if ablation_path.exists():
            ablation_df = pd.read_csv(ablation_path)

    # Build PDF
    out_path = Path(args.output)
    with PdfPages(out_path) as pdf:
        _cover_page(pdf, meta)
        _fewshot_page(pdf, records)
        if ablation_df is not None:
            _ablation_page(pdf, ablation_df)
        _summary_page(pdf, records, meta)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "MachiningFM 2.0 — Downstream Evaluation Report"
        d["Author"]  = "MachiningFM"
        d["Subject"] = "PHM2010 Tool Wear Regression"

    print(f"Report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
