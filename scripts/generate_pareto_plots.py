#!/usr/bin/env python
"""
Generate 4 Pareto front scatter plots for the paper:
  1. Accuracy vs Stability (R²adj vs bootstrap CV median)
  2. Accuracy vs Generalisation Gap
  3. Accuracy vs Total Time (log scale)
  4. Accuracy vs Regularity (Lipschitz tail ratio)

Outputs PNG (300 dpi) + PDF to figures/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_MARKERS = {
    "ridge": ("s", "#1f77b4"),
    "dt": ("^", "#ff7f0e"),
    "rf": ("D", "#2ca02c"),
    "xgb": ("v", "#d62728"),
    "chebypoly": ("o", "#9467bd"),
    "chebytree": ("P", "#8c564b"),
    "erbf": ("*", "#e377c2"),
    "tabpfn": ("X", "#17becf"),
}

# Nudge offsets (dx, dy in points) per model to avoid label overlap.
# Adjust as needed after visual inspection.
LABEL_OFFSETS = {
    "ridge": (7, 4),
    "dt": (7, 4),
    "rf": (7, 4),
    "xgb": (7, 4),
    "chebypoly": (7, 4),
    "chebytree": (7, 4),
    "erbf": (7, 4),
    "tabpfn": (7, 4),
}


def pareto_front(x, y, x_lower_better=True, y_higher_better=True):
    """Return indices of Pareto-optimal points.

    Convention: a point dominates another if it is at least as good on both
    axes and strictly better on at least one.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (x[j] <= x[i]) if x_lower_better else (x[j] >= x[i])
            y_better = (y[j] >= y[i]) if y_higher_better else (y[j] <= y[i])
            x_strict = (x[j] < x[i]) if x_lower_better else (x[j] > x[i])
            y_strict = (y[j] > y[i]) if y_higher_better else (y[j] < y[i])
            if x_better and y_better and (x_strict or y_strict):
                is_pareto[i] = False
                break
    return is_pareto


def plot_pareto(ax, models, xvals, yvals, xlabel, ylabel,
                invert_x=False, log_x=False, x_lower_better=True):
    """Scatter models, draw Pareto front, label points."""
    for m, xv, yv in zip(models, xvals, yvals):
        marker, color = MODEL_MARKERS.get(m, ("o", "gray"))
        ax.scatter(xv, yv, marker=marker, color=color, s=140,
                   zorder=3, edgecolors="k", linewidths=0.5)
        dx, dy = LABEL_OFFSETS.get(m, (7, 4))
        ax.annotate(m, (xv, yv), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8.5)

    # Pareto front
    xarr, yarr = np.array(xvals), np.array(yvals)
    mask = pareto_front(xarr, yarr, x_lower_better=x_lower_better)
    if mask.sum() > 1:
        fx, fy = xarr[mask], yarr[mask]
        order = np.argsort(fx)
        ax.plot(fx[order], fy[order], "k--", alpha=0.5, linewidth=1.2, zorder=2)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    if invert_x:
        ax.invert_xaxis()
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto front scatter plots")
    parser.add_argument("--summary-dir", type=Path,
                        default=PROJECT_ROOT / "results" / "benchmark_summary",
                        help="Directory containing summary.csv (default: results/benchmark_summary)")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "figures",
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    SUMMARY = args.summary_dir / "summary.csv"
    STABILITY_CSV = args.summary_dir / "prediction_stability.csv"
    REGULARITY_CSVS = [
        args.summary_dir / "probe_regularity.csv",
        args.summary_dir / "probe_regularity_tabpfn.csv",
    ]
    OUT_DIR = args.output_dir

    plt.style.use("seaborn-v0_8-whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY)

    # ---- Plot 1: Accuracy vs Stability ----
    if STABILITY_CSV.exists():
        stab = pd.read_csv(STABILITY_CSV)
        stab_agg = stab.groupby("model")["cv_median"].median().reset_index()
        stab_agg.columns = ["model", "stability_cv_median"]
        merged = summary.merge(stab_agg, on="model", how="inner")

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        plot_pareto(ax, merged["model"].values,
                    merged["stability_cv_median"].values,
                    merged["val_r2_adj_mean"].values,
                    xlabel="Prediction Instability (CV median, lower → more stable)",
                    ylabel="Mean Adjusted R²",
                    invert_x=True, x_lower_better=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"pareto_accuracy_stability.{ext}"
            fig.savefig(out, dpi=300)
            print(f"Saved: {out}", flush=True)
        plt.close(fig)
    else:
        print(f"SKIP stability plot: {STABILITY_CSV} not found", flush=True)

    # ---- Plot 2: Accuracy vs Gap ----
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plot_pareto(ax, summary["model"].values,
                summary["gap_mean"].values,
                summary["val_r2_adj_mean"].values,
                xlabel="Mean Generalisation Gap (lower → less overfitting)",
                ylabel="Mean Adjusted R²",
                invert_x=True, x_lower_better=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"pareto_accuracy_gap.{ext}"
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}", flush=True)
    plt.close(fig)

    # ---- Plot 3: Accuracy vs Time (log) ----
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plot_pareto(ax, summary["model"].values,
                summary["total_time_mean"].values,
                summary["val_r2_adj_mean"].values,
                xlabel="Mean Total Time per Dataset (s, log scale)",
                ylabel="Mean Adjusted R²",
                invert_x=False, log_x=True, x_lower_better=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"pareto_accuracy_time.{ext}"
        fig.savefig(out, dpi=300)
        print(f"Saved: {out}", flush=True)
    plt.close(fig)

    # ---- Plot 4: Accuracy vs Regularity ----
    reg_parts = [pd.read_csv(p) for p in REGULARITY_CSVS if p.exists()]
    if reg_parts:
        reg = pd.concat(reg_parts, ignore_index=True)
        reg_agg = reg.groupby("model")["lip_tail_ratio"].median().reset_index()
        reg_agg.columns = ["model", "tail_ratio_median"]
        merged = summary.merge(reg_agg, on="model", how="inner")

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        plot_pareto(ax, merged["model"].values,
                    merged["tail_ratio_median"].values,
                    merged["val_r2_adj_mean"].values,
                    xlabel="Lipschitz Tail Ratio (lower → smoother)",
                    ylabel="Mean Adjusted R²",
                    invert_x=True, x_lower_better=True)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            out = OUT_DIR / f"pareto_accuracy_regularity.{ext}"
            fig.savefig(out, dpi=300)
            print(f"Saved: {out}", flush=True)
        plt.close(fig)
    else:
        print("SKIP regularity plot: no regularity CSVs found", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
