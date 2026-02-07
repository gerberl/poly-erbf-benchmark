#!/usr/bin/env python
"""Generate box+strip plot of adjusted R² by model.

Loads all joblib results from a benchmark run, ranks models by mean adjusted R²,
and produces a box-and-strip plot.

Usage:
    python scripts/generate_boxstrip_r2adj.py results/my_run
    python scripts/generate_boxstrip_r2adj.py results/my_run --output-dir figures/

Requires: matplotlib, seaborn, pandas, joblib
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]


def load_results(results_dir):
    """Load all joblib results into a DataFrame."""
    cols = ["model_name", "dataset_name", "r2_val_adj", "gap", "r2_val", "mae_val", "rmse_val"]
    records = []
    for f in sorted(results_dir.glob("*.joblib")):
        r = joblib.load(f)
        records.append(pd.Series(r)[cols])
    return pd.DataFrame(records)


def rank_by_on(bm_df, by, on, lower_is_better=True):
    """Rank models per dataset on a given metric."""
    bm_df = bm_df.assign(rank=(
        bm_df.groupby(by=by)[on]
        .rank("dense", ascending=lower_is_better)
        .astype("int")
    ))
    return bm_df


def main():
    parser = argparse.ArgumentParser(description="Generate box+strip plot of adjusted R²")
    parser.add_argument("results_dir", type=Path,
                        help="Directory containing .joblib result files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures",
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    RESULTS_DIR = args.results_dir
    OUTPUT_DIR = args.output_dir

    print(f"Loading results from {RESULTS_DIR} ...", flush=True)
    df = load_results(RESULTS_DIR)
    print(f"  Loaded {len(df)} results ({df['model_name'].nunique()} models, "
          f"{df['dataset_name'].nunique()} datasets)", flush=True)

    # Compute mean ranks for ordering
    rank_df = rank_by_on(df, by="dataset_name", on="r2_val_adj", lower_is_better=False)
    mean_ranks = (
        rank_df.groupby("model_name")
        .agg(rank_mean=("rank", "mean"), rank_std=("rank", "std"))
        .reset_index()
        .sort_values("rank_mean")
    )
    model_order = mean_ranks["model_name"].tolist()
    model_colors = dict(zip(model_order, sns.color_palette("Set3", len(model_order))))

    # Plot settings
    sns.set(style="ticks", context="notebook", font_scale=0.8)
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Roboto", "Fira Sans"]
    rcParams["font.size"] = 13
    rcParams["axes.titlesize"] = 15
    rcParams["axes.labelsize"] = 13
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12

    model_r2_adj = df[["model_name", "r2_val_adj"]]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.boxplot(
        data=model_r2_adj, x="model_name", y="r2_val_adj",
        order=model_order, notch=True,
        hue=model_r2_adj["model_name"], palette=model_colors,
        showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=model_r2_adj, x="model_name", y="r2_val_adj",
        order=model_order,
        linewidth=0.5, jitter=0.3, size=7, alpha=0.4,
        hue=model_r2_adj["model_name"], palette=model_colors,
        ax=ax,
    )
    ax.set_xlabel("Model Name")
    ax.set_ylabel(r"Mean Adjusted R$^2$")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / "boxstripplot-rank-meanr2adj.png"
    out_pdf = OUTPUT_DIR / "boxstripplot-rank-meanr2adj.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    print(f"  Saved: {out_png}", flush=True)
    print(f"  Saved: {out_pdf}", flush=True)
    plt.close(fig)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
