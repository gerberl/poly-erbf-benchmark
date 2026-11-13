"""Regenerate the final gap-rank and two Pareto figures from corrected July results; preserve existing styles.
Run from any directory with micromamba run -n poly_erbf_benchmark python <script>.
Time Pareto retains saved total_time; inference normalisation is not used here.
"""
from pathlib import Path
import importlib.util
ROOT = Path(__file__).resolve().parents[1]

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

STAB = ROOT / "expl/stability_denominator_sweep_11jul26_fixprefilter.csv"
YCOL = "sd_over_p99_p1"
OUTDIR = ROOT / "figures"
COMPETITIVE = ["chebypoly", "chebytree", "xgb", "rf", "ebm", "erbf"]
SMOOTH = {"chebypoly", "chebytree", "erbf"}
TREES = {"rf", "xgb", "dt"}
FAM_COLOR = {"smooth": "#4C72B0", "tree": "#C44E52", "other": "#8C8C8C"}

sns.set(style="ticks", context="paper", font_scale=1.0)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Fira Sans", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

rankplot = load_module("rankplot", "making_plots/generate_rank_linedot_plots_04feb26.py")
rankplot.OUTPUT_DIR = OUTDIR
d = pd.read_csv(ROOT/"results/benchmark_summary/results_detailed.csv")
summary = rankplot.compute_ranks_and_summary(d, "gap", lower_is_better=True)
pivot = d.pivot(index="dataset", columns="model", values="gap")
expected = pivot.rank(axis=1, ascending=True, method="min", na_option="bottom").mean()
assert np.allclose(summary.set_index("model").rank_mean.sort_index(), expected.sort_index())
rankplot.make_rank_plot(summary, "gap", "Mean rank", "linedot_rank_gap", lower_is_better=True)
pareto = load_module("pareto", "making_plots/pareto_plot_helpers_01feb26.py")
means = d.groupby("model")[["val_r2_adj", "gap", "total_time"]].mean()
plt.style.use("seaborn-v0_8-whitegrid")
for metric, name, label in [("gap", "pareto_accuracy_gap", "Mean generalisation gap"), ("total_time", "pareto_accuracy_time", "Mean total time per dataset (s, log scale)")]:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    pareto.plot_pareto(ax, means.index, means[metric].to_numpy(), means.val_r2_adj.to_numpy(),
        xlabel=label, ylabel=r"Mean $\bar{R}^2$", invert_x=True, log_x=metric=="total_time",
        x_lower_better=True, label_offsets={**pareto.LABEL_OFFSETS_GAP, "ebm": (7, -12)} if metric=="gap" else None)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR/f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("Saved", name, flush=True)
print("COMPLETE", flush=True)
