#!/usr/bin/env python3
"""
Generate publication-ready plots for the paper:
  1. Combined Figure 1: (a) mean rank on R²adj all models, (b) CPU-only
  2. Standalone gap ranking line-dot plot
  3. Boxstripplot of adjusted R² distributions (for appendix)

Also prints Nemenyi CD values, non-significant groups, and summary tables.

Usage:
    python making_plots/generate_rank_linedot_plots_04feb26.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.stats import studentized_range, friedmanchisquare

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_REPO = SCRIPT_DIR.parent
REPO_ROOT = PAPER_REPO.parent
RESULTS_CSV = REPO_ROOT / 'results/benchmark-A/benchmark_summary/results_detailed.csv'
OUTPUT_DIR = PAPER_REPO / 'figures'

# --- Style ---
sns.set(style='ticks', context='paper', font_scale=1.0)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Roboto', 'Fira Sans', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
})

# Model display names — use code names matching \mn{...} in the paper
MODEL_LABELS = {
    'tabpfn': 'tabpfn',
    'erbf': 'erbf',
    'chebytree': 'chebytree',
    'xgb': 'xgb',
    'chebypoly': 'chebypoly',
    'rf': 'rf',
    'dt': 'dt',
    'ridge': 'ridge',
}

CPU_MODELS = ['ridge', 'dt', 'rf', 'xgb', 'erbf', 'chebypoly', 'chebytree']


def load_data():
    """Load results and compute ranks for R²adj and gap."""
    df = pd.read_csv(RESULTS_CSV)
    # Keep only what we need
    df = df[['model', 'dataset', 'val_r2_adj', 'val_r2', 'gap']].copy()
    df = df.dropna(subset=['val_r2_adj', 'gap'])
    return df


def compute_ranks_and_summary(df, metric, lower_is_better=False):
    """Compute per-dataset ranks and aggregate summary.

    Models missing from a dataset receive worst rank (= total number of models)
    so that all models are ranked over the same set of datasets.
    """
    df = df.copy()
    all_models = sorted(df['model'].unique())
    all_datasets = sorted(df['dataset'].unique())
    k = len(all_models)

    # Rank present models per dataset
    df['rank'] = (
        df.groupby('dataset')[metric]
        .rank('dense', ascending=lower_is_better)
        .astype(int)
    )

    # Fill missing model/dataset pairs with worst rank
    missing_rows = []
    for dataset in all_datasets:
        present = set(df.loc[df['dataset'] == dataset, 'model'])
        for m in set(all_models) - present:
            missing_rows.append({'model': m, 'dataset': dataset, metric: np.nan, 'rank': k})
    if missing_rows:
        df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    df['is_win'] = df.groupby('dataset')['rank'].transform(lambda x: x == x.min())
    df['is_second'] = df['rank'] == 2

    summary = (
        df.groupby('model')
        .agg(
            rank_mean=('rank', 'mean'),
            rank_std=('rank', 'std'),
            rank_min=('rank', 'min'),
            rank_max=('rank', 'max'),
            rank_q25=('rank', lambda x: x.quantile(0.25)),
            rank_q75=('rank', lambda x: x.quantile(0.75)),
            wins=('is_win', 'sum'),
            seconds=('is_second', 'sum'),
            metric_mean=(metric, 'mean'),
            metric_median=(metric, 'median'),
            metric_std=(metric, 'std'),
            n_datasets=('dataset', 'count'),
        )
        .reset_index()
        .sort_values('rank_mean')
    )
    summary['wins'] = summary['wins'].astype(int)
    summary['seconds'] = summary['seconds'].astype(int)
    return summary


def linedotplot(
    data, x, y, err=None, err_low=None, err_high=None,
    xlabel=None, ylabel='', title=None,
    xlim=None, xlim_pad=0.3,
    width=4.5, height_per_row=0.45,
    dot_size=7, dot_linewidth=0.5, dot_edgecolor='w',
    palette='flare',
    annotate=True, annot_fmt='.2f', annot_offset=0.35, annot_size=7,
    x_tick_spacing=None,
    ax=None,
):
    """Horizontal dot plot for rankings.

    Error bars: if err_low and err_high are given, draws asymmetric bars
    from data[err_low] to data[err_high].  Otherwise falls back to
    symmetric ±data[err].
    """
    if ax is None:
        height = height_per_row * len(data)
        fig, ax = plt.subplots(
            figsize=(width, height), constrained_layout=True,
        )
    else:
        fig = ax.get_figure()

    colors = sns.color_palette(palette, n_colors=len(data))

    sns.stripplot(
        data=data, x=x, y=y, hue=y, orient='h',
        size=dot_size, linewidth=dot_linewidth, edgecolor=dot_edgecolor,
        jitter=False, palette=palette, legend=False, ax=ax,
    )

    # error bars
    if err_low is not None and err_high is not None:
        for i, (lo, hi, colour) in enumerate(
            zip(data[err_low], data[err_high], colors)
        ):
            ax.plot(
                [lo, hi], [i, i],
                color=colour, linewidth=1.0, solid_capstyle='round',
            )
    elif err is not None:
        for i, (x_val, err_val, colour) in enumerate(
            zip(data[x], data[err], colors)
        ):
            ax.plot(
                [x_val - err_val, x_val + err_val], [i, i],
                color=colour, linewidth=1.0, solid_capstyle='round',
            )

    # value annotations
    if annotate:
        for i, val in enumerate(data[x]):
            ax.text(
                val, i + annot_offset, f'{val:{annot_fmt}}',
                ha='center', size=annot_size, color='0.5',
            )

    # axis cosmetics
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.xaxis.set_ticks_position('bottom')
    if x_tick_spacing is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1))

    if xlim is None:
        xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - xlim_pad, xlim[1] + xlim_pad)

    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color='lightgray', linestyle=':', linewidth=0.8)
    ax.spines['bottom'].set_position(('outward', 10))
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='y', length=0, width=0)

    return fig, ax


def make_rank_plot(summary, metric_name, xlabel, filename, lower_is_better=False, cd=None,
                   height_per_row=0.45, width=4.5, annot_offset=0.35):
    """Generate and save a linedot rank plot with wins annotations."""
    # Build y-labels with wins
    summary = summary.copy()
    summary['label'] = summary.apply(
        lambda r: f"{MODEL_LABELS.get(r['model'], r['model'])}  ({r['wins']:.0f}/{r['seconds']:.0f})",
        axis=1
    )

    fig, ax = linedotplot(
        summary, x='rank_mean', y='label',
        err_low='rank_q25', err_high='rank_q75',
        xlabel=xlabel, ylabel='',
        width=width, height_per_row=height_per_row,
        annot_offset=annot_offset,
    )

    if cd is not None:
        _add_cd_bar(ax, cd)

    fig.savefig(
        OUTPUT_DIR / f'{filename}.pdf',
        bbox_inches='tight', pad_inches=0.05,
    )
    fig.savefig(
        OUTPUT_DIR / f'{filename}.png', dpi=300,
        bbox_inches='tight', pad_inches=0.05,
    )
    print(f"Saved {OUTPUT_DIR / filename}.pdf/.png", flush=True)
    plt.close(fig)

    return summary


def boxstripplot(
    df, x, y, order=None, palette='Set3',
    notch=True, jitter=0.3, strip_size=5, strip_alpha=0.45,
    xlabel=None, ylabel=None,
    width=4.5, height=3.2, ax=None,
):
    """Boxplot with overlaid strip/dot plot (publication-ready)."""
    if order is None:
        order = df[x].unique()

    colors = dict(zip(order, sns.color_palette(palette, len(order))))

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    else:
        fig = ax.get_figure()

    sns.boxplot(
        data=df, x=x, y=y, order=order,
        notch=notch, hue=df[x], palette=colors,
        showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=df, x=x, y=y, order=order,
        linewidth=0.4, jitter=jitter, size=strip_size, alpha=strip_alpha,
        hue=df[x], palette=colors, ax=ax,
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig, ax


def make_boxstripplot(df, acc_summary, filename='boxstripplot_r2adj'):
    """Generate and save a boxstripplot of R²adj distributions."""
    # Map model names to code names and order by mean rank
    df = df.copy()
    df['model_label'] = df['model'].map(MODEL_LABELS).fillna(df['model'])
    model_order = [
        MODEL_LABELS.get(m, m)
        for m in acc_summary['model']
    ]

    fig, ax = boxstripplot(
        df, x='model_label', y='val_r2_adj', order=model_order,
        ylabel=r'$\bar{R}^2$', xlabel='',
        width=4.5, height=3.0,
    )

    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    fig.savefig(
        OUTPUT_DIR / f'{filename}.pdf',
        bbox_inches='tight', pad_inches=0.05,
    )
    fig.savefig(
        OUTPUT_DIR / f'{filename}.png', dpi=300,
        bbox_inches='tight', pad_inches=0.05,
    )
    print(f"Saved {OUTPUT_DIR / filename}.pdf/.png", flush=True)
    plt.close(fig)


def compute_nemenyi_cd(k, n, alpha=0.05):
    """Compute Nemenyi critical difference.

    CD = q_alpha * sqrt(k*(k+1) / (6*n))
    where q_alpha = studentized_range.isf(alpha, k, inf) / sqrt(2).
    """
    q_alpha = studentized_range.isf(alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))


def nemenyi_groups(names, mean_ranks, cd):
    """Find non-significant groups from Nemenyi post-hoc test.

    Returns list of groups (each a tuple of model names) whose pairwise
    rank differences are all within CD.  Groups are maximal contiguous
    intervals in rank order and may overlap.
    """
    idx = np.argsort(mean_ranks)
    sorted_names = [names[i] for i in idx]
    sorted_ranks = [mean_ranks[i] for i in idx]

    groups = []
    n = len(sorted_names)
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
            j += 1
        if j > i:
            group = tuple(sorted_names[i:j + 1])
            if not any(set(group).issubset(set(g)) for g in groups):
                groups.append(group)

    return groups


def _add_cd_bar(ax, cd, fontsize=6.5):
    """Add a Nemenyi critical difference reference bar below the bottom model."""
    # In seaborn horizontal stripplots, y=0 is top, y=n-1 is bottom.
    # ylim is (bottom_y, top_y) with bottom > top (inverted).
    ylim = ax.get_ylim()  # e.g. (7.5, -0.5) for 8 models
    y_bottom = ylim[0]    # larger value = bottom of plot
    y = y_bottom + 0.6    # below last model
    x_start = 1.0
    x_end = x_start + cd
    ax.plot([x_start, x_end], [y, y], color='0.3', linewidth=1.5,
            solid_capstyle='butt', clip_on=False)
    # End ticks
    tick_h = 0.12
    ax.plot([x_start, x_start], [y - tick_h, y + tick_h],
            color='0.3', linewidth=1.0, clip_on=False)
    ax.plot([x_end, x_end], [y - tick_h, y + tick_h],
            color='0.3', linewidth=1.0, clip_on=False)
    # Label (below bar in inverted y)
    ax.text((x_start + x_end) / 2, y + 0.25, f'CD = {cd:.2f}',
            ha='center', va='top', fontsize=fontsize, color='0.3')
    # Expand ylim slightly to make room
    ax.set_ylim(y + 0.55, ylim[1])


def make_combined_accuracy_plot(acc_all, acc_cpu, cd_all=None, cd_cpu=None,
                                filename='linedot_rank_combined'):
    """Figure 1: (a) all-models accuracy ranking, (b) CPU-only accuracy ranking."""
    acc_all = acc_all.copy()
    acc_cpu = acc_cpu.copy()
    def _label(r):
        name = MODEL_LABELS.get(r['model'], r['model'])
        return f"{name}  ({r['wins']:.0f}/{r['seconds']:.0f})"

    acc_all['label'] = acc_all.apply(_label, axis=1)
    acc_cpu['label'] = acc_cpu.apply(_label, axis=1)

    n_max = max(len(acc_all), len(acc_cpu))
    row_h = 0.30
    fig_h = row_h * n_max + 0.6
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, fig_h),
        constrained_layout=True,
    )

    compact = dict(
        dot_size=6, dot_linewidth=0.4, dot_edgecolor='w',
        annot_offset=-0.25, annot_size=6.5, xlim_pad=0.3,
    )

    linedotplot(
        acc_all, x='rank_mean', y='label',
        err_low='rank_q25', err_high='rank_q75',
        xlabel='Mean rank', ylabel='',
        ax=ax_a, **compact,
    )
    ax_a.set_title('(a) All models', fontsize=8, loc='left')

    linedotplot(
        acc_cpu, x='rank_mean', y='label',
        err_low='rank_q25', err_high='rank_q75',
        xlabel='Mean rank', ylabel='',
        ax=ax_b, **compact,
    )
    ax_b.set_title('(b) CPU-only', fontsize=8, loc='left')

    if cd_all is not None:
        _add_cd_bar(ax_a, cd_all)
    if cd_cpu is not None:
        _add_cd_bar(ax_b, cd_cpu)

    for path_suffix in ('pdf', 'png'):
        kw = dict(bbox_inches='tight', pad_inches=0.05)
        if path_suffix == 'png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'{filename}.{path_suffix}', **kw)
    print(f"Saved {OUTPUT_DIR / filename}.pdf/.png", flush=True)
    plt.close(fig)


def print_table(summary, metric_name, metric_col_label):
    """Print a summary table for verification."""
    print(f"\n{'='*70}", flush=True)
    print(f"Table: {metric_name}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Model':<12} {'Wins':>5} {'Mean Rank':>10} "
          f"{'Median':>8} {'Mean':>8} {'Std':>8}", flush=True)
    print('-' * 60, flush=True)
    for _, r in summary.iterrows():
        print(f"{r['model']:<12} {r['wins']:5.0f} {r['rank_mean']:10.2f} "
              f"{r['metric_median']:8.3f} {r['metric_mean']:8.3f} {r['metric_std']:8.3f}",
              flush=True)


if __name__ == '__main__':
    print("Loading data...", flush=True)
    df = load_data()
    print(f"  {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets", flush=True)

    # --- All-models accuracy ---
    acc_all = compute_ranks_and_summary(df, 'val_r2_adj', lower_is_better=False)
    print_table(acc_all, 'Predictive accuracy — all models (adj. R²)', 'R²adj')

    # --- CPU-only accuracy ---
    df_cpu = df[df['model'].isin(CPU_MODELS)].copy()
    acc_cpu = compute_ranks_and_summary(df_cpu, 'val_r2_adj', lower_is_better=False)
    print_table(acc_cpu, 'Predictive accuracy — CPU-only (adj. R²)', 'R²adj')

    # --- Gap (all models) ---
    gap_all = compute_ranks_and_summary(df, 'gap', lower_is_better=True)
    print_table(gap_all, 'Generalisation gap — all models', 'Gap')

    # --- Nemenyi critical differences (scipy) ---
    wide_all = df.pivot(index='dataset', columns='model', values='val_r2_adj').dropna()
    k_all, n_all = wide_all.shape[1], wide_all.shape[0]
    cd_all = compute_nemenyi_cd(k_all, n_all)

    wide_cpu = df_cpu.pivot(index='dataset', columns='model', values='val_r2_adj').dropna()
    k_cpu, n_cpu = wide_cpu.shape[1], wide_cpu.shape[0]
    cd_cpu = compute_nemenyi_cd(k_cpu, n_cpu)

    print(f"\n  Nemenyi CD (all {k_all} models, {n_all} datasets) = {cd_all:.4f}", flush=True)
    print(f"  Nemenyi CD (CPU {k_cpu} models, {n_cpu} datasets) = {cd_cpu:.4f}", flush=True)

    # Friedman test
    stat, p = friedmanchisquare(*[wide_all[col] for col in wide_all.columns])
    print(f"  Friedman test (all models): chi2={stat:.2f}, p={p:.2e}", flush=True)

    # Non-significant groups
    for label, summary, cd_val in [
        ('All models (accuracy)', acc_all, cd_all),
        ('CPU-only (accuracy)', acc_cpu, cd_cpu),
        ('All models (gap)', gap_all, cd_all),
    ]:
        groups = nemenyi_groups(
            summary['model'].values, summary['rank_mean'].values, cd_val
        )
        print(f"\n  {label} non-significant groups:", flush=True)
        for g in groups:
            print(f"    {{{', '.join(g)}}}", flush=True)

    # --- Figure 1: combined accuracy (a) all, (b) CPU-only ---
    make_combined_accuracy_plot(acc_all, acc_cpu, cd_all=cd_all, cd_cpu=cd_cpu)

    # --- Standalone gap figure (tighter spacing) ---
    make_rank_plot(
        gap_all, metric_name='Generalisation gap',
        xlabel='Mean rank', filename='linedot_rank_gap',
        cd=cd_all, height_per_row=0.35, width=4.0, annot_offset=-0.28,
    )

    # --- Individual accuracy plots (tighter, annotations above dots) ---
    make_rank_plot(acc_all, 'Adjusted R² (all)', 'Mean rank', 'linedot_rank_r2adj',
                   height_per_row=0.35, width=4.0, annot_offset=-0.28)
    make_rank_plot(acc_cpu, 'Adjusted R² (CPU)', 'Mean rank', 'linedot_rank_r2adj_cpu',
                   height_per_row=0.35, width=4.0, annot_offset=-0.28)

    # --- Boxstripplot of R²adj distributions ---
    make_boxstripplot(df, acc_all)

    print("\nDone.", flush=True)
