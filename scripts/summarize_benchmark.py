#!/usr/bin/env python
"""
Summarize benchmark results from joblib files.

Creates benchmark_summary/ directory with summary CSVs and summary.md.

Usage:
    python scripts/summarize_benchmark.py results/my_run

    # With CD plots
    python scripts/summarize_benchmark.py results/my_run --cd

    # Exclude specific models
    python scripts/summarize_benchmark.py results/my_run --exclude-models tabpfn

    # Custom output directory
    python scripts/summarize_benchmark.py results/my_run --output-dir my_summary/

    # Include detailed hyperparameters CSV
    python scripts/summarize_benchmark.py results/my_run --detailed

Output:
    results/my_run/benchmark_summary/
        summary.md
        summary.csv
        wins_by_accuracy_metric.csv
        summary_by_stratum.csv
        summary_by_size_stratum.csv
        stratum_size_matrix.csv
        summary_by_target_type.csv
        results_detailed.csv     (with --detailed)
"""

import argparse
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dataset_sizes_cache() -> pd.DataFrame:
    """Load the dataset sizes cache CSV."""
    cache_path = Path(__file__).parent.parent / 'benchmark' / 'data' / 'dataset_sizes_cache.csv'
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return pd.DataFrame()


# Global cache for dataset info
_DATASET_SIZES_CACHE = None


def get_dataset_info_from_cache(dataset_name: str) -> dict:
    """Get dataset info from the sizes cache."""
    global _DATASET_SIZES_CACHE
    if _DATASET_SIZES_CACHE is None:
        _DATASET_SIZES_CACHE = load_dataset_sizes_cache()

    if len(_DATASET_SIZES_CACHE) > 0:
        row = _DATASET_SIZES_CACHE[_DATASET_SIZES_CACHE['dataset'] == dataset_name]
        if len(row) > 0:
            return row.iloc[0].to_dict()
    return {}


def get_dataset_stratum(dataset_name: str) -> str:
    """Get stratum for a dataset from the loader registry."""
    # Try cache first
    info = get_dataset_info_from_cache(dataset_name)
    if 'stratum' in info:
        return info['stratum']
    # Fallback to loader
    from perbf.data.loader import get_dataset_info
    info = get_dataset_info(dataset_name)
    return info['stratum']


def get_dataset_size_stratum(dataset_name: str) -> str:
    """Get size stratum for a dataset: Small (<1K), Medium (1K-10K), Large (>=10K)."""
    info = get_dataset_info_from_cache(dataset_name)
    n = info.get('n_samples', 0)
    if n < 1000:
        return 'Small (<1K)'
    elif n < 10000:
        return 'Medium (1K-10K)'
    else:
        return 'Large (>=10K)'


def get_dataset_n_samples(dataset_name: str) -> int:
    """Get n_samples for a dataset from cache."""
    info = get_dataset_info_from_cache(dataset_name)
    return info.get('n_samples', 0)


def is_ordinal_dataset(dataset_name: str) -> bool:
    """Check if a dataset is ordinal regression from cache or loader."""
    # Try cache first
    info = get_dataset_info_from_cache(dataset_name)
    if 'ordinal' in info:
        return bool(info['ordinal'])
    # Fallback to loader
    try:
        from perbf.data.loader import is_ordinal_dataset as _is_ordinal
        return _is_ordinal(dataset_name)
    except ImportError:
        return False


def is_discrete_dataset(dataset_name: str) -> bool:
    """Check if a dataset has discrete targets (counts, integers) from cache or loader."""
    # Try cache first
    info = get_dataset_info_from_cache(dataset_name)
    if 'discrete' in info:
        return bool(info['discrete'])
    # Fallback to loader
    try:
        from perbf.data.loader import is_discrete_dataset as _is_discrete
        return _is_discrete(dataset_name)
    except ImportError:
        return False


def _extract_hyperparams(fold_results: list, model_name: str) -> dict:
    """Extract best hyperparameters from fold results.

    Returns aggregated hyperparameters:
    - Numeric params: mean across folds
    - Categorical params: mode (most common)
    """
    if not fold_results:
        return {}

    # Collect params from all folds
    all_params = [fr.get('best_params', {}) for fr in fold_results if fr.get('best_params')]
    if not all_params:
        return {}

    # Get all param names
    param_names = set()
    for p in all_params:
        param_names.update(p.keys())

    result = {}
    for name in param_names:
        values = [p.get(name) for p in all_params if name in p]
        if not values:
            continue

        # Prefix with hp_ to distinguish from other columns
        col_name = f'hp_{name}'

        # Check if numeric or categorical
        if all(isinstance(v, (int, float)) for v in values):
            # Numeric: use mean
            result[col_name] = np.mean(values)
        else:
            # Categorical: use mode (most common)
            from collections import Counter
            result[col_name] = Counter(values).most_common(1)[0][0]

    return result


def _extract_model_info(fold_results: list) -> dict:
    """Extract model complexity info from fold results.

    Returns mean of numeric model_info fields across folds.
    """
    if not fold_results:
        return {}

    # Collect model_info from all folds
    all_info = [fr.get('model_info', {}) for fr in fold_results if fr.get('model_info')]
    if not all_info:
        return {}

    # Get all field names
    field_names = set()
    for info in all_info:
        field_names.update(info.keys())

    result = {}
    for name in field_names:
        values = [info.get(name) for info in all_info if name in info and info[name] is not None]
        if not values:
            continue

        # Prefix with mi_ (model info)
        col_name = f'mi_{name}'

        # Mean for numeric, mode for categorical
        if all(isinstance(v, (int, float)) for v in values):
            result[col_name] = np.mean(values)
        else:
            from collections import Counter
            result[col_name] = Counter(values).most_common(1)[0][0]

    return result


def load_results(results_dir: Path, include_hyperparams: bool = False) -> pd.DataFrame:
    """Load all joblib results into a DataFrame.

    Args:
        results_dir: Directory containing joblib result files
        include_hyperparams: If True, extract tuned hyperparameters and model info
                            from fold_results (slower but more detailed)

    Returns:
        DataFrame with one row per model×dataset combination
    """
    rows = []
    for f in results_dir.glob("*.joblib"):
        try:
            # Extract model and dataset from filename
            parts = f.stem.split('_')
            model = parts[0]
            dataset = '_'.join(parts[1:-1])  # Remove date suffix

            res = joblib.load(f)
            # Handle both dict and object formats
            if hasattr(res, '__dict__'):
                res = res.__dict__

            row = {
                'model': res.get('model_name', res.get('model', model)),
                'dataset': res.get('dataset_name', res.get('dataset', dataset)),
                'val_r2': res.get('r2_val', res.get('val_r2', np.nan)),
                'val_r2_std': res.get('r2_val_std', np.nan),
                'val_r2_median': res.get('r2_val_median', np.nan),
                'val_r2_adj': res.get('r2_val_adj', np.nan),
                'val_mae': res.get('mae_val', res.get('val_mae', np.nan)),
                'val_mae_std': res.get('mae_val_std', np.nan),
                'val_rmse': res.get('rmse_val', res.get('val_rmse', np.nan)),
                'val_rmse_std': res.get('rmse_val_std', np.nan),
                'train_r2': res.get('r2_train', res.get('train_r2', np.nan)),
                'gap': res.get('gap', np.nan),
                # Timing
                'tune_time': res.get('tune_time', np.nan),
                'train_time': res.get('train_time', np.nan),
                'eval_time': res.get('predict_time', res.get('eval_time', np.nan)),
                'total_time': res.get('total_time', np.nan),
            }

            # Compute gap if not present
            if pd.isna(row['gap']) and not pd.isna(row['train_r2']) and not pd.isna(row['val_r2']):
                row['gap'] = row['train_r2'] - row['val_r2']

            # Add stratum info
            row['stratum'] = get_dataset_stratum(row['dataset'])
            row['size_stratum'] = get_dataset_size_stratum(row['dataset'])
            row['n_samples'] = get_dataset_n_samples(row['dataset'])
            row['ordinal'] = is_ordinal_dataset(row['dataset'])
            row['discrete'] = is_discrete_dataset(row['dataset'])

            # Get n_features from cache
            info = get_dataset_info_from_cache(row['dataset'])
            row['n_features'] = info.get('n_features', np.nan)

            # Effective sample size after subsampling (for ERBF/TabPFN)
            row['n_samples_used'] = res.get('n_samples_used', row['n_samples'])
            row['subsampled'] = res.get('subsampled', False)

            # Per-instance inference time (microseconds)
            if not pd.isna(row['eval_time']) and row['n_samples'] > 0:
                row['eval_time_per_instance_us'] = (row['eval_time'] / row['n_samples']) * 1e6
            else:
                row['eval_time_per_instance_us'] = np.nan

            # Stress test flags
            row['no_tune'] = res.get('no_tune', False)
            row['feature_selected'] = 'feature_selection' in res and res['feature_selection'] is not None
            row['n_features_original'] = res.get('n_features_original', row.get('n_features', np.nan))

            # Extract hyperparameters and model info if requested
            if include_hyperparams:
                fold_results = res.get('fold_results', [])
                model_name = row['model']

                # Tuned hyperparameters (hp_* columns)
                hyperparams = _extract_hyperparams(fold_results, model_name)
                row.update(hyperparams)

                # Model complexity info (mi_* columns)
                model_info = _extract_model_info(fold_results)
                row.update(model_info)

            rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to load {f.name}: {e}", flush=True)

    return pd.DataFrame(rows)


def compute_ranks(df: pd.DataFrame, metric: str = 'val_r2', higher_better: bool = True) -> pd.DataFrame:
    """Compute per-dataset ranks for each model."""
    ranks = []
    for dataset, group in df.groupby('dataset'):
        group = group.copy()
        if higher_better:
            group['rank'] = group[metric].rank(ascending=False, method='min')
        else:
            group['rank'] = group[metric].rank(ascending=True, method='min')
        ranks.append(group[['model', 'dataset', 'rank']])
    return pd.concat(ranks, ignore_index=True)


def compute_wins(df: pd.DataFrame, metric: str = 'val_r2', higher_better: bool = True) -> dict:
    """Count wins per model (rank=1 counts)."""
    ranks_df = compute_ranks(df, metric, higher_better)
    wins = ranks_df[ranks_df['rank'] == 1].groupby('model').size().to_dict()
    return wins


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics per model."""
    # Compute ranks for R² adjusted (primary metric for cross-dataset comparison)
    ranks_df = compute_ranks(df, 'val_r2_adj', higher_better=True)
    mean_ranks = ranks_df.groupby('model')['rank'].mean()

    # Compute ranks for MAE and RMSE (scale-free comparison)
    # Note: MAE/RMSE values themselves are scale-dependent and NOT aggregated
    mae_ranks_df = compute_ranks(df, 'val_mae', higher_better=False)
    mae_mean_ranks = mae_ranks_df.groupby('model')['rank'].mean()
    rmse_ranks_df = compute_ranks(df, 'val_rmse', higher_better=False)
    rmse_mean_ranks = rmse_ranks_df.groupby('model')['rank'].mean()

    # Compute wins (R² adjusted - higher better) and gap wins (lower better)
    wins = compute_wins(df, 'val_r2_adj', higher_better=True)
    # Exclude ridge from gap wins - it underfits, so low gap is not meaningful
    df_no_ridge = df[df['model'] != 'ridge']
    gap_wins = compute_wins(df_no_ridge, 'gap', higher_better=False)
    mae_wins = compute_wins(df, 'val_mae', higher_better=False)
    rmse_wins = compute_wins(df, 'val_rmse', higher_better=False)

    # Aggregate metrics
    # Note: MAE/RMSE mean/std/median are NOT included - they are scale-dependent
    # and meaningless when aggregated across datasets with different target scales
    summary = df.groupby('model').agg({
        'val_r2': ['mean', 'std', 'median', 'count'],
        'val_r2_adj': ['mean', 'std', 'median'],
        'gap': ['mean', 'std', 'median'],
        # Timing
        'tune_time': ['mean', 'sum'],
        'train_time': ['mean', 'sum'],
        'eval_time': ['mean', 'sum'],
        'total_time': ['mean', 'sum'],
        'eval_time_per_instance_us': ['mean', 'median'],
    })

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.rename(columns={'val_r2_count': 'n_datasets'})

    # Add ranks and wins
    summary['mean_rank'] = mean_ranks
    summary['wins'] = summary.index.map(lambda x: wins.get(x, 0))
    summary['gap_wins'] = summary.index.map(lambda x: gap_wins.get(x, 0))
    # MAE/RMSE: mean rank and wins (scale-free)
    summary['mae_mean_rank'] = mae_mean_ranks
    summary['mae_wins'] = summary.index.map(lambda x: mae_wins.get(x, 0))
    summary['rmse_mean_rank'] = rmse_mean_ranks
    summary['rmse_wins'] = summary.index.map(lambda x: rmse_wins.get(x, 0))

    # Reorder columns
    cols = ['n_datasets', 'wins', 'gap_wins', 'mean_rank',
            'val_r2_mean', 'val_r2_std', 'val_r2_median',
            'val_r2_adj_mean', 'val_r2_adj_std', 'val_r2_adj_median',
            'mae_wins', 'mae_mean_rank', 'rmse_wins', 'rmse_mean_rank',
            'gap_mean', 'gap_std', 'gap_median',
            # Timing
            'tune_time_mean', 'tune_time_sum',
            'train_time_mean', 'train_time_sum',
            'eval_time_mean', 'eval_time_sum',
            'total_time_mean', 'total_time_sum',
            # Per-instance inference
            'eval_time_per_instance_us_mean', 'eval_time_per_instance_us_median']
    summary = summary[[c for c in cols if c in summary.columns]]

    return summary.sort_values('mean_rank')


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if pd.isna(seconds):
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_summary(summary: pd.DataFrame) -> str:
    """Format summary as readable table."""
    lines = []

    def get_val(row, col):
        return row[col] if col in row.index else np.nan

    # Main metrics table (wins and rank based on R² adjusted)
    lines.append("=== Performance Metrics (mean±std / median) ===")
    lines.append("Note: Wins and Rank computed using R² adjusted for cross-dataset comparability")
    lines.append(f"{'Model':<12} {'N':>4} {'R²adjW':>6} {'GapW':>5} {'Rank':>6} {'R² mean±std':>14} {'med':>6} {'R²adj mean±std':>16} {'med':>6} {'Gap mean±std':>12} {'med':>6}")
    lines.append("-" * 120)

    for model, row in summary.iterrows():
        r2_str = f"{get_val(row, 'val_r2_mean'):.3f}±{get_val(row, 'val_r2_std'):.3f}"
        r2_med = f"{get_val(row, 'val_r2_median'):.3f}" if not pd.isna(get_val(row, 'val_r2_median')) else "N/A"
        r2_adj_str = f"{get_val(row, 'val_r2_adj_mean'):.3f}±{get_val(row, 'val_r2_adj_std'):.3f}"
        r2_adj_med = f"{get_val(row, 'val_r2_adj_median'):.3f}" if not pd.isna(get_val(row, 'val_r2_adj_median')) else "N/A"
        gap_str = f"{get_val(row, 'gap_mean'):.3f}±{get_val(row, 'gap_std'):.3f}" if not pd.isna(get_val(row, 'gap_mean')) else "N/A"
        gap_med = f"{get_val(row, 'gap_median'):.3f}" if not pd.isna(get_val(row, 'gap_median')) else "N/A"
        gap_wins = int(get_val(row, 'gap_wins')) if not pd.isna(get_val(row, 'gap_wins')) else 0

        lines.append(f"{model:<12} {int(get_val(row, 'n_datasets')):>4} {int(get_val(row, 'wins')):>6} {gap_wins:>5} {get_val(row, 'mean_rank'):>6.2f} {r2_str:>14} {r2_med:>6} {r2_adj_str:>16} {r2_adj_med:>6} {gap_str:>12} {gap_med:>6}")

    # MAE/RMSE table (wins and mean rank - scale-free metrics)
    # Note: MAE/RMSE values are scale-dependent, so we only show ranks and wins
    lines.append("")
    lines.append("=== Error Metrics (wins & mean rank - scale-free) ===")
    lines.append("Note: MAE/RMSE values are scale-dependent; only ranks are meaningful across datasets")
    lines.append(f"{'Model':<12} {'MAE Wins':>10} {'MAE Rank':>10} {'RMSE Wins':>11} {'RMSE Rank':>11}")
    lines.append("-" * 60)

    for model, row in summary.iterrows():
        mae_wins = int(get_val(row, 'mae_wins')) if not pd.isna(get_val(row, 'mae_wins')) else 0
        mae_rank = f"{get_val(row, 'mae_mean_rank'):.2f}" if not pd.isna(get_val(row, 'mae_mean_rank')) else "N/A"
        rmse_wins = int(get_val(row, 'rmse_wins')) if not pd.isna(get_val(row, 'rmse_wins')) else 0
        rmse_rank = f"{get_val(row, 'rmse_mean_rank'):.2f}" if not pd.isna(get_val(row, 'rmse_mean_rank')) else "N/A"

        lines.append(f"{model:<12} {mae_wins:>10} {mae_rank:>10} {rmse_wins:>11} {rmse_rank:>11}")

    # Smoothness metrics removed (22Jan26) - current metrics don't confirm hypothesis
    # Lipschitz measures prediction steepness, not model inductive bias
    # A model that fits well will have steep predictions if data has steep gradients
    # TODO: Revisit with synthetic test functions where ground truth is known

    # Timing table
    lines.append("")
    lines.append("=== Timing (per dataset avg / total) ===")
    lines.append(f"{'Model':<12} {'Tune':>14} {'Train':>14} {'Eval':>14} {'Total':>14}")
    lines.append("-" * 70)

    for model, row in summary.iterrows():
        tune_str = f"{format_time(get_val(row, 'tune_time_mean'))}/{format_time(get_val(row, 'tune_time_sum'))}"
        train_str = f"{format_time(get_val(row, 'train_time_mean'))}/{format_time(get_val(row, 'train_time_sum'))}"
        eval_str = f"{format_time(get_val(row, 'eval_time_mean'))}/{format_time(get_val(row, 'eval_time_sum'))}"
        total_str = f"{format_time(get_val(row, 'total_time_mean'))}/{format_time(get_val(row, 'total_time_sum'))}"

        lines.append(f"{model:<12} {tune_str:>14} {train_str:>14} {eval_str:>14} {total_str:>14}")

    # Per-instance inference time
    lines.append("")
    lines.append("=== Per-Instance Inference Time ===")
    lines.append(f"{'Model':<12} {'Mean (µs)':>12} {'Median (µs)':>12}")
    lines.append("-" * 38)

    for model, row in summary.iterrows():
        mean_us = get_val(row, 'eval_time_per_instance_us_mean')
        median_us = get_val(row, 'eval_time_per_instance_us_median')
        mean_str = f"{mean_us:.1f}" if not pd.isna(mean_us) else "N/A"
        median_str = f"{median_us:.1f}" if not pd.isna(median_us) else "N/A"
        lines.append(f"{model:<12} {mean_str:>12} {median_str:>12}")

    return "\n".join(lines)


def compute_wins_by_accuracy_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute wins for each model across accuracy metrics only.

    Note: Gap is excluded (will be analyzed separately via Pareto front).
    R²adj is excluded (redundant with R² for wins).
    """
    metrics = {
        'R²': ('val_r2', False),       # higher better
        'MAE': ('val_mae', True),       # lower better
        'RMSE': ('val_rmse', True),     # lower better
    }

    results = {model: {} for model in df['model'].unique()}

    for metric_name, (col, lower_better) in metrics.items():
        for dataset, grp in df.groupby('dataset'):
            grp = grp.dropna(subset=[col])
            if len(grp) == 0:
                continue
            if lower_better:
                best = grp.loc[grp[col].idxmin(), 'model']
            else:
                best = grp.loc[grp[col].idxmax(), 'model']
            results[best][metric_name] = results[best].get(metric_name, 0) + 1

    # Convert to DataFrame
    rows = []
    for model in sorted(results.keys()):
        row = {'model': model}
        total = 0
        for metric in metrics.keys():
            wins = results[model].get(metric, 0)
            row[metric] = wins
            total += wins
        row['Total'] = total
        rows.append(row)

    wins_df = pd.DataFrame(rows)
    # Reorder columns
    wins_df = wins_df[['model', 'R²', 'MAE', 'RMSE', 'Total']]
    wins_df = wins_df.sort_values('Total', ascending=False).reset_index(drop=True)
    return wins_df


def summarize_by_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate per-stratum summary: wins and mean rank per model.

    Returns DataFrame with columns: stratum, model, n_datasets, r2_adj_wins, gap_wins, mean_rank, r2_adj_mean, r2_adj_median, gap_median
    """
    rows = []

    for stratum in sorted(df['stratum'].unique()):
        stratum_df = df[df['stratum'] == stratum]
        n_datasets = stratum_df['dataset'].nunique()

        # Compute R² adjusted wins and mean rank
        r2_adj_wins = compute_wins(stratum_df, 'val_r2_adj', higher_better=True)

        # Compute gap wins excluding ridge (it underfits, low gap is meaningless)
        stratum_no_ridge = stratum_df[stratum_df['model'] != 'ridge']
        gap_wins = compute_wins(stratum_no_ridge, 'gap', higher_better=False)

        ranks_df = compute_ranks(stratum_df, 'val_r2_adj', higher_better=True)
        mean_ranks = ranks_df.groupby('model')['rank'].mean()

        for model in stratum_df['model'].unique():
            model_df = stratum_df[stratum_df['model'] == model]
            rows.append({
                'stratum': stratum,
                'model': model,
                'n_datasets': len(model_df),
                'r2_adj_wins': r2_adj_wins.get(model, 0),
                'gap_wins': gap_wins.get(model, 0),  # Ridge excluded from competition
                'mean_rank': mean_ranks.get(model, np.nan),
                'r2_adj_mean': model_df['val_r2_adj'].mean(),
                'r2_adj_median': model_df['val_r2_adj'].median(),
                'gap_median': model_df['gap'].median(),
            })

    result = pd.DataFrame(rows)
    return result.sort_values(['stratum', 'mean_rank'])


def format_stratum_summary(df: pd.DataFrame, stratum_df: pd.DataFrame) -> str:
    """Format per-stratum summary as readable table."""
    lines = []

    # Stratum descriptions (single point of truth in benchmark/data/strata.py)
    from perbf.data.strata import STRATUM_NAMES_SHORT
    stratum_names = STRATUM_NAMES_SHORT

    for stratum in sorted(stratum_df['stratum'].unique()):
        s_df = stratum_df[stratum_df['stratum'] == stratum].copy()
        n_datasets = s_df['n_datasets'].max()

        lines.append(f"\n=== Stratum {stratum}: {stratum_names.get(stratum, 'Unknown')} ===")
        lines.append(f"Datasets: {n_datasets}")
        lines.append(f"{'Model':<12} {'R²adjW':>6} {'Gap*':>5} {'Rank':>6} {'R²adj mean':>11} {'med':>6} {'Gap med':>8}")
        lines.append("-" * 62)

        for _, row in s_df.iterrows():
            r2_adj_mean = f"{row['r2_adj_mean']:.3f}" if not pd.isna(row['r2_adj_mean']) else "N/A"
            r2_adj_med = f"{row['r2_adj_median']:.3f}" if not pd.isna(row['r2_adj_median']) else "N/A"
            gap_med = f"{row['gap_median']:.3f}" if not pd.isna(row['gap_median']) else "N/A"
            lines.append(f"{row['model']:<12} {int(row['r2_adj_wins']):>6} {int(row['gap_wins']):>5} {row['mean_rank']:>6.2f} {r2_adj_mean:>11} {r2_adj_med:>6} {gap_med:>8}")

    return "\n".join(lines)


def summarize_by_size_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate per-size-stratum summary: wins and mean rank per model.

    Size strata: Small (<1K), Medium (1K-10K), Large (>=10K)

    Returns DataFrame with columns: size_stratum, model, n_datasets, r2_adj_wins, gap_wins, mean_rank, r2_adj_mean, r2_adj_median, gap_median
    """
    rows = []

    # Order size strata logically
    size_order = ['Small (<1K)', 'Medium (1K-10K)', 'Large (>=10K)']
    available = [s for s in size_order if s in df['size_stratum'].unique()]

    for size_stratum in available:
        stratum_df = df[df['size_stratum'] == size_stratum]
        n_datasets = stratum_df['dataset'].nunique()

        # Compute R² adjusted wins and mean rank
        r2_adj_wins = compute_wins(stratum_df, 'val_r2_adj', higher_better=True)

        # Compute gap wins excluding ridge (it underfits, low gap is meaningless)
        stratum_no_ridge = stratum_df[stratum_df['model'] != 'ridge']
        gap_wins = compute_wins(stratum_no_ridge, 'gap', higher_better=False)

        ranks_df = compute_ranks(stratum_df, 'val_r2_adj', higher_better=True)
        mean_ranks = ranks_df.groupby('model')['rank'].mean()

        for model in stratum_df['model'].unique():
            model_df = stratum_df[stratum_df['model'] == model]
            rows.append({
                'size_stratum': size_stratum,
                'model': model,
                'n_datasets': len(model_df),
                'r2_adj_wins': r2_adj_wins.get(model, 0),
                'gap_wins': gap_wins.get(model, 0),  # Ridge excluded from competition
                'mean_rank': mean_ranks.get(model, np.nan),
                'r2_adj_mean': model_df['val_r2_adj'].mean(),
                'r2_adj_median': model_df['val_r2_adj'].median(),
                'gap_median': model_df['gap'].median(),
            })

    result = pd.DataFrame(rows)
    # Sort by size stratum order, then by mean rank
    result['size_order'] = result['size_stratum'].map({s: i for i, s in enumerate(size_order)})
    result = result.sort_values(['size_order', 'mean_rank']).drop(columns=['size_order'])
    return result


def format_size_stratum_summary(df: pd.DataFrame, size_stratum_df: pd.DataFrame) -> str:
    """Format per-size-stratum summary as readable table."""
    lines = []

    # Size stratum descriptions
    descriptions = {
        'Small (<1K)': 'n < 1,000 samples',
        'Medium (1K-10K)': '1,000 <= n < 10,000 samples',
        'Large (>=10K)': 'n >= 10,000 samples',
    }

    # Order size strata logically
    size_order = ['Small (<1K)', 'Medium (1K-10K)', 'Large (>=10K)']
    available = [s for s in size_order if s in size_stratum_df['size_stratum'].unique()]

    for size_stratum in available:
        s_df = size_stratum_df[size_stratum_df['size_stratum'] == size_stratum].copy()
        n_datasets = s_df['n_datasets'].max()

        lines.append(f"\n=== Size: {size_stratum} ({descriptions.get(size_stratum, '')}) ===")
        lines.append(f"Datasets: {n_datasets}")
        lines.append(f"{'Model':<12} {'R²adjW':>6} {'Gap*':>5} {'Rank':>6} {'R²adj mean':>11} {'med':>6} {'Gap med':>8}")
        lines.append("-" * 62)

        for _, row in s_df.iterrows():
            r2_adj_mean = f"{row['r2_adj_mean']:.3f}" if not pd.isna(row['r2_adj_mean']) else "N/A"
            r2_adj_med = f"{row['r2_adj_median']:.3f}" if not pd.isna(row['r2_adj_median']) else "N/A"
            gap_med = f"{row['gap_median']:.3f}" if not pd.isna(row['gap_median']) else "N/A"
            lines.append(f"{row['model']:<12} {int(row['r2_adj_wins']):>6} {int(row['gap_wins']):>5} {row['mean_rank']:>6.2f} {r2_adj_mean:>11} {r2_adj_med:>6} {gap_med:>8}")

    return "\n".join(lines)


def summarize_stratum_size_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate stratum x size matrix showing winner model in each cell.

    Returns DataFrame with rows=strata (S1-S4), cols=size strata, values=winner model.
    """
    size_order = ['Small (<1K)', 'Medium (1K-10K)', 'Large (>=10K)']
    stratum_order = ['S1', 'S2', 'S3', 'S4']

    results = {}
    for stratum in stratum_order:
        results[stratum] = {}
        for size in size_order:
            cell_df = df[(df['stratum'] == stratum) & (df['size_stratum'] == size)]
            n_datasets = cell_df['dataset'].nunique()

            if n_datasets == 0:
                results[stratum][size] = '-'
                continue

            # Find winner by R² adjusted wins
            wins = compute_wins(cell_df, 'val_r2_adj', higher_better=True)
            if wins:
                winner = max(wins, key=wins.get)
                win_count = wins[winner]
                results[stratum][size] = f"{winner} ({win_count}/{n_datasets})"
            else:
                results[stratum][size] = '-'

    # Convert to DataFrame
    matrix = pd.DataFrame(results).T
    matrix = matrix[size_order]  # Ensure column order
    matrix.index.name = 'stratum'
    return matrix


def format_stratum_size_matrix(matrix: pd.DataFrame, df: pd.DataFrame) -> str:
    """Format stratum x size matrix as readable table."""
    lines = []
    lines.append("\n=== Stratum x Size Matrix (R²adj winner) ===")
    lines.append("Cell format: winner_model (wins/n_datasets)")

    # Get dataset counts per cell for reference
    size_order = ['Small (<1K)', 'Medium (1K-10K)', 'Large (>=10K)']

    # Header
    lines.append(f"{'Stratum':<8} {'Small (<1K)':<20} {'Medium (1K-10K)':<20} {'Large (>=10K)':<20}")
    lines.append("-" * 70)

    for stratum in matrix.index:
        row_vals = [f"{matrix.loc[stratum, s]:<20}" for s in size_order]
        lines.append(f"{stratum:<8} {''.join(row_vals)}")

    # Add dataset count reference
    lines.append("")
    lines.append("Dataset counts per cell:")
    for stratum in ['S1', 'S2', 'S3', 'S4']:
        counts = []
        for size in size_order:
            n = df[(df['stratum'] == stratum) & (df['size_stratum'] == size)]['dataset'].nunique()
            counts.append(str(n))
        lines.append(f"  {stratum}: {', '.join(counts)}")

    return "\n".join(lines)


def summarize_by_target_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate per-target-type summary: wins and mean rank per model.

    Target types: Continuous, Discrete (counts/integers, includes ordinal)

    Returns DataFrame with columns: target_type, model, n_datasets, r2_adj_wins, gap_wins, mean_rank, r2_adj_mean, r2_adj_median, gap_median
    """
    rows = []

    for target_type, is_discrete in [('Continuous', False), ('Discrete', True)]:
        type_df = df[df['discrete'] == is_discrete]
        if len(type_df) == 0:
            continue

        n_datasets = type_df['dataset'].nunique()

        # Compute R² adjusted wins and mean rank
        r2_adj_wins = compute_wins(type_df, 'val_r2_adj', higher_better=True)

        # Compute gap wins excluding ridge (it underfits, low gap is meaningless)
        type_no_ridge = type_df[type_df['model'] != 'ridge']
        gap_wins = compute_wins(type_no_ridge, 'gap', higher_better=False)

        ranks_df = compute_ranks(type_df, 'val_r2_adj', higher_better=True)
        mean_ranks = ranks_df.groupby('model')['rank'].mean()

        for model in type_df['model'].unique():
            model_df = type_df[type_df['model'] == model]
            rows.append({
                'target_type': target_type,
                'model': model,
                'n_datasets': len(model_df),
                'r2_adj_wins': r2_adj_wins.get(model, 0),
                'gap_wins': gap_wins.get(model, 0),
                'mean_rank': mean_ranks.get(model, np.nan),
                'r2_adj_mean': model_df['val_r2_adj'].mean(),
                'r2_adj_median': model_df['val_r2_adj'].median(),
                'gap_median': model_df['gap'].median(),
            })

    result = pd.DataFrame(rows)
    return result.sort_values(['target_type', 'mean_rank'])


def format_target_type_summary(df: pd.DataFrame, target_type_df: pd.DataFrame) -> str:
    """Format per-target-type summary as readable table."""
    lines = []

    descriptions = {
        'Continuous': 'Standard continuous regression targets',
        'Discrete': 'Discrete targets (counts, integers, ordinal ratings)',
    }

    for target_type in ['Continuous', 'Discrete']:
        t_df = target_type_df[target_type_df['target_type'] == target_type].copy()
        if len(t_df) == 0:
            continue

        n_datasets = t_df['n_datasets'].max()

        lines.append(f"\n=== Target Type: {target_type} ({descriptions.get(target_type, '')}) ===")
        lines.append(f"Datasets: {n_datasets}")

        # List discrete datasets with ordinal flag noted
        if target_type == 'Discrete':
            discrete_datasets = df[df['discrete'] == True]['dataset'].unique()
            ordinal_datasets = set(df[df['ordinal'] == True]['dataset'].unique())
            dataset_list = []
            for d in sorted(discrete_datasets):
                if d in ordinal_datasets:
                    dataset_list.append(f"{d} (ordinal)")
                else:
                    dataset_list.append(d)
            lines.append(f"Discrete datasets: {', '.join(dataset_list)}")

        lines.append(f"{'Model':<12} {'R²adjW':>6} {'Gap*':>5} {'Rank':>6} {'R²adj mean':>11} {'med':>6} {'Gap med':>8}")
        lines.append("-" * 62)

        for _, row in t_df.iterrows():
            r2_adj_mean = f"{row['r2_adj_mean']:.3f}" if not pd.isna(row['r2_adj_mean']) else "N/A"
            r2_adj_med = f"{row['r2_adj_median']:.3f}" if not pd.isna(row['r2_adj_median']) else "N/A"
            gap_med = f"{row['gap_median']:.3f}" if not pd.isna(row['gap_median']) else "N/A"
            lines.append(f"{row['model']:<12} {int(row['r2_adj_wins']):>6} {int(row['gap_wins']):>5} {row['mean_rank']:>6.2f} {r2_adj_mean:>11} {r2_adj_med:>6} {gap_med:>8}")

    return "\n".join(lines)


def create_dataset_model_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset × model pivot table with adjusted R² values.

    Rows: datasets sorted by stratum then name
    Columns: models sorted by mean rank (best first)
    Values: adjusted R²

    Returns:
        DataFrame with datasets as rows, models as columns, R²adj as values
    """
    # Compute model rankings (mean rank across datasets)
    ranks_df = compute_ranks(df, 'val_r2_adj', higher_better=True)
    model_mean_ranks = ranks_df.groupby('model')['rank'].mean().sort_values()
    model_order = model_mean_ranks.index.tolist()

    # Create pivot table
    pivot = df.pivot_table(
        index='dataset',
        columns='model',
        values='val_r2_adj',
        aggfunc='first'  # In case of duplicates, take first
    )

    # Reorder columns by model ranking (best first)
    pivot = pivot[[m for m in model_order if m in pivot.columns]]

    # Add stratum for sorting
    dataset_strata = df.drop_duplicates('dataset').set_index('dataset')['stratum']
    pivot['_stratum'] = pivot.index.map(dataset_strata)

    # Sort by stratum then dataset name
    pivot = pivot.sort_values(['_stratum', pivot.index.name])

    # Move stratum to first column position (for reference)
    stratum_col = pivot.pop('_stratum')
    pivot.insert(0, 'stratum', stratum_col)

    return pivot


def format_dataset_model_pivot(pivot: pd.DataFrame) -> str:
    """Format dataset × model pivot as readable table."""
    lines = []
    lines.append("\n=== Dataset × Model: Adjusted R² ===")
    lines.append("Rows sorted by stratum/dataset, columns by model ranking (best first)")
    lines.append("")

    # Get model columns (exclude stratum)
    model_cols = [c for c in pivot.columns if c != 'stratum']

    # Header
    header = f"{'Dataset':<35} {'Str':>3}"
    for model in model_cols:
        header += f" {model:>9}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    current_stratum = None
    for dataset, row in pivot.iterrows():
        stratum = row['stratum']

        # Add separator between strata
        if current_stratum is not None and stratum != current_stratum:
            lines.append("")
        current_stratum = stratum

        row_str = f"{dataset:<35} {stratum:>3}"
        for model in model_cols:
            val = row[model]
            if pd.isna(val):
                row_str += f" {'---':>9}"
            else:
                row_str += f" {val:>9.3f}"
        lines.append(row_str)

    return "\n".join(lines)


def generate_cd_plots(df: pd.DataFrame, output_dir: Path, alpha: float = 0.05) -> list:
    """
    Generate Critical Difference plots using autorank package.

    Requires: pip install autorank

    Returns list of lines to include in summary.md.
    """
    lines = []
    try:
        from autorank import autorank, plot_stats, create_report
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: autorank not installed. Run: pip install autorank")
        return lines

    output_dir.mkdir(parents=True, exist_ok=True)

    lines.append("")
    lines.append(f"=== Critical Difference Analysis (alpha={alpha}) ===")
    lines.append("Friedman test with Nemenyi post-hoc (autorank)")

    # Metrics to plot (column, higher_is_better, display_name, exclude_models)
    # Note: R² removed - adjusted R² is sufficient for cross-dataset comparison
    metrics = [
        ('val_r2_adj', True, 'R² adjusted', []),
        ('gap', False, 'Generalization Gap', ['ridge']),  # Exclude ridge (underfits)
    ]

    for col, higher_better, name, exclude_models in metrics:
        # Pivot to wide format: rows=datasets, columns=models, values=metric
        df_metric = df.dropna(subset=[col])

        # Exclude specified models (e.g., ridge for gap - it underfits)
        if exclude_models:
            df_metric = df_metric[~df_metric['model'].isin(exclude_models)]
        if len(df_metric) == 0:
            continue

        # Drop duplicates (keep last) in case of re-runs
        df_metric = df_metric.drop_duplicates(subset=['dataset', 'model'], keep='last')
        wide = df_metric.pivot(index='dataset', columns='model', values=col)

        # For metrics where lower is better, negate
        if not higher_better:
            wide = -wide

        # Drop any columns/rows with all NaN
        wide = wide.dropna(axis=1, how='all').dropna(axis=0, how='all')

        if wide.shape[1] < 2:
            print(f"Skipping {name}: not enough models with data")
            continue

        print(f"\nGenerating CD plot for {name}...", flush=True)

        try:
            result = autorank(wide, alpha=alpha, verbose=False)

            # Add to summary lines
            lines.append("")
            lines.append(f"--- {name} ---")
            lines.append(f"Critical Difference (CD): {result.cd:.3f}")
            if exclude_models:
                lines.append(f"Note: {', '.join(exclude_models)} excluded from comparison")

            # Show available columns from rankdf
            rank_cols = [c for c in ['meanrank', 'mean', 'std', 'median', 'mad'] if c in result.rankdf.columns]
            lines.append("Rankings:")
            lines.append(result.rankdf[rank_cols].to_string())

            # Also print to console
            print(f"  Critical Difference (CD): {result.cd:.3f}")
            print(f"  Rankings:\n{result.rankdf[rank_cols].to_string()}")

            # Generate plot
            fig = plt.figure(figsize=(10, 4))
            plot_stats(result)

            safe_name = name.replace(' ', '_').replace('²', '2')
            fig_path = output_dir / f"cd_plot_{safe_name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            print(f"  Saved: {fig_path}")
            lines.append(f"Plot: {fig_path.name}")

            # Save report and include in summary
            # Note: create_report prints to stdout, need to capture it
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            create_report(result)
            report = buffer.getvalue()
            sys.stdout = old_stdout

            if report:
                report_path = output_dir / f"cd_report_{safe_name}.txt"
                report_path.write_text(report)
                # Add report to summary (indent for readability)
                lines.append("")
                lines.append("Statistical interpretation:")
                for report_line in report.strip().split('\n'):
                    lines.append(f"  {report_line}")

        except Exception as e:
            import traceback
            print(f"  Error generating CD plot for {name}: {e}")
            traceback.print_exc()
            lines.append(f"Error generating CD for {name}: {e}")

    return lines


def run_summary(df: pd.DataFrame, output_dir: Path, generate_cd: bool = False,
                alpha: float = 0.05, label: str = "") -> list:
    """Run summary analysis and save outputs.

    Args:
        df: DataFrame with results
        output_dir: Directory to save outputs
        generate_cd: Whether to generate Critical Difference plots
        alpha: Significance level for CD test
        label: Optional label for output (e.g., "with TabPFN")

    Returns:
        List of output lines for summary.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    if label:
        lines.append(f"# Benchmark Summary ({label})")
        lines.append("")

    n_discrete = df[df['discrete'] == True]['dataset'].nunique()
    n_continuous = df[df['discrete'] == False]['dataset'].nunique()
    info_line = f"Loaded {len(df)} results for {df['dataset'].nunique()} datasets ({n_continuous} continuous, {n_discrete} discrete), {df['model'].nunique()} models"
    lines.append(info_line)

    # Stress test info (no tuning, feature selection, subsampling)
    n_no_tune = df['no_tune'].sum() if 'no_tune' in df.columns else 0
    n_feat_sel = df['feature_selected'].sum() if 'feature_selected' in df.columns else 0
    n_subsampled = df['subsampled'].sum() if 'subsampled' in df.columns else 0

    if n_no_tune > 0 or n_feat_sel > 0 or n_subsampled > 0:
        lines.append("")
        lines.append("Note: Some results used simplified evaluation:")
        if n_no_tune > 0:
            no_tune_datasets = df[df['no_tune'] == True]['dataset'].unique()
            lines.append(f"  - {n_no_tune} results with no hyperparameter tuning (sensible defaults)")
            lines.append(f"    Datasets: {', '.join(sorted(no_tune_datasets))}")
        if n_feat_sel > 0:
            feat_sel_datasets = df[df['feature_selected'] == True]['dataset'].unique()
            lines.append(f"  - {n_feat_sel} results with k-best Spearman feature selection (high-d)")
            lines.append(f"    Datasets: {', '.join(sorted(feat_sel_datasets))}")
        if n_subsampled > 0:
            sub_datasets = df[df['subsampled'] == True]['dataset'].unique()
            lines.append(f"  - {n_subsampled} results with row subsampling (high-n)")
            lines.append(f"    Datasets: {', '.join(sorted(sub_datasets))}")

    # Main rankings include all datasets (continuous + ordinal)
    lines.append("")

    summary = summarize(df)
    lines.append(format_summary(summary))

    # Save CSVs
    summary.to_csv(output_dir / "summary.csv")

    # Wins-by-accuracy-metric breakdown
    wins_df = compute_wins_by_accuracy_metric(df)
    wins_df.to_csv(output_dir / "wins_by_accuracy_metric.csv", index=False)
    lines.append("")
    lines.append("=== Wins by Accuracy Metric ===")
    lines.append(wins_df.to_string(index=False))

    # Stratum summary
    stratum_df = summarize_by_stratum(df)
    stratum_df.to_csv(output_dir / "summary_by_stratum.csv", index=False)
    lines.append(format_stratum_summary(df, stratum_df))

    # Size stratum summary
    size_stratum_df = summarize_by_size_stratum(df)
    size_stratum_df.to_csv(output_dir / "summary_by_size_stratum.csv", index=False)
    lines.append(format_size_stratum_summary(df, size_stratum_df))

    # Stratum x size matrix
    stratum_size_matrix = summarize_stratum_size_matrix(df)
    stratum_size_matrix.to_csv(output_dir / "stratum_size_matrix.csv")
    lines.append(format_stratum_size_matrix(stratum_size_matrix, df))

    # Target type summary (stratified by continuous vs discrete)
    target_type_df = summarize_by_target_type(df)
    if len(target_type_df) > 0 and n_discrete > 0:
        target_type_df.to_csv(output_dir / "summary_by_target_type.csv", index=False)
        lines.append(format_target_type_summary(df, target_type_df))

    # Dataset × Model pivot table with adjusted R²
    pivot = create_dataset_model_pivot(df)
    pivot.to_csv(output_dir / "dataset_model_r2adj.csv")
    lines.append(format_dataset_model_pivot(pivot))

    # Generate CD plots if requested
    if generate_cd:
        cd_lines = generate_cd_plots(df, output_dir, alpha=alpha)
        lines.extend(cd_lines)

    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results")
    parser.add_argument("results_dir", type=Path, help="Directory with joblib results")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for summary files (default: <results_dir>/benchmark_summary)")
    parser.add_argument("--cd", action="store_true", help="Generate Critical Difference plots")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for CD test (default: 0.05)")
    parser.add_argument("--exclude-models", nargs='+', default=[], metavar='MODEL',
                        help="Exclude specific models from analysis (e.g., --exclude-models tabpfn erbfb)")
    parser.add_argument("--detailed", action="store_true",
                        help="Save results_detailed.csv with hyperparameters and model info")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: {args.results_dir} not found")
        return

    print(f"Loading results from {args.results_dir}...", flush=True)
    df = load_results(args.results_dir)

    if len(df) == 0:
        print("No results found!")
        return

    # Apply any explicit model exclusions
    if args.exclude_models:
        excluded_str = ', '.join(sorted(args.exclude_models))
        df = df[~df['model'].isin(args.exclude_models)]
        print(f"Excluded models: {excluded_str}", flush=True)
        if len(df) == 0:
            print("No results remaining after exclusion!")
            return

    # Output directory
    output_dir = args.output_dir or (args.results_dir / "benchmark_summary")
    print(f"Output directory: {output_dir}", flush=True)

    lines = run_summary(df, output_dir, generate_cd=args.cd, alpha=args.alpha)
    for line in lines:
        print(line, flush=True)

    # Save summary.md
    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved summary to {md_path}")

    # Save detailed results if requested
    if args.detailed:
        print("\nLoading detailed results with hyperparameters...", flush=True)
        df_detailed = load_results(args.results_dir, include_hyperparams=True)
        detailed_path = output_dir / "results_detailed.csv"
        df_detailed.to_csv(detailed_path, index=False)
        print(f"Saved detailed results to {detailed_path}")

    print(f"\n\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
