"""
Result loading and parsing utilities.

Load joblib results and extract comprehensive metrics for analysis.

Created: 16Jan26
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


def load_result(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a single result file.

    Parameters
    ----------
    filepath : str or Path
        Path to .joblib file

    Returns
    -------
    result : dict
        Result dictionary from nested_cv_tune_and_evaluate
    """
    return joblib.load(filepath)


def load_all_results(directory: Union[str, Path], pattern: str = "*.joblib") -> List[Dict[str, Any]]:
    """
    Load all result files from a directory.

    Parameters
    ----------
    directory : str or Path
        Directory containing .joblib files
    pattern : str
        Glob pattern for result files

    Returns
    -------
    results : list of dict
        List of result dictionaries
    """
    directory = Path(directory)
    results = []

    for filepath in sorted(directory.glob(pattern)):
        try:
            result = load_result(filepath)
            result['_filepath'] = str(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    return results


def extract_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary row from a single result.

    Returns comprehensive metrics including core, complexity, and stability.

    Parameters
    ----------
    result : dict
        Result from nested_cv_tune_and_evaluate

    Returns
    -------
    row : dict
        Flat dictionary suitable for DataFrame row
    """
    row = {}

    # === Basic info ===
    row['model'] = result.get('model_name', 'unknown')
    row['dataset'] = result.get('dataset_name', 'unknown')

    # === Core metrics ===
    row['r2_val'] = result.get('r2_val')
    row['r2_val_std'] = result.get('r2_val_std')
    row['r2_val_median'] = result.get('r2_val_median')
    row['r2_val_trimmed'] = result.get('r2_val_trimmed')
    row['r2_train'] = result.get('r2_train')
    row['r2_val_adj'] = result.get('r2_val_adj')
    row['gap'] = result.get('gap')
    # Failure tracking
    row['n_failed_folds'] = result.get('n_failed_folds', 0)
    # RMSE/MAE/MAPE
    row['rmse_val'] = result.get('rmse_val')
    row['rmse_val_std'] = result.get('rmse_val_std')
    row['mae_val'] = result.get('mae_val')
    row['mae_val_std'] = result.get('mae_val_std')
    row['mape_val'] = result.get('mape_val')
    # Note: Residual stats and Lipschitz metrics removed 23Jan26
    # Timing
    row['total_time'] = result.get('total_time')
    row['train_time'] = result.get('train_time')
    row['predict_time'] = result.get('predict_time')
    row['n_features_used'] = result.get('n_features_used')

    # === Per-fold metrics (if available) ===
    fold_results = result.get('fold_results', [])
    if fold_results:
        test_r2s = [f.get('test_r2', np.nan) for f in fold_results]
        gaps = [f.get('gap', np.nan) for f in fold_results]

        # Stability: coefficient of variation of R²
        if np.nanmean(test_r2s) != 0:
            row['r2_fold_cv'] = np.nanstd(test_r2s) / abs(np.nanmean(test_r2s))
        else:
            row['r2_fold_cv'] = np.nan

        row['gap_std'] = np.nanstd(gaps)
        row['n_folds'] = len(fold_results)

        # === Model complexity (from first fold's model_info) ===
        first_info = fold_results[0].get('model_info', {})
        row['n_params'] = first_info.get('n_params')

        # Model-specific complexity
        model_name = row['model']
        if model_name == 'erbf':
            row['n_rbf'] = first_info.get('n_rbf')
            row['n_width_params'] = first_info.get('n_width_params')
        elif model_name in ('rf', 'xgb'):
            row['n_estimators'] = first_info.get('n_estimators')
            row['n_leaves_total'] = first_info.get('n_leaves_total')
        elif model_name in ('dt', 'chebytree'):
            row['n_leaves'] = first_info.get('n_leaves')
            row['max_depth_actual'] = first_info.get('max_depth_actual')
        elif model_name == 'chebypoly':
            row['complexity'] = first_info.get('complexity')
            row['regressor'] = first_info.get('regressor')

        # === Hyperparameter stability ===
        # Check if key hyperparams vary across folds
        all_params = [f.get('best_params', {}) for f in fold_results]
        if all_params and all_params[0]:
            # Get a representative numeric hyperparam
            first_params = all_params[0]
            for key in first_params:
                values = [p.get(key) for p in all_params if key in p]
                if values and isinstance(values[0], (int, float)):
                    row[f'hp_{key}_mean'] = np.mean(values)
                    row[f'hp_{key}_std'] = np.std(values)
                    break  # Just track first numeric param for simplicity

    # === Preprocessing info ===
    prep_info = result.get('preprocessing')
    if prep_info:
        row['n_t1'] = prep_info.get('n_t1')
        row['n_t2'] = prep_info.get('n_t2')
        row['n_interactions'] = prep_info.get('n_interactions')

    # === Config info ===
    config = result.get('config', {})
    row['outer_splits'] = config.get('outer_splits')
    row['outer_repeats'] = config.get('outer_repeats')
    row['n_trials'] = config.get('n_trials')

    return row


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create summary DataFrame from list of results.

    Parameters
    ----------
    results : list of dict
        Results from load_all_results

    Returns
    -------
    df : DataFrame
        Summary table with one row per experiment
    """
    rows = [extract_summary_row(r) for r in results]
    df = pd.DataFrame(rows)

    # Sort by dataset, then by model for consistent ordering
    if 'dataset' in df.columns and 'model' in df.columns:
        df = df.sort_values(['dataset', 'model'])

    return df


def add_rankings(df: pd.DataFrame, metric: str = 'r2_val', ascending: bool = False) -> pd.DataFrame:
    """
    Add per-dataset rankings based on a metric.

    Parameters
    ----------
    df : DataFrame
        Summary DataFrame with 'dataset' and metric columns
    metric : str
        Column to rank by
    ascending : bool
        If True, lower is better

    Returns
    -------
    df : DataFrame
        DataFrame with 'rank' and 'is_winner' columns added
    """
    df = df.copy()
    df['rank'] = df.groupby('dataset')[metric].rank(ascending=ascending, method='min')
    df['is_winner'] = df['rank'] == 1
    return df


def compute_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per model across datasets.

    Parameters
    ----------
    df : DataFrame
        Summary DataFrame with rankings

    Returns
    -------
    summary : DataFrame
        One row per model with aggregated statistics
    """
    agg_funcs = {
        'r2_val': ['mean', 'std'],
        'r2_val_median': 'mean',  # Robust alternative
        'r2_val_trimmed': 'mean',  # Robust alternative
        'gap': ['mean', 'std'],
        'n_failed_folds': 'sum',  # Total failed folds across datasets
        'total_time': ['mean', 'sum'],
        'n_params': 'mean',
        'rank': 'mean',
        'is_winner': 'mean',  # Win rate
    }

    # Filter to columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    summary = df.groupby('model').agg(agg_funcs)
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns]

    # Rename for clarity
    rename_map = {
        'is_winner_mean': 'win_rate',
        'rank_mean': 'avg_rank',
    }
    summary = summary.rename(columns=rename_map)

    # Sort by mean R²
    if 'r2_val_mean' in summary.columns:
        summary = summary.sort_values('r2_val_mean', ascending=False)

    return summary
