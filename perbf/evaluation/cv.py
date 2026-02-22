"""
Evaluation Module - 4x5 repeated k-fold evaluation for benchmark models.

Provides robust evaluation with:
- 4x5 repeated k-fold CV (20 evaluation folds)
- Separate train/predict timing
- Train-val gap calculation
- Per-fold and aggregated results

Usage:
    from perbf.evaluation.cv import evaluate_model, run_benchmark, get_eval_cv

    # Single model evaluation
    results = evaluate_model(model, X, y)

    # Multiple model comparison
    df = run_benchmark(models_dict, X, y, dataset_name='friedman1')

Created: 14Jan26
"""

import time
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error

from perbf.preprocessing import FoldPreprocessor


# =============================================================================
# CV SETUP
# =============================================================================

def get_eval_cv(n_splits: int = 4, n_repeats: int = 5, random_state: int = 42):
    """
    Get default evaluation CV splitter (4x5 = 20 folds).

    Parameters
    ----------
    n_splits : int
        Number of folds per repeat
    n_repeats : int
        Number of repeats
    random_state : int
        Random seed

    Returns
    -------
    cv : RepeatedKFold
        CV splitter
    """
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_model(
    model,
    X,  # DataFrame or ndarray
    y,  # array-like
    cv=None,
    scale: bool = True,
    return_fold_results: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model using repeated k-fold CV.

    Parameters
    ----------
    model : estimator
        sklearn-compatible model (will be cloned for each fold)
    X : DataFrame or array-like
        Features (DataFrame preserves categorical dtypes for FoldPreprocessor)
    y : array-like
        Target
    cv : CV splitter, optional
        Cross-validation splitter (default: 4x5 RepeatedKFold)
    scale : bool
        Whether to standardize features
    return_fold_results : bool
        Whether to return per-fold results
    verbose : bool
        Whether to print progress

    Returns
    -------
    results : dict
        Dictionary with:
        - r2_val: Mean validation R²
        - r2_val_std: Std of validation R²
        - r2_train: Mean training R²
        - gap: Mean train-val gap
        - train_time: Mean training time per fold
        - predict_time: Mean prediction time per fold
        - fold_results: (optional) List of per-fold results
    """
    if cv is None:
        cv = get_eval_cv()

    # Preserve DataFrame/ndarray type for categorical detection in FoldPreprocessor
    is_dataframe = hasattr(X, 'iloc')
    y_arr = np.asarray(y)

    # For cv.split, need array-like
    X_for_split = X.values if is_dataframe else np.asarray(X)

    train_scores = []
    val_scores = []
    train_times = []
    predict_times = []
    fold_results = []

    n_folds = cv.get_n_splits(X_for_split)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_for_split)):
        if verbose:
            print(f"[fold {fold+1}/{n_folds}]", end=' ', flush=True)

        # Handle both DataFrame and ndarray slicing
        if is_dataframe:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X_for_split[train_idx], X_for_split[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Preprocess: fit on train, transform both (no leakage)
        prep = FoldPreprocessor(scale=scale)
        X_train = prep.fit_transform(X_train, y_train)
        X_val = prep.transform(X_val)

        # Clone model
        if hasattr(model, 'get_params'):
            model_clone = model.__class__(**model.get_params())
        else:
            model_clone = model.__class__()

        # Fit with timing
        t0 = time.perf_counter()
        model_clone.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        train_times.append(train_time)

        # Predict with timing
        t0 = time.perf_counter()
        y_train_pred = model_clone.predict(X_train)
        y_val_pred = model_clone.predict(X_val)
        predict_time = time.perf_counter() - t0
        predict_times.append(predict_time)

        # Score
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        train_scores.append(train_r2)
        val_scores.append(val_r2)

        if return_fold_results:
            fold_result = {
                'fold': fold,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'gap': train_r2 - val_r2,
                'train_time': train_time,
                'predict_time': predict_time,
            }

            fold_results.append(fold_result)

    if verbose:
        print()  # Newline after fold progress

    results = {
        'r2_val': np.mean(val_scores),
        'r2_val_std': np.std(val_scores),
        'r2_train': np.mean(train_scores),
        'gap': np.mean(train_scores) - np.mean(val_scores),
        'train_time': np.mean(train_times),
        'predict_time': np.mean(predict_times),
    }

    if return_fold_results:
        results['fold_results'] = fold_results

    return results


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

# Default scaling by model type
# False for models with internal scaling (avoid double-scaling)
# Note: Replicated from optuna_cv.py to avoid circular imports
DEFAULT_SCALE_MAP = {
    'ridge': True,
    'dt': False,
    'rf': False,
    'xgb': False,
    'tabpfn': True,
    'erbf': False,  # Internal StandardScaler
    'chebypoly': False,  # Internal MinMaxScaler
    'chebytree': False,  # Internal MinMaxScaler
}


def run_benchmark(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "dataset",
    cv=None,
    scale_map: Optional[Dict[str, bool]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run benchmark evaluation on multiple models.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: model}
    X : array-like
        Features
    y : array-like
        Target
    dataset_name : str
        Name of dataset for results
    cv : CV splitter, optional
        Cross-validation splitter (default: 4x5 RepeatedKFold)
    scale_map : dict, optional
        Dictionary specifying which models need scaling
    verbose : bool
        Whether to print progress

    Returns
    -------
    results : DataFrame
        Results for all models with columns:
        dataset, model, r2_val, r2_val_std, r2_train, gap,
        train_time, predict_time, total_time
    """
    if cv is None:
        cv = get_eval_cv()

    if scale_map is None:
        scale_map = DEFAULT_SCALE_MAP.copy()

    all_results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Shape: {X.shape}")
        print(f"CV: {cv.get_n_splits(X)} folds")
        print(f"{'='*60}")

    for name, model in models.items():
        if verbose:
            print(f"  {name}...", end=' ', flush=True)

        t0 = time.time()
        try:
            scale = scale_map.get(name, True)
            result = evaluate_model(model, X, y, cv=cv, scale=scale)
            elapsed = time.time() - t0

            all_results.append({
                'dataset': dataset_name,
                'model': name,
                'r2_val': result['r2_val'],
                'r2_val_std': result['r2_val_std'],
                'r2_train': result['r2_train'],
                'gap': result['gap'],
                'train_time': result['train_time'],
                'predict_time': result['predict_time'],
                'total_time': elapsed,
            })

            if verbose:
                print(f"R²={result['r2_val']:.4f} ± {result['r2_val_std']:.3f}, "
                      f"gap={result['gap']:.4f}, train={result['train_time']:.3f}s")

        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            all_results.append({
                'dataset': dataset_name,
                'model': name,
                'r2_val': np.nan,
                'r2_val_std': np.nan,
                'r2_train': np.nan,
                'gap': np.nan,
                'train_time': np.nan,
                'predict_time': np.nan,
                'total_time': np.nan,
            })

    return pd.DataFrame(all_results)


def summarize_results(df: pd.DataFrame, by: str = 'model') -> pd.DataFrame:
    """
    Summarize benchmark results across datasets or models.

    Parameters
    ----------
    df : DataFrame
        Results from run_benchmark
    by : str
        Group by 'model' or 'dataset'

    Returns
    -------
    summary : DataFrame
        Aggregated results
    """
    agg = df.groupby(by).agg({
        'r2_val': ['mean', 'std'],
        'gap': ['mean', 'std'],
        'train_time': 'mean',
    }).round(4)

    agg.columns = ['r2_mean', 'r2_std', 'gap_mean', 'gap_std', 'train_time_mean']
    return agg.sort_values('r2_mean', ascending=False)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from perbf.data.loader import load_dataset

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    print("=" * 60)
    print("Evaluation Module - Quick Test")
    print("=" * 60)

    # Load a test dataset
    X, y, meta = load_dataset('friedman1')
    print(f"\nDataset: friedman1 ({meta.n_samples} samples, {meta.n_features} features)")

    # Test single model evaluation
    print("\n--- Testing Ridge evaluation (4x5 CV) ---")
    model = Ridge(alpha=1.0)
    results = evaluate_model(model, X, y)
    print(f"R²: {results['r2_val']:.4f} ± {results['r2_val_std']:.3f}")
    print(f"Gap: {results['gap']:.4f}")
    print(f"Train time: {results['train_time']:.4f}s")

    # Test benchmark runner
    print("\n--- Testing benchmark runner ---")
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RF-100': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    df = run_benchmark(models, X, y, dataset_name='friedman1')
    print("\nResults:")
    print(df.to_string(index=False))
