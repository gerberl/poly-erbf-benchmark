"""
Batch runner for benchmark experiments.

Core execution engine that iterates over models and datasets,
running nested CV with Optuna tuning for each combination.

Parallelization strategy:
- Models for each dataset run in PARALLEL (n_parallel_models)
- Datasets are processed SEQUENTIALLY (simpler, predictable memory)
- Each model's CV folds run SEQUENTIALLY (n_jobs=1) to avoid nested parallelism

Created: 16Jan26
"""

import json
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Suppress noisy warnings (before any imports that might trigger them)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*disp.*iprint.*L-BFGS-B.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*omp_set_nested.*deprecated.*')

import numpy as np
from joblib import Parallel, delayed

from .progress_tracker import ProgressTracker, format_date


# Dataset preprocessing configuration (runs BEFORE nested CV)
PREPROCESSING_CONFIG = {
    'na_threshold': 0.5,        # Drop features where NA fraction > tol
    'quasi_constant_tol': 0.95, # Drop features where mode_fraction > tol
    'prefilter': True,          # Remove uninformative features (Spearman + MI rescue)
    'prefilter_threshold': 0.05,  # Minimum |Spearman ρ| for prefilter
    'prefilter_d_min': 25,      # Minimum d to apply prefilter (skip for low-d datasets)
    # MI rescue parameters: rescue features with low Spearman but high MI (non-monotonic)
    'prefilter_spearman_bottom_pctl': 30,  # Bottom percentile for rescue consideration
    'prefilter_mi_top_pctl': 30,           # Top MI percentile to qualify for rescue
    'max_features': None,       # Hard cap on features (k-best MI), None to disable
    'max_samples': None,        # Hard cap on samples (random subsample), None to disable
}


def preprocess_dataset(
    X: 'pd.DataFrame',
    y: np.ndarray,
    config: Dict[str, Any],
    verbose: bool = False,
) -> tuple:
    """
    Apply dataset-level preprocessing (runs BEFORE cross-validation).

    Stages (all run conditionally based on config):
    0. NA cleaning: drop features with >50% missing values
    1. Subsampling: random subsample if n > max_samples
    2. Prefilter: remove uninformative features (Spearman |ρ| < threshold)
    3. Dim reduction: select top-k features by mutual information

    Categorical handling:
    - Prefilter: categoricals pass through (only numerics filtered by Spearman)
    - Dim reduction: uses Mutual Information (handles mixed types properly)

    Parameters
    ----------
    X : DataFrame
        Feature matrix (all benchmark datasets return DataFrame)
    y : ndarray
        Target vector
    config : dict
        Preprocessing config with keys:
        - prefilter: bool (remove uninformative features)
        - prefilter_threshold: float (min |Spearman ρ|)
        - max_features: int or None (hard cap on features)
        - max_samples: int or None (hard cap on samples)
        - random_state: int
    verbose : bool
        Print progress

    Returns
    -------
    X_out : DataFrame
        Preprocessed features
    y_out : ndarray
        Preprocessed target (potentially subsampled)
    preprocess_info : dict
        Metadata about preprocessing applied
    """
    from perbf.preprocessing import select_k_best_mi, prefilter_combined, drop_high_na, drop_quasi_constant

    y = np.asarray(y).ravel()
    random_state = config.get('random_state', 42)

    # Get preprocessing options with defaults from PREPROCESSING_CONFIG
    prefilter_enabled = config.get('prefilter', PREPROCESSING_CONFIG['prefilter'])
    prefilter_threshold = config.get('prefilter_threshold', PREPROCESSING_CONFIG['prefilter_threshold'])
    prefilter_d_min = config.get('prefilter_d_min', PREPROCESSING_CONFIG['prefilter_d_min'])
    max_features = config.get('max_features', PREPROCESSING_CONFIG['max_features'])
    max_samples = config.get('max_samples', PREPROCESSING_CONFIG['max_samples'])

    n_original, d_original = X.shape
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    preprocess_info = {
        'n_samples_original': n_original,
        'n_features_original': d_original,
        'prefilter_enabled': prefilter_enabled,
        'prefilter_threshold': prefilter_threshold,
        'prefilter_d_min': prefilter_d_min,
        'max_features': max_features,
        'max_samples': max_samples,
        'n_categorical_cols': len(cat_cols),
    }

    # Step 0a: Drop high-NA features
    na_threshold = config.get('na_threshold', PREPROCESSING_CONFIG['na_threshold'])
    kept_cols, dropped_na = drop_high_na(X, tol=na_threshold)
    if dropped_na:
        if verbose:
            print(f"  High-NA: dropping {len(dropped_na)} features (>{na_threshold*100:.0f}% missing): {dropped_na}", flush=True)
        X = X[kept_cols]
    preprocess_info['dropped_na'] = dropped_na
    preprocess_info['n_after_na_drop'] = X.shape[1]

    # Step 0b: Drop quasi-constant features (mode_fraction > tol)
    quasi_constant_tol = config.get('quasi_constant_tol', PREPROCESSING_CONFIG['quasi_constant_tol'])
    kept_cols, dropped_quasi = drop_quasi_constant(X, tol=quasi_constant_tol)
    if dropped_quasi:
        if verbose:
            print(f"  Quasi-constant: dropping {len(dropped_quasi)} features (mode_frac > {quasi_constant_tol}): {dropped_quasi}", flush=True)
        X = X[kept_cols]
    preprocess_info['dropped_quasi_constant'] = dropped_quasi
    preprocess_info['n_after_quasi_constant'] = X.shape[1]

    # Step 1: Subsampling (for efficiency - reduces data before correlation computation)
    if max_samples is not None and n_original > max_samples:
        if verbose:
            print(f"  Subsampling: {n_original:,} -> {max_samples:,} samples", flush=True)
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_original, size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        preprocess_info['subsampling_applied'] = True
        preprocess_info['subsampled_from'] = n_original
    else:
        preprocess_info['subsampling_applied'] = False

    # Step 2: Quality prefilter (remove uninformative features)
    # Uses Spearman with MI rescue for non-monotonic features
    # CATEGORICALS PASS THROUGH - only filter numeric columns
    d_current = X.shape[1]
    spearman_bottom_pctl = config.get('prefilter_spearman_bottom_pctl', 30)
    mi_top_pctl = config.get('prefilter_mi_top_pctl', 30)

    if prefilter_enabled and d_current >= prefilter_d_min:
        num_cols = X.select_dtypes(exclude=['category', 'object']).columns.tolist()
        present_cat_cols = [c for c in cat_cols if c in X.columns]

        if num_cols:  # Only filter if there are numeric columns
            # Use prefilter_combined with MI rescue on numeric columns only
            X_num = X[num_cols]
            kept_num_cols, debug_info = prefilter_combined(
                X_num, y,
                spearman_threshold=prefilter_threshold,
                spearman_bottom_pctl=spearman_bottom_pctl,
                mi_top_pctl=mi_top_pctl,
                random_state=random_state,
            )

            # Final columns: ALL categoricals + filtered numerics
            kept_cols = present_cat_cols + kept_num_cols
            n_kept = len(kept_cols)

            if n_kept < d_current:
                n_rescued = len(debug_info.get('rescued', []))
                if verbose:
                    msg = f"  Prefilter: {d_current} -> {n_kept} features "
                    msg += f"(kept {len(present_cat_cols)} cats, {len(num_cols)}->{len(kept_num_cols)} nums"
                    if n_rescued > 0:
                        msg += f", {n_rescued} MI-rescued"
                    msg += ")"
                    print(msg, flush=True)
                X = X[kept_cols]
                preprocess_info['prefilter_applied'] = True
                preprocess_info['n_after_prefilter'] = n_kept
                preprocess_info['prefilter_debug'] = debug_info
            else:
                preprocess_info['prefilter_applied'] = False
        else:
            preprocess_info['prefilter_applied'] = False
    else:
        preprocess_info['prefilter_applied'] = False

    # Step 3: Dimensionality reduction (hard cap on features)
    # Uses Mutual Information - handles mixed types (numeric + categorical) properly
    d_current = X.shape[1]
    if max_features is not None and d_current > max_features:
        if verbose:
            print(f"  Dim reduction: {d_current} -> {max_features} features (k-best MI)", flush=True)

        # select_k_best_mi handles DataFrames and categoricals properly
        X, selected_cols, selected_mi = select_k_best_mi(
            X, y, k=max_features, random_state=random_state
        )

        preprocess_info['dim_reduction_applied'] = True
        preprocess_info['dim_reduction_method'] = 'mi_k_best'
        preprocess_info['dim_reduction_k'] = max_features
        preprocess_info['selected_columns'] = selected_cols
        preprocess_info['top_mi_scores'] = selected_mi[:10].tolist()
    else:
        preprocess_info['dim_reduction_applied'] = False

    preprocess_info['n_samples_final'] = len(y)
    preprocess_info['n_features_final'] = X.shape[1]

    return X, y, preprocess_info


# Nested CV configuration
DEFAULT_CONFIG = {
    # Cross-validation structure
    'outer_splits': 5,
    'outer_repeats': 1,     # 5 total folds (single shuffled CV)
    'inner_splits': None,   # Adaptive: 3-fold for n>=1000, 5-fold for smaller datasets
    'n_trials': 30,
    'random_state': 42,
    'timeout_per_fold': None,

    # Parallelization
    'n_jobs': 1,            # Sequential outer folds (parallelism at model level instead)
    'n_parallel_models': -1,  # Parallel models per dataset (-1 = all cores)
}


def _run_single_experiment(
    model_name: str,
    dataset_name: str,
    exp_config: Dict[str, Any],
    result_file: Path,
    skip_existing: bool,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a single (model, dataset) experiment (called in parallel).

    Flow:
    1. Load dataset (or use provided X, y)
    2. Apply preprocessing (subsampling, prefilter, dim reduction)
    3. Run nested CV with hyperparameter tuning
    4. Merge preprocessing metadata into CV results
    5. Save final result as joblib

    If X and y are provided, uses them directly. Otherwise loads dataset by name.
    This allows stress tests to pre-process data (feature selection, subsampling).

    Parameters
    ----------
    model_name : str
        Model name (key in MODEL_FACTORIES)
    dataset_name : str
        Dataset name (for load_dataset or labeling)
    exp_config : dict
        Experiment configuration (outer_splits, n_trials, etc.)
    result_file : Path
        Path to save result joblib
    skip_existing : bool
        If True, skip if result_file exists
    X : ndarray, optional
        Pre-loaded/preprocessed features. If None, loads from dataset_name.
    y : ndarray, optional
        Pre-loaded/preprocessed target. If None, loads from dataset_name.
    extra_metadata : dict, optional
        Additional metadata to add to result (e.g., feature_selection info)

    Returns
    -------
    dict with status and results or error info
    """
    from perbf.data.loader import load_dataset
    from perbf.tuning.optuna_cv import nested_cv_tune_and_evaluate

    experiment_id = f"{model_name}|{dataset_name}"

    # Skip if exists
    if skip_existing and result_file.exists():
        print(f"[SKIP] {experiment_id}", flush=True)
        return {
            'status': 'skipped',
            'model': model_name,
            'dataset': dataset_name,
        }

    # Run experiment
    print(f"[START] {experiment_id}", flush=True)
    t0 = time.time()
    try:
        # === DATA LOADING & PREPROCESSING ===
        # meta: DatasetMetadata object (n_samples, n_features, stratum, etc.) from loader
        # preprocess_info: dict with preprocessing stats (subsampling, prefilter, dim reduction applied)
        preprocess_info = None
        if X is None or y is None:
            X, y, meta = load_dataset(dataset_name)      # meta = dataset characteristics

            # Apply preprocessing (stages run conditionally inside function)
            X, y, preprocess_info = preprocess_dataset(X, y, exp_config, verbose=True)
        else:
            meta = None  # Pre-processed data provided, metadata comes from extra_metadata instead

        # === METADATA ASSEMBLY ===
        # result_metadata: Extra info to merge into nested_cv result (preprocessing stats, no_tune flag)
        # Built separately because nested_cv returns its own dict (r2_val, gap, fold_results, etc.)
        result_metadata = extra_metadata.copy() if extra_metadata else {}
        if preprocess_info:
            result_metadata['preprocessing'] = preprocess_info   # What preprocessing was applied
            result_metadata['no_tune'] = exp_config.get('no_tune', False)

        # If we have metadata to add, disable auto-save (we'll merge and save manually)
        save_path_arg = None if result_metadata else str(result_file)

        # === NESTED CV EXECUTION ===
        # result: dict with CV metrics (r2_val, gap, rmse_val, fold_results, etc.)
        # Don't save fitted models for TabPFN (200-400MB each, not needed for analysis)
        save_model = model_name != 'tabpfn'
        result = nested_cv_tune_and_evaluate(
            model_name=model_name,
            X=X,
            y=y,
            outer_splits=exp_config['outer_splits'],
            outer_repeats=exp_config['outer_repeats'],
            inner_splits=exp_config['inner_splits'],
            n_trials=exp_config['n_trials'],
            random_state=exp_config['random_state'],
            timeout_per_fold=exp_config.get('timeout_per_fold'),
            n_jobs=exp_config['n_jobs'],  # Sequential folds (1)
            # Note: All dataset-level preprocessing done in preprocess_dataset() above
            save_path=save_path_arg,
            dataset_name=dataset_name,
            verbose=False,  # Suppress per-fold output in parallel
            no_tune=exp_config.get('no_tune', False),
            default_params=exp_config.get('default_params'),
            save_model=save_model,
        )

        # === FINAL ASSEMBLY & SAVE ===
        # Merge preprocessing metadata into nested_cv result
        # (nested_cv auto-saves if save_path_arg provided, otherwise we save here)
        if result_metadata:
            result.update(result_metadata)    # result now has: CV metrics + preprocessing info
            import joblib
            joblib.dump(result, result_file)

        elapsed = time.time() - t0
        print(f"[DONE] {experiment_id} R²={result['r2_val']:.4f} gap={result['gap']:.4f} ({elapsed:.1f}s)", flush=True)

        # Return summary for batch tracker (not the full result)
        return {
            'status': 'completed',
            'model': model_name,
            'dataset': dataset_name,
            'r2_val': result['r2_val'],
            'gap': result['gap'],
            'time': elapsed,
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] {experiment_id} {str(e)[:50]} ({elapsed:.1f}s)", flush=True)
        return {
            'status': 'failed',
            'model': model_name,
            'dataset': dataset_name,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'time': elapsed,
        }


def run_benchmark_batch(
    models: List[str],
    datasets: List[str],
    save_dir: str,
    config: Optional[Dict[str, Any]] = None,
    model_configs: Optional[Dict[str, Dict]] = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run benchmark across multiple models and datasets.

    Config hierarchy (three levels of merging):
    1. DEFAULT_CONFIG + PREPROCESSING_CONFIG (base defaults)
    2. + config (CLI overrides from run_benchmark.py)
    3. + model_configs[model] (per-model settings like n_trials)

    Final exp_config passed to each _run_single_experiment contains all merged settings.

    Parameters
    ----------
    models : list of str
        Model names (keys in MODEL_FACTORIES)
    datasets : list of str
        Dataset names (for load_dataset)
    save_dir : str
        Directory to save results (will be created if needed)
    config : dict, optional
        Override default config for all models
    model_configs : dict, optional
        Per-model config overrides, e.g., {'tabpfn': {'n_trials': 0}}
    skip_existing : bool
        If True, skip experiments with existing result files
    verbose : bool
        Print progress

    Returns
    -------
    summary : dict
        Summary with 'completed', 'failed', 'skipped' experiment lists
    """
    # === CONFIG MERGING (three-level hierarchy) ===
    # 1. run_config = base defaults for all experiments
    run_config = DEFAULT_CONFIG.copy()           # Nested CV settings (outer_splits, n_trials, etc.)
    run_config.update(PREPROCESSING_CONFIG)      # Preprocessing defaults (prefilter, max_features, etc.)
    if config:
        run_config.update(config)                # CLI overrides from run_benchmark.py

    # Setup save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config_file = save_path / 'run_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'models': models,
            'datasets': datasets,
            'config': run_config,
            'model_configs': model_configs,
            'start_time': datetime.now().isoformat(),
        }, f, indent=2)

    # Calculate total experiments
    total_experiments = len(models) * len(datasets)
    tracker = ProgressTracker(total_experiments)

    # Track results
    completed = []
    failed = []
    skipped = []

    # Error log
    error_log_path = save_path / 'errors.log'

    # Get parallel workers setting (joblib handles negative values)
    n_parallel = run_config.get('n_parallel_models', -1)

    if verbose:
        print("\n" + "="*60, flush=True)
        print(f"Benchmark Run: {len(models)} models × {len(datasets)} datasets = {total_experiments} experiments", flush=True)
        print(f"Save directory: {save_path}", flush=True)
        # Show per-model trials if using model_configs, otherwise default
        if model_configs:
            trials_info = {m: model_configs.get(m, {}).get('n_trials', run_config.get('n_trials', 30)) for m in models}
            print(f"Config: outer={run_config['outer_splits']}×{run_config['outer_repeats']} folds, "
                  f"inner={run_config['inner_splits']} folds, trials={trials_info}", flush=True)
        else:
            print(f"Config: outer={run_config['outer_splits']}×{run_config['outer_repeats']} folds, "
                  f"inner={run_config['inner_splits']} folds, n_trials={run_config.get('n_trials', 30)}", flush=True)
        print(f"Parallelization: {n_parallel} workers", flush=True)
        print("="*60 + "\n", flush=True)

    # Build flat list of ALL (model, dataset) tasks
    all_tasks = []

    for dataset_name in datasets:
        for model_name in models:
            result_file = save_path / f"{model_name}_{dataset_name}.joblib"

            # 2. exp_config = run_config + model-specific overrides (e.g., n_trials per model)
            exp_config = run_config.copy()
            if model_configs and model_name in model_configs:
                exp_config.update(model_configs[model_name])   # Per-model settings (e.g., erbf: 30 trials, rf: 25 trials)

            all_tasks.append({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'exp_config': exp_config,
                'result_file': result_file,
            })

    if verbose:
        print(f"Launching {len(all_tasks)} experiments across {n_parallel} workers...\n", flush=True)

    # Run ALL tasks in parallel - joblib schedules optimally
    t0_all = time.time()
    all_results = Parallel(n_jobs=n_parallel, verbose=10 if verbose else 0)(
        delayed(_run_single_experiment)(
            model_name=task['model_name'],
            dataset_name=task['dataset_name'],
            exp_config=task['exp_config'],
            result_file=task['result_file'],
            skip_existing=skip_existing,
        )
        for task in all_tasks
    )
    total_elapsed = time.time() - t0_all

    # Process all results
    for res in all_results:
        model_name = res['model']
        dataset_name = res['dataset']
        experiment_name = f"{model_name}_{dataset_name}"

        if res['status'] == 'skipped':
            skipped.append({'model': model_name, 'dataset': dataset_name})
            tracker.update(experiment_name, 0, success=True)

        elif res['status'] == 'completed':
            completed.append({
                'model': model_name,
                'dataset': dataset_name,
                'r2_val': res['r2_val'],
                'gap': res['gap'],
                'time': res['time'],
            })
            tracker.update(experiment_name, res['time'], success=True)

        elif res['status'] == 'failed':
            failed.append({
                'model': model_name,
                'dataset': dataset_name,
                'error': res['error'],
            })
            tracker.update(experiment_name, res.get('time', 0), success=False)

            # Log error
            error_msg = f"{datetime.now()} | {model_name} | {dataset_name} | {res['error']}\n"
            with open(error_log_path, 'a') as f:
                f.write(error_msg)
                if 'traceback' in res:
                    f.write(res['traceback'] + "\n")

    if verbose:
        print(f"\nAll experiments completed in {total_elapsed:.1f}s", flush=True)

    # Print summary
    if verbose:
        tracker.print_summary()
        print(f"\nCompleted: {len(completed)}, Failed: {len(failed)}, Skipped: {len(skipped)}", flush=True)
        if failed:
            print(f"\nFailed experiments (see {error_log_path}):", flush=True)
            for f in failed[:5]:  # Show first 5
                print(f"  - {f['model']} × {f['dataset']}: {f['error'][:50]}", flush=True)
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more", flush=True)

    return {
        'completed': completed,
        'failed': failed,
        'skipped': skipped,
        'save_dir': str(save_path),
        'summary': tracker.get_summary(),
    }


