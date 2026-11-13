#!/usr/bin/env python
"""
Prediction stability under retraining via bootstrap resampling.

For each (model, dataset), loads winning hyperparameters from benchmark results,
bootstrap-resamples training data N times, refits, predicts on fixed holdout,
and computes coefficient of variation per test point.

Measures sensitivity of the learned function to training-set composition --
distinct from generalisation gap (train vs test performance) and surface
regularity (local smoothness).

Usage:
    python scripts/compute_prediction_stability_01feb26.py \
        results/rerun-benchmark-25Jan26/benchmark-A/ \
        --n-bootstrap 100 --n-jobs 17

    # Quick test:
    python scripts/compute_prediction_stability_01feb26.py \
        results/rerun-benchmark-25Jan26/benchmark-A/ \
        --datasets esol --models ridge --n-bootstrap 10

Output: results/prediction_stability_01feb26.csv
"""

# BLAS thread control (before numpy)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import time
import zlib
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedKFold

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from perbf.data.loader import load_dataset
from perbf.preprocessing import FoldPreprocessor, apply_fold_feature_selection
from perbf.tuning.optuna_cv import MODEL_FACTORIES, DEFAULT_SCALE_MAP
from utils.batch_runner import preprocess_dataset

# Suppress sklearn version warnings from unpickling
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

N_BOOTSTRAP_DEFAULT = 100


class MockTrial:
    """Replay stored hyperparameters as if from an Optuna trial."""

    def __init__(self, params):
        self.params = params

    def suggest_float(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_int(self, name, *args, **kwargs):
        return self.params[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self.params[name]


def reconstruct_preprocessing_config(result):
    """Build preprocess_dataset config from stored preprocessing metadata."""
    pp = result.get('preprocessing', {})
    return {
        'na_threshold': 0.5,
        'quasi_constant_tol': 0.95,
        'prefilter': pp.get('prefilter_enabled', True),
        'prefilter_threshold': pp.get('prefilter_threshold', 0.05),
        'prefilter_d_min': pp.get('prefilter_d_min', 25),
        'prefilter_spearman_bottom_pctl': 30,
        'prefilter_mi_top_pctl': 30,
        'max_features': pp.get('max_features'),
        'max_samples': pp.get('max_samples'),
        'random_state': 42,
    }


def _fit_predict_one(model_name, best_params, X_train_arr, y_train_arr,
                      X_test_arr, seed):
    """Single bootstrap iteration: resample, fit, predict. Returns 1-d predictions."""
    rng = np.random.default_rng(seed)
    n_train = X_train_arr.shape[0]
    idx = rng.integers(0, n_train, size=n_train)
    X_b = X_train_arr[idx]
    y_b = y_train_arr[idx]

    if model_name == 'tabpfn':
        from tabpfn import TabPFNRegressor
        # Fix the seed across bootstraps so the measured dispersion is purely
        # data-resampling-induced -- apples-to-apples with the tuned CPU models
        # (fixed seeds, resampled data) and the TabPFN probe. The internal-seed
        # channel is a separate, much smaller effect (see the 09jul26 probe).
        model = TabPFNRegressor(random_state=0)
    else:
        model = MODEL_FACTORIES[model_name](MockTrial(best_params))

    # XGB factory uses early stopping callbacks that require eval_set.
    # Strip them and use the stored best iteration as n_estimators.
    if hasattr(model, 'callbacks') and model.callbacks:
        n_est = best_params.get('_n_estimators', model.n_estimators)
        model.set_params(callbacks=None, n_estimators=n_est)

    model.fit(X_b, y_b)
    return model.predict(X_test_arr)


def summarise_predictions(predictions, y_range=None):
    """Compute CV, std, and SD/range summaries from an (n_bootstrap, n_test) matrix.

    SD/range normalises each test point's prediction SD by the target range
    (max - min of the fold's training targets). Unlike per-instance CV (std/|mean|),
    it has a single well-defined denominator, so it stays finite and comparable even
    where the mean prediction is near zero -- the reviewer-requested robust stability
    metric, reported alongside CV rather than replacing it.
    """
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0, ddof=1)
    n_test = predictions.shape[1]

    # Coefficient of variation: std / |mean|
    abs_mean = np.abs(pred_mean)
    safe_mask = abs_mean > 1e-10
    cv_values = np.full(n_test, np.nan)
    cv_values[safe_mask] = pred_std[safe_mask] / abs_mean[safe_mask]

    summary = {
        'cv_median': np.nanmedian(cv_values),
        'cv_mean': np.nanmean(cv_values),
        'cv_q90': np.nanpercentile(cv_values, 90),
        'cv_max': np.nanmax(cv_values),
        'std_median': np.median(pred_std),
        'std_mean': np.mean(pred_std),
        'std_q90': np.percentile(pred_std, 90),
        'std_max': np.max(pred_std),
        'n_test': n_test,
        'n_near_zero': int((~safe_mask).sum()),
    }

    # SD / target-range: invariant, near-zero-robust stability metric.
    if y_range is not None and y_range > 0:
        summary.update({
            'sd_over_range_median': float(np.median(pred_std) / y_range),
            'sd_over_range_mean': float(np.mean(pred_std) / y_range),
            'sd_over_range_q90': float(np.percentile(pred_std, 90) / y_range),
            'sd_over_range_max': float(np.max(pred_std) / y_range),
            'y_range': float(y_range),
        })
    else:
        summary.update({
            'sd_over_range_median': np.nan,
            'sd_over_range_mean': np.nan,
            'sd_over_range_q90': np.nan,
            'sd_over_range_max': np.nan,
            'y_range': np.nan,
        })

    return summary


def prepare_model_jobs(model_name, result_file, X_pp, y_pp, fold_indices,
                        n_bootstrap, base_seed):
    """
    Prepare a model for bootstrap: load params, preprocess fold 0, return
    (model_name, best_params, X_train_arr, y_train_arr, X_test_arr, r2_val, r2_val_std)
    or None if the model should be skipped.
    """
    result = joblib.load(result_file)
    fold_results = result.get('fold_results', [])
    if not fold_results:
        print(f"    {model_name}: no fold results -- skipping", flush=True)
        return None

    fold_r = fold_results[0]
    trainval_idx, test_idx = fold_indices[0]

    # Slice data
    if hasattr(X_pp, 'iloc'):
        X_trainval = X_pp.iloc[trainval_idx]
        X_test = X_pp.iloc[test_idx]
    else:
        X_trainval = X_pp[trainval_idx]
        X_test = X_pp[test_idx]
    y_trainval = y_pp[trainval_idx]

    # Per-fold feature selection -- SAME helper the benchmark uses, so the
    # bootstrap refits operate on the identical feature space as the stored
    # model. Reconstruct the fs_config from stored metadata (mirrors batch_runner).
    pp = result.get('preprocessing', {})
    fs_config = None
    if pp.get('prefilter_enabled', False) or pp.get('max_features') is not None:
        fs_config = {
            'prefilter': pp.get('prefilter_enabled', False),
            'prefilter_threshold': pp.get('prefilter_threshold', 0.05),
            'prefilter_d_min': pp.get('prefilter_d_min', 25),
            'spearman_bottom_pctl': pp.get('prefilter_spearman_bottom_pctl', 30),
            'mi_top_pctl': pp.get('prefilter_mi_top_pctl', 30),
            'max_features': pp.get('max_features'),
            'random_state': result.get('config', {}).get('random_state', 42),
        }
    X_trainval, X_test, _ = apply_fold_feature_selection(
        X_trainval, X_test, y_trainval, fs_config
    )

    # Fold-level preprocessing
    result_scale = result.get('config', {}).get('scale')
    scale = result_scale if result_scale is not None else DEFAULT_SCALE_MAP.get(model_name, False)

    outer_prep = FoldPreprocessor(scale=scale)
    X_trainval_t = outer_prep.fit_transform(X_trainval, y_trainval)
    X_test_t = outer_prep.transform(X_test)

    # Extract best_params
    if model_name == 'tabpfn':
        best_params = {}
    else:
        best_params = dict(fold_r.get('best_params') or {})
        if not best_params:
            print(f"    {model_name}: no best_params in fold 0 -- skipping", flush=True)
            return None

        # Inject fitted attributes that avoid expensive auto-detection on refit
        model_info = fold_r.get('model_info', {})

        # XGB: use early-stopped iteration count instead of training 2000 rounds
        if model_name == 'xgb':
            best_iter = model_info.get('best_iteration')
            if best_iter is not None:
                best_params['_n_estimators'] = best_iter + 1  # 0-indexed -> count

        # ERBF: if n_rbf was 'auto', use the resolved count to skip auto-detection
        if model_name == 'erbf' and best_params.get('n_rbf_auto'):
            resolved_n_rbf = model_info.get('n_rbf')
            if resolved_n_rbf is not None:
                best_params['n_rbf_auto'] = False
                best_params['n_rbf'] = int(resolved_n_rbf)

    # Feature count sanity check
    if model_name != 'tabpfn':
        stored_model = fold_r.get('model')
        if stored_model is not None:
            expected_feats = getattr(stored_model, 'n_features_in_', X_test_t.shape[1])
            if expected_feats != X_test_t.shape[1]:
                print(f"    {model_name}: feature mismatch ({expected_feats} vs {X_test_t.shape[1]}) -- skipping", flush=True)
                return None

    X_train_arr = X_trainval_t.values if hasattr(X_trainval_t, 'values') else np.asarray(X_trainval_t)
    y_train_arr = np.asarray(y_trainval)
    X_test_arr = X_test_t.values if hasattr(X_test_t, 'values') else np.asarray(X_test_t)

    return {
        'model_name': model_name,
        'best_params': best_params,
        'X_train_arr': X_train_arr,
        'y_train_arr': y_train_arr,
        'X_test_arr': X_test_arr,
        # Target range of the fold-0 training targets: denominator for SD/range
        # (matches the probe's y_ref = fold train targets).
        'y_range': float(np.max(y_train_arr) - np.min(y_train_arr)),
        'r2_val': result.get('r2_val'),
        'r2_val_std': result.get('r2_val_std'),
        'r2_val_adj': result.get('r2_val_adj'),
        'gap': result.get('gap'),
        'rmse_val': result.get('rmse_val'),
        'mae_val': result.get('mae_val'),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Prediction stability under retraining via bootstrap resampling')
    parser.add_argument('results_dir', type=Path,
                        help='Directory with .joblib result files')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP_DEFAULT,
                        help=f'Number of bootstrap resamples (default: {N_BOOTSTRAP_DEFAULT})')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel workers for bootstrap fit-predict jobs')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'results' / 'prediction_stability_01feb26.csv')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Only process these models (default: all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Only process these datasets (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for bootstrap resampling')
    args = parser.parse_args()

    # Discover result files, group by dataset
    joblib_files = sorted(args.results_dir.glob('*.joblib'))
    if not joblib_files:
        print(f"No .joblib files found in {args.results_dir}", flush=True)
        sys.exit(1)

    model_names_ordered = ['chebytree', 'chebypoly', 'tabpfn', 'ridge', 'erbf', 'ebm', 'xgb', 'rf', 'dt']
    by_dataset = defaultdict(list)
    for f in joblib_files:
        stem = f.stem
        for model in model_names_ordered:
            if stem.startswith(model + '_'):
                dataset = stem[len(model) + 1:]
                by_dataset[dataset].append((model, f))
                break

    # Filter by --models and --datasets
    if args.datasets:
        filtered = {k: v for k, v in by_dataset.items() if k in args.datasets}
        missing = set(args.datasets) - set(filtered.keys())
        if missing:
            print(f"WARNING: no results for datasets: {sorted(missing)}", flush=True)
        by_dataset = filtered

    if args.models:
        model_set = set(args.models)
        by_dataset = {
            k: [(m, f) for m, f in v if m in model_set]
            for k, v in by_dataset.items()
        }
        by_dataset = {k: v for k, v in by_dataset.items() if v}

    total_pairs = sum(len(v) for v in by_dataset.values())
    print(f"Found {total_pairs} (model, dataset) pairs across {len(by_dataset)} datasets", flush=True)
    print(f"Settings: n_bootstrap={args.n_bootstrap}, n_jobs={args.n_jobs}, seed={args.seed}", flush=True)

    records = []

    for ds_idx, (dataset_name, model_files) in enumerate(sorted(by_dataset.items())):
        print(f"\n[{ds_idx+1}/{len(by_dataset)}] {dataset_name} ({len(model_files)} models)", flush=True)

        # Load dataset
        print(f"  Loading dataset...", end=' ', flush=True)
        X_raw, y_raw, meta = load_dataset(dataset_name)
        print(f"({X_raw.shape[0]}x{X_raw.shape[1]})", flush=True)

        # Reconstruct preprocessing from first result's metadata
        first_result = joblib.load(model_files[0][1])
        config = reconstruct_preprocessing_config(first_result)
        print(f"  Preprocessing...", end=' ', flush=True)
        X_pp, y_pp, _ = preprocess_dataset(X_raw, y_raw, config, verbose=False)
        print(f"-> ({X_pp.shape[0]}x{X_pp.shape[1]})", flush=True)

        # Verify shapes
        expected_n = first_result.get('n_samples_used', X_pp.shape[0])
        expected_d = first_result.get('n_features_used', X_pp.shape[1])
        if X_pp.shape[0] != expected_n or X_pp.shape[1] != expected_d:
            print(f"  WARNING: shape mismatch! got {X_pp.shape}, expected ({expected_n}, {expected_d})", flush=True)
            print(f"  Skipping dataset", flush=True)
            continue

        # Outer CV splits (matching optuna_cv.py)
        cv_config = first_result.get('config', {})
        outer_splits = cv_config.get('outer_splits', 5)
        outer_repeats = cv_config.get('outer_repeats', 1)
        random_state = cv_config.get('random_state', 42)

        outer_cv = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats,
                                  random_state=random_state)
        X_for_split = X_pp.values if hasattr(X_pp, 'values') else X_pp
        fold_indices = [(trainval_idx, test_idx)
                        for trainval_idx, test_idx in outer_cv.split(X_for_split)]

        # Prepare all models for this dataset (load params, preprocess fold 0)
        prepared = {}
        for model_name, result_file in sorted(model_files):
            # Deterministic per-pair seed: Python's built-in hash() of strings is
            # salted per process (PYTHONHASHSEED), so it was NOT reproducible across
            # runs. crc32 of the name is stable, making the bootstrap reproducible.
            pair_seed = args.seed + zlib.crc32(f"{model_name}|{dataset_name}".encode()) % (2**31)
            info = prepare_model_jobs(
                model_name, result_file, X_pp, y_pp, fold_indices,
                args.n_bootstrap, pair_seed,
            )
            if info is not None:
                prepared[model_name] = (info, pair_seed)

        if not prepared:
            continue

        # Dispatch per model: bootstrap jobs parallelised, progress printed per model
        t0_ds = time.perf_counter()
        for model_name, (info, pair_seed) in prepared.items():
            jobs = []
            for b in range(args.n_bootstrap):
                boot_seed = pair_seed + b + 1
                jobs.append((model_name, info['best_params'],
                             info['X_train_arr'], info['y_train_arr'],
                             info['X_test_arr'], boot_seed))

            n_jobs_model = 1 if model_name == 'tabpfn' else args.n_jobs
            print(f"    {model_name}: {args.n_bootstrap} bootstraps...", end=' ', flush=True)
            t0_m = time.perf_counter()
            preds_list = Parallel(n_jobs=n_jobs_model, verbose=10)(
                delayed(_fit_predict_one)(*job) for job in jobs
            )
            elapsed_m = time.perf_counter() - t0_m

            preds = np.array(preds_list)
            stability = summarise_predictions(preds, info['y_range'])
            print(f"cv_median={stability['cv_median']:.4f} "
                  f"std_median={stability['std_median']:.4f} "
                  f"({elapsed_m:.1f}s)", flush=True)

            records.append({
                'model': model_name,
                'dataset': dataset_name,
                'n_bootstrap': args.n_bootstrap,
                'n_test': stability['n_test'],
                'cv_median': stability['cv_median'],
                'cv_mean': stability['cv_mean'],
                'cv_q90': stability['cv_q90'],
                'cv_max': stability['cv_max'],
                'std_median': stability['std_median'],
                'std_mean': stability['std_mean'],
                'std_q90': stability['std_q90'],
                'std_max': stability['std_max'],
                'sd_over_range_median': stability['sd_over_range_median'],
                'sd_over_range_mean': stability['sd_over_range_mean'],
                'sd_over_range_q90': stability['sd_over_range_q90'],
                'sd_over_range_max': stability['sd_over_range_max'],
                'y_range': stability['y_range'],
                'n_near_zero': stability['n_near_zero'],
                'r2_val': info['r2_val'],
                'r2_val_std': info['r2_val_std'],
                'r2_val_adj': info['r2_val_adj'],
                'gap': info['gap'],
                'rmse_val': info['rmse_val'],
                'mae_val': info['mae_val'],
                'time_seconds': elapsed_m,
            })

        print(f"  Dataset done in {time.perf_counter() - t0_ds:.1f}s", flush=True)

        # Incremental save after each dataset
        df_partial = pd.DataFrame(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_partial.to_csv(args.output, index=False)
        print(f"  [{len(records)} rows saved to {args.output}]", flush=True)

    # Final save
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}", flush=True)

    # Summary
    if len(df) > 0:
        print("\n" + "=" * 80, flush=True)
        print("Prediction stability (CV median) per model:", flush=True)
        print("=" * 80, flush=True)
        summary = df.groupby('model')['cv_median'].agg(['mean', 'median', 'count'])
        summary = summary.sort_values('median')
        for model, row in summary.iterrows():
            print(f"  {model:12s}  mean={row['mean']:.4f}  median={row['median']:.4f}  n={int(row['count'])}", flush=True)

        print(f"\nPrediction std (median across test points) per model:", flush=True)
        print("=" * 80, flush=True)
        s2 = df.groupby('model')['std_median'].agg(['mean', 'median'])
        s2 = s2.sort_values('median')
        for model, row in s2.iterrows():
            print(f"  {model:12s}  mean={row['mean']:.4f}  median={row['median']:.4f}", flush=True)


if __name__ == '__main__':
    main()
