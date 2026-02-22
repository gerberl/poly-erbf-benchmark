#!/usr/bin/env python
"""
Compute prediction surface regularity via synthetic perturbation probes.

Uses the probe-based approach: for each test point, generate m synthetic
neighbours at distance δ in random directions, predict on them, compute
|Δŷ|/δ quotients. All metrics (Lipschitz tail ratio, violation probability,
per-point max) are extracted from the same probe data.

δ = delta_frac × median kNN distance (adaptive to scale and dimensionality).

Reproduces the exact benchmark pipeline (dataset preprocessing → outer CV splits →
fold-level preprocessing → predict) to obtain (X_test, ŷ) for each fold.

Output: results/probe_regularity.csv

Usage:
    python scripts/compute_probe_regularity.py \
        results/benchmark-A/

    # Only specific datasets:
    python scripts/compute_probe_regularity.py \
        results/benchmark-A/ --datasets esol airfoil

NOTE: fiat_500_price is skipped due to FoldPreprocessor TargetEncoder incompatibility.
"""

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from perbf.analysis.discontinuity_smoothness import compute_probe_regularity
from perbf.data.loader import load_dataset
from perbf.preprocessing import FoldPreprocessor
from perbf.tuning.optuna_cv import DEFAULT_SCALE_MAP
from utils.batch_runner import preprocess_dataset

# Suppress sklearn version warnings from unpickling
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

DELTA_FRAC = 0.1
M_PROBES = 30
TAU_MULTIPLIER = 0.3


def reconstruct_preprocessing_config(result):
    """Build preprocess_dataset config from stored preprocessing metadata."""
    pp = result.get('preprocessing', {})
    config = {
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
    return config


def main():
    parser = argparse.ArgumentParser(description='Compute probe-based regularity across benchmark results')
    parser.add_argument('results_dir', type=Path, help='Directory with .joblib result files')
    parser.add_argument('--include-tabpfn', action='store_true', help='Include TabPFN (retrains per fold)')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'results' / 'probe_regularity.csv')
    parser.add_argument('--verify-n', type=int, default=3, help='Number of datasets to spot-check R² (0 to skip)')
    parser.add_argument('--datasets', nargs='+', default=None, help='Only process these datasets (default: all)')
    parser.add_argument('--delta-frac', type=float, default=DELTA_FRAC, help='δ = frac × median kNN dist')
    parser.add_argument('--m-probes', type=int, default=M_PROBES, help='Probes per point')
    args = parser.parse_args()

    # Discover result files, group by dataset
    joblib_files = sorted(args.results_dir.glob('*.joblib'))
    if not joblib_files:
        print(f"No .joblib files found in {args.results_dir}", flush=True)
        sys.exit(1)

    by_dataset = defaultdict(list)
    for f in joblib_files:
        stem = f.stem
        for model in ['chebytree', 'chebypoly', 'tabpfn', 'ridge', 'erbf', 'xgb', 'rf', 'dt']:
            if stem.startswith(model + '_'):
                dataset = stem[len(model) + 1:]
                by_dataset[dataset].append((model, f))
                break

    if args.datasets:
        filtered = {k: v for k, v in by_dataset.items() if k in args.datasets}
        missing = set(args.datasets) - set(filtered.keys())
        if missing:
            print(f"WARNING: no results found for datasets: {sorted(missing)}", flush=True)
        by_dataset = filtered

    print(f"Found {sum(len(v) for v in by_dataset.values())} result files across {len(by_dataset)} datasets", flush=True)
    print(f"Settings: delta_frac={args.delta_frac}, m_probes={args.m_probes}", flush=True)

    records = []
    n_verified = 0

    for ds_idx, (dataset_name, model_files) in enumerate(sorted(by_dataset.items())):
        print(f"\n[{ds_idx+1}/{len(by_dataset)}] {dataset_name} ({len(model_files)} models)", flush=True)

        # Load dataset
        try:
            print(f"  Loading dataset...", end=' ', flush=True)
            X_raw, y_raw, meta = load_dataset(dataset_name)
            print(f"({X_raw.shape[0]}x{X_raw.shape[1]})", flush=True)
        except Exception as e:
            print(f"\n  SKIP: failed to load dataset: {e}", flush=True)
            continue

        # Reconstruct preprocessing from first result's metadata
        first_result = joblib.load(model_files[0][1])
        config = reconstruct_preprocessing_config(first_result)
        print(f"  Preprocessing...", end=' ', flush=True)
        X_pp, y_pp, _ = preprocess_dataset(X_raw, y_raw, config, verbose=False)
        d_effective = X_pp.shape[1]
        print(f"-> ({X_pp.shape[0]}x{d_effective})", flush=True)

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

        outer_cv = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state)
        X_for_split = X_pp.values if hasattr(X_pp, 'values') else X_pp
        fold_indices = list(enumerate(outer_cv.split(X_for_split)))

        do_verify = n_verified < args.verify_n

        for model_name, result_file in model_files:
            if model_name == 'tabpfn' and not args.include_tabpfn:
                continue

            result = joblib.load(result_file)
            fold_results = result.get('fold_results', [])

            result_scale = result.get('config', {}).get('scale')
            if result_scale is not None:
                scale = result_scale
            else:
                scale = DEFAULT_SCALE_MAP.get(model_name, False)

            fold_metrics = []
            r2_mismatches = []
            print(f"  {model_name}: ", end='', flush=True)

            for fold_idx, (trainval_idx, test_idx) in fold_indices:
                if fold_idx >= len(fold_results):
                    break

                fold_r = fold_results[fold_idx]

                # Slice data
                if hasattr(X_pp, 'iloc'):
                    X_trainval = X_pp.iloc[trainval_idx]
                    X_test = X_pp.iloc[test_idx]
                else:
                    X_trainval = X_pp[trainval_idx]
                    X_test = X_pp[test_idx]
                y_trainval = y_pp[trainval_idx]
                y_test = y_pp[test_idx]

                # Fold-level preprocessing
                outer_prep = FoldPreprocessor(scale=scale)
                X_trainval_t = outer_prep.fit_transform(X_trainval, y_trainval)
                X_test_t = outer_prep.transform(X_test)

                # Get model and predictions
                if model_name == 'tabpfn':
                    from tabpfn import TabPFNRegressor
                    model = TabPFNRegressor()
                    model.fit(X_trainval_t, y_trainval)
                    y_pred = model.predict(X_test_t)
                else:
                    model = fold_r.get('model')
                    if model is None:
                        break

                    expected_feats = getattr(model, 'n_features_in_', X_test_t.shape[1])
                    if expected_feats != X_test_t.shape[1]:
                        print(f"feature mismatch (model expects {expected_feats}, "
                              f"got {X_test_t.shape[1]}) -- skipping", flush=True)
                        break

                    try:
                        y_pred = model.predict(X_test_t)
                    except Exception as e:
                        print(f"predict failed ({e}) -- skipping", flush=True)
                        break

                # Spot-check R²
                if do_verify:
                    from sklearn.metrics import r2_score
                    r2_computed = r2_score(y_test, y_pred)
                    r2_stored = fold_r.get('test_r2')
                    if r2_stored is not None and abs(r2_computed - r2_stored) > 1e-4:
                        r2_mismatches.append((fold_idx, r2_computed, r2_stored))

                # Compute clip range from training data
                X_train_arr = X_trainval_t.values if hasattr(X_trainval_t, 'values') else np.asarray(X_trainval_t)
                clip_range = np.column_stack([X_train_arr.min(axis=0), X_train_arr.max(axis=0)])

                # Compute probe regularity
                X_test_arr = X_test_t.values if hasattr(X_test_t, 'values') else np.asarray(X_test_t)
                fm = compute_probe_regularity(
                    model, X_test_arr, y_pred=y_pred,
                    delta_frac=args.delta_frac, m=args.m_probes,
                    clip_range=clip_range, tau_multiplier=TAU_MULTIPLIER,
                )
                fold_metrics.append(fm)
                print('.', end='', flush=True)

            if not fold_metrics:
                print(" no valid folds", flush=True)
                continue

            if do_verify and r2_mismatches:
                print(f" R2 MISMATCH on {len(r2_mismatches)} folds:", flush=True)
                for fi, rc, rs in r2_mismatches[:3]:
                    print(f"    fold {fi}: computed={rc:.6f} stored={rs:.6f} diff={abs(rc-rs):.2e}", flush=True)

            # Average metrics across folds
            avg = {}
            for key in fold_metrics[0]:
                vals = [fm[key] for fm in fold_metrics]
                avg[key] = np.nanmean(vals)

            tail_str = f"{avg.get('lip_tail_ratio_q90', np.nan):.3f}" if not np.isnan(avg.get('lip_tail_ratio_q90', np.nan)) else 'nan'
            v_str = f"{avg['violation_prob']:.4f}" if not np.isnan(avg['violation_prob']) else 'nan'
            print(f" tail={tail_str} V={v_str} delta={avg['delta']:.4f}", flush=True)

            records.append({
                'model': model_name,
                'dataset': dataset_name,
                'd_effective': d_effective,
                'r2_val': result.get('r2_val'),
                'lip_tail_ratio': avg['lip_tail_ratio'],
                'lip_tail_ratio_q90': avg.get('lip_tail_ratio_q90', np.nan),
                'lip_q50': avg['lip_q50'],
                'lip_q75': avg['lip_q75'],
                'lip_q90': avg['lip_q90'],
                'lip_q95': avg['lip_q95'],
                'lip_q99': avg['lip_q99'],
                'lip_mean': avg['lip_mean'],
                'lip_std': avg['lip_std'],
                'violation_prob': avg['violation_prob'],
                'per_point_max_q95': avg['per_point_max_q95'],
                'tau': avg['tau'],
                'delta': avg['delta'],
                'median_knn_dist': avg['median_knn_dist'],
                'n_probes': int(avg['n_probes']),
                'n_folds': len(fold_metrics),
            })

        if do_verify:
            n_verified += 1

    # Save CSV
    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}", flush=True)

    # Summary
    if len(df) > 0:
        print("\n" + "=" * 80, flush=True)
        print("Lipschitz tail ratio (probe-based, q99/q90) per model:", flush=True)
        print("=" * 80, flush=True)
        summary = df.groupby('model')['lip_tail_ratio_q90'].agg(['mean', 'median', 'count'])
        summary = summary.sort_values('mean')
        for model, row in summary.iterrows():
            print(f"  {model:12s}  mean={row['mean']:.3f}  median={row['median']:.3f}  n={int(row['count'])}", flush=True)

        print(f"\nViolation probability per model (all datasets):", flush=True)
        print("=" * 80, flush=True)
        v_summary = df.groupby('model')['violation_prob'].agg(['mean', 'median'])
        v_summary = v_summary.sort_values('mean')
        for model, row in v_summary.iterrows():
            print(f"  {model:12s}  mean={row['mean']:.4f}  median={row['median']:.4f}", flush=True)

        print(f"\nPer-point max quotient q95 per model:", flush=True)
        print("=" * 80, flush=True)
        pm_summary = df.groupby('model')['per_point_max_q95'].agg(['mean', 'median'])
        pm_summary = pm_summary.sort_values('mean')
        for model, row in pm_summary.iterrows():
            print(f"  {model:12s}  mean={row['mean']:.2f}  median={row['median']:.2f}", flush=True)


if __name__ == '__main__':
    main()
