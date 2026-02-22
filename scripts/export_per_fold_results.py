#!/usr/bin/env python3
"""
Export per-fold results from joblib files to a flat CSV for reproducibility.

Produces one row per (model, dataset, fold) with metrics, timing, and
selected hyperparameters. Intended for the public reproducibility repository
so reviewers can verify aggregated results without loading joblib files.

Usage:
    python scripts/export_per_fold_results.py results/benchmark-A
    python scripts/export_per_fold_results.py results/benchmark-A -o results/per_fold_results.csv
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd


def extract_fold_rows(result: dict) -> list[dict]:
    """Extract per-fold rows from a single joblib result dict."""
    model = result['model_name']
    dataset = result['dataset_name']
    pp = result.get('preprocessing', {})

    rows = []
    for fr in result['fold_results']:
        # Selected hyperparameters (flat dict, exclude internal keys)
        bp = fr.get('best_params', {})
        mi = fr.get('model_info', {})

        row = {
            'model': model,
            'dataset': dataset,
            'fold': fr['fold'],
            # Metrics
            'train_r2': fr.get('train_r2'),
            'train_r2_adj': fr.get('train_r2_adj'),
            'test_r2': fr.get('test_r2'),
            'test_r2_adj': fr.get('test_r2_adj'),
            'gap': fr.get('gap'),
            'test_mae': fr.get('test_mae'),
            'test_rmse': fr.get('test_rmse'),
            'train_mae': fr.get('train_mae'),
            'train_rmse': fr.get('train_rmse'),
            # Timing (seconds)
            'tune_time': fr.get('tune_time'),
            'train_time': fr.get('train_time'),
            'predict_time': fr.get('predict_time'),
            'n_pruned': fr.get('n_pruned'),
            # Preprocessing metadata
            'n_samples_original': pp.get('n_samples_original'),
            'n_samples_final': pp.get('n_samples_final'),
            'n_features_original': pp.get('n_features_original'),
            'n_features_final': pp.get('n_features_final'),
            'subsampling_applied': pp.get('subsampling_applied'),
            'prefilter_applied': pp.get('prefilter_applied'),
            'dim_reduction_applied': pp.get('dim_reduction_applied'),
        }

        # Add selected hyperparameters with hp_ prefix
        for k, v in bp.items():
            row[f'hp_{k}'] = v

        # Add key model info with mi_ prefix (skip model_name, redundant)
        for k, v in mi.items():
            if k != 'model_name' and not isinstance(v, (list, dict)):
                row[f'mi_{k}'] = v

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description='Export per-fold results to CSV')
    parser.add_argument('results_dir', type=Path, help='Directory containing .joblib files')
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help='Output CSV path (default: <results_dir>/per_fold_results.csv)')
    args = parser.parse_args()

    joblib_files = sorted(args.results_dir.glob('*.joblib'))
    if not joblib_files:
        print(f"No .joblib files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(joblib_files)} joblib files from {args.results_dir}...", flush=True)

    all_rows = []
    errors = []
    for f in joblib_files:
        try:
            r = joblib.load(f)
            all_rows.extend(extract_fold_rows(r))
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    if errors:
        print(f"WARNING: {len(errors)} files failed to load:", flush=True)
        for err in errors:
            print(f"  {err}", flush=True)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(['model', 'dataset', 'fold']).reset_index(drop=True)

    output_path = args.output or args.results_dir / 'per_fold_results.csv'
    df.to_csv(output_path, index=False)

    n_models = df['model'].nunique()
    n_datasets = df['dataset'].nunique()
    n_folds = df.groupby(['model', 'dataset'])['fold'].count().median()
    print(f"Exported {len(df)} rows ({n_models} models x {n_datasets} datasets x {n_folds:.0f} folds)", flush=True)
    print(f"Output: {output_path}", flush=True)

    # Summary check
    print(f"\nColumns: {', '.join(df.columns[:15])}... ({len(df.columns)} total)", flush=True)


if __name__ == '__main__':
    main()
