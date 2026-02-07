#!/usr/bin/env python
"""
Verify all datasets load correctly and contain expected data.

Usage:
    python scripts/verify_datasets.py              # Verify all
    python scripts/verify_datasets.py --stratum S1 # Verify S1 only
    python scripts/verify_datasets.py --quick      # Just check loading (no detailed stats)
    python scripts/verify_datasets.py --verbose    # Show feature names and sample data

Created: 17Jan26
Updated: 17Jan26 - Added flush=True for live output, enhanced feature reporting
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def verify_dataset(name, quick=False, verbose=False):
    """Verify a single dataset loads and has valid data."""
    from benchmark.data.loader import load_dataset, DATASET_REGISTRY

    info = DATASET_REGISTRY.get(name, {})
    result = {
        'name': name,
        'stratum': info.get('stratum', '?'),
        'source': info.get('source', '?'),
        'description': info.get('description', '')[:60],
        'status': 'OK',
        'error': None,
        'n_samples': None,
        'n_features': None,
        'n_numerical': None,
        'n_categorical': None,
        'y_nan_pct': None,
        'X_nan_pct': None,
        'y_min': None,
        'y_max': None,
        'y_mean': None,
        'feature_names': None,
    }

    try:
        X, y, meta = load_dataset(name)
        result['n_samples'] = X.shape[0]
        result['n_features'] = X.shape[1]
        result['n_numerical'] = meta.n_numerical
        result['n_categorical'] = meta.n_categorical

        if isinstance(X, pd.DataFrame):
            result['feature_names'] = list(X.columns)

        if not quick:
            # Check for NaN in target
            y_nan = np.isnan(y).sum() if hasattr(y, '__len__') else 0
            result['y_nan_pct'] = round(100 * y_nan / len(y), 1)

            # Check for NaN in features
            if isinstance(X, pd.DataFrame):
                X_nan = X.isna().sum().sum()
            else:
                X_nan = np.isnan(X).sum()
            result['X_nan_pct'] = round(100 * X_nan / X.size, 1)

            # Target stats
            y_clean = y[~np.isnan(y)] if hasattr(y, '__len__') else y
            if len(y_clean) > 0:
                result['y_min'] = float(y_clean.min())
                result['y_max'] = float(y_clean.max())
                result['y_mean'] = float(y_clean.mean())

            # Sanity checks
            if result['y_nan_pct'] == 100:
                result['status'] = 'FAIL'
                result['error'] = "All y values are NaN"
            elif result['y_nan_pct'] > 50:
                result['status'] = 'WARN'
                result['error'] = f"High NaN in y: {result['y_nan_pct']}%"
            elif result['n_samples'] < 50:
                result['status'] = 'WARN'
                result['error'] = f"Very small dataset: {result['n_samples']} samples"
            elif result['n_features'] == 0:
                result['status'] = 'FAIL'
                result['error'] = "No features"
            elif result['y_min'] == result['y_max']:
                result['status'] = 'WARN'
                result['error'] = "Constant target"

    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = str(e)[:100]

    return result


def format_result_line(result, show_features=False):
    """Format a single result for display."""
    status_icon = {'OK': '+', 'WARN': '!', 'FAIL': 'X'}[result['status']]

    if result['status'] != 'OK':
        line = f"[{status_icon}] {result['name']:40} {result['stratum']} {result['status']}: {result['error']}"
    else:
        # Basic info
        line = f"[{status_icon}] {result['name']:40} {result['stratum']} "
        line += f"n={result['n_samples']:>6} d={result['n_features']:>4}"

        # Feature type breakdown
        if result['n_numerical'] is not None:
            line += f" (num={result['n_numerical']}, cat={result['n_categorical']})"

        # Target range
        if result['y_min'] is not None:
            line += f" y=[{result['y_min']:.2g}, {result['y_max']:.2g}]"

    return line


def main():
    parser = argparse.ArgumentParser(description='Verify dataset loading')
    parser.add_argument('--stratum', type=str, help='Only verify specific stratum (S1-S5)')
    parser.add_argument('--quick', action='store_true', help='Quick check (loading only, no stats)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show feature names')
    parser.add_argument('--save', type=str, help='Save results to CSV')
    parser.add_argument('datasets', nargs='*', help='Specific datasets to verify')
    args = parser.parse_args()

    from benchmark.data.loader import DATASET_REGISTRY

    # Filter datasets
    if args.datasets:
        datasets = args.datasets
    elif args.stratum:
        datasets = [name for name, info in DATASET_REGISTRY.items()
                   if info.get('stratum') == args.stratum]
    else:
        datasets = list(DATASET_REGISTRY.keys())

    print(f"Verifying {len(datasets)} datasets...", flush=True)
    print("=" * 80, flush=True)

    results = []
    for i, name in enumerate(sorted(datasets)):
        print(f"[{i+1:3}/{len(datasets)}] Loading {name}...", end='', flush=True)

        result = verify_dataset(name, quick=args.quick, verbose=args.verbose)
        results.append(result)

        # Print status on same line
        if result['status'] == 'OK':
            print(f" OK n={result['n_samples']:>6} d={result['n_features']:>4}", end='', flush=True)
            if result['n_numerical'] is not None:
                print(f" (num={result['n_numerical']}, cat={result['n_categorical']})", end='', flush=True)
            if result['y_min'] is not None:
                print(f" y=[{result['y_min']:.2g}, {result['y_max']:.2g}]", end='', flush=True)
            print(flush=True)
        else:
            print(f" {result['status']}: {result['error']}", flush=True)

        # Show feature names if verbose and successful
        if args.verbose and result['status'] == 'OK' and result['feature_names']:
            feat_str = ', '.join(result['feature_names'][:10])
            if len(result['feature_names']) > 10:
                feat_str += f", ... ({len(result['feature_names'])} total)"
            print(f"       Features: {feat_str}", flush=True)

    # Summary
    print("=" * 80, flush=True)
    ok = sum(1 for r in results if r['status'] == 'OK')
    warn = sum(1 for r in results if r['status'] == 'WARN')
    fail = sum(1 for r in results if r['status'] == 'FAIL')
    print(f"\nSummary: {ok} OK, {warn} WARN, {fail} FAIL out of {len(results)} datasets", flush=True)

    # Stratum breakdown
    print("\n--- By Stratum ---", flush=True)
    for stratum in ['S1', 'S2', 'S3', 'S4', 'S5']:
        stratum_results = [r for r in results if r['stratum'] == stratum]
        if stratum_results:
            stratum_ok = sum(1 for r in stratum_results if r['status'] == 'OK')
            print(f"  {stratum}: {stratum_ok}/{len(stratum_results)} OK", flush=True)

    # Source breakdown
    print("\n--- By Source ---", flush=True)
    sources = sorted(set(r['source'] for r in results if r['source']))
    for source in sources:
        source_results = [r for r in results if r['source'] == source]
        source_ok = sum(1 for r in source_results if r['status'] == 'OK')
        print(f"  {source}: {source_ok}/{len(source_results)} OK", flush=True)

    if fail > 0:
        print(f"\n--- FAILURES ({fail}) ---", flush=True)
        for r in results:
            if r['status'] == 'FAIL':
                print(f"  {r['name']}: {r['error']}", flush=True)

    if warn > 0:
        print(f"\n--- WARNINGS ({warn}) ---", flush=True)
        for r in results:
            if r['status'] == 'WARN':
                print(f"  {r['name']}: {r['error']}", flush=True)

    # Save to CSV if requested
    if args.save:
        df = pd.DataFrame(results)
        # Don't save feature_names list to CSV
        if 'feature_names' in df.columns:
            df = df.drop(columns=['feature_names'])
        df.to_csv(args.save, index=False)
        print(f"\nResults saved to: {args.save}", flush=True)

    # Return exit code
    return 1 if fail > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
