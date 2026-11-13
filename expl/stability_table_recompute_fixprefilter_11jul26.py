#!/usr/bin/env python
"""
Step 1 (R2#3/R4#2 revision) -- denominator robustness for the SD/range headline.

The headline stability metric is SD/range = std_median / range(y_train_fold0).
Range, SD, and IQR of the fold-0 training targets are all per-dataset constants,
so WITHIN a dataset the model ranking is algebraically identical for all three
denominators. What CAN move: (a) the per-model MEDIAN across datasets and its
induced sort order; (b) the pooled Spearman rho(gap, SD/denom). This script
recomputes SD/range, SD/SD(y_train), SD/IQR(y_train) on the identical fold-0
y_train vector the stability run used and reports both.

Faithfulness guarantee: my recomputed range(y_train) must equal the SoT CSV
`y_range` per dataset (assert) -- proof I am on the same vector before trusting SD/IQR.

Usage:
    micromamba run -n poly_erbf_benchmark python expl/stability_denominator_robustness_10jul26.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from perbf.data.loader import load_dataset
from utils.batch_runner import preprocess_dataset

RESULTS_DIR = PROJECT_ROOT / 'results' / 'revision-A'
SOT_CSV = PROJECT_ROOT / 'results' / 'stability' / \
    'prediction_stability_revisionA_9models_55ds_11jul26_fixprefilter.csv'

# Ordering used to identify model from joblib stem (matches stability script).
MODEL_NAMES = ['chebytree', 'chebypoly', 'tabpfn', 'ridge', 'erbf', 'ebm', 'xgb', 'rf', 'dt']


def reconstruct_preprocessing_config(result):
    """Verbatim copy from compute_prediction_stability_01feb26.py (keeps y_pp identical)."""
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


def fold0_y_train(dataset_name, a_joblib):
    """Reproduce the fold-0 training-target vector exactly as the stability run did."""
    X_raw, y_raw, meta = load_dataset(dataset_name)
    first_result = joblib.load(a_joblib)
    config = reconstruct_preprocessing_config(first_result)
    X_pp, y_pp, _ = preprocess_dataset(X_raw, y_raw, config, verbose=False)

    cv_config = first_result.get('config', {})
    outer_cv = RepeatedKFold(
        n_splits=cv_config.get('outer_splits', 5),
        n_repeats=cv_config.get('outer_repeats', 1),
        random_state=cv_config.get('random_state', 42),
    )
    X_for_split = X_pp.values if hasattr(X_pp, 'values') else X_pp
    trainval_idx, _ = next(iter(outer_cv.split(X_for_split)))
    y_trainval = y_pp[trainval_idx]                       # line 205 of stability script
    return np.asarray(y_trainval, dtype=float)


def main():
    print("=== Step 1: denominator robustness for SD/range ===", flush=True)
    df = pd.read_csv(SOT_CSV)
    datasets = sorted(df['dataset'].unique())
    print(f"SoT CSV: {len(df)} rows, {len(datasets)} datasets, "
          f"{df['model'].nunique()} models", flush=True)

    # One joblib per dataset (any model -> same per-dataset preprocessing/y_range).
    joblib_by_ds = {}
    for f in sorted(RESULTS_DIR.glob('*.joblib')):
        for m in MODEL_NAMES:
            if f.stem.startswith(m + '_'):
                ds = f.stem[len(m) + 1:]
                joblib_by_ds.setdefault(ds, f)
                break

    # Denominator family (all per-dataset constants -> within-dataset rankings identical):
    #   range = p100-p0 (max-min); p99_p1; p95_p5 (robust "trimmed ranges"); iqr = p75-p25; sd.
    denom = {}   # dataset -> dict of denominators
    for i, ds in enumerate(datasets):
        jb = joblib_by_ds.get(ds)
        if jb is None:
            print(f"  [{i+1}/{len(datasets)}] {ds}: NO JOBLIB -- skip", flush=True)
            continue
        y = fold0_y_train(ds, jb)
        p1, p5, p25, p75, p95, p99 = np.percentile(y, [1, 5, 25, 75, 95, 99])
        rng = float(y.max() - y.min())
        denom[ds] = {
            'range': rng,
            'p99_p1': float(p99 - p1),
            'p95_p5': float(p95 - p5),
            'iqr': float(p75 - p25),
            'sd': float(np.std(y, ddof=0)),
        }

        csv_range = float(df.loc[df['dataset'] == ds, 'y_range'].iloc[0])
        ok = np.isclose(rng, csv_range, rtol=1e-6, atol=1e-9)
        flag = 'OK ' if ok else '*** MISMATCH ***'
        infl = rng / denom[ds]['p95_p5'] if denom[ds]['p95_p5'] > 0 else float('nan')
        print(f"  [{i+1}/{len(datasets)}] {ds:28s} range={rng:.6g} "
              f"(csv {csv_range:.6g}) {flag} range/(p95-p5)={infl:.2f}", flush=True)
        if not ok:
            raise SystemExit(f"y_range mismatch on {ds}: recomputed {rng} vs csv {csv_range}")

    # Attach denominators; recompute SD/denom per (model,dataset).
    DENOMS = ['range', 'p99_p1', 'p95_p5', 'iqr', 'sd']
    df = df[df['dataset'].isin(denom)].copy()
    for dn in DENOMS:
        df[dn] = df['dataset'].map(lambda d, k=dn: denom[d][k])
        df[f'sd_over_{dn}'] = df['std_median'] / df[dn]
    df['sd_over_range'] = df['std_median'] / df['range']
    df['sd_over_iqr'] = df['std_median'] / df['iqr']

    # Sanity: recomputed sd_over_range must reproduce the CSV column.
    max_rel = float(((df['sd_over_range'] - df['sd_over_range_median']).abs()
                     / df['sd_over_range_median'].abs()).max())
    print(f"\nRecomputed SD/range vs CSV sd_over_range_median: max rel diff = "
          f"{max_rel:.2e} (expect ~0)", flush=True)

    print("\n=== Per-model median across datasets, with induced sort order ===",
          flush=True)
    per_model = df.groupby('model').agg(**{
        **{f'sd_over_{dn}': (f'sd_over_{dn}', 'median') for dn in DENOMS},
        'cv': ('cv_median', 'median'),
    })
    orders = {dn: list(per_model.sort_values(f'sd_over_{dn}').index) for dn in DENOMS}
    print(per_model.sort_values('sd_over_range').to_string(
        float_format=lambda x: f"{x:.4f}"), flush=True)
    for dn in DENOMS:
        print(f"  sort order by SD/{dn:7s}: {orders[dn]}", flush=True)
    for dn in DENOMS:
        print(f"  order(range)==order({dn})? {orders['range'] == orders[dn]}", flush=True)

    print("\n=== Pooled Spearman rho(gap, .) across all (model,dataset) points ===",
          flush=True)
    print("  (range is p100-p0; p99_p1 and p95_p5 are robust trimmed ranges)", flush=True)
    for dn in DENOMS:
        rho, _ = spearmanr(df['gap'], df[f'sd_over_{dn}'])
        print(f"  rho(gap, SD/{dn:7s}) = {rho:.4f}  (n={len(df)})", flush=True)
    rho_cv, _ = spearmanr(df['gap'], df['cv_median'])
    print(f"  rho(gap, CV       ) = {rho_cv:.4f}  (n={len(df)})", flush=True)

    print("\n=== Worst range-inflation datasets (range / (p95-p5)) ===", flush=True)
    infl = sorted(((denom[d]['range'] / denom[d]['p95_p5'], d) for d in denom),
                  reverse=True)
    for ratio, d in infl[:10]:
        print(f"  {ratio:6.2f}x  {d}", flush=True)

    # ---- Paper table numbers under the chosen p99-p1 denominator ----
    print("\n=== tab:stability numbers (p99-p1 headline), per model ===", flush=True)
    tab = df.groupby('model').agg(
        n_ds=('dataset', 'nunique'),
        mean_gap=('gap', 'mean'),
        median_gap=('gap', 'median'),
        median_cov=('cv_median', 'median'),
        sd_over_span=('sd_over_p99_p1', 'median'),
    ).sort_values('sd_over_span')
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)
    cpu8 = tab[tab.index != 'tabpfn']
    print(f"\n  CPU-8 order by SD/span: {list(cpu8.index)}", flush=True)
    rho_span, _ = spearmanr(df['gap'], df['sd_over_p99_p1'])
    rho_cov, _ = spearmanr(df['gap'], df['cv_median'])
    print(f"  pooled rho(gap, SD/span)={rho_span:.4f}  rho(gap, CoV)={rho_cov:.4f}",
          flush=True)

    out = PROJECT_ROOT / 'expl' / 'stability_denominator_sweep_11jul26_fixprefilter.csv'
    df.to_csv(out, index=False)
    print(f"\n  saved per-(model,dataset) sweep -> {out}", flush=True)

    # Within-dataset ranking identity (algebraic; verify empirically as a check).
    print("\n=== Within-dataset ranking identity check ===", flush=True)
    mism = 0
    for ds, g in df.groupby('dataset'):
        r1 = g.set_index('model')['sd_over_range'].rank()
        r2 = g.set_index('model')['sd_over_iqr'].rank()
        if not r1.equals(r2):
            mism += 1
    print(f"  datasets where SD/range and SD/IQR give a different within-dataset "
          f"model order: {mism} (expect 0)", flush=True)

    print("\n=== DONE ===", flush=True)


if __name__ == '__main__':
    main()
