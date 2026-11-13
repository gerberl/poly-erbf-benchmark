"""
ERBF convergence stability experiment (R2#9).

Tests whether width optimisation converges to similar solutions despite non-convexity.
Runs ERBF with 5 different random seeds on 5 representative datasets.

Usage:
    micromamba run -n poly_erbf_benchmark python expl/erbf_convergence_stability_24mar26.py
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from erbf import ERBFRegressor
from perbf.data.loader import load_dataset

DATASETS = ['esol', 'concrete_strength', 'power_plant', 'superconduct', 'california_housing']
SEEDS = [0, 7, 42, 123, 999]

results = []

for ds_name in DATASETS:
    X, y, meta = load_dataset(ds_name)
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # Subsample large datasets
    if len(y) > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), 5000, replace=False)
        X, y = X[idx], y[idx]

    # Cap features at 50 (MI-based)
    if X.shape[1] > 50:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42)
        top_idx = np.argsort(mi)[-50:]
        X = X[:, top_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n{ds_name} (n={len(y)}, d={X.shape[1]})", flush=True)
    seed_r2s = []
    seed_gaps = []

    for seed in SEEDS:
        t0 = time.time()
        model = ERBFRegressor(
            n_rbf='auto', alpha=0.2,
            center_init='lipschitz', width_init='local_ridge',
            width_optim_iters=30, width_mode='full',
            width_optim='gradient', standardize=True,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        gap = train_r2 - test_r2
        elapsed = time.time() - t0

        seed_r2s.append(test_r2)
        seed_gaps.append(gap)

        print(f"  seed={seed}: R2={test_r2:.4f}, gap={gap:.4f}, time={elapsed:.1f}s", flush=True)

        results.append({
            'dataset': ds_name, 'seed': seed,
            'train_r2': train_r2, 'test_r2': test_r2,
            'gap': gap, 'time': elapsed,
        })

    print(f"  Summary: R2={np.mean(seed_r2s):.4f} +/- {np.std(seed_r2s):.4f}, "
          f"gap={np.mean(seed_gaps):.4f} +/- {np.std(seed_gaps):.4f}", flush=True)

df = pd.DataFrame(results)
df.to_csv('expl/erbf_convergence_stability_results_24mar26.csv', index=False)

print("\n=== Overall summary ===", flush=True)
summary = df.groupby('dataset').agg(
    r2_mean=('test_r2', 'mean'), r2_std=('test_r2', 'std'),
    gap_mean=('gap', 'mean'), gap_std=('gap', 'std'),
).round(4)
print(summary.to_string(), flush=True)
print(f"\nMean R2 std across datasets: {summary['r2_std'].mean():.4f}", flush=True)
