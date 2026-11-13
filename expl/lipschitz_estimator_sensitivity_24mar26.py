"""
Lipschitz estimator sensitivity experiment (R2#11).

Tests whether ERBF is robust to center placement strategy.
Compares lipschitz vs kmeans initialization (lipschitz uses 5-NN Lipschitz estimates;
kmeans is independent of the estimator). If results are similar, the Lipschitz
estimator quality is not a critical bottleneck.

Also tests with multiple random seeds to assess stochastic sensitivity.

Usage:
    micromamba run -n poly_erbf_benchmark python expl/lipschitz_estimator_sensitivity_24mar26.py
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
CENTER_INITS = ['lipschitz', 'kmeans']
WIDTH_INITS = ['local_ridge', 'local_variance']
SEEDS = [42, 123, 999]

results = []

for ds_name in DATASETS:
    X, y, meta = load_dataset(ds_name)
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if len(y) > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), 5000, replace=False)
        X, y = X[idx], y[idx]

    if X.shape[1] > 50:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42)
        top_idx = np.argsort(mi)[-50:]
        X = X[:, top_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n{ds_name} (n={len(y)}, d={X.shape[1]})", flush=True)

    for ci in CENTER_INITS:
        for wi in WIDTH_INITS:
            for seed in SEEDS:
                t0 = time.time()
                model = ERBFRegressor(
                    n_rbf='auto', alpha=0.2,
                    center_init=ci, width_init=wi,
                    width_optim_iters=30, width_mode='full',
                    width_optim='gradient', standardize=True,
                    random_state=seed,
                )
                model.fit(X_train, y_train)
                train_r2 = model.score(X_train, y_train)
                test_r2 = model.score(X_test, y_test)
                gap = train_r2 - test_r2
                elapsed = time.time() - t0

                print(f"  {ci}/{wi}/seed={seed}: R2={test_r2:.4f}, gap={gap:.4f}, {elapsed:.1f}s", flush=True)

                results.append({
                    'dataset': ds_name, 'center_init': ci, 'width_init': wi,
                    'seed': seed, 'test_r2': test_r2, 'gap': gap, 'time': elapsed,
                })

df = pd.DataFrame(results)
df.to_csv('expl/lipschitz_estimator_sensitivity_results_24mar26.csv', index=False)

print("\n=== Summary by center_init ===", flush=True)
summary = df.groupby(['dataset', 'center_init']).agg(
    r2_mean=('test_r2', 'mean'), r2_std=('test_r2', 'std'),
).round(4).unstack('center_init')
print(summary.to_string(), flush=True)

print("\n=== Overall R2 range per dataset ===", flush=True)
for ds in DATASETS:
    sub = df[df['dataset'] == ds]['test_r2']
    print(f"  {ds}: {sub.min():.4f} - {sub.max():.4f} (range={sub.max()-sub.min():.4f})", flush=True)
