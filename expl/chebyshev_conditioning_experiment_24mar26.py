"""
Chebyshev conditioning experiment (R2#10 reviewer feedback).

Checks whether Chebyshev basis + raw product interactions is better-conditioned
than a monomial basis under the same conditions.

For each of 5 datasets, we construct:
  1. Chebyshev main effects only (degree 5)
  2. Chebyshev main effects + raw product interactions (x_i * x_j, i < j)
  3. Monomial main effects only (degree 5)
  4. Monomial main effects + raw product interactions

All features are scaled to [-1, 1] before basis construction.
Condition number of the design matrix (numpy.linalg.cond) is reported.

Note: we cap to 10 features (MI-selected top-10) to keep the interaction
matrix tractable (10 choose 2 = 45 pairs) and comparisons fair across
datasets with different dimensionalities.
"""

import sys
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.polynomial import polyvander
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, '/Users/gerberl/Dropbox/00.new.structure/rke-doing/poly_erbf_benchmark')
from perbf.data.loader import load_dataset

DATASETS = ['esol', 'concrete_strength', 'power_plant', 'superconduct', 'california_housing']
DEGREE = 5
MAX_FEATURES = 10  # cap for tractable interaction matrix


def select_top_features(X, y, max_features):
    """MI-based top-k feature selection. Ensures numpy array output."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[1] <= max_features:
        return X
    mi = mutual_info_regression(X, y, random_state=0)
    idx = np.argsort(mi)[::-1][:max_features]
    return X[:, idx]


def scale_to_minus1_plus1(X):
    """Scale each feature to [-1, 1] using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(X)


def build_chebyshev_main(X, degree):
    """
    Chebyshev main-effects design matrix.
    For each feature x_j, expand T_0(x_j), T_1(x_j), ..., T_degree(x_j).
    Returns matrix of shape (n, n_features * (degree + 1)).
    The intercept column T_0=1 is included once (from first feature only)
    to avoid duplicating the bias; alternatively we keep all columns and
    let the condition number reflect the raw redundancy.
    We keep all columns (including repeated T_0=1) to match what a naive
    implementation would produce, which is the fairest comparison.
    """
    n, d = X.shape
    blocks = []
    for j in range(d):
        # chebvander returns (n, degree+1) for 1-d input
        block = chebvander(X[:, j], degree)  # shape (n, degree+1)
        blocks.append(block)
    return np.hstack(blocks)  # (n, d*(degree+1))


def build_monomial_main(X, degree):
    """
    Monomial main-effects design matrix.
    For each feature x_j, expand [1, x_j, x_j^2, ..., x_j^degree].
    Returns matrix of shape (n, n_features * (degree + 1)).
    Same structure as Chebyshev version for direct comparison.
    """
    n, d = X.shape
    blocks = []
    for j in range(d):
        block = polyvander(X[:, j], degree)  # shape (n, degree+1)
        blocks.append(block)
    return np.hstack(blocks)  # (n, d*(degree+1))


def build_raw_interactions(X):
    """
    Raw product interaction terms: x_i * x_j for all pairs i < j.
    These are the same (raw feature products) regardless of whether the
    main-effects basis is Chebyshev or monomial -- that is the point of
    the comparison.
    Returns matrix of shape (n, n_features*(n_features-1)/2).
    """
    n, d = X.shape
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append(X[:, i] * X[:, j])
    if not pairs:
        return np.empty((n, 0))
    return np.column_stack(pairs)


def condition_number(A):
    """Condition number of matrix A (ratio of largest to smallest singular value)."""
    return np.linalg.cond(A)


def run_experiment():
    results = []
    print(f"Chebyshev vs Monomial conditioning experiment", flush=True)
    print(f"Degree={DEGREE}, max_features={MAX_FEATURES}", flush=True)
    print(f"{'Dataset':<30} {'n':>7} {'d_orig':>7} {'d_used':>7} "
          f"{'cond_cheby_main':>18} {'cond_cheby_interact':>20} "
          f"{'cond_mono_main':>16} {'cond_mono_interact':>18}", flush=True)
    print("-" * 120, flush=True)

    for ds_name in DATASETS:
        X, y, meta = load_dataset(ds_name)
        n_orig, d_orig = X.shape
        print(f"Loaded {ds_name}: n={n_orig}, d={d_orig}", flush=True)

        # Feature selection
        X_sel = select_top_features(X, y, MAX_FEATURES)
        d_used = X_sel.shape[1]

        # Scale to [-1, 1]
        X_sc = scale_to_minus1_plus1(X_sel)

        # Subsample if very large to keep SVD tractable
        if n_orig > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_orig, size=5000, replace=False)
            X_sc = X_sc[idx]
            print(f"  Subsampled to 5000 rows for SVD tractability", flush=True)

        # Build design matrices
        cheby_main = build_chebyshev_main(X_sc, DEGREE)
        mono_main = build_monomial_main(X_sc, DEGREE)
        interactions = build_raw_interactions(X_sc)

        cheby_interact = np.hstack([cheby_main, interactions])
        mono_interact = np.hstack([mono_main, interactions])

        print(f"  Matrix shapes: cheby_main={cheby_main.shape}, "
              f"cheby_interact={cheby_interact.shape}, "
              f"mono_main={mono_main.shape}, "
              f"mono_interact={mono_interact.shape}", flush=True)

        # Condition numbers
        cond_cm = condition_number(cheby_main)
        cond_ci = condition_number(cheby_interact)
        cond_mm = condition_number(mono_main)
        cond_mi = condition_number(mono_interact)

        ratio_main = cond_mm / cond_cm
        ratio_interact = cond_mi / cond_ci

        print(f"  cond_cheby_main={cond_cm:.3e}  cond_cheby_interact={cond_ci:.3e}", flush=True)
        print(f"  cond_mono_main={cond_mm:.3e}   cond_mono_interact={cond_mi:.3e}", flush=True)
        print(f"  Ratios (mono/cheby): main={ratio_main:.2f}x, interact={ratio_interact:.2f}x", flush=True)

        row = {
            'dataset': ds_name,
            'n_samples': n_orig,
            'n_features_orig': d_orig,
            'n_features_used': d_used,
            'degree': DEGREE,
            'n_cols_main': cheby_main.shape[1],
            'n_cols_interact': cheby_interact.shape[1],
            'cond_cheby_main': cond_cm,
            'cond_cheby_interact': cond_ci,
            'cond_mono_main': cond_mm,
            'cond_mono_interact': cond_mi,
            'ratio_main_mono_over_cheby': ratio_main,
            'ratio_interact_mono_over_cheby': ratio_interact,
        }
        results.append(row)

        print(f"{'  ' + ds_name:<30} {n_orig:>7} {d_orig:>7} {d_used:>7} "
              f"{cond_cm:>18.3e} {cond_ci:>20.3e} "
              f"{cond_mm:>16.3e} {cond_mi:>18.3e}", flush=True)
        print(flush=True)

    df = pd.DataFrame(results)

    out_csv = '/Users/gerberl/Dropbox/00.new.structure/rke-doing/poly_erbf_benchmark/expl/chebyshev_conditioning_results_24mar26.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}", flush=True)

    # Summary table
    print("\n=== SUMMARY ===", flush=True)
    print(f"{'Dataset':<30} {'cond_cheby_main':>16} {'cond_mono_main':>16} "
          f"{'ratio_main':>12} {'cond_cheby_int':>16} {'cond_mono_int':>14} {'ratio_int':>11}", flush=True)
    print("-" * 90, flush=True)
    for row in results:
        print(f"{row['dataset']:<30} {row['cond_cheby_main']:>16.3e} {row['cond_mono_main']:>16.3e} "
              f"{row['ratio_main_mono_over_cheby']:>12.2f}x "
              f"{row['cond_cheby_interact']:>16.3e} {row['cond_mono_interact']:>14.3e} "
              f"{row['ratio_interact_mono_over_cheby']:>10.2f}x", flush=True)

    # Overall geometric mean ratios
    ratios_main = [r['ratio_main_mono_over_cheby'] for r in results]
    ratios_int = [r['ratio_interact_mono_over_cheby'] for r in results]
    gm_main = np.exp(np.mean(np.log(ratios_main)))
    gm_int = np.exp(np.mean(np.log(ratios_int)))
    print(f"\nGeometric mean ratio (mono/cheby): main={gm_main:.2f}x, interact={gm_int:.2f}x", flush=True)
    print("(> 1 means Chebyshev is better-conditioned)", flush=True)


if __name__ == '__main__':
    run_experiment()
