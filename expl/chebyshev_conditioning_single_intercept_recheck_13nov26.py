"""Re-check of appendix tab:conditioning (R2.4).

The 24mar26 experiment kept T_0 = 1 (and x^0 = 1) once per feature, so every design matrix
has d identical constant columns and is exactly rank-deficient; its condition numbers
(1e18-1e21) are beyond 1/eps and the mono/cheby ratios compare round-off. Here the same
setup (top-10 MI features, [-1,1] min-max scaling, degree 5, 5000-row subsample, raw
products) is computed both ways: 'dup' = original duplicated intercepts, 'one' = a single
intercept, as in the ChebyshevRegressor described in methods (1 + d*c columns).
"""
import sys
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from numpy.polynomial.polynomial import polyvander

sys.path.insert(0, "expl")
from chebyshev_conditioning_experiment_24mar26 import (
    load_dataset, select_top_features, scale_to_minus1_plus1, build_raw_interactions,
    DEGREE, MAX_FEATURES,
)

DATASETS = pd.read_csv("expl/chebyshev_conditioning_results_expanded_25mar26.csv")
EPS_LIMIT = 1 / np.finfo(float).eps


def main_effects(X, vander, single_intercept):
    blocks = [vander(X[:, j], DEGREE) for j in range(X.shape[1])]
    if single_intercept:
        blocks = [blocks[0]] + [b[:, 1:] for b in blocks[1:]]
    return np.hstack(blocks)


rows = []
for ds in DATASETS.dataset:
    X, y, _ = load_dataset(ds)
    Xs = scale_to_minus1_plus1(select_top_features(X, y, MAX_FEATURES))
    if Xs.shape[0] > 5000:
        Xs = Xs[np.random.default_rng(42).choice(Xs.shape[0], 5000, replace=False)]
    inter = build_raw_interactions(Xs)
    r = {"dataset": ds, "d": Xs.shape[1]}
    for tag, single in [("dup", False), ("one", True)]:
        cm, mm = main_effects(Xs, chebvander, single), main_effects(Xs, polyvander, single)
        k = {"cm": np.linalg.cond(cm), "mm": np.linalg.cond(mm),
             "ci": np.linalg.cond(np.hstack([cm, inter])), "mi": np.linalg.cond(np.hstack([mm, inter]))}
        r.update({f"{tag}_{key}": v for key, v in k.items()})
        r[f"{tag}_ratio_main"] = k["mm"] / k["cm"]
        r[f"{tag}_ratio_int"] = k["mi"] / k["ci"]
    rows.append(r)
    print(f"{ds:28s} dup: cheby={r['dup_cm']:.1e} ratio_main={r['dup_ratio_main']:.2f} | "
          f"one: cheby_main={r['one_cm']:.1e} mono_main={r['one_mm']:.1e} "
          f"cheby_int={r['one_ci']:.1e} mono_int={r['one_mi']:.1e} "
          f"ratio_main={r['one_ratio_main']:.1f} ratio_int={r['one_ratio_int']:.2f}", flush=True)

out = pd.DataFrame(rows)
assert len(out) == len(DATASETS)
out = out.merge(DATASETS[["dataset", "ratio_main", "ratio_int"]].rename(
    columns={"ratio_main": "paper_ratio_main", "ratio_int": "paper_ratio_int"}), on="dataset", validate="1:1")
out.to_csv("expl/chebyshev_conditioning_single_intercept_recheck_13nov26.csv", index=False)

print(f"\n1/eps = {EPS_LIMIT:.1e}")
for tag in ["dup", "one"]:
    sing = (out[[f"{tag}_cm", f"{tag}_mm", f"{tag}_ci", f"{tag}_mi"]] > EPS_LIMIT).sum().to_dict()
    for c in ["ratio_main", "ratio_int"]:
        v = out[f"{tag}_{c}"]
        print(f"{tag} {c}: >1 in {(v > 1).sum()}/{len(v)}, geomean {np.exp(np.log(v).mean()):.2f}")
    print(f"{tag} matrices with cond > 1/eps: {sing}")
print("Spearman(paper ratio_main, one ratio_main) =",
      round(out.paper_ratio_main.corr(out.one_ratio_main, method="spearman"), 2))
