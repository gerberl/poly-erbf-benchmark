"""Re-verify ALL matched-accuracy paper numbers on the prefilter-FIXED stability SoT.

Reproduces every figure in results.tex I am about to edit, from one consistent source
(the fixed stability CSV, which carries r2_val_adj, gap, and sd_over_range_median):
  - gap smooth-vs-tree matched-accuracy win-rate + threshold sweep       (results:128, appendix:437)
  - stability smooth-vs-tree matched-accuracy win-rate + threshold sweep (results:222)
  - stability sub-split vs xgb / vs rf                                   (results:223)
  - per-model steadiness score (Copeland)                               (results:224)
  - stability mean-rank (CPU7, no ridge): chebypoly/rf, CD, joint-most-stable count (results:228)

Within-dataset "steadier" is invariant to the SD normaliser (per-dataset constant),
so sd_over_range_median gives the same wins as the p99-p1 headline.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import friedmanchisquare

SOT = "results/stability/prediction_stability_revisionA_9models_55ds_11jul26_fixprefilter.csv"
ACC, GAP, STAB, THRESH = "r2_val_adj", "gap", "sd_over_range_median", 0.02
SMOOTH = ["chebypoly", "chebytree", "erbf"]
TREE = ["xgb", "rf"]
COMPETITIVE = ["erbf", "chebytree", "xgb", "chebypoly", "ebm", "rf"]
ALL9 = ["ridge", "dt", "rf", "xgb", "ebm", "erbf", "chebypoly", "chebytree", "tabpfn"]
QN = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102}

df = pd.read_csv(SOT)[["model", "dataset", ACC, GAP, STAB]].dropna()


def matched(a, b, thresh=THRESH):
    da = df[df.model == a].rename(columns={ACC: "aa", GAP: "ga", STAB: "sa"})
    db = df[df.model == b].rename(columns={ACC: "ab", GAP: "gb", STAB: "sb"})
    p = da.merge(db, on="dataset")
    return p[(p.aa - p.ab).abs() <= thresh]


def pooled(outcome_a, outcome_b, thresh=THRESH, pairs=None):
    pairs = pairs or [(s, t) for s in SMOOTH for t in TREE]
    tot = win = 0
    for a, b in pairs:
        p = matched(a, b, thresh)
        tot += len(p)
        win += int((p[outcome_a] < p[outcome_b]).sum())
    return win, tot


print("=" * 70)
print("GAP smooth-vs-tree (win = smooth has SMALLER gap)")
w, n = pooled("ga", "gb")
print(f"  pooled: {w}/{n} = {w/n:.1%}")
print("  threshold sweep:")
for t in [0.01, 0.02, 0.03, 0.05]:
    w, n = pooled("ga", "gb", t)
    print(f"    |dR2|<={t:.2f}: {w:3d}/{n:3d} = {w/n:.0%}")

print("=" * 70)
print("STABILITY smooth-vs-tree (win = smooth STEADIER, lower SD)")
w, n = pooled("sa", "sb")
print(f"  pooled: {w}/{n} = {w/n:.1%}")
print("  threshold sweep:")
for t in [0.01, 0.02, 0.03, 0.05]:
    w, n = pooled("sa", "sb", t)
    print(f"    |dR2|<={t:.2f}: {w:3d}/{n:3d} = {w/n:.0%}")
print("  sub-split:")
for tm in TREE:
    w, n = pooled("sa", "sb", pairs=[(s, tm) for s in SMOOTH])
    print(f"    smooth vs {tm}: {w}/{n} = {w/n:.0%}")

print("=" * 70)
print("PER-MODEL steadiness score (Copeland, all unordered pairs)")


def per_model(models):
    wins = {m: 0 for m in models}; comps = {m: 0 for m in models}
    for a, b in combinations(models, 2):
        p = matched(a, b)
        for _, r in p.iterrows():
            comps[a] += 1; comps[b] += 1
            if r.sa < r.sb: wins[a] += 1
            elif r.sb < r.sa: wins[b] += 1
    return pd.DataFrame({"model": models,
                         "n": [comps[m] for m in models],
                         "wins": [wins[m] for m in models],
                         "rate": [round(wins[m]/comps[m], 3) if comps[m] else np.nan for m in models]
                         }).sort_values("rate", ascending=False)


print("  competitive-6:")
print(per_model(COMPETITIVE).to_string(index=False))

print("=" * 70)
print("STABILITY mean-rank, competitive-7 (no ridge), 55 datasets")
CPU7 = ["dt", "rf", "xgb", "ebm", "erbf", "chebypoly", "chebytree"]
piv = df.pivot(index="dataset", columns="model", values=STAB)[CPU7].dropna(how="any")
ranks = piv.rank(axis=1, ascending=True, method="average")
tbl = ranks.mean().round(2).sort_values()
k, N = len(CPU7), len(piv)
cd = QN[k] * np.sqrt(k * (k + 1) / (6 * N))
chi2, p = friedmanchisquare(*[piv[m].values for m in CPU7])
print(tbl.to_string())
print(f"  N={N}, CD(0.05)={cd:.2f}, Friedman chi2={chi2:.1f}, p={p:.1e}")
# jointly most stable (rank-1 chebypoly or rf) count
top2 = ranks[["chebypoly", "rf"]].min(axis=1)
others = ranks.drop(columns=["chebypoly", "rf"]).min(axis=1)
joint = int((top2 < others).sum())
print(f"  chebypoly/rf jointly the steadiest on {joint} of {N} datasets")
print(f"  chebypoly mean rank={tbl['chebypoly']}, rf={tbl['rf']}, "
      f"diff to next={tbl.iloc[2]-tbl.iloc[1]:.2f}")
