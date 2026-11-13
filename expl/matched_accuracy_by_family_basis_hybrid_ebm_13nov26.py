import pandas as pd

SOT = "results/stability/prediction_stability_revisionA_9models_55ds_11jul26_fixprefilter.csv"
ACC, GAP, STAB = "r2_val_adj", "gap", "sd_over_range_median"
raw = pd.read_csv(SOT)
df = raw[["model", "dataset", ACC, GAP, STAB]].dropna()
print("rows", len(raw), "after dropna", len(df))


def pooled(group, tree, out, thresh):
    tot = win = 0
    for a in group:
        for b in tree:
            da = df[df.model == a].set_index("dataset")
            db = df[df.model == b].set_index("dataset")
            p = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
            p = p[(p[f"{ACC}_a"] - p[f"{ACC}_b"]).abs() <= thresh]
            tot += len(p)
            win += int((p[f"{out}_a"] < p[f"{out}_b"]).sum())
    return win, tot


GROUPS = {
    "basis+hybrid (paper)": ["chebypoly", "chebytree", "erbf"],
    "smooth-basis only": ["chebypoly", "erbf"],
    "chebytree only": ["chebytree"],
    "ebm only": ["ebm"],
}
for out in [GAP, STAB]:
    print(f"\n== {out}: win = smaller vs {{xgb, rf}}")
    for t in [0.005, 0.01, 0.02, 0.05, 0.10]:
        cells = [f"{k}: {w}/{n}={w/n:.0%}" for k, g in GROUPS.items() for w, n in [pooled(g, ["xgb", "rf"], out, t)]]
        print(f"  t={t}: " + " | ".join(cells))
