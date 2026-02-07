"""Compute Table 5 (computational cost) from benchmark joblib results.

Usage:
    python scripts/compute_timing_table.py results/benchmark-A
"""
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

warnings.filterwarnings("ignore")


def main(results_dir):
    results_dir = Path(results_dir)
    data = defaultdict(lambda: {"tune": [], "train": [], "predict_ms_per_1k": []})

    for f in sorted(results_dir.glob("*.joblib")):
        r = joblib.load(f)
        m = r["model_name"]
        data[m]["tune"].append(r.get("tune_time", 0))
        data[m]["train"].append(r.get("train_time", 0))

        # predict_time is total wall-clock across all outer folds
        # n_samples_used / 5 folds gives approx n_test per fold; * 5 folds = n_samples_used total test
        folds = r.get("fold_results", [])
        total_pred_s = r.get("predict_time", 0)
        n_used = r.get("n_samples_used", 0)
        # each sample appears as test exactly once across 5 folds
        total_n_test = n_used
        if total_n_test > 0:
            ms_per_1k = (total_pred_s / total_n_test) * 1000 * 1000
        else:
            ms_per_1k = np.nan
        data[m]["predict_ms_per_1k"].append(ms_per_1k)

    print(f"{'Model':12s}  {'Tune (s)':>10s}  {'Train (s)':>10s}  {'Predict (ms/1K)':>16s}  n")
    print("-" * 60)
    for m in sorted(data, key=lambda m: np.mean(data[m]["tune"])):
        d = data[m]
        print(
            f"{m:12s}  {np.mean(d['tune']):10.1f}  {np.mean(d['train']):10.2f}"
            f"  {np.nanmean(d['predict_ms_per_1k']):16.1f}  {len(d['tune'])}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
