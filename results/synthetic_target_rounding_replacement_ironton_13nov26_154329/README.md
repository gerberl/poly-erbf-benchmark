# Synthetic target-rounding replacement experiment (ironton, 13 Nov 2026)

Status: **COMPLETE**: 135/135 fits in 8 s (2026-11-13 15:43:30 -> 15:43:38 UTC). All 12 validation checks
pass (`validation.json`). The manuscript and `tab:erbf-ordinal-synth` were NOT updated.

This is a documented **replacement** for the paper's nine-condition rounding table. The generating script and
results for that table could not be found. This run is not an exact reproduction, and no settings were adjusted
to match the historical numbers.

## Protocol (full detail in protocol.json)

- Data: `np.random.default_rng(42)`, n = 1000, generated in this order: x1 ~ U(0,1), x2 ~ U(0,1), 3 standard-normal noise
  features, then y = sin(2 pi x1) + cos(3 pi x2) + N(0, 0.1^2). X has 5 columns, 2 informative.
- Targets: continuous y plus 8 rounded versions with L in {3, 5, 7, 10, 15, 21, 50, 100}, on an equal grid from
  y.min() to y.max(): `index = clip(floor((y-lo)/step + 0.5), 0, L-1)`, `y_L = lo + index*step`. The grid is
  defined once from the generated y, as part of constructing the synthetic outcome before CV. It is not a learned
  preprocessing step. Predictions are not rounded. Occupied levels were counted, not assumed, and equal L for every condition.
- CV: one `KFold(5, shuffle=True, random_state=42)` partition, materialised once (`folds.npz`) and reused for
  every model and condition.
- Models: ERBF, XGB and DT at `perbf/defaults.py` MODEL_DEFAULTS, with random_state=42; XGB also gets n_jobs=1, verbosity=0.
  There is no tuning. Resolved parameter dictionaries and source hashes are in protocol.json.
- Scaling: ERBF uses its internal standardisation (`standardize=True`, fitted inside `fit()` on the training fold).
  No model gets external scaling, so nothing is scaled before CV and nothing is scaled twice.
- Differences from the March diagnostic (`expl/erbf_ordinal_diagnostic_24mar26.py`, read but not executed):
  that script used cos(2 pi x2), noise SD 0.05 and only 5 levels (`round(5y)/5`). It also applied a full-data
  StandardScaler before CV, which ERBF then repeated internally. The present run follows the brief instead:
  cos(3 pi x2), SD 0.1, an equal-width grid and fold-internal scaling only.

Environment: `poly_erbf_benchmark` (Python 3.12.12, numpy 2.3.5, scipy 1.17.0, sklearn 1.6.1, xgboost 3.1.3).
`environment.yml` is identical to `environment_lock_linux_stability_09jul26.yml`. ERBF is an editable install,
git 73755ed, with a clean working tree. The project directory on ironton has no .git, so file hashes are recorded instead.

## Results: mean of the 5 fold test R2 values (fold SD, ddof=1)

| condition | ERBF | XGB | DT | ERBF - XGB (folds ERBF ahead) |
|---|---|---|---|---|
| L3 | 0.797 (0.040) | **0.855** (0.031) | 0.809 (0.054) | -0.059 (0/5) |
| L5 | **0.894** (0.011) | 0.886 (0.016) | 0.847 (0.022) | +0.007 (4/5) |
| L7 | **0.943** (0.010) | 0.923 (0.008) | 0.878 (0.021) | +0.019 (5/5) |
| L10 | **0.966** (0.005) | 0.945 (0.006) | 0.909 (0.009) | +0.021 (5/5) |
| L15 | **0.979** (0.003) | 0.960 (0.005) | 0.918 (0.007) | +0.019 (5/5) |
| L21 | **0.984** (0.003) | 0.964 (0.005) | 0.921 (0.009) | +0.019 (5/5) |
| L50 | **0.977** (0.021) | 0.969 (0.005) | 0.932 (0.009) | +0.008 (4/5) |
| L100 | **0.986** (0.003) | 0.969 (0.004) | 0.927 (0.011) | +0.017 (5/5) |
| continuous | **0.987** (0.003) | 0.969 (0.004) | 0.929 (0.008) | +0.018 (5/5) |

ERBF - DT is positive in every condition except L3 (-0.013, ERBF ahead in 1/5 folds). MSE, target variance,
train R2 and per-fold paired differences are in `summary.csv`, `paired_comparisons.csv` and `summarise_output.txt`.

## Findings

1. **Rounding changes ERBF's relative performance only at the coarsest level.** From L >= 7 up to the continuous
   target, ERBF leads XGB by about 0.02 in R2 in every fold. The lead narrows at L5 (+0.007) and **reverses at L3**,
   where ERBF trails XGB by 0.059 in all five folds and is also slightly behind DT. The mean MSE tells the same
   story (L3: ERBF 0.301, XGB 0.216). The ranking flip therefore does not come from R2's denominator, since every model in a condition shares the same target variance.
2. **Train R2 suggests a mechanism, but it is a conjecture.** At L3, XGB fits its training folds almost perfectly
   (train R2 0.996), while ERBF reaches only 0.856. This is consistent with the tree ensemble representing a
   three-valued step target that the smooth model cannot follow. This run does not test that explanation.
3. **The L50 entry is a seed artefact, not a rounding effect.** ERBF fold 0 at L50 scores 0.939 (train 0.944),
   against 0.98-0.99 for the other folds. In an out-of-protocol diagnostic, refitting that fold with seeds 0, 7
   and 123 gives 0.989-0.990. The continuous target fails similarly with seed 123 (0.945), and L100 more mildly
   with seed 0 (0.979) (`diagnostic_L50_fold0_erbf_seeds_out_of_protocol.txt`). With default settings, ERBF
   occasionally lands in a poorer solution, losing about 0.045 R2 on this problem. The protocol result (seed 42) is
   reported unchanged. The dip at L50 makes the mean curve non-monotone, and it should not be read as a property of L = 50.
4. Comparison with the historical table (context only, not a reproduction target): the qualitative pattern agrees
   (ERBF behind XGB only at 3 levels, ahead from 5 levels onwards, DT last), but the numbers differ. For example,
   the historical L3 gap is -0.004 against -0.059 here, and the historical continuous R2s are 0.987 / 0.978 / 0.931.
   The data-generating settings of the historical run are unknown.

## Limitations

- One synthetic dataset and one 5-fold partition. Fold SD describes spread across folds; it is not a confidence interval from independent repetitions.
- The evaluated target changes across conditions, and coarse rounding raises its variance (1.49 at L3 against 0.98 continuous).
  Changes in R2 across conditions therefore mix model fit with changes in the target. Comparisons between models within a condition are cleaner.
- Single seed (42) per model. Finding 3 shows that ERBF's seed sensitivity can exceed the gaps between conditions at fine L.
- This experiment does not establish why real datasets with discrete targets behave differently.

## Output inventory

data.npz (X, y_continuous, y_L*, grid_L*, levels_nominal, levels_occupied); folds.npz (train_k/test_k);
fold_results.csv (135 rows); predictions.csv (27,000 rows: held-out predictions with row_index, condition target and
continuous y); summary.csv; paired_comparisons.csv (90 rows); validation.json; summarise_output.txt;
protocol.json; environment.yml; package_versions.txt; copy of the runner
(`synthetic_target_rounding_replacement_ironton_13nov26.py`, sha256 c133702c...2133);
diagnostic_L50_fold0_erbf_seeds_out_of_protocol.txt.
Summariser: `expl/synthetic_target_rounding_replacement_summarise_ironton_13nov26.py` (sha256 a5b58ef1...2584).
Log: `logs/synthetic_target_rounding_replacement_ironton_13nov26_154329.log`.
