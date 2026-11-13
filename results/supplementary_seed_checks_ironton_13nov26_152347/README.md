# ERBF seed/initialisation sensitivity -- corrected feature selection (ironton, 13 Nov 2026)

Status: **COMPLETE** -- 102/102 fits (85 corrected + 17 superconduct original-procedure baseline),
148 s wall time, 2026-11-13 15:23:48 -> 15:26:17 UTC. Manuscript and conditioning table NOT updated.

## What was run

Corrected rerun of `expl/erbf_convergence_stability_24mar26.py` (Study A, seeds) and
`expl/lipschitz_estimator_sensitivity_24mar26.py` (Study B, initialisation). Neither original was
executed. The March scripts selected the top-50 mutual-information (MI) features on all 5000 sampled
rows before the 80/20 split. This leaks test rows into feature selection; superconduct (79 features)
is the only one of the five datasets affected. In the corrected procedure (`train_only`), MI is computed on the
4000 training rows only. Superconduct was also rerun with the March all-rows selection
(`all_rows_leaky`), for comparison only.

All other settings follow the March protocol: subsample 5000 rows with `RandomState(42)` if n > 5000,
split with `random_state=42`, MI `random_state=42`, `np.argsort(mi)[-50:]`, and the ERBF settings in `protocol.json`.
Sample SD (ddof=1) throughout.

- Runner: `expl/erbf_seed_init_sensitivity_corrected_fs_ironton_13nov26.py`
  (sha256 d7659536b0165d27f672f4d565edbe71d43676bb35ea83ba4d842c9cef3d28ad)
- Comparison: `expl/erbf_seed_init_sensitivity_compare_ironton_13nov26.py`
  (sha256 de3d67ed8dd4b59958414b509aaa3d6e082d2ac73c7b26cb169005a04695ed90); its printed output is in
  `compare_output.txt`
- Log: `logs/supplementary_seed_checks_ironton_13nov26_152347.log`

## Environment

`poly_erbf_benchmark` micromamba env on ironton. Python 3.12.12; the imported versions are numpy 2.3.5, scipy 1.17.0,
scikit-learn 1.6.1 (pip; it overrides the conda sklearn 1.8.0 / numpy 2.4.1 that are also installed),
pandas 2.3.3. ERBF 0.1.0 is an editable install from `my-own-packages-modules/erbf`, git HEAD 73755ed
(27 Feb 2026), with a clean working tree. Per-file source hashes are in `protocol.json`.

- `environment.yml` is identical to `environment_lock_linux_stability_09jul26.yml` (only the prefix line was excluded from the diff).
- In site-packages, scikit-learn, numpy and scipy are dated 19 Jan 2026, and erbf/poly_basis_ml 7 Feb 2026.
  Packages added after 24 Mar (shap, dash, aplr, ipython-related) are not imported by ERBF or the loader.
  The core numerical stack has therefore not changed since before the March runs.
- The `poly_erbf_benchmark` development directory on ironton is a Dropbox copy **without a .git**, so no
  repository commit can be recorded. File hashes for the loader, the March scripts and the March CSVs are in `protocol.json`.

Input provenance (raw-data hashes, sampled-row/train/test/selected-feature indices with shape/dtype/sha256,
MI vectors): `inputs_provenance.json`, `input_indices.npz`.

## Results

### Study A -- seed sensitivity (lipschitz / local_ridge, 5 seeds), test R2

| dataset | corrected mean (SD) | corrected gap mean (SD) | March mean (SD) | March gap mean (SD) |
|---|---|---|---|---|
| esol | 0.9082 (0.0021) | 0.0196 (0.0014) | 0.9101 (0.0039) | 0.0166 (0.0034) |
| concrete_strength | 0.8451 (0.0084) | 0.0553 (0.0086) | 0.8496 (0.0119) | 0.0474 (0.0100) |
| power_plant | 0.9384 (0.0011) | 0.0104 (0.0010) | 0.9384 (0.0011) | 0.0104 (0.0010) |
| superconduct | 0.8484 (0.0026) | 0.0248 (0.0061) | 0.8527 (0.0046) | 0.0226 (0.0059) |
| california_housing | 0.7686 (0.0077) | 0.0371 (0.0033) | 0.7686 (0.0077) | 0.0371 (0.0033) |

Superconduct with the original (leaky) procedure on ironton: 0.8473 (0.0051), gap 0.0273 (0.0052).

### Study B -- configuration means (3 seeds each), test R2, corrected

| dataset | lip/ridge | lip/var | km/ridge | km/var | range of 4 config means | range of 12 fits |
|---|---|---|---|---|---|---|
| esol | 0.9073 | 0.8936 | 0.8994 | 0.8982 | 0.0137 | 0.0301 |
| concrete_strength | 0.8491 | 0.8486 | 0.8708 | 0.8492 | 0.0222 | 0.0451 |
| power_plant | 0.9381 | 0.9384 | 0.9382 | 0.9399 | 0.0019 | 0.0036 |
| superconduct | 0.8472 | 0.8498 | 0.8247 | 0.8404 | **0.0252** | 0.0350 |
| california_housing | 0.7681 | 0.7629 | 0.7455 | 0.7513 | 0.0226 | 0.0368 |

March config-mean ranges: 0.0119, 0.0222, 0.0019, 0.0220, 0.0226. Superconduct leaky on ironton: 0.0156.
Within-config seed SDs are in `summary.csv`; the largest corrected one is 0.0155 (concrete km/var, identical
to March) and the next is 0.0139 (esol lip/var).

### Paper bounds

- Max Study A seed SD about 0.012: **holds** (corrected max 0.0084, concrete; March max was 0.0119).
- Config-mean range below 0.025: **narrowly fails** for superconduct (0.0252). The other four datasets hold, the largest being 0.0226.
  The excess (0.0002) is far smaller than the seed noise in each configuration mean (within-config SD up to 0.008 on
  superconduct, so the SE of a 3-seed mean is about 0.005). Read it as "about 0.025", not as a qualitative change.
  The paper's sentence still needs wording that does not claim a strict "< 0.025".

### Effect of the feature-selection fix (superconduct, same environment)

Over the 14 distinct (config, seed) pairs, corrected minus original test R2 has mean -0.0004 and SD 0.0074 (range -0.017 to
+0.009; 7 positive, 7 negative). Study A mean +0.0011. The largest single change is the km/ridge config mean
(0.8328 -> 0.8247), and that change is what pushes the config-mean range from 0.0156 to 0.0252. The correction moves
individual fits by about as much as changing the seed does, and shows no systematic direction.

### Reproduction of March -- PARTIAL, unexplained

| subset | exact matches (abs diff < 1e-10) | max abs diff |
|---|---|---|
| power_plant (no FS, subsampled) | 16/17 (the other differs by 5e-10) | 5e-10 |
| california_housing (no FS, subsampled) | 15/17 (others differ by ~1e-9) | 4e-9 |
| esol (no FS, n=1128) | 3/17 | 0.0225 |
| concrete_strength (no FS, n=1030) | 5/17 | 0.0174 |
| superconduct, original leaky procedure | 4/17 | 0.0157 |

- Feature selection cannot explain the mismatch: esol and concrete do not use it and still diverge.
- The mismatches are concentrated in `center_init='lipschitz'` (0 of 22 esol/concrete Lipschitz fits match
  exactly) plus a few kmeans/local_ridge fits. kmeans/local_variance matches everywhere except superconduct.
- On ironton these fits are deterministic. A fresh process with a different PYTHONHASHSEED reproduces this run's
  values bit-for-bit (esol and concrete, lipschitz, seed 42). In both this run and March, the Study A rows equal the
  overlapping Study B rows. So each source is internally deterministic, and the two disagree.
- Cause NOT established. Candidates that would discriminate: (i) March CSVs produced on another machine
  (BLAS/CPU differences in kNN tie-breaking or the gradient optimiser) -- check whether March ran on the Mac;
  (ii) different ERBF code in March (uncommitted edits on the producing machine) -- diff ERBF source between machines;
  (iii) ties in Lipschitz kNN estimates on datasets with duplicate rows (esol, concrete) -- count duplicate rows and
  test whether perturbing tie order reproduces the March values. No settings were changed to obtain agreement.

## What was not checked

- Which machine produced the March CSVs (not recorded; the 25 Mar mtime is only the Dropbox sync time).
- Whether the March CSVs reproduce on the Mac with today's code (reported by Codex as not exactly; not re-examined here).
- Optimiser convergence itself: these studies measure sensitivity to seed and initialisation, not convergence.
- Only one train/test split and n = 3 to 5 seeds per cell, so SDs and ranges are themselves noisy.

## Files

convergence_corrected.csv (25), initialisation_corrected.csv (60), superconduct_original_procedure.csv (17),
summary.csv, superconduct_paired_comparison.csv, march_comparison.csv (extra: per-fit ironton vs March),
protocol.json, inputs_provenance.json, input_indices.npz, environment.yml, environment_nobuilds.yml,
package_versions.txt (pip freeze), compare_output.txt.
