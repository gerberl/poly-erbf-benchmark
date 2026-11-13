"""
Validate and summarise the outputs of erbf_seed_init_sensitivity_corrected_fs_ironton_13nov26.py,
and compare against the March CSVs. Sample SD (ddof=1) throughout.

Usage:
    ~/micromamba/envs/poly_erbf_benchmark/bin/python \
        expl/erbf_seed_init_sensitivity_compare_ironton_13nov26.py <outdir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[1]
KEY = ['study', 'selection_procedure', 'dataset', 'center_init', 'width_init', 'seed']
CFG = ['center_init', 'width_init']
SEED_SD_BOUND, CFG_RANGE_BOUND = 0.012, 0.025

a = pd.read_csv(out / 'convergence_corrected.csv')
b = pd.read_csv(out / 'initialisation_corrected.csv')
o = pd.read_csv(out / 'superconduct_original_procedure.csv')
assert (len(a), len(b), len(o)) == (25, 60, 17), (len(a), len(b), len(o))
assert (len(o[o.study == 'A']), len(o[o.study == 'B'])) == (5, 12)
df = pd.concat([a, b, o], ignore_index=True)
assert not df.duplicated(KEY).any()
assert np.isfinite(df[['train_r2', 'test_r2', 'gap', 'elapsed_s']].to_numpy()).all()
assert np.allclose(df.gap, df.train_r2 - df.test_r2)

march_a = pd.read_csv(repo / 'expl/erbf_convergence_stability_results_24mar26.csv').assign(
    study='A', center_init='lipschitz', width_init='local_ridge', selection_procedure='march_csv')
march_b = pd.read_csv(repo / 'expl/lipschitz_estimator_sensitivity_results_24mar26.csv').assign(
    study='B', selection_procedure='march_csv')
assert (len(march_a), len(march_b)) == (25, 60)
march = pd.concat([march_a, march_b], ignore_index=True)

allruns = pd.concat([df, march], ignore_index=True)
sd = lambda s: s.std(ddof=1)

sa = (allruns[allruns.study == 'A'].groupby(['study', 'selection_procedure', 'dataset'])
      .agg(n_fits=('test_r2', 'size'), test_r2_mean=('test_r2', 'mean'), test_r2_sd=('test_r2', sd),
           gap_mean=('gap', 'mean'), gap_sd=('gap', sd)).reset_index().assign(summary='A_seed'))
sb = (allruns[allruns.study == 'B'].groupby(['study', 'selection_procedure', 'dataset'] + CFG)
      .agg(n_fits=('test_r2', 'size'), test_r2_mean=('test_r2', 'mean'), test_r2_sd=('test_r2', sd),
           gap_mean=('gap', 'mean'), gap_sd=('gap', sd)).reset_index().assign(summary='B_config'))
rng_cfg = (sb.groupby(['study', 'selection_procedure', 'dataset']).test_r2_mean
           .agg(lambda s: s.max() - s.min()).rename('config_mean_range'))
rng_fit = (allruns[allruns.study == 'B'].groupby(['study', 'selection_procedure', 'dataset']).test_r2
           .agg(lambda s: s.max() - s.min()).rename('individual_fit_range'))
max_cfg_sd = sb.groupby(['study', 'selection_procedure', 'dataset']).test_r2_sd.max().rename('max_within_config_seed_sd')
sr = pd.concat([rng_cfg, rng_fit, max_cfg_sd], axis=1).reset_index().assign(summary='B_dataset_range')
summary = pd.concat([sa, sb, sr], ignore_index=True)
cols = ['summary', 'study', 'selection_procedure', 'dataset', 'center_init', 'width_init', 'n_fits',
        'test_r2_mean', 'test_r2_sd', 'gap_mean', 'gap_sd', 'config_mean_range', 'individual_fit_range',
        'max_within_config_seed_sd']
summary = summary[cols]
summary.to_csv(out / 'summary.csv', index=False)

# corrected vs original procedure, superconduct, same Linux env
pk = ['study', 'dataset', 'center_init', 'width_init', 'seed']
sc = df[df.dataset == 'superconduct']
paired = (sc[sc.selection_procedure == 'train_only'][pk + ['train_r2', 'test_r2', 'gap']]
          .merge(sc[sc.selection_procedure == 'all_rows_leaky'][pk + ['train_r2', 'test_r2', 'gap']],
                 on=pk, suffixes=('_corrected', '_original'), validate='one_to_one'))
assert len(paired) == 17
paired['test_r2_diff_corrected_minus_original'] = paired.test_r2_corrected - paired.test_r2_original
paired['gap_diff_corrected_minus_original'] = paired.gap_corrected - paired.gap_original
paired.to_csv(out / 'superconduct_paired_comparison.csv', index=False)

# Linux vs March: corrected for non-superconduct; both procedures for superconduct
mk = ['study', 'dataset', 'center_init', 'width_init', 'seed']
mcols = mk + ['test_r2', 'gap']
lin = df[mcols + ['selection_procedure']]
mc = lin.merge(march[mcols], on=mk, suffixes=('_ironton', '_march'), validate='many_to_one')
assert len(mc) == 85 + 17, len(mc)
mc['test_r2_diff_ironton_minus_march'] = mc.test_r2_ironton - mc.test_r2_march
mc['gap_diff_ironton_minus_march'] = mc.gap_ironton - mc.gap_march
mc['exact_match_1e-10'] = mc.test_r2_diff_ironton_minus_march.abs() < 1e-10
mc.to_csv(out / 'march_comparison.csv', index=False)

pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print('=== Study A (seed SD, ddof=1) ===')
print(sa.pivot(index='dataset', columns='selection_procedure',
               values=['test_r2_mean', 'test_r2_sd', 'gap_mean', 'gap_sd']).to_string())
print('\n=== Study B config means (test R2) ===')
print(sb.pivot_table(index=['dataset'] + CFG, columns='selection_procedure', values='test_r2_mean').to_string())
print('\n=== Study B config SD (test R2) ===')
print(sb.pivot_table(index=['dataset'] + CFG, columns='selection_procedure', values='test_r2_sd').to_string())
print('\n=== Study B ranges ===')
print(sr.pivot(index='dataset', columns='selection_procedure',
               values=['config_mean_range', 'individual_fit_range']).to_string())
print('\n=== superconduct paired: corrected - original (same env) ===')
print(paired[pk[:1] + pk[2:] + ['test_r2_corrected', 'test_r2_original', 'test_r2_diff_corrected_minus_original',
                                 'gap_diff_corrected_minus_original']].to_string(index=False))
print('\n=== ironton vs March, by procedure/dataset ===')
print(mc.groupby(['selection_procedure', 'dataset']).agg(
    n=('seed', 'size'), exact=('exact_match_1e-10', 'sum'),
    max_abs_diff=('test_r2_diff_ironton_minus_march', lambda s: s.abs().max()),
    mean_diff=('test_r2_diff_ironton_minus_march', 'mean')).to_string())
cor = summary.selection_procedure == 'train_only'
max_seed_sd = sa[sa.selection_procedure == 'train_only'].test_r2_sd.max()
max_cfg_rng = sr[sr.selection_procedure == 'train_only'].config_mean_range.max()
print(f'\nBounds (corrected): max Study A seed SD = {max_seed_sd:.4f} (paper ~{SEED_SD_BOUND}) -> '
      f'{"HOLDS" if max_seed_sd <= SEED_SD_BOUND + 5e-4 else "EXCEEDED"}; '
      f'max Study B config-mean range = {max_cfg_rng:.4f} (paper <{CFG_RANGE_BOUND}) -> '
      f'{"HOLDS" if max_cfg_rng < CFG_RANGE_BOUND else "EXCEEDED"}')
for p in ['march_csv', 'all_rows_leaky']:
    print(f'  [{p}] max seed SD = {sa[sa.selection_procedure == p].test_r2_sd.max():.4f}; '
          f'max config-mean range = {sr[sr.selection_procedure == p].config_mean_range.max():.4f}')
