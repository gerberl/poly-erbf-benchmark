"""
Validate and summarise synthetic_target_rounding_replacement_ironton_13nov26.py outputs.

Usage:
    ~/micromamba/envs/poly_erbf_benchmark/bin/python \
        expl/synthetic_target_rounding_replacement_summarise_ironton_13nov26.py <outdir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

out = Path(sys.argv[1])
LEVELS = [3, 5, 7, 10, 15, 21, 50, 100]
CONDS = [f'L{L}' for L in LEVELS] + ['continuous']
MODELS = ['erbf', 'xgb', 'dt']

fr = pd.read_csv(out / 'fold_results.csv')
pr = pd.read_csv(out / 'predictions.csv')
data, folds = np.load(out / 'data.npz'), np.load(out / 'folds.npz')
proto = json.loads((out / 'protocol.json').read_text())
checks = {}

checks['135_unique_results'] = len(fr) == 135 and not fr.duplicated(['condition', 'model', 'fold']).any()
checks['finite_scores'] = bool(np.isfinite(fr[['test_r2', 'test_mse', 'train_r2', 'test_target_var']].to_numpy()).all())
checks['finite_predictions'] = bool(np.isfinite(pr.y_pred).all())
checks['27000_prediction_rows'] = len(pr) == 27000
cnt = pr.groupby(['condition', 'model']).row_index.agg(lambda s: (len(s) == 1000) and s.nunique() == 1000)
checks['each_row_heldout_once_per_condition_model'] = bool(cnt.all()) and len(cnt) == 27

test_idx = [folds[f'test_{k}'] for k in range(5)]
same = all(np.array_equal(np.sort(g.row_index.to_numpy()), np.sort(test_idx[k]))
           for (c, m, k), g in pr.groupby(['condition', 'model', 'fold']))
partition = np.array_equal(np.sort(np.concatenate(test_idx)), np.arange(1000)) and all(
    len(np.intersect1d(folds[f'train_{k}'], test_idx[k])) == 0 and
    len(folds[f'train_{k}']) + len(test_idx[k]) == 1000 for k in range(5))
checks['identical_folds_all_conditions_models'] = same and partition

y = data['y_continuous']
checks['continuous_target_unchanged'] = (hashlib := __import__('hashlib')).sha256(
    np.ascontiguousarray(y).tobytes()).hexdigest() == proto['data']['y_sha256']
on_grid, occ_ok = True, True
for L in LEVELS:
    yL, grid = data[f'y_L{L}'], data[f'grid_L{L}']
    d = np.abs(yL[:, None] - grid[None, :]).min(1)
    on_grid &= bool(d.max() < 1e-12) and len(grid) == L
    occ_ok &= len(np.unique(yL)) == proto['rounding']['levels_occupied'][f'L{L}']
checks['rounded_targets_on_grid'] = on_grid
checks['occupied_levels_recorded_and_match'] = occ_ok
tgt_ok = all(np.allclose(g.y_target.to_numpy(), (y if c == 'continuous' else data[f'y_{c}'])[g.row_index.to_numpy()])
             for c, g in pr.groupby('condition'))
checks['prediction_targets_match_data'] = tgt_ok

rec = pr.groupby(['condition', 'model', 'fold']).apply(
    lambda g: pd.Series({'r2_rec': r2_score(g.y_target, g.y_pred), 'mse_rec': mean_squared_error(g.y_target, g.y_pred)}),
    include_groups=False).reset_index()
mm = fr.merge(rec, on=['condition', 'model', 'fold'], validate='one_to_one')
checks['fold_scores_reproduce_from_predictions'] = bool(
    np.allclose(mm.test_r2, mm.r2_rec, atol=1e-12) and np.allclose(mm.test_mse, mm.mse_rec, atol=1e-12))

summ = (fr.groupby(['condition', 'model']).agg(
    levels_nominal=('levels_nominal', 'first'), levels_occupied=('levels_occupied', 'first'),
    n_folds=('fold', 'size'), test_r2_mean=('test_r2', 'mean'), test_r2_sd=('test_r2', lambda s: s.std(ddof=1)),
    test_r2_min=('test_r2', 'min'), test_r2_max=('test_r2', 'max'),
    test_mse_mean=('test_mse', 'mean'), test_target_var_mean=('test_target_var', 'mean'),
    train_r2_mean=('train_r2', 'mean'), elapsed_fit_s_mean=('elapsed_fit_s', 'mean')).reset_index())
rsumm = mm.groupby(['condition', 'model']).r2_rec.mean()
checks['summary_reproduces_from_predictions_mean_of_fold_r2'] = bool(
    np.allclose(summ.set_index(['condition', 'model']).test_r2_mean, rsumm.reindex(summ.set_index(['condition', 'model']).index)))
summ['condition'] = pd.Categorical(summ.condition, CONDS, ordered=True)
summ['model'] = pd.Categorical(summ.model, MODELS, ordered=True)
summ = summ.sort_values(['condition', 'model'])
summ.to_csv(out / 'summary.csv', index=False)

w = fr.pivot_table(index=['condition', 'fold'], columns='model', values='test_r2').reset_index()
pc = pd.concat([
    w[['condition', 'fold']].assign(comparison='erbf_minus_xgb', r2_erbf=w.erbf, r2_other=w.xgb, diff=w.erbf - w.xgb),
    w[['condition', 'fold']].assign(comparison='erbf_minus_dt', r2_erbf=w.erbf, r2_other=w.dt, diff=w.erbf - w.dt)])
assert len(pc) == 90
pc['condition'] = pd.Categorical(pc.condition, CONDS, ordered=True)
pc = pc.sort_values(['comparison', 'condition', 'fold'])
pc.to_csv(out / 'paired_comparisons.csv', index=False)

(out / 'validation.json').write_text(json.dumps(checks, indent=2))
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print('=== validation ===')
for k, v in checks.items():
    print(f'  {"PASS" if v else "FAIL"}  {k}')
print('\n=== mean test R2 (mean of 5 fold R2), [fold SD ddof=1] ===')
t = summ.pivot(index='condition', columns='model', values='test_r2_mean')
s = summ.pivot(index='condition', columns='model', values='test_r2_sd')
print(pd.concat({'mean': t, 'sd': s}, axis=1).to_string())
print('\n=== mean test MSE and target variance ===')
print(pd.concat([summ.pivot(index='condition', columns='model', values='test_mse_mean'),
                 summ.groupby('condition', observed=True).test_target_var_mean.first().rename('target_var')],
                axis=1).to_string())
print('\n=== paired diffs (mean, SD ddof=1, folds ERBF ahead out of 5) ===')
print(pc.groupby(['comparison', 'condition'], observed=True)['diff'].agg(
    mean='mean', sd=lambda s: s.std(ddof=1), n_pos=lambda s: int((s > 0).sum())).unstack('comparison').to_string())
print('\n=== train R2 ===')
print(summ.pivot(index='condition', columns='model', values='train_r2_mean').to_string())
assert all(checks.values()), {k: v for k, v in checks.items() if not v}
print('\nALL CHECKS PASS')
