"""
Replacement synthetic target-rounding experiment (ironton, 13 Nov 2026).

Documented replacement for the paper's tab:erbf-ordinal-synth (generating script/results not found).
NOT an exact reproduction. One dataset (default_rng(42)), y = sin(2 pi x1) + cos(3 pi x2) + N(0, 0.1^2),
eight rounded targets (L levels on an equal grid over [min y, max y]) plus continuous y,
one materialised KFold(5, shuffle, 42) partition reused everywhere, ERBF / XGB / DT at MODEL_DEFAULTS.

Scaling: ERBF uses its internal standardisation (standardize=True default; scaler fitted in fit(),
i.e. on the training fold only). No external scaling for any model. Predictions are not rounded.

Usage:
    ~/micromamba/envs/poly_erbf_benchmark/bin/python \
        expl/synthetic_target_rounding_replacement_ironton_13nov26.py --outdir <dir>
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import erbf
from erbf import ERBFRegressor
import perbf.defaults as defaults_mod
from perbf.defaults import MODEL_DEFAULTS

SEED, N, NOISE_SD = 42, 1000, 0.1
LEVELS = [3, 5, 7, 10, 15, 21, 50, 100]
CONDITIONS = [f'L{L}' for L in LEVELS] + ['continuous']
MODELS = ['erbf', 'xgb', 'dt']
N_SPLITS = 5


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_array(a):
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def generate():
    rng = np.random.default_rng(SEED)
    x1 = rng.uniform(0, 1, N)
    x2 = rng.uniform(0, 1, N)
    noise_features = rng.standard_normal((N, 3))
    X = np.column_stack([x1, x2, noise_features])
    y = np.sin(2 * np.pi * x1) + np.cos(3 * np.pi * x2) + rng.normal(0, NOISE_SD, N)
    return X, y


def round_target(y, L):
    lo, hi = y.min(), y.max()
    step = (hi - lo) / (L - 1)
    index = np.clip(np.floor((y - lo) / step + 0.5), 0, L - 1).astype(int)
    grid = lo + np.arange(L) * step
    return lo + index * step, grid, index


def make_model(name):
    if name == 'erbf':
        return ERBFRegressor(**MODEL_DEFAULTS['erbf'], random_state=SEED)
    if name == 'xgb':
        return XGBRegressor(**MODEL_DEFAULTS['xgb'], random_state=SEED, n_jobs=1, verbosity=0)
    if name == 'dt':
        return DecisionTreeRegressor(**MODEL_DEFAULTS['dt'], random_state=SEED)
    raise ValueError(name)


def jsonable(d):
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else repr(v)) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    shutil.copy2(runner, out / runner.name)

    X, y = generate()
    assert X.shape == (N, 5) and np.isfinite(X).all() and np.isfinite(y).all()
    y_orig = y.copy()
    targets, grids, occupied = {'continuous': y}, {}, {}
    for L in LEVELS:
        yL, grid, idx = round_target(y, L)
        assert np.allclose(yL, grid[idx]) and np.isin(idx, np.arange(L)).all()
        targets[f'L{L}'], grids[f'L{L}'] = yL, grid
        occupied[f'L{L}'] = int(len(np.unique(idx)))
    occupied['continuous'] = int(len(np.unique(y)))
    np.savez(out / 'data.npz', X=X, y_continuous=y,
             **{f'y_{c}': targets[c] for c in CONDITIONS if c != 'continuous'},
             **{f'grid_{c}': g for c, g in grids.items()},
             levels_nominal=np.array(LEVELS),
             levels_occupied=np.array([occupied[f'L{L}'] for L in LEVELS]))

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = [(tr, te) for tr, te in kf.split(X)]
    all_te = np.concatenate([te for _, te in folds])
    assert np.array_equal(np.sort(all_te), np.arange(N))
    np.savez(out / 'folds.npz', **{f'train_{k}': tr for k, (tr, _) in enumerate(folds)},
             **{f'test_{k}': te for k, (_, te) in enumerate(folds)})

    erbf_src = Path(erbf.__file__).parent
    git = lambda *a: subprocess.run(['git', '-C', str(erbf_src), *a], capture_output=True, text=True).stdout.strip()
    protocol = {
        'description': 'Replacement synthetic target-rounding experiment (not a reproduction of tab:erbf-ordinal-synth)',
        'command': ' '.join([sys.executable] + sys.argv), 'cwd': os.getcwd(),
        'host': socket.gethostname(), 'platform': platform.platform(), 'python': sys.version,
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'data': {'rng': 'np.random.default_rng(42)', 'n': N,
                 'generation_order': ['x1=uniform(0,1,n)', 'x2=uniform(0,1,n)', 'noise=standard_normal((n,3))',
                                      'y=sin(2pi x1)+cos(3pi x2)+normal(0,0.1,n)'],
                 'noise_sd': NOISE_SD, 'X_sha256': sha256_array(X), 'y_sha256': sha256_array(y),
                 'X_shape': list(X.shape), 'X_dtype': str(X.dtype), 'y_dtype': str(y.dtype)},
        'rounding': {'levels_nominal': LEVELS,
                     'rule': 'lo=y.min(); hi=y.max(); step=(hi-lo)/(L-1); '
                             'index=clip(floor((y-lo)/step+0.5),0,L-1); y_L=lo+index*step',
                     'note': 'grid defined once from the full generated y as part of constructing the synthetic '
                             'outcome, before CV; not a learned preprocessing step. Predictions are not rounded.',
                     'levels_occupied': occupied,
                     'target_sha256': {c: sha256_array(targets[c]) for c in CONDITIONS},
                     'target_variance_full': {c: float(np.var(targets[c])) for c in CONDITIONS}},
        'cv': {'splitter': 'KFold(n_splits=5, shuffle=True, random_state=42)', 'materialised_once': True,
               'fold_test_sizes': [len(te) for _, te in folds],
               'test_idx_sha256': [sha256_array(te) for _, te in folds]},
        'scaling': 'none external; ERBF internal standardize (fitted on training fold inside fit)',
        'model_defaults_source': {'path': defaults_mod.__file__, 'sha256': sha256_file(defaults_mod.__file__),
                                  'MODEL_DEFAULTS_used': {m: MODEL_DEFAULTS[m] for m in MODELS}},
        'resolved_params': {m: jsonable(make_model(m).get_params()) for m in MODELS},
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'pandas': pd.__version__,
                     'erbf': getattr(erbf, '__version__', None)},
        'thread_env': {k: os.environ[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'runner': {'path': str(runner), 'sha256': sha256_file(runner)},
        'erbf_source': {'dir': str(erbf_src), 'git_head': git('rev-parse', 'HEAD'),
                        'git_status_porcelain': git('status', '--porcelain'),
                        'files_sha256': {p.name: sha256_file(p) for p in sorted(erbf_src.glob('*.py'))}},
    }
    assert protocol['resolved_params']['erbf']['standardize'] is True
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f"START {protocol['started']} host={protocol['host']} sklearn={sklearn.__version__} "
          f"xgboost={xgboost.__version__} fits={len(CONDITIONS) * len(MODELS) * N_SPLITS} out={out}", flush=True)
    print(f"occupied levels: {occupied}", flush=True)

    res_f, pred_f = out / 'fold_results.csv', out / 'predictions.csv'
    i, t_all = 0, time.time()
    for cond in CONDITIONS:
        yc = targets[cond]
        for model_name in MODELS:
            for k, (tr, te) in enumerate(folds):
                i += 1
                t0 = time.time()
                m = make_model(model_name).fit(X[tr], yc[tr])
                elapsed_fit = time.time() - t0
                p_te, p_tr = m.predict(X[te]), m.predict(X[tr])
                assert np.isfinite(p_te).all() and np.isfinite(p_tr).all()
                row = {'condition': cond, 'levels_nominal': np.nan if cond == 'continuous' else int(cond[1:]),
                       'levels_occupied': occupied[cond], 'model': model_name, 'fold': k,
                       'n_train': len(tr), 'n_test': len(te),
                       'test_r2': r2_score(yc[te], p_te), 'test_mse': mean_squared_error(yc[te], p_te),
                       'test_target_var': float(np.var(yc[te])), 'train_r2': r2_score(yc[tr], p_tr),
                       'elapsed_fit_s': elapsed_fit, 'elapsed_total_s': time.time() - t0}
                pd.DataFrame([row]).to_csv(res_f, mode='a', header=not res_f.exists(), index=False)
                pd.DataFrame({'condition': cond, 'model': model_name, 'fold': k, 'row_index': te,
                              'y_target': yc[te], 'y_continuous': y[te], 'y_pred': p_te}).to_csv(
                    pred_f, mode='a', header=not pred_f.exists(), index=False)
                print(f"[{i:3d}/135] {cond:10s} {model_name:4s} fold={k} test_r2={row['test_r2']:.4f} "
                      f"train_r2={row['train_r2']:.4f} mse={row['test_mse']:.5f} {elapsed_fit:.2f}s", flush=True)

    assert np.array_equal(y, y_orig)
    protocol['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    protocol['total_elapsed_s'] = time.time() - t_all
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f"COMPLETE {protocol['finished']} fits={i} total={protocol['total_elapsed_s']:.0f}s", flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print('FAILED', flush=True)
        raise
