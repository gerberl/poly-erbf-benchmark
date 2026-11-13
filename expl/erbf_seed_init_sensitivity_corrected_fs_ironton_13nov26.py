"""
Corrected rerun of the two March ERBF supplementary diagnostics (ironton, 13 Nov 2026).

Protocol reference (NOT executed; they overwrite their CSVs):
    expl/erbf_convergence_stability_24mar26.py       -> Study A (seed sensitivity)
    expl/lipschitz_estimator_sensitivity_24mar26.py  -> Study B (initialisation sensitivity)

Fix: for d > 50 the March scripts ran mutual_info_regression on ALL (sampled) rows before the
80/20 split (feature-selection leakage). Here the 'train_only' procedure computes MI on the
training rows only. The 'all_rows_leaky' procedure reproduces the March selection on
superconduct alone, as a reproducibility baseline -- not a replacement result.

Usage:
    ~/micromamba/envs/poly_erbf_benchmark/bin/python \
        expl/erbf_seed_init_sensitivity_corrected_fs_ironton_13nov26.py --outdir <dir>
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

import erbf
from erbf import ERBFRegressor
from perbf.data.loader import load_dataset
import perbf.data.loader as loader_mod

DATASETS = ['esol', 'concrete_strength', 'power_plant', 'superconduct', 'california_housing']
BASELINE_DATASETS = ['superconduct']
N_MAX, SAMPLE_SEED, SPLIT_SEED, TEST_SIZE, MI_SEED, D_MAX = 5000, 42, 42, 0.2, 42, 50

COMMON = dict(n_rbf='auto', alpha=0.2, width_optim_iters=30, width_mode='full',
              width_optim='gradient', standardize=True)
SEEDS_A = [0, 7, 42, 123, 999]
CENTER_INITS_B = ['lipschitz', 'kmeans']
WIDTH_INITS_B = ['local_ridge', 'local_variance']
SEEDS_B = [42, 123, 999]

FIT_FILES = {
    ('A', 'train_only'): 'convergence_corrected.csv',
    ('B', 'train_only'): 'initialisation_corrected.csv',
    ('A', 'all_rows_leaky'): 'superconduct_original_procedure.csv',
    ('B', 'all_rows_leaky'): 'superconduct_original_procedure.csv',
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_array(a):
    a = np.ascontiguousarray(a)
    return sha256_bytes(a.tobytes())


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def arr_meta(a):
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': sha256_array(a)}


def prepare_inputs(ds_name, procedure):
    X, y, _ = load_dataset(ds_name)
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    prov = {'dataset': ds_name, 'procedure': procedure,
            'raw_X': arr_meta(X), 'raw_y': arr_meta(y)}
    assert np.isfinite(X).all() and np.isfinite(y).all(), ds_name

    sample_idx = None
    if len(y) > N_MAX:
        sample_idx = np.random.RandomState(SAMPLE_SEED).choice(len(y), N_MAX, replace=False)
        X, y = X[sample_idx], y[sample_idx]
    prov['sample_idx'] = None if sample_idx is None else arr_meta(sample_idx)

    n = len(y)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=SPLIT_SEED)

    sel = None
    if X.shape[1] > D_MAX:
        if procedure == 'all_rows_leaky':
            mi = mutual_info_regression(X, y, random_state=MI_SEED)
        elif procedure == 'train_only':
            mi = mutual_info_regression(X[train_idx], y[train_idx], random_state=MI_SEED)
        else:
            raise ValueError(procedure)
        sel = np.argsort(mi)[-D_MAX:]
        prov['mi'] = arr_meta(mi)
    Xs = X if sel is None else X[:, sel]
    prov['selected_features'] = None if sel is None else arr_meta(sel)

    # same call pattern as March (split after selection) must give identical partitions
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    assert np.array_equal(X_tr, Xs[train_idx]) and np.array_equal(X_te, Xs[test_idx])
    assert np.array_equal(y_tr, y[train_idx]) and np.array_equal(y_te, y[test_idx])
    assert len(np.intersect1d(train_idx, test_idx)) == 0 and len(train_idx) + len(test_idx) == n

    prov.update({'train_idx': arr_meta(train_idx), 'test_idx': arr_meta(test_idx),
                 'X_train': arr_meta(X_tr), 'X_test': arr_meta(X_te),
                 'y_train': arr_meta(y_tr), 'y_test': arr_meta(y_te),
                 'n': int(n), 'd_raw': int(X.shape[1]), 'd_used': int(Xs.shape[1])})
    arrays = {'sample_idx': sample_idx, 'train_idx': train_idx, 'test_idx': test_idx,
              'selected_features': sel}
    return (X_tr, X_te, y_tr, y_te), prov, arrays


def fit_jobs():
    jobs = []
    for procedure, datasets in [('train_only', DATASETS), ('all_rows_leaky', BASELINE_DATASETS)]:
        for ds in datasets:
            jobs += [(ds, 'A', procedure, s, 'lipschitz', 'local_ridge') for s in SEEDS_A]
            jobs += [(ds, 'B', procedure, s, ci, wi)
                     for ci in CENTER_INITS_B for wi in WIDTH_INITS_B for s in SEEDS_B]
    return jobs


def provenance_static(runner_path):
    erbf_src = Path(erbf.__file__).parent
    import subprocess
    git = lambda *a: subprocess.run(['git', '-C', str(erbf_src), *a], capture_output=True, text=True).stdout.strip()
    try:
        import numba
        numba_v = numba.__version__
    except ImportError:
        numba_v = None
    repo = Path(__file__).resolve().parents[1]
    return {
        'host': socket.gethostname(), 'platform': platform.platform(),
        'python': sys.version, 'executable': sys.executable,
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                     'sklearn': sklearn.__version__, 'pandas': pd.__version__,
                     'erbf': getattr(erbf, '__version__', None), 'numba': numba_v},
        'thread_env': {k: os.environ[k] for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'runner': {'path': str(runner_path), 'sha256': sha256_file(runner_path)},
        'erbf_source': {'dir': str(erbf_src), 'git_head': git('rev-parse', 'HEAD'),
                        'git_status_porcelain': git('status', '--porcelain'),
                        'files_sha256': {p.name: sha256_file(p) for p in sorted(erbf_src.glob('*.py'))}},
        'perbf_loader': {'path': loader_mod.__file__, 'sha256': sha256_file(loader_mod.__file__)},
        'march_reference_sha256': {p: sha256_file(repo / p) for p in [
            'expl/erbf_convergence_stability_24mar26.py',
            'expl/lipschitz_estimator_sensitivity_24mar26.py',
            'expl/erbf_convergence_stability_results_24mar26.csv',
            'expl/lipschitz_estimator_sensitivity_results_24mar26.csv']},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    runner_path = Path(__file__).resolve()

    jobs = fit_jobs()
    assert len(jobs) == 102, len(jobs)
    protocol = {
        'description': 'ERBF seed/initialisation sensitivity, corrected (train-only) MI feature selection, '
                       'plus superconduct all-rows (leaky, March) baseline',
        'datasets': DATASETS, 'baseline_datasets': BASELINE_DATASETS,
        'subsample': {'n_max': N_MAX, 'rng': 'np.random.RandomState(42).choice(n, 5000, replace=False)'},
        'split': {'test_size': TEST_SIZE, 'random_state': SPLIT_SEED},
        'feature_selection': {'d_max': D_MAX, 'score': 'mutual_info_regression', 'random_state': MI_SEED,
                              'select': 'np.argsort(mi)[-50:]',
                              'procedures': {'train_only': 'MI on training rows only (corrected)',
                                             'all_rows_leaky': 'MI on all sampled rows before split (March, leaky)'}},
        'model_common': COMMON,
        'study_A': {'center_init': 'lipschitz', 'width_init': 'local_ridge', 'seeds': SEEDS_A},
        'study_B': {'center_init': CENTER_INITS_B, 'width_init': WIDTH_INITS_B, 'seeds': SEEDS_B},
        'elapsed_definition': 'wall time from model construction through fit and both score() calls (as March)',
        'n_fits': {'corrected_A': 25, 'corrected_B': 60, 'baseline': 17, 'total': len(jobs)},
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        **provenance_static(runner_path),
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))

    print(f"START {protocol['started']} host={protocol['host']} sklearn={sklearn.__version__} "
          f"numpy={np.__version__} scipy={scipy.__version__} jobs={len(jobs)} out={out}", flush=True)

    cache, inputs_prov, idx_arrays = {}, [], {}
    t_all = time.time()
    for i, (ds, study, procedure, seed, ci, wi) in enumerate(jobs, 1):
        key = (ds, procedure)
        if key not in cache:
            cache[key], prov, arrays = prepare_inputs(ds, procedure)
            inputs_prov.append(prov)
            idx_arrays.update({f'{ds}__{procedure}__{k}': v for k, v in arrays.items() if v is not None})
            (out / 'inputs_provenance.json').write_text(json.dumps(inputs_prov, indent=2))
            np.savez(out / 'input_indices.npz', **idx_arrays)
            print(f"INPUTS {ds} [{procedure}] n={prov['n']} d_raw={prov['d_raw']} d_used={prov['d_used']}", flush=True)
        X_tr, X_te, y_tr, y_te = cache[key]

        t0 = time.time()
        model = ERBFRegressor(**COMMON, center_init=ci, width_init=wi, random_state=seed)
        model.fit(X_tr, y_tr)
        train_r2 = model.score(X_tr, y_tr)
        test_r2 = model.score(X_te, y_te)
        elapsed = time.time() - t0
        row = {'dataset': ds, 'study': study, 'selection_procedure': procedure, 'seed': seed,
               'center_init': ci, 'width_init': wi, 'train_r2': train_r2, 'test_r2': test_r2,
               'gap': train_r2 - test_r2, 'elapsed_s': elapsed,
               'n_train': len(y_tr), 'n_test': len(y_te), 'd_used': X_tr.shape[1]}
        f = out / FIT_FILES[(study, procedure)]
        pd.DataFrame([row]).to_csv(f, mode='a', header=not f.exists(), index=False)
        print(f"[{i:3d}/{len(jobs)}] {ds:18s} {study} {procedure:14s} {ci}/{wi} seed={seed:<3d} "
              f"train={train_r2:.4f} test={test_r2:.4f} gap={train_r2 - test_r2:.4f} {elapsed:.1f}s", flush=True)

    counts = {fn: len(pd.read_csv(out / fn)) for fn in sorted(set(FIT_FILES.values()))}
    assert counts == {'convergence_corrected.csv': 25, 'initialisation_corrected.csv': 60,
                      'superconduct_original_procedure.csv': 17}, counts
    protocol['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    protocol['total_elapsed_s'] = time.time() - t_all
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f"COMPLETE {protocol['finished']} fits={len(jobs)} total={protocol['total_elapsed_s']:.0f}s counts={counts}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print('FAILED', flush=True)
        raise
