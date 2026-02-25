#!/usr/bin/env python
"""
Run benchmark with configurable dataset selection and models.

Usage:
    python scripts/run_benchmark.py --test                   # Quick test (2 models x 2 datasets)
    python scripts/run_benchmark.py                          # All datasets, proportional trials
    python scripts/run_benchmark.py --n-trials 30            # Override with uniform 30 trials for all
    python scripts/run_benchmark.py --models ridge erbf      # Specific models
    python scripts/run_benchmark.py --exclude-models tabpfn  # Exclude models (run TabPFN separately)
    python scripts/run_benchmark.py --datasets esol superconduct  # Specific datasets
    python scripts/run_benchmark.py --high-d-high-n          # High-d/high-n datasets only

    # Preprocessing options
    python scripts/run_benchmark.py --no-prefilter           # Disable Spearman/MI prefilter
    python scripts/run_benchmark.py --no-dim-reduction       # Disable k-best feature selection
    python scripts/run_benchmark.py --max-features 50        # Custom feature cap (default: 100)
    python scripts/run_benchmark.py --max-samples 10000      # Subsample large datasets

    # Fast mode (no tuning)
    python scripts/run_benchmark.py --no-tune                # Use sensible defaults (see benchmark/defaults.py)

Trial allocation (default: proportional based on search space complexity):
    ridge: 20, dt: 25, rf: 30, xgb: 50, erbf: 30, chebypoly: 25, chebytree: 30
    Use --n-trials N to force uniform N trials for all models.

TabPFN Note:
    TabPFN uses local GPU mode. When running TabPFN:
    - n_jobs is forced to 1 (sequential outer folds)
    Consider running TabPFN separately: --models tabpfn --output-dir results/.../tabpfn_run
"""

# === BLAS thread control (must be before numpy import) ===
# Prevent BLAS from spawning threads that compete with joblib workers.
# Without this, each joblib worker spawns ~20-30 BLAS threads, causing thrashing.
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Suppress noisy warnings
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*disp.*iprint.*L-BFGS-B.*', category=DeprecationWarning)

# Suppress RDKit warnings (molecule parsing noise)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.batch_runner import run_benchmark_batch, DEFAULT_CONFIG
from scripts.utils.result_loader import load_all_results, create_summary_dataframe, add_rankings
from perbf.tuning.optuna_cv import MODEL_TRIAL_COUNTS
from perbf.defaults import MODEL_DEFAULTS


def log_package_versions(save_path=None):
    """Log versions of key packages for reproducibility."""
    import sys
    import platform

    packages = [
        'sklearn', 'numpy', 'pandas', 'xgboost', 'optuna',
        'joblib', 'pmlb', 'scipy', 'tabpfn_client'
    ]

    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"---"
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            lines.append(f"{pkg}: {mod.__version__}")
        except ImportError:
            lines.append(f"{pkg}: not installed")
        except AttributeError:
            lines.append(f"{pkg}: (no version attr)")
        except Exception as e:
            lines.append(f"{pkg}: error ({e})")

    version_text = "\n".join(lines)
    print(f"Package versions:\n{version_text}\n")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(version_text + "\n")
        print(f"Versions saved to: {save_path}\n")

    return version_text


# All benchmark models
ALL_MODELS = ['ridge', 'dt', 'rf', 'xgb', 'erbf', 'chebypoly', 'chebytree', 'tabpfn']


def get_model_configs(n_trials_override: int = None, no_tune: bool = False) -> dict:
    """Build model configs with proportional or uniform trial counts.

    Args:
        n_trials_override: If set, use this for all models (uniform).
                          If None, use proportional MODEL_TRIAL_COUNTS.
        no_tune: If True, skip tuning and use sensible defaults from perbf.defaults.

    Returns:
        Dict mapping model name to config overrides.
    """
    configs = {}
    for model in ALL_MODELS:
        if no_tune:
            # No tuning: use sensible defaults
            configs[model] = {
                'n_trials': 0,
                'no_tune': True,
                'default_params': MODEL_DEFAULTS.get(model, {}),
            }
        elif n_trials_override is not None:
            # Uniform: same trials for all (except tabpfn)
            trials = 0 if model == 'tabpfn' else n_trials_override
            configs[model] = {'n_trials': trials}
        else:
            # Proportional: based on search space complexity
            trials = MODEL_TRIAL_COUNTS.get(model, 30)
            configs[model] = {'n_trials': trials}
    return configs


def get_all_datasets():
    """Get all registered datasets."""
    from perbf.data.loader import get_benchmark_datasets
    return sorted(get_benchmark_datasets())


# High-d and high-n stress test datasets (for Benchmark B)
HIGH_D_HIGH_N_DATASETS = [
    # High-d (d > 50)
    'qsar_tid_11',                        # d=1024
    'Allstate_Claims_Severity',           # d=130, n=188K
    'pmlb_4544_GeographicalOriginalofMusic',  # d=117
    'friedman1_d100',                     # d=100
    'pmlb_588_fri_c4_1000_100',          # d=100
    'superconduct',                       # d=79
    # High-n (n > 50K)
    'nyc-taxi-green-dec-2016',           # n=582K
    'medical_charges',                    # n=163K
    'diamonds',                           # n=54K
]


def main():
    parser = argparse.ArgumentParser(description='Run benchmark on tabular regression datasets')
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        help='Models to run (default: all)')
    parser.add_argument('--exclude-models', nargs='+', default=[],
                        help='Models to exclude (e.g., --exclude-models tabpfn)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: ALL datasets)')
    parser.add_argument('--high-d-high-n', action='store_true',
                        help='Use only high-d/high-n stress test datasets (9 datasets)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Uniform trials for all models (default: proportional per model)')
    parser.add_argument('--trials-proportional', action='store_true', default=True,
                        help='Use proportional trials based on search space (default)')
    parser.add_argument('--n-jobs', type=int, default=-2,
                        help='Number of parallel jobs (default: -2)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/run_TIMESTAMP)')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                        help='Skip experiments with existing results')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip tuning, use sensible defaults (fast mode for large datasets)')
    parser.add_argument('--test', action='store_true',
                        help='Quick test with 2 models x 2 datasets')
    # Preprocessing options
    parser.add_argument('--no-prefilter', action='store_true',
                        help='Disable uninformative feature removal (Spearman/MI prefilter)')
    parser.add_argument('--no-dim-reduction', action='store_true',
                        help='Disable dimensionality reduction (k-best feature selection)')
    parser.add_argument('--max-features', type=int, default=100,
                        help='Max features for dim reduction (default: 100, only if d > this)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples (subsample if n > this, default: None = no limit)')
    args = parser.parse_args()

    # Get datasets
    if args.datasets:
        datasets = args.datasets
    elif args.high_d_high_n:
        datasets = HIGH_D_HIGH_N_DATASETS
    else:
        datasets = get_all_datasets()

    # Quick test mode
    if args.test:
        models = ['ridge', 'erbf']
        datasets = ['concrete_strength', 'esol']
        n_trials_override = 5  # Uniform 5 for test
        no_tune = False
        print("TEST MODE: Running 2 models x 2 datasets with 5 trials")
    else:
        models = args.models
        n_trials_override = args.n_trials  # None = proportional, int = uniform
        no_tune = args.no_tune
        if no_tune:
            print("NO-TUNE MODE: Using sensible defaults (see benchmark/defaults.py)")

    # Apply exclusions
    if args.exclude_models:
        models = [m for m in models if m not in args.exclude_models]
        print(f"Excluding models: {args.exclude_models}")

    # TabPFN special handling: force sequential execution (API rate limits)
    tabpfn_in_models = 'tabpfn' in models
    if tabpfn_in_models and args.n_jobs != 1:
        print("NOTE: TabPFN detected - forcing n_jobs=1 (API rate limits)")
        args.n_jobs = 1

    # Setup output directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = project_root / f'results/run_{timestamp}'

    # Log package versions for reproducibility
    log_package_versions(save_path=save_dir / 'package_versions.txt')

    # Build config
    config = DEFAULT_CONFIG.copy()
    config['n_parallel_models'] = args.n_jobs  # Parallel experiments
    # Preprocessing options
    config['prefilter'] = not args.no_prefilter
    config['max_features'] = None if args.no_dim_reduction else args.max_features
    config['max_samples'] = args.max_samples

    # Build model-specific configs with trial counts
    model_configs = get_model_configs(n_trials_override, no_tune=no_tune)

    # Run benchmark
    print(f"\nBenchmark")
    print(f"  Models: {models}")
    print(f"  Datasets ({len(datasets)}): {datasets}")
    if no_tune:
        trials_desc = "NO TUNING (sensible defaults)"
    elif n_trials_override is not None:
        trials_desc = f"{n_trials_override} trials (uniform)"
    else:
        trials_summary = {m: model_configs[m]['n_trials'] for m in models if m in model_configs}
        trials_desc = f"proportional trials: {trials_summary}"
    inner_desc = "adaptive (3 if n>=1000 else 5)" if config['inner_splits'] is None else f"{config['inner_splits']}-fold"
    print(f"  Config: {config['outer_splits']}x{config['outer_repeats']} CV, "
          f"{inner_desc} inner, {trials_desc}")
    # Preprocessing info
    preproc_parts = []
    if config['prefilter']:
        preproc_parts.append("prefilter")
    if config['max_features']:
        preproc_parts.append(f"max_features={config['max_features']}")
    if config['max_samples']:
        preproc_parts.append(f"max_samples={config['max_samples']}")
    if preproc_parts:
        print(f"  Preprocessing: {', '.join(preproc_parts)}")
    else:
        print(f"  Preprocessing: disabled")
    print(f"  Output: {save_dir}\n")

    summary = run_benchmark_batch(
        models=models,
        datasets=datasets,
        save_dir=str(save_dir),
        config=config,
        model_configs=model_configs,
        skip_existing=args.skip_existing,
        verbose=not args.quiet,
    )

    # Quick summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    results = load_all_results(save_dir)
    if results:
        df = create_summary_dataframe(results)
        df = add_rankings(df)

        # Quick model summary (just R² and gap)
        model_summary = df.groupby('model').agg({
            'r2_val': ['mean', 'std'],
            'gap': 'mean',
            'rank': 'mean',
        }).round(4)
        model_summary.columns = ['r2_mean', 'r2_std', 'gap_mean', 'avg_rank']
        model_summary = model_summary.sort_values('avg_rank')

        print("\n--- Quick Summary (sorted by rank) ---")
        print(model_summary.to_string())
        print(f"\nResults: {len(df)} model×dataset combinations")
    else:
        print("No results found (all experiments may have failed)")

    print(f"\nResults saved to: {save_dir}")
    print(f"\nFor full analysis run:")
    print(f"  python scripts/summarize_benchmark.py {save_dir}")
    return summary


if __name__ == '__main__':
    main()
