# Polynomial and Ellipsoidal RBF Benchmark

Reproducibility repository for the paper: *Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression*.


## Repository structure

```
perbf/              # Library modules (data loading, tuning, evaluation, analysis)
scripts/            # Runnable scripts (benchmark runner, summarisation, figure generation)
making_plots/       # Figure generation scripts (linedot rank plots)
results/            # Pre-computed summary CSVs from the paper's experiments
figures/            # Generated figures used in the paper
```

## Setup

```bash
# Create environment
micromamba create -n polyerbfbench -f environment.yml
micromamba activate polyerbfbench

# Install custom model packages
pip install erbf poly-basis-ml

# Verify
python -c "from erbf import ERBFRegressor; print('OK')"
```

TabPFN requires a separate install (`pip install tabpfn`). The benchmark runs without it (TabPFN is simply skipped).

## Running the benchmark

All scripts add the repo root to `sys.path`, so `perbf/` is importable without installation. Alternatively, `pip install -e .` uses the included `pyproject.toml`.

```bash
# Quick test (2 models x 2 datasets)
python scripts/run_benchmark.py --test

# All models, preprocessed (max 50 features, 50k samples)
python scripts/run_benchmark.py \
    --max-features 50 --max-samples 50000 \
    --output-dir results/my_run_A --n-jobs=-1

# Full-scale data, no preprocessing
python scripts/run_benchmark.py \
    --models ridge dt xgb chebypoly chebytree \
    --datasets diamonds nyc-taxi-green-dec-2016 particulate-matter-ukair-2017 \
               qsar_tid_11 superconduct friedman1_d100 \
    --no-preprocess --output-dir results/my_run_B --n-jobs=-1

# Run specific models or datasets
python scripts/run_benchmark.py --models erbf chebypoly --datasets superconduct esol

# Summarise results
python scripts/summarize_benchmark.py results/my_run_A
```

## Regenerating figures from pre-computed results

The `results/benchmark_summary/` directory contains the summary CSVs used in the paper. To regenerate figures:

```bash
python scripts/generate_cd_plots.py
python scripts/generate_pareto_plots.py
python scripts/generate_boxstrip_r2adj.py
python scripts/compute_timing_table.py
```

Figures are written to `figures/`.

## Models

| Key | Model | Package |
|-----|-------|---------|
| `ridge` | Ridge regression | scikit-learn |
| `dt` | Decision tree | scikit-learn |
| `rf` | Random forest | scikit-learn |
| `xgb` | XGBoost | xgboost |
| `erbf` | Ellipsoidal RBF network | erbf |
| `chebypoly` | Chebyshev polynomial regression | poly-basis-ml |
| `chebytree` | Chebyshev model tree | poly-basis-ml |
| `tabpfn` | TabPFN | tabpfn (optional) |

## Dataset strata

Datasets are grouped into four strata by expected target-function smoothness:

| Stratum | Domain |
|---------|--------|
| S1 | Engineering / simulation | 
| S2 | Behavioural / social | Trees |
| S3 | Physics / chemistry / life science | 
| S4 | Economic / pricing (threshold-heavy) | 

## Licence

MIT. See [LICENSE](LICENSE).
