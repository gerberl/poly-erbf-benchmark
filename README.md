# Polynomial and Ellipsoidal RBF Benchmark

Reproducibility repository for *Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression*. Updated for the second KBS revision, 13 November 2026.

## Paper and supplementary material

- [Current clean KBS manuscript](paper/manuscript-kbs-clean.pdf)
- [Scientific Supplementary Material](paper/supplementary-material.pdf): dataset catalogue, source IDs/URLs and selected hyperparameter distributions
- [Main LaTeX source archive](paper/manuscript-latex-source.zip)
- [Supplementary LaTeX source archive](paper/supplementary-latex-source.zip)
- [arXiv record](https://arxiv.org/abs/2602.22422) (may contain the earlier preprint until its update is processed)

## What is included

| Location | Contents |
| --- | --- |
| `perbf/` | Current dataset loaders, preprocessing, tuning and evaluation code |
| `scripts/` | Benchmark runner, summarisation, bootstrap and export scripts |
| `results/benchmark_summary/` | Corrected revision-A summaries: 494 model-dataset results, nine models, 55 datasets |
| `results/per_fold_results.csv` | 2,470 outer-fold records, including accuracy, timing and selected hyperparameters |
| `results/scalability-fixpipeline-11jul26/` | 200 outer-fold records for the five-model, eight-dataset full-scale check and its run configuration |
| `results/stability/` | Corrected bootstrap summaries: eight CPU models on 55 datasets and TabPFN on 44 |
| `figures/` | Six final figure panels in PNG and PDF formats; the removed gap-stability panel is not included |
| `expl/` | Supplementary experiment scripts, conditioning checks and corrected timing/stability analyses |
| `results/supplementary_seed_checks_ironton_13nov26_152347/` | Corrected seed/initialisation results, provenance and historical comparison |
| `results/synthetic_target_rounding_replacement_ironton_13nov26_154329/` | Synthetic inputs, folds, 135 fit results, predictions, validation and provenance |

## Datasets

The complete 55-dataset collection is **not vendored here**. The Supplementary Material gives the dataset catalogue and links/IDs; `perbf/data/loader.py` retrieves data from the original providers and generates synthetic benchmarks. Source datasets retain their providers' licences and access conditions. The synthetic target-rounding experiment includes its generated input data and split indices directly. Large fitted-estimator joblib files are not shipped; the main benchmark's fold-level numerical results and selected hyperparameters are provided as CSV instead.

## Setup and benchmark execution

```bash
micromamba create -n polyerbfbench -f environment.yml
micromamba activate polyerbfbench
pip install erbf poly-basis-ml
pip install -e .

# Main benchmark configuration: 50-feature and 50,000-sample caps
python scripts/run_benchmark.py --max-features 50 --max-samples 50000 \
    --output-dir results/my_revision_A --n-jobs 1

# Summarise a completed benchmark
python scripts/summarize_benchmark.py results/my_revision_A
```

TabPFN uses local GPU inference and requires the corresponding `tabpfn` package and model access. No paid inference API is used in the paper. Models depend on `erbf`, `poly-basis-ml`, scikit-learn, XGBoost and InterpretML. The supplementary run directories contain the Linux environment exports and source hashes used for those experiments; these are provenance records, not portable cross-platform lock files.

Feature selection is fitted separately inside each inner training split, then refitted on the outer training fold. The Spearman-filter index alignment is corrected. EBM uses `interactions=0`; ERBF and ChebyPoly are the smooth-basis group, ChebyTree is reported separately as a hybrid, and EBM as an additive bin-based comparator. Domain strata are exploratory.

## Rechecking supplementary results

Run commands from the repository root with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python expl/matched_accuracy_by_family_basis_hybrid_ebm_13nov26.py
PYTHONPATH=. python expl/synthetic_target_rounding_replacement_summarise_ironton_13nov26.py \
    results/synthetic_target_rounding_replacement_ironton_13nov26_154329
```

The rounding summary script validates saved predictions and rewrites derived summaries; it does not fit models. Its runner and the seed/initialisation runner accept a new output directory for fresh experiments. Main-bootstrap reruns require the benchmark joblib outputs produced by a new benchmark run. The saved CSVs support inspecting published results without those estimator files.

Historical March seed/initialisation results are retained only as comparison inputs: they reproduced partially, and the cause is unresolved. The paper uses the corrected Linux November results. Conditioning uses one intercept; six numerically singular cases are identified separately. The old expanded conditioning CSV is retained as the dataset-list input, not as the final diagnostic. The rounding experiment retains the seed-sensitive 50-level fit rather than excluding it.

`figures/` contains the exact submitted panels. Run `python making_plots/generate_rank_linedot_plots_04feb26.py` for accuracy ranks and the accuracy distribution, then `python scripts/regenerate_final_gap_and_tradeoff_figures_13nov26.py` for gap ranks and the two Pareto panels. These regenerate from the supplied corrected CSV; font availability may affect typography. Older generic plotting scripts are retained for exploration. Corrected batch prediction times are in `expl/corrected_timing_per_dataset_13nov26.csv` and its model summary; prediction time covers training plus test rows across all five folds and is normalised per 1,000 predicted rows.

## Citation

```bibtex
@misc{gerber2026revisiting,
  title={Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression},
  author={Luciano Gerber and Huw Lloyd},
  year={2026},
  eprint={2602.22422},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.22422}
}
```

Code is released under the MIT licence; see LICENSE. This does not relicense third-party datasets.
