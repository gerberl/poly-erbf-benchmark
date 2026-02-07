"""
Stratum definitions for benchmark datasets.

ARCHITECTURE - Single Points of Truth:
======================================

1. THIS FILE (strata.py):
   - Stratum DEFINITIONS only (S1-S4 names, characteristics, expected winners)
   - Does NOT list individual datasets

2. loader.py:
   - Dataset REGISTRATIONS (the PRIMARY source of truth for datasets)
   - Each dataset's: stratum, ordinal flag, description, loader function
   - To add/modify/remove a dataset, edit loader.py

3. dataset_sizes_cache.csv:
   - DERIVED cache, regenerated from loader.py via regenerate_metadata_cache()
   - Contains: dataset, n_samples, n_features, stratum, source, ordinal
   - Used by summarize_benchmark.py for fast lookups
   - Regenerate after changing loader.py: `python -c "from benchmark.data.loader import regenerate_metadata_cache; regenerate_metadata_cache()"`

4. notes-plans/dataset_registry_17jan26.md:
   - Human-readable documentation (may drift from code - loader.py is authoritative)

Updated: 25Jan26 - Clarified architecture, S2 count 12→11 (online_news_popularity excluded)

Canonical table:

| Stratum | Count | Domain | Smoothness Hypothesis | Expected Winner |
|---------|-------|--------|----------------------|-----------------|
| S1 | 13 | Engineering/Simulation | Smooth by design | Smooth models |
| S2 | 11 | Behavioural/Social | Threshold-prone (human decisions) | Trees |
| S3 | 16 | Physics/Chemistry/Life | Smooth with phase transitions | Smooth models |
| S4 | 16 | Economic/Pricing | Threshold-heavy (explicit rules) | Trees |

Total: 56 datasets
Excluded: online_news_popularity (S2, no signal), pmlb_227_cpu_small (S1, duplicate of cpu_act)

Groupings:
- Smooth strata (S1+S3): 29 datasets - expect ERBF, Chebyshev to win
- Threshold strata (S2+S4): 27 datasets - expect XGBoost, RF to win
"""

# Stratum definitions
# Counts as of 25Jan26: S1=13, S2=11, S3=16, S4=16 (Total=56)
# Removed: online_news_popularity (S2), pmlb_227_cpu_small (S1, duplicate of cpu_act)
STRATA = {
    'S1': {
        'name': 'Engineering/Simulation',
        'characteristic': 'Smooth by design',
        'expected_winner': 'Smooth models (ERBF, Chebyshev)',
        'count': 13,  # Updated 25Jan26: was 14, pmlb_227_cpu_small removed (duplicate)
    },
    'S2': {
        'name': 'Behavioural/Social',
        'characteristic': 'Threshold-prone (human decisions)',
        'expected_winner': 'Tree models (XGBoost, RF)',
        'count': 11,  # Updated 25Jan26: was 12, online_news_popularity excluded
    },
    'S3': {
        'name': 'Physics/Chemistry/Life',
        'characteristic': 'Smooth with phase transitions',
        'expected_winner': 'Smooth models (ERBF, Chebyshev)',
        'count': 16,
    },
    'S4': {
        'name': 'Economic/Pricing',
        'characteristic': 'Threshold-heavy (explicit rules)',
        'expected_winner': 'Tree models (XGBoost, RF)',
        'count': 16,
    },
}

# Groupings for paper
SMOOTH_STRATA = ['S1', 'S3']  # 29 datasets
THRESHOLD_STRATA = ['S2', 'S4']  # 27 datasets

# Simple name lookups for summarize_benchmark.py
STRATUM_NAMES = {
    'S1': 'Engineering/Simulation (Smooth)',
    'S2': 'Behavioural/Social (Threshold-prone)',
    'S3': 'Physics/Chemistry/Life (Smooth)',
    'S4': 'Economic/Pricing (Threshold-heavy)',
}

STRATUM_NAMES_SHORT = {
    'S1': 'Engineering/Simulation',
    'S2': 'Behavioural/Social',
    'S3': 'Physics/Chemistry/Life',
    'S4': 'Economic/Pricing',
}
