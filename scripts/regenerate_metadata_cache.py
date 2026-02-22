#!/usr/bin/env python
"""
Regenerate dataset metadata cache from DATASET_REGISTRY.

This script loads all datasets in the registry and extracts metadata
(n_samples, n_features, stratum, source, ordinal) into a CSV cache
for fast lookups by get_benchmark_datasets_by_size() and other functions.

Usage:
    python scripts/regenerate_metadata_cache.py

Output:
    benchmark/data/dataset_sizes_cache.csv

Run this whenever:
- New datasets are added to DATASET_REGISTRY
- Dataset source URLs/IDs are corrected
- You need to verify cache is in sync with registry

Created: 23Jan26
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perbf.data import regenerate_metadata_cache

if __name__ == '__main__':
    print("=" * 70)
    print("Dataset Metadata Cache Regeneration")
    print("=" * 70)
    print()

    df = regenerate_metadata_cache()

    print()
    print("=" * 70)
    print("Cache regeneration complete!")
    print("=" * 70)
    print()
    print(f"Total datasets: {len(df)}")
    print(f"Output: benchmark/data/dataset_sizes_cache.csv")
    print()
    print("Size distribution:")
    print(df['n_samples'].describe().to_string())
