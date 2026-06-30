"""Utility modules for benchmark scripts."""

from .progress_tracker import ProgressTracker
from .result_loader import (
    load_result,
    load_all_results,
    extract_summary_row,
    create_summary_dataframe,
)
from .batch_runner import run_benchmark_batch, DEFAULT_CONFIG
from .scaling_data import (
    make_scaling_dataset,
    make_friedman_extended,
    make_smooth_nonlinear,
    make_piecewise_linear,
    SCALING_GRID_FULL,
    SCALING_GRID_REDUCED,
    SCALING_GRID_QUICK,
    # Real-world dataset support
    load_real_dataset,
    subsample_dataset,
    make_real_scaling_dataset,
    REAL_DATASETS_FOR_SCALING,
    REAL_SCALING_GRID,
)

__all__ = [
    'ProgressTracker',
    'load_result',
    'load_all_results',
    'extract_summary_row',
    'create_summary_dataframe',
    'run_benchmark_batch',
    'DEFAULT_CONFIG',
    # Scaling experiment utilities
    'make_scaling_dataset',
    'make_friedman_extended',
    'make_smooth_nonlinear',
    'make_piecewise_linear',
    'SCALING_GRID_FULL',
    'SCALING_GRID_REDUCED',
    'SCALING_GRID_QUICK',
    # Real-world dataset support
    'load_real_dataset',
    'subsample_dataset',
    'make_real_scaling_dataset',
    'REAL_DATASETS_FOR_SCALING',
    'REAL_SCALING_GRID',
]
