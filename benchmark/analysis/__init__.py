"""Analysis utilities for benchmark results."""

from .discontinuity_smoothness import (
    # Dataset-level metrics (characterize data)
    compute_discontinuity_profile,
    llqd_distribution,
    ntj_stats,
    llfrr,
    gtv,
    cqjs_binned,
    # Unified roughness metrics (same formula for data/model comparison)
    compute_knn_roughness,
    compute_roughness_amplification,
    # Model-level metrics (characterize predictions)
    compute_knn_jump,
    compute_path_metrics,
    compute_violation_probability,
    compute_local_lipschitz_quantiles,
    compute_model_smoothness_profile,
)
from .model_complexity import extract_model_info

__all__ = [
    # Dataset-level
    'compute_discontinuity_profile',
    'llqd_distribution',
    'ntj_stats',
    'llfrr',
    'gtv',
    'cqjs_binned',
    # Unified roughness (data/model comparison)
    'compute_knn_roughness',
    'compute_roughness_amplification',
    # Model-level
    'compute_knn_jump',
    'compute_path_metrics',
    'compute_violation_probability',
    'compute_local_lipschitz_quantiles',
    'compute_model_smoothness_profile',
    # Model complexity
    'extract_model_info',
]
