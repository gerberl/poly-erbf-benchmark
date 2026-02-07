"""Cross-validation evaluation utilities."""

from .cv import (
    get_eval_cv,
    evaluate_model,
    run_benchmark,
    summarize_results,
    DEFAULT_SCALE_MAP,
)
from .metrics import (
    adjusted_r2,
    rmse,
    mae,
    mape,
    compute_regression_metrics,
)

__all__ = [
    'get_eval_cv',
    'evaluate_model',
    'run_benchmark',
    'summarize_results',
    'DEFAULT_SCALE_MAP',
    # Metrics
    'adjusted_r2',
    'rmse',
    'mae',
    'mape',
    'compute_regression_metrics',
]
