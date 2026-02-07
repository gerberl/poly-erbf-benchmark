"""
Evaluation metrics for model performance assessment.

Functions for computing various evaluation metrics including:
- Adjusted R² (dataset-aware, for fair cross-dataset aggregation)
- Standard regression metrics (RMSE, MAE, MAPE) - using sklearn implementations

Created: 23Jan26
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def adjusted_r2(r2_score: float, n_samples: int, n_features: int) -> float:
    """
    Compute adjusted R² using dataset dimensionality.

    Adjusts for dataset dimensionality (input features), not model parameters.
    This allows fair aggregation across datasets with different dimensionalities,
    as all models on the same dataset use the same adjustment.

    Ref: Varoquaux et al. approach for cross-dataset aggregation.

    Parameters
    ----------
    r2_score : float
        Unadjusted R² score
    n_samples : int
        Number of samples in test set
    n_features : int
        Number of input features (d)

    Returns
    -------
    r2_adj : float
        Adjusted R² score, or np.nan if n_samples <= n_features + 1
    """
    if n_samples > n_features + 1:
        return 1 - (1 - r2_score) * (n_samples - 1) / (n_samples - n_features - 1)
    else:
        return np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Uses sklearn.metrics.root_mean_squared_error.
    """
    return float(root_mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Uses sklearn.metrics.mean_absolute_error.
    """
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Uses sklearn.metrics.mean_absolute_percentage_error.
    Returns as percentage (0-100), or np.nan if computation fails.

    Note: sklearn's MAPE uses epsilon=1e-10 internally and returns values
    in [0, 1] range. We multiply by 100 to match our convention.
    """
    try:
        # sklearn returns MAPE in [0, 1] range, we want [0, 100]
        mape_val = float(mean_absolute_percentage_error(y_true, y_pred)) * 100

        # Return nan if result is infinite or invalid
        if np.isinf(mape_val) or mape_val > 1e6:
            return np.nan
        return mape_val
    except (ValueError, ZeroDivisionError):
        return np.nan


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2_score: float,
    n_features: int = None,
) -> dict:
    """
    Compute comprehensive regression metrics.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    r2_score : float
        Pre-computed R² score
    n_features : int, optional
        Number of input features for adjusted R² computation

    Returns
    -------
    metrics : dict
        Dictionary with rmse, mae, mape, and optionally r2_adj
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
    }

    if n_features is not None:
        metrics['r2_adj'] = adjusted_r2(r2_score, len(y_true), n_features)

    return metrics
