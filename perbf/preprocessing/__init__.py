"""Preprocessing utilities for the benchmark.

Two-level preprocessing:
1. Dataset-level: Run ONCE before CV in batch_runner.preprocess_dataset() (NA, subsample, prefilter, dim reduction)
2. Fold-level: FoldPreprocessor runs per-fold for scaling/encoding (no leakage)
"""

import numpy as np
from scipy.stats import spearmanr
from .fold_preprocessor import FoldPreprocessor, preprocess_fold


def drop_high_na(X, tol=0.5):
    """
    Drop features with high fraction of missing values.

    Parameters
    ----------
    X : DataFrame
        Feature matrix
    tol : float, default=0.5
        Drop features where NA fraction > tol

    Returns
    -------
    kept_columns : list of str
        Column names to keep
    dropped_columns : list of str
        Column names that were dropped
    """
    na_fracs = X.isna().mean()
    dropped = na_fracs[na_fracs > tol].index.tolist()
    kept = [col for col in X.columns if col not in set(dropped)]
    return kept, dropped


def drop_quasi_constant(X, tol=0.95):
    """
    Drop features where a single value dominates (quasi-constant).

    A feature is dropped if the most frequent value appears in more than
    `tol` fraction of samples. This catches:
    - Constant features (100% same value)
    - Near-constant features (e.g., 97% same value with tol=0.95)

    This is a cheap first-pass filter before Spearman/MI prefiltering.
    Unlike variance-based thresholds, this is scale-independent.

    Parameters
    ----------
    X : DataFrame
        Feature matrix
    tol : float, default=0.95
        Drop features where mode_fraction > tol

    Returns
    -------
    kept_columns : list of str
        Column names to keep
    dropped_columns : list of str
        Column names that were dropped
    """
    mode_fracs = {col: X[col].value_counts(normalize=True).iloc[0] for col in X.columns}
    dropped = [col for col, frac in mode_fracs.items() if frac > tol]
    kept = [col for col in X.columns if col not in set(dropped)]
    return kept, dropped


def select_k_best_mi(X, y, k=100, random_state=42):
    """
    Select k best features using Mutual Information with target.

    Handles mixed types (numeric + categorical) properly:
    - Categoricals: OrdinalEncoder + discrete_features flag for MI
    - Numerics: continuous MI estimation

    This is preferred over Spearman for feature selection because:
    1. MI captures nonlinear relationships
    2. Handles categoricals without target leakage (no TargetEncoder)
    3. Well-established in feature selection literature

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Feature matrix (must be DataFrame to detect categoricals)
    y : array-like of shape (n_samples,)
        Target values
    k : int, default=100
        Number of features to select
    random_state : int, default=42
        Random state for MI estimation

    Returns
    -------
    X_selected : DataFrame of shape (n_samples, k)
        Selected features (preserves dtypes including categoricals)
    selected_columns : list of str
        Column names of selected features
    mi_scores : array of shape (k,)
        MI scores of selected features (sorted descending)
    """
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import OrdinalEncoder

    y = np.asarray(y).ravel()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])

    columns = list(X.columns)
    n_features = len(columns)

    if k >= n_features:
        return X.copy(), columns, np.ones(n_features)

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()

    # Create encoded copy for MI computation
    X_encoded = X.copy()
    if cat_cols:
        # OrdinalEncoder for categoricals (just to make numeric for MI)
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))

    # Mark which features are discrete (categorical)
    discrete_mask = [col in cat_cols for col in columns]

    # Handle NaN: fill with median for numerics, mode for categoricals
    X_for_mi = X_encoded.copy()
    for col in columns:
        if col in cat_cols:
            X_for_mi[col] = X_for_mi[col].fillna(X_for_mi[col].mode().iloc[0] if len(X_for_mi[col].mode()) > 0 else 0)
        else:
            X_for_mi[col] = X_for_mi[col].fillna(X_for_mi[col].median())

    # Compute MI scores
    mi_scores = mutual_info_regression(
        X_for_mi.values.astype(float),
        y,
        discrete_features=discrete_mask,
        random_state=random_state
    )

    # Select top k by MI
    selected_indices = np.argsort(mi_scores)[-k:][::-1]
    selected_cols = [columns[i] for i in selected_indices]
    selected_mi = mi_scores[selected_indices]

    # Return ORIGINAL DataFrame columns (preserves dtypes)
    return X[selected_cols].copy(), selected_cols, selected_mi


def select_k_best_spearman(X, y, k=100, random_state=42):
    """
    Select k best features using Spearman correlation with target.

    NOTE: For mixed-type data, prefer select_k_best_mi() which handles
    categoricals without target leakage.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        Feature matrix (numeric only, or will use TargetEncoder for categoricals)
    y : array-like of shape (n_samples,)
        Target values
    k : int, default=100
        Number of features to select
    random_state : int, default=42
        Random state for TargetEncoder (if categoricals present)

    Returns
    -------
    X_selected : DataFrame or array of shape (n_samples, k)
        Selected features (same type as input)
    selected_columns : list of str or array of int
        Column names (if DataFrame) or indices (if array) of selected features
    correlations : array of shape (k,)
        Spearman correlations of selected features (sorted descending)
    """
    import pandas as pd

    y = np.asarray(y).ravel()
    is_dataframe = isinstance(X, pd.DataFrame)

    if is_dataframe:
        X_df = X
        columns = list(X_df.columns)
        n_features = len(columns)

        # Handle categoricals with TargetEncoder for correlation computation
        cat_cols = X_df.select_dtypes(include=['category', 'object']).columns.tolist()
        if cat_cols:
            from sklearn.preprocessing import TargetEncoder
            X_numeric = X_df.copy()
            encoder = TargetEncoder(smooth='auto', random_state=random_state)
            X_cat_encoded = encoder.fit_transform(X_df[cat_cols], y)
            for i, col in enumerate(cat_cols):
                X_numeric[col] = X_cat_encoded[:, i]
            X_for_corr = X_numeric.values.astype(float)
        else:
            X_for_corr = X_df.values.astype(float)
    else:
        X_for_corr = np.asarray(X).astype(float)
        columns = None
        n_features = X_for_corr.shape[1]

    if k >= n_features:
        if is_dataframe:
            return X_df.copy(), columns, np.ones(n_features)
        else:
            return X_for_corr, np.arange(n_features), np.ones(n_features)

    # Compute Spearman correlation for each feature
    correlations = np.zeros(n_features)
    for i in range(n_features):
        col_data = X_for_corr[:, i]
        # Handle constant features and NaN
        valid_mask = ~np.isnan(col_data)
        if valid_mask.sum() > 10 and np.std(col_data[valid_mask]) > 1e-10:
            corr, _ = spearmanr(col_data[valid_mask], y[valid_mask])
            correlations[i] = abs(corr) if not np.isnan(corr) else 0.0
        else:
            correlations[i] = 0.0

    # Select top k by absolute correlation
    selected_indices = np.argsort(correlations)[-k:][::-1]
    selected_corrs = correlations[selected_indices]

    if is_dataframe:
        selected_columns = [columns[i] for i in selected_indices]
        X_selected = X_df[selected_columns].copy()
        return X_selected, selected_columns, selected_corrs
    else:
        X_selected = X_for_corr[:, selected_indices]
        return X_selected, selected_indices, selected_corrs


def prefilter_by_spearman(X, y, threshold=0.05):
    """
    Remove features with weak Spearman correlation to target.

    A fast, conservative prefilter to remove noise-floor features.
    Features with |r| < threshold have essentially no monotonic relationship.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (numeric columns only)
    y : array-like
        Target values
    threshold : float, default=0.05
        Minimum |Spearman r| to keep

    Returns
    -------
    list of str
        Feature names with |r| >= threshold

    Note
    ----
    May remove synergistic features (those only useful in interaction).
    Use conservative threshold (0.03-0.10) to minimize this risk.
    """
    import pandas as pd
    correlations = X.corrwith(pd.Series(y), method='spearman').abs()
    return correlations[correlations >= threshold].index.tolist()


def prefilter_combined(X, y, spearman_threshold=0.05, spearman_bottom_pctl=30.0,
                       mi_top_pctl=30.0, random_state=42):
    """
    Spearman prefilter with MI-based rescue for non-monotonic features.

    Logic:
    1. Keep if |Spearman r| >= spearman_threshold (monotonic relationship)
    2. RESCUE if: Spearman in bottom Q% AND MI in top P% (non-monotonic rescue)

    The rescue catches features like x^2 that have near-zero Spearman correlation
    but high mutual information due to non-monotonic relationships.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (numeric columns only)
    y : array-like
        Target values
    spearman_threshold : float, default=0.05
        Minimum |Spearman r| to pass main filter
    spearman_bottom_pctl : float, default=30.0
        Bottom percentile for rescue consideration
    mi_top_pctl : float, default=30.0
        Top MI percentile for rescue
    random_state : int, default=42
        Random seed for MI computation

    Returns
    -------
    tuple of (list, dict)
        - feature_names: List of features to keep
        - debug_info: Dict with 'passed_spearman', 'rescued', 'dropped' lists
    """
    import pandas as pd
    from sklearn.feature_selection import mutual_info_regression

    d = X.shape[1]

    # Spearman correlations and ranks
    spearman_r = X.corrwith(pd.Series(y), method='spearman').abs()
    spearman_rank = spearman_r.rank(ascending=False)  # 1 = highest correlation
    spearman_pctl = (spearman_rank / d) * 100  # percentile (low = good)

    # Features passing main Spearman filter
    pass_spearman = set(spearman_r[spearman_r >= spearman_threshold].index)

    # Features failing Spearman but potentially rescuable
    fail_spearman = set(X.columns) - pass_spearman
    in_bottom_spearman = set(spearman_pctl[spearman_pctl > (100 - spearman_bottom_pctl)].index)
    candidates_for_rescue = fail_spearman & in_bottom_spearman

    # Compute MI only for rescue candidates (efficiency)
    # IMPORTANT: Sort candidates for deterministic MI estimation (k-NN MI is order-sensitive)
    rescued = set()
    if candidates_for_rescue:
        candidates_sorted = sorted(candidates_for_rescue)
        mi = pd.Series(
            mutual_info_regression(X[candidates_sorted], y, random_state=random_state),
            index=candidates_sorted
        )
        mi_rank = mi.rank(ascending=False)
        mi_pctl = (mi_rank / len(candidates_for_rescue)) * 100

        # Rescue if in top MI percentile among candidates
        rescued = set(mi_pctl[mi_pctl <= mi_top_pctl].index)

    keep = sorted(pass_spearman | rescued)
    dropped = sorted(set(X.columns) - set(keep))

    debug_info = {
        'passed_spearman': sorted(pass_spearman),
        'rescued': sorted(rescued),
        'dropped': dropped,
    }

    return keep, debug_info


__all__ = [
    'FoldPreprocessor',
    'preprocess_fold',
    'drop_high_na',
    'drop_quasi_constant',
    'select_k_best_mi',
    'select_k_best_spearman',
    'prefilter_by_spearman',
    'prefilter_combined',
]
