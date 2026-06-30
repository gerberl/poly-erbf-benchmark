"""
Interactive Chebyshev Wrappers - PolyBasis models with tabular_fs interactions.

STATUS: EXPERIMENTAL / DEFERRED
===============================
Initial experiments (21Jan26) showed:
- No clear R2 benefit over baseline Chebyshev without interactions
- Bug: feature mismatch between fit() and predict() in 'tiered' mode
- Contrast interactions cause numerical instability
- MI-filtering is 6-44x slower than variance-based

See: notes-plans/chebyshev_interactions_deferred_21jan26.md

Benchmark-specific wrappers that compose:
- tabular_fs: MI-based feature selection and interaction generation
- poly_basis_ml: Chebyshev polynomial regression

These wrappers handle the fit/transform separation properly to avoid leakage:
- fit(): Runs tabular_fs pipeline (uses y), stores selected features/interactions
- predict(): Applies stored feature selection, generates interactions, predicts

KNOWN BUG (interactions='tiered'):
- fit() trains on X_final (all features + interactions)
- predict() uses only T1 features + interactions (via _transform_features)
- This causes shape mismatch during cross-validation

Usage (NOT RECOMMENDED - use PolyBasisRegressor directly instead):
    from perbf.models.interactive_chebyshev import (
        InteractiveChebyshevRegressor,
        InteractiveChebyshevModelTree,
    )

    model = InteractiveChebyshevRegressor(complexity=5, interactions='tiered')
    model.fit(X, y)
    y_pred = model.predict(X_test)

Created: 15Jan26
Updated: 21Jan26 - marked as experimental
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from typing import Optional, Literal


class InteractiveChebyshevRegressor(BaseEstimator, RegressorMixin):
    """
    Chebyshev polynomial regressor with automatic interaction generation.

    Wraps PolyBasisRegressor with tabular_fs interaction generation.
    Interactions are selected based on MI during fit, then applied during predict.

    Parameters
    ----------
    complexity : int, default=5
        Maximum polynomial degree for Chebyshev basis
    regressor : str, default='Ridge'
        Inner regressor ('Ridge', 'ElasticNet', 'Lasso')
    alpha : float, default=1.0
        Regularization strength
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (ignored for Ridge/Lasso)
    interactions : str or None, default=None
        Interaction generation method:
        - None: No interactions (just polynomial on original features)
        - 'tiered': Use tabular_fs tiered pipeline (T1xT1, T1xT2)
        - 'product': Generate product interactions only
        - 'contrast': Generate contrast interactions only
        - 'both': Generate both product and contrast
    t1_quantile : float, default=0.80
        Top quantile for T1 tier (used with 'tiered')
    interaction_mi_threshold : float, default=0.3
        MI threshold for keeping interactions (fraction of median T1 MI)
    clip_input : bool, default=True
        Whether to clip inputs to [-1, 1] for Chebyshev basis
    random_state : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    base_features_ : list
        Selected base feature names (T1 + T2 for tiered, all for others)
    interaction_features_ : list
        Generated interaction feature names
    feature_names_in_ : list
        Original input feature names
    poly_model_ : PolyBasisRegressor
        Fitted polynomial model
    """

    def __init__(
        self,
        complexity: int = 5,
        regressor: str = 'Ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        interactions: Optional[Literal['tiered', 'product', 'contrast', 'both']] = None,
        t1_quantile: float = 0.80,
        interaction_mi_threshold: float = 0.3,
        clip_input: bool = True,
        random_state: int = 42,
    ):
        self.complexity = complexity
        self.regressor = regressor
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.interactions = interactions
        self.t1_quantile = t1_quantile
        self.interaction_mi_threshold = interaction_mi_threshold
        self.clip_input = clip_input
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model: select features, generate interactions, fit polynomial.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        from poly_basis_ml import PolyBasisRegressor

        # Convert to DataFrame for tabular_fs
        X_df = self._to_dataframe(X)
        y_arr = np.asarray(y).ravel()

        self.feature_names_in_ = list(X_df.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Generate features based on interaction mode
        if self.interactions is None:
            # No interactions - use all features directly
            X_final = X_df
            self.base_features_ = list(X_df.columns)
            self.interaction_features_ = []
            self.interaction_specs_ = []

        elif self.interactions == 'tiered':
            # Full tabular_fs pipeline with MI-based selection
            X_final, pipeline_result = self._run_tiered_pipeline(X_df, y_arr)
            self.base_features_ = pipeline_result['base_features']
            self.interaction_features_ = pipeline_result['interaction_features']
            self.interaction_specs_ = pipeline_result['interaction_specs']
            self.tiers_ = pipeline_result['tiers']

        else:
            # Simple interaction generation (product, contrast, or both)
            X_final, specs = self._generate_simple_interactions(X_df, y_arr)
            self.base_features_ = list(X_df.columns)
            self.interaction_features_ = [s['name'] for s in specs]
            self.interaction_specs_ = specs

        # Fit polynomial model
        self.poly_model_ = PolyBasisRegressor(
            basis_name='Chebyshev',
            complexity=self.complexity,
            regressor=self.regressor,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            clip_input=self.clip_input,
        )
        self.poly_model_.fit(X_final.values, y_arr)

        return self

    def predict(self, X):
        """
        Predict using fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        check_is_fitted(self, ['poly_model_', 'base_features_'])

        X_df = self._to_dataframe(X)
        X_final = self._transform_features(X_df)

        return self.poly_model_.predict(X_final.values)

    def _to_dataframe(self, X):
        """Convert input to DataFrame with consistent column names."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            X_arr = np.asarray(X)
            if hasattr(self, 'feature_names_in_'):
                cols = self.feature_names_in_
            else:
                cols = [f'x{i}' for i in range(X_arr.shape[1])]
            return pd.DataFrame(X_arr, columns=cols)

    def _run_tiered_pipeline(self, X_df, y):
        """Run tabular_fs tiered pipeline for feature selection and interactions."""
        from tabular_fs import run_pipeline

        result = run_pipeline(
            X_df, y,
            prefilter='combined',
            t1_quantile=self.t1_quantile,
            interaction_types=['contrast', 'product'],
            interaction_mi_threshold=self.interaction_mi_threshold,
            random_state=self.random_state,
            verbose=False,
        )

        # X_final from run_pipeline is X_t1 + interactions (NOT X_t1_t2)
        # So we use X_t1 columns as base features to match
        base_features = list(result['X_t1'].columns)
        interaction_features = list(result['X_interactions'].columns)

        # Parse interaction specs from column names
        interaction_specs = []
        for col in interaction_features:
            parts = col.split('__')
            if len(parts) == 3:
                feat_a, itype, feat_b = parts
                interaction_specs.append({
                    'name': col,
                    'feat_a': feat_a,
                    'feat_b': feat_b,
                    'type': itype,
                })

        X_final = result['X_final']

        return X_final, {
            'base_features': base_features,
            'interaction_features': interaction_features,
            'interaction_specs': interaction_specs,
            'tiers': result['tiers'],
        }

    def _generate_simple_interactions(self, X_df, y):
        """Generate simple pairwise interactions without MI-based selection."""
        from tabular_fs import generate_interactions, get_tier_pairs, compute_mi

        cols = list(X_df.columns)

        # Determine interaction types
        if self.interactions == 'product':
            itypes = ['product']
        elif self.interactions == 'contrast':
            itypes = ['contrast']
        elif self.interactions == 'both':
            itypes = ['product', 'contrast']
        else:
            itypes = []

        # Generate all pairwise interactions
        pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i+1, len(cols))]

        interaction_specs = []
        interaction_dfs = []

        for itype in itypes:
            X_int = generate_interactions(X_df, pairs, interaction_type=itype)
            interaction_dfs.append(X_int)

            for col in X_int.columns:
                parts = col.split('__')
                if len(parts) == 3:
                    interaction_specs.append({
                        'name': col,
                        'feat_a': parts[0],
                        'feat_b': parts[2],
                        'type': parts[1],
                    })

        if interaction_dfs:
            X_interactions = pd.concat(interaction_dfs, axis=1)

            # Filter by MI if threshold is set
            if self.interaction_mi_threshold > 0:
                mi_int = compute_mi(X_interactions, y, random_state=self.random_state)
                mi_base = compute_mi(X_df, y, random_state=self.random_state)
                threshold = self.interaction_mi_threshold * mi_base.median()
                keep = mi_int[mi_int >= threshold].index.tolist()
                X_interactions = X_interactions[keep]
                interaction_specs = [s for s in interaction_specs if s['name'] in keep]

            X_final = pd.concat([X_df, X_interactions], axis=1)
        else:
            X_final = X_df

        return X_final, interaction_specs

    def _transform_features(self, X_df):
        """Transform features for prediction (apply stored selection + interactions)."""
        from tabular_fs import INTERACTION_TYPES

        # Select base features
        X_base = X_df[self.base_features_].copy()

        # Generate interactions from stored specs
        if self.interaction_specs_:
            interaction_cols = {}
            for spec in self.interaction_specs_:
                func = INTERACTION_TYPES[spec['type']]
                a_vals = X_df[spec['feat_a']].values
                b_vals = X_df[spec['feat_b']].values
                interaction_cols[spec['name']] = func(a_vals, b_vals, 1e-6)  # eps is positional

            X_int = pd.DataFrame(interaction_cols, index=X_df.index)
            X_final = pd.concat([X_base, X_int], axis=1)
        else:
            X_final = X_base

        return X_final

    @property
    def coef_(self):
        """Coefficients from fitted polynomial model."""
        check_is_fitted(self, 'poly_model_')
        return self.poly_model_.coef_

    @property
    def intercept_(self):
        """Intercept from fitted polynomial model."""
        check_is_fitted(self, 'poly_model_')
        return self.poly_model_.intercept_


class InteractiveChebyshevModelTree(BaseEstimator, RegressorMixin):
    """
    Chebyshev ModelTree with automatic interaction generation.

    Wraps PolyBasisModelTreeRegressor with tabular_fs interaction generation.

    Parameters
    ----------
    complexity : int, default=3
        Maximum polynomial degree for leaf Chebyshev models
    max_depth : int, default=5
        Maximum depth of the routing tree
    min_samples_leaf : int, default=20
        Minimum samples per leaf
    regressor : str, default='Ridge'
        Inner regressor for leaf models
    alpha : float, default=1.0
        Regularization strength
    interactions : str or None, default=None
        Interaction generation method (same as InteractiveChebyshevRegressor)
    t1_quantile : float, default=0.80
        Top quantile for T1 tier
    interaction_mi_threshold : float, default=0.3
        MI threshold for keeping interactions
    random_state : int, default=42
        Random seed
    """

    def __init__(
        self,
        complexity: int = 3,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        regressor: str = 'Ridge',
        alpha: float = 1.0,
        interactions: Optional[Literal['tiered', 'product', 'contrast', 'both']] = None,
        t1_quantile: float = 0.80,
        interaction_mi_threshold: float = 0.3,
        random_state: int = 42,
    ):
        self.complexity = complexity
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.regressor = regressor
        self.alpha = alpha
        self.interactions = interactions
        self.t1_quantile = t1_quantile
        self.interaction_mi_threshold = interaction_mi_threshold
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the model."""
        from poly_basis_ml import PolyBasisModelTreeRegressor

        # Convert to DataFrame
        X_df = self._to_dataframe(X)
        y_arr = np.asarray(y).ravel()

        self.feature_names_in_ = list(X_df.columns)
        self.n_features_in_ = len(self.feature_names_in_)

        # Use the same interaction logic as the regressor
        self._interaction_handler = InteractiveChebyshevRegressor(
            interactions=self.interactions,
            t1_quantile=self.t1_quantile,
            interaction_mi_threshold=self.interaction_mi_threshold,
            random_state=self.random_state,
        )

        # Fit interaction handler (to get features)
        self._interaction_handler.fit(X_df, y_arr)

        # Get transformed features
        X_final = self._interaction_handler._transform_features(X_df)

        # Store feature info
        self.base_features_ = self._interaction_handler.base_features_
        self.interaction_features_ = self._interaction_handler.interaction_features_
        self.interaction_specs_ = self._interaction_handler.interaction_specs_

        # Fit model tree
        self.tree_model_ = PolyBasisModelTreeRegressor(
            basis_name='Chebyshev',
            complexity=self.complexity,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            regressor=self.regressor,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        self.tree_model_.fit(X_final.values, y_arr)

        return self

    def predict(self, X):
        """Predict using fitted model."""
        check_is_fitted(self, ['tree_model_', '_interaction_handler'])

        X_df = self._to_dataframe(X)
        X_final = self._interaction_handler._transform_features(X_df)

        return self.tree_model_.predict(X_final.values)

    def _to_dataframe(self, X):
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            X_arr = np.asarray(X)
            if hasattr(self, 'feature_names_in_'):
                cols = self.feature_names_in_
            else:
                cols = [f'x{i}' for i in range(X_arr.shape[1])]
            return pd.DataFrame(X_arr, columns=cols)

    @property
    def n_leaves_(self):
        """Number of leaves in the tree."""
        check_is_fitted(self, 'tree_model_')
        return self.tree_model_.n_leaves_

    @property
    def leaf_models_(self):
        """Leaf polynomial models."""
        check_is_fitted(self, 'tree_model_')
        return self.tree_model_.leaf_models_
