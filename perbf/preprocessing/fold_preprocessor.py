"""
Fold-level preprocessing for proper CV without leakage.

Uses sklearn Pipeline + ColumnTransformer for standard, well-tested preprocessing.

Handles transformations that must be fit on training data only:
- TargetEncoder for categoricals (uses y)
- Median imputation (uses train statistics)
- StandardScaler (uses train mean/std)

Usage:
    from perbf.preprocessing import FoldPreprocessor

    # Create with model-specific settings
    prep = FoldPreprocessor(scale=True)

    # Fit on training fold
    prep.fit(X_train, y_train)

    # Transform both splits
    X_train_t = prep.transform(X_train)
    X_val_t = prep.transform(X_val)

Created: 15Jan26
Updated: 23Jan26 - Replaced custom code with sklearn Pipeline + ColumnTransformer
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer


class FoldPreprocessor:
    """
    Preprocessing pipeline for CV folds using sklearn primitives.

    Encapsulates transformations that must be fit on training data only
    to prevent information leakage.

    Parameters
    ----------
    scale : bool, default=True
        Whether to apply StandardScaler
    categorical_cols : list of str, optional
        Not used (kept for API compatibility). Columns auto-detected via dtype.
    impute_strategy : str, default='median'
        Strategy for SimpleImputer ('median', 'mean', 'most_frequent')

    Attributes
    ----------
    preprocessor_ : ColumnTransformer
        Fitted sklearn pipeline
    """

    def __init__(
        self,
        scale: bool = True,
        categorical_cols: Optional[list] = None,
        impute_strategy: str = 'median'
    ):
        self.scale = scale
        self.categorical_cols = categorical_cols  # Kept for API compatibility
        self.impute_strategy = impute_strategy

        # Fitted components (set in fit())
        self.preprocessor_ = None
        self._input_columns = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> 'FoldPreprocessor':
        """
        Fit preprocessing on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training target (needed for TargetEncoder)

        Returns
        -------
        self
        """
        # Convert to DataFrame for column type detection
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        self._input_columns = list(X_df.columns)
        y_arr = np.asarray(y)

        # Build preprocessing pipeline
        self.preprocessor_ = self._make_preprocessor()

        # Fit (TargetEncoder needs y)
        self.preprocessor_.fit(X_df, y_arr)

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted preprocessors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to transform

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed features
        """
        if self.preprocessor_ is None:
            raise RuntimeError("FoldPreprocessor not fitted. Call fit() first.")

        # Convert to DataFrame with same columns
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self._input_columns)
        else:
            X_df = X.copy()

        return self.preprocessor_.transform(X_df)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _make_preprocessor(self) -> ColumnTransformer:
        """Build sklearn ColumnTransformer pipeline."""

        # Numeric pipeline: impute → scale
        if self.scale:
            numeric_transformer = Pipeline([
                ('impute', SimpleImputer(strategy=self.impute_strategy)),
                ('scale', StandardScaler()),
            ])
        else:
            numeric_transformer = SimpleImputer(strategy=self.impute_strategy)

        # Categorical pipeline: impute → encode → scale
        # Impute missing categoricals first (most_frequent = mode), then encode to numeric
        cat_steps = [
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', TargetEncoder(smooth='auto', random_state=42, target_type='continuous')),
        ]
        if self.scale:
            cat_steps.append(('scale', StandardScaler()))
        categorical_transformer = Pipeline(cat_steps)

        # ColumnTransformer: auto-detect column types
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer,
                 make_column_selector(dtype_exclude=['object', 'category'])),
                ('cat', categorical_transformer,
                 make_column_selector(dtype_include=['object', 'category'])),
            ],
            remainder='drop',
            verbose_feature_names_out=False,
        )

        return preprocessor

    def get_feature_names(self) -> list:
        """Return feature names after transformation."""
        if self.preprocessor_ is None:
            raise RuntimeError("FoldPreprocessor not fitted.")
        try:
            return self.preprocessor_.get_feature_names_out().tolist()
        except AttributeError:
            # Fallback if get_feature_names_out not available
            return [f'f{i}' for i in range(
                self.preprocessor_.transform(
                    pd.DataFrame([[0]*len(self._input_columns)], columns=self._input_columns)
                ).shape[1]
            )]


# Convenience function for simple cases
def preprocess_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    scale: bool = True
) -> tuple:
    """
    Preprocess a single CV fold.

    Convenience wrapper for FoldPreprocessor.

    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_val : array-like
        Validation features
    scale : bool
        Whether to standardize

    Returns
    -------
    X_train_t : ndarray
        Transformed training features
    X_val_t : ndarray
        Transformed validation features
    """
    prep = FoldPreprocessor(scale=scale)
    X_train_t = prep.fit_transform(X_train, y_train)
    X_val_t = prep.transform(X_val)
    return X_train_t, X_val_t
