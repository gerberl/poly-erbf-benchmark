"""
Tuning Module - Optuna-based hyperparameter tuning for benchmark models.

Adapted from cpet project patterns:
- Model factory pattern for clean Optuna integration
- Fold-aware pruning for efficient search

Two-level preprocessing:
- Dataset-level: In batch_runner.preprocess_dataset() before CV (NA, subsample, prefilter, dim reduction)
- Fold-level: FoldPreprocessor per-fold (scaling, encoding to prevent leakage)

Usage:
    from benchmark.tuning.optuna_cv import tune_model, get_best_model, MODEL_FACTORIES
    from benchmark.tuning.optuna_cv import nested_cv_tune_and_evaluate

    # Tune on a dataset
    study = tune_model('rf', X, y, n_trials=50)
    best_model = get_best_model('rf', study)

    # Or full nested CV with preprocessing
    results = nested_cv_tune_and_evaluate('rf', X, y, n_jobs=-2)

Created: 14Jan26
Updated: 23Jan26 - Removed DatasetPreprocessor (now in batch_runner.preprocess_dataset)
"""

import warnings
from typing import Callable, Dict, Any, Optional, List

# Suppress noisy warnings (applied in worker processes too)
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.delayed.*')
warnings.filterwarnings('ignore', message='.*disp.*iprint.*L-BFGS-B.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*omp_set_nested.*deprecated.*')

# Suppress RDKit warnings
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from benchmark.preprocessing import FoldPreprocessor
from benchmark.evaluation.metrics import adjusted_r2, rmse, mae, mape
from benchmark.analysis.model_complexity import extract_model_info

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def ridge_factory(trial: optuna.Trial) -> Ridge:
    """Create Ridge regressor with Optuna hyperparameters."""
    alpha = trial.suggest_float('alpha', 1e-3, 1e3, log=True)
    return Ridge(alpha=alpha)


def dt_factory(trial: optuna.Trial) -> DecisionTreeRegressor:
    """Create Decision Tree with Optuna hyperparameters.

    Searches over:
    - max_depth: 1 (stump) to 20 (deep tree)
    - min_samples_leaf: 0.5%-10% of samples
    """
    return DecisionTreeRegressor(
        max_depth=trial.suggest_int('max_depth', 1, 20),  # Start from 1 (stump)
        min_samples_leaf=trial.suggest_float('min_samples_leaf', 0.005, 0.1),  # 0.5%-10% of samples
        min_samples_split=trial.suggest_float('min_samples_split', 0.01, 0.1),  # 1%-10%
        random_state=42
    )


def rf_factory(trial: optuna.Trial) -> RandomForestRegressor:
    """Create Random Forest with Optuna hyperparameters.

    Searches over (3 params - simplified 20Jan26, ranges expanded 21Jan26):
    - n_estimators: 50-500 (expanded from 300 for complex datasets)
    - max_depth: 3-20 (expanded from 15 for complex datasets)
    - max_features: 0.3, 0.5, 0.7, or sqrt (expanded for more flexibility)

    Fixed (min_samples_leaf almost always at minimum in tuning runs):
    - min_samples_leaf=0.005: chosen ~100% of time, no need to tune
    """
    return RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 500),
        max_depth=trial.suggest_int('max_depth', 3, 20),
        min_samples_leaf=0.005,  # Fixed: almost always chosen in benchmark
        max_features=trial.suggest_categorical('max_features', [0.3, 0.5, 0.7, 'sqrt']),
        random_state=42,
        n_jobs=1  # Single-threaded; parallelism at outer fold level
    )


def xgb_factory(trial: optuna.Trial):
    """Create XGBoost with Optuna hyperparameters.

    Searches over (6 params):
    - max_depth: 1 (stumps) to 9
    - learning_rate: 0.01-0.3
    - subsample/colsample: 0.6-1.0
    - reg_lambda (L2): 1e-3 to 1e3
    - min_child_weight: 1-100 (log scale) - reduces overfitting

    Note: reg_alpha (L1) removed - redundant with reg_lambda.
    Note: gamma removed - scale-dependent, can break small-variance targets.
    Updated 21Jan26: added min_child_weight, early_stopping=10 (was 50).

    Uses early stopping with 2000 max estimators.
    """
    from xgboost import XGBRegressor
    from xgboost.callback import EarlyStopping
    return XGBRegressor(
        n_estimators=2000,  # High, rely on early stopping
        max_depth=trial.suggest_int('max_depth', 1, 9),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 1e3, log=True),
        min_child_weight=trial.suggest_float('min_child_weight', 1, 100, log=True),
        random_state=42,
        n_jobs=1,
        callbacks=[EarlyStopping(rounds=10)]
    )


# TabPFN: Always use local GPU mode (assumes tabpfn package installed)


# =============================================================================
# MODEL FACTORIES
# =============================================================================


def tabpfn_factory(trial: optuna.Trial):
    """Create TabPFN - no hyperparameters to tune.

    Uses local GPU mode (assumes tabpfn package installed).
    Dataset subsampling is handled upstream in nested_cv_tune_and_evaluate.
    """
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor()


def tabpfnv2_factory(trial: optuna.Trial):
    """Create TabPFN with v2 weights (Apache 2.0 licence).

    Uses ModelVersion.V2 for the peer-reviewed model described by
    Hollmann et al. (2025).  10K sample hard limit (vs 50K for v2.5).
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2)


def erbf_factory(trial: optuna.Trial):
    """Create ERBF regressor with Optuna hyperparameters.

    Single-layer ellipsoidal RBF network with gradient-optimized widths.

    Searches over:
    - n_rbf: 'auto' or 10-80 (matches auto clip range [10, 80])
    - alpha: 1e-3 to 1e3 regularization strength
    - center_init: 'lipschitz' or 'kmeans' (dataset-dependent optimal)
    - width_init: 'local_ridge' or 'local_variance' (dataset-dependent optimal)

    Fixed:
    - width_optim_iters=30: 99.3% of R² at 2.1x faster than 100
    - width_mode='full': K*d params (per-RBF per-feature widths)

    Note: center_init/width_init restored to tunable after 20Jan26 benchmark
    showed significant regressions on some datasets (e.g., power_grid_stability
    -0.10 R² with fixed lipschitz+local_ridge vs optimal kmeans+local_variance).
    """
    from erbf import ERBFRegressor

    # n_rbf: 'auto' or integer 10-80 (matches auto clip range)
    use_auto_rbf = trial.suggest_categorical('n_rbf_auto', [True, False])
    n_rbf = 'auto' if use_auto_rbf else trial.suggest_int('n_rbf', 10, 80)

    # center_init and width_init: restored to tunable (20Jan26)
    center_init = trial.suggest_categorical('center_init', ['lipschitz', 'kmeans'])
    width_init = trial.suggest_categorical('width_init', ['local_ridge', 'local_variance'])

    return ERBFRegressor(
        n_rbf=n_rbf,
        alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True),
        center_init=center_init,
        width_init=width_init,
        width_optim_iters=30,  # Fixed: 99.3% of R² at 2.1x faster
        width_mode='full',
        width_optim='gradient',
        standardize=True,
        random_state=42
    )



# erbf_boosted_factory: removed 02Feb26 -- BoostedERBFRegressor not in clean
# erbf 0.1.0 API.  See notes-plans/erbfb_removal_rationale_21jan26.md

def chebyshev_poly_factory(trial: optuna.Trial):
    """Create Chebyshev polynomial regressor with Optuna hyperparameters.

    Searches over:
    - complexity: 1 (linear) to 14 (high-order polynomial)
    - alpha: Ridge regularization strength
    - include_interactions: let Optuna discover if interactions help
    - max_interaction_complexity: 1 (raw product) or 2 (with T_2 expansion)

    Fixed:
    - regressor='Ridge': ElasticNet removed (20Jan26) - coordinate descent
      extremely slow at low alpha + low l1_ratio (9.6s vs 0.01s per fit)
    - interaction_types=['product']: other types tested 25Jan26, minimal benefit

    On small datasets, Optuna naturally finds simpler configs (low complexity,
    no interactions) because they have better CV scores.

    Updated 25Jan26: Tested extended interaction types (product, contrast, addition)
    and expansion. Median improvement only +0.003 R². Reverted to minimal search
    space with just max_interaction_complexity=[1,2] tweak.
    See notes-plans/chebypoly_interaction_exploration_25jan26.md
    """
    from poly_basis_ml import ChebyshevRegressor

    complexity = trial.suggest_int('complexity', 1, 14)
    alpha = trial.suggest_float('alpha', 1e-3, 1e3, log=True)
    include_interactions = trial.suggest_categorical('include_interactions', [False, True])

    if include_interactions:
        # Minimal expansion test: 1 = raw product, 2 = with T_2 expansion
        max_interaction_complexity = trial.suggest_categorical('max_interaction_complexity', [1, 2])
        expand_interactions = (max_interaction_complexity > 1)
    else:
        max_interaction_complexity = 1
        expand_interactions = False

    return ChebyshevRegressor(
        complexity=complexity,
        alpha=alpha,
        clip_input=True,
        include_interactions=include_interactions,
        interaction_types=['product'] if include_interactions else None,
        expand_interactions=expand_interactions,
        max_interaction_complexity=max_interaction_complexity,
    )


# chebyshev_td_factory: removed 02Feb26 -- total_degree mode not yet in
# clean poly-basis-ml 0.1.0 API.  Was experimental, not in main benchmark.


def chebyshev_modeltree_factory(trial: optuna.Trial):
    """Create Chebyshev ModelTree regressor with Optuna hyperparameters.

    Searches over:
    - complexity: 1 (linear) to 6 (moderate polynomial per leaf)
    - max_depth: 1 (single split) to 12 (deep partitioning)
    - min_samples_leaf: minimum samples per leaf (as fraction)
    - alpha: Ridge regularization strength

    Fixed:
    - regressor='Ridge': ElasticNet removed (20Jan26) - 11x slower at low alpha

    Note: ModelTree doesn't support interactions (each leaf fits on subset),
    so we keep complexity lower (1-6) to avoid overfitting small leaves.

    Extreme cases naturally discovered by Optuna:
    - max_depth=1, complexity=1: Two linear models (simplest)
    - max_depth=1, complexity=6: Two polynomial surfaces
    - max_depth=12, complexity=1: Many linear pieces (piecewise linear)

    Updated 22Jan26: Extended max_depth to 12 and min_samples_leaf floor to 0.01
    based on experiments showing large datasets benefit from deeper trees with
    smaller leaves. See expl/test_chebytree_high_depth_22jan26.py.
    """
    from poly_basis_ml import ChebyshevModelTreeRegressor

    return ChebyshevModelTreeRegressor(
        complexity=trial.suggest_int('complexity', 1, 6),  # 1=linear leaves
        alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True),
        max_depth=trial.suggest_int('max_depth', 1, 12),  # Extended from 8 (22Jan26)
        min_samples_leaf=trial.suggest_float('min_samples_leaf', 0.01, 0.1),  # Extended from 0.02 (22Jan26)
        random_state=42,
    )



# Registry of model factories
# Note: erbfb removed 21Jan26 - see notes-plans/erbfb_removal_rationale_21jan26.md
MODEL_FACTORIES: Dict[str, Callable] = {
    'ridge': ridge_factory,
    'dt': dt_factory,
    'rf': rf_factory,
    'xgb': xgb_factory,
    'tabpfn': tabpfn_factory,
    'tabpfnv2': tabpfnv2_factory,
    'erbf': erbf_factory,
    'chebypoly': chebyshev_poly_factory,
    # 'cheby_td': chebyshev_td_factory,  # 30Jan26: experimental, not in main benchmark
    'chebytree': chebyshev_modeltree_factory,
}

# Models that need early stopping (special handling)
EARLY_STOPPING_MODELS = {'xgb'}

# Models that don't need tuning
NO_TUNING_MODELS = {'tabpfn', 'tabpfnv2'}

# Proportional trial counts based on search space complexity
# Formula: n_trials = max(20, 10 * n_continuous + 5 * n_categorical)
# See notes-plans/search_space_simplification_17jan26.md for rationale
# Updated 20Jan26: reduced RF and ERBFB based on benchmark analysis
# Updated 25Jan26: increased chebypoly for extended interaction search space
MODEL_TRIAL_COUNTS = {
    'ridge': 20,      # 1 param
    'dt': 25,         # 3 params
    'rf': 25,         # 3 params (simplified 20Jan26: n_estimators, max_depth, max_features)
    'xgb': 50,        # 6 params (added min_child_weight 21Jan26)
    'erbf': 30,       # 4 params: n_rbf, alpha, center_init, width_init (restored 20Jan26)
    'chebypoly': 30,  # 4 params: complexity, alpha, include_interactions, max_interaction_complexity (25Jan26)
    # 'cheby_td': 20,   # 2 params: complexity (1-4), alpha
    'chebytree': 30,  # 4 params: complexity, max_depth, min_samples_leaf, alpha (ElasticNet removed 20Jan26)
    'tabpfn': 0,      # No tuning needed
    'tabpfnv2': 0,    # No tuning needed (v2 weights)
}

# Default scaling by model type
# False for models with internal scaling (avoid double-scaling)
# - erbf: StandardScaler internally (standardize=True)
# - chebypoly/chebytree: MinMaxScaler to basis interval
DEFAULT_SCALE_MAP = {
    'ridge': True,
    'dt': False,
    'rf': False,
    'xgb': False,
    'tabpfn': True,
    'tabpfnv2': True,
    'erbf': False,  # Internal StandardScaler
    'chebypoly': False,  # Internal MinMaxScaler
    # 'cheby_td': False,   # Internal MinMaxScaler
    'chebytree': False,  # Internal MinMaxScaler
}


# =============================================================================
# TUNING UTILITIES
# =============================================================================
# Note: extract_model_info() moved to benchmark/analysis/model_complexity.py (23Jan26)

def create_regression_objective(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    n_splits: int = 5,
    scale: bool = True,
    random_state: int = 42,
    early_stopping: bool = False
) -> Callable:
    """
    Create Optuna objective for regression with fold-aware pruning.

    Uses FoldPreprocessor to properly fit TargetEncoder + Imputer + Scaler
    on each inner fold's training data only (no leakage).

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target
    model_factory : callable
        Function that takes Optuna trial and returns model
    n_splits : int
        Number of CV folds for tuning
    scale : bool
        Whether to standardize features
    random_state : int
        Random seed
    early_stopping : bool
        Whether model supports early stopping (XGBoost)

    Returns
    -------
    objective : callable
        Optuna objective function
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_arr = np.asarray(y)
    is_dataframe = hasattr(X, 'iloc')
    # For cv.split, need array-like
    X_for_split = X.values if is_dataframe else np.asarray(X)

    def objective(trial: optuna.Trial) -> float:
        model = model_factory(trial)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_for_split)):
            # Handle both DataFrame and ndarray slicing
            if is_dataframe:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X_for_split[train_idx], X_for_split[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            # Preprocess: fit on train, transform both (no leakage)
            prep = FoldPreprocessor(scale=scale)
            X_train_t = prep.fit_transform(X_train, y_train)
            X_val_t = prep.transform(X_val)

            # Fit model
            if early_stopping and hasattr(model, 'fit'):
                # XGBoost with early stopping (callback set in factory)
                model.fit(
                    X_train_t, y_train,
                    eval_set=[(X_val_t, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train_t, y_train)

            # Score (R²)
            score = model.score(X_val_t, y_val)
            scores.append(score)

            # Report intermediate result for pruning
            trial.report(np.mean(scores), fold)

            # Prune if unpromising
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def tune_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    scale: Optional[bool] = None,
    random_state: int = 42,
    timeout: Optional[int] = None,
    show_progress: bool = True
) -> optuna.Study:
    """
    Tune model hyperparameters using Optuna.

    Parameters
    ----------
    model_name : str
        Name of model (key in MODEL_FACTORIES)
    X : array-like
        Features
    y : array-like
        Target
    n_trials : int
        Number of Optuna trials
    n_splits : int
        Number of CV folds
    scale : bool, optional
        Whether to standardize features (default: model-specific)
    random_state : int
        Random seed
    timeout : int, optional
        Timeout in seconds
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    study : optuna.Study
        Completed Optuna study (None if model doesn't need tuning)
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FACTORIES.keys())}")

    if model_name in NO_TUNING_MODELS:
        print(f"  {model_name} has no hyperparameters to tune")
        return None

    if scale is None:
        scale = DEFAULT_SCALE_MAP.get(model_name, True)

    model_factory = MODEL_FACTORIES[model_name]
    early_stopping = model_name in EARLY_STOPPING_MODELS

    objective = create_regression_objective(
        X, y, model_factory,
        n_splits=n_splits,
        scale=scale,
        random_state=random_state,
        early_stopping=early_stopping
    )

    # Create study with TPE sampler and median pruner
    study = optuna.create_study(
        direction='maximize',  # Maximize R²
        sampler=TPESampler(seed=random_state, multivariate=True, warn_independent_sampling=False),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress
    )

    return study


def get_best_model(model_name: str, study: Optional[optuna.Study]):
    """
    Create model with best hyperparameters from study.

    Parameters
    ----------
    model_name : str
        Name of model
    study : optuna.Study or None
        Completed study (None for models without tuning)

    Returns
    -------
    model : estimator
        Model with best hyperparameters
    """
    if study is None:
        # No tuning needed (e.g., TabPFN)
        if model_name == 'tabpfn':
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor()
        raise ValueError(f"No study provided for {model_name}")

    # Create a mock trial with best params
    class MockTrial:
        def __init__(self, params):
            self.params = params

        def suggest_float(self, name, *args, **kwargs):
            return self.params[name]

        def suggest_int(self, name, *args, **kwargs):
            return self.params[name]

        def suggest_categorical(self, name, *args, **kwargs):
            return self.params[name]

    mock_trial = MockTrial(study.best_params)
    return MODEL_FACTORIES[model_name](mock_trial)


def get_default_model(model_name: str, params: Optional[Dict[str, Any]] = None):
    """
    Get model with default or custom hyperparameters.

    Parameters
    ----------
    model_name : str
        Name of model
    params : dict, optional
        Custom hyperparameters to override defaults (from benchmark.defaults)

    Returns
    -------
    model : estimator
        Model with specified hyperparameters
    """
    params = params or {}

    if model_name == 'ridge':
        return Ridge(alpha=params.get('alpha', 1.0))

    if model_name == 'dt':
        return DecisionTreeRegressor(
            max_depth=params.get('max_depth', 10),
            min_samples_leaf=params.get('min_samples_leaf', 0.01),
            min_samples_split=params.get('min_samples_split', 0.01),
            random_state=42
        )

    if model_name == 'rf':
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 10),
            max_features=params.get('max_features', 0.5),
            min_samples_leaf=params.get('min_samples_leaf', 0.01),
            random_state=42,
            n_jobs=1
        )

    if model_name == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.9),
            colsample_bytree=params.get('colsample_bytree', 0.9),
            reg_lambda=params.get('reg_lambda', 1.0),
            min_child_weight=params.get('min_child_weight', 10),  # Conservative default (updated 21Jan26)
            random_state=42
        )

    if model_name == 'tabpfn':
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor()

    if model_name == 'tabpfnv2':
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(ModelVersion.V2)

    if model_name == 'erbf':
        from erbf import ERBFRegressor
        return ERBFRegressor(
            n_rbf=params.get('n_rbf', 'auto'),
            center_init=params.get('center_init', 'lipschitz'),
            width_init=params.get('width_init', 'local_ridge'),
            alpha=params.get('alpha', 0.2),
            standardize=True,
            random_state=42
        )

    if model_name == 'chebypoly':
        from poly_basis_ml import ChebyshevRegressor

        include_interactions = params.get('include_interactions', True)
        max_interaction_complexity = params.get('max_interaction_complexity', 1)
        expand_interactions = (max_interaction_complexity > 1)

        return ChebyshevRegressor(
            complexity=params.get('complexity', 7),
            alpha=params.get('alpha', 0.5),
            clip_input=True,
            include_interactions=include_interactions,
            interaction_types=['product'] if include_interactions else None,
            expand_interactions=expand_interactions,
            max_interaction_complexity=max_interaction_complexity,
        )

    if model_name == 'chebytree':
        from poly_basis_ml import ChebyshevModelTreeRegressor
        return ChebyshevModelTreeRegressor(
            complexity=params.get('complexity', 2),
            alpha=params.get('alpha', 0.1),
            max_depth=params.get('max_depth', 4),
            min_samples_leaf=params.get('min_samples_leaf', 0.05),
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# NESTED CV WITH PRUNING (PARALLEL)
# =============================================================================

def _run_single_outer_fold(
    fold_idx: int,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    X,  # DataFrame or ndarray
    y_arr: np.ndarray,
    model_name: str,
    inner_splits: int,
    n_trials: int,
    scale: bool,
    random_state: int,
    timeout_per_fold: Optional[int],
    no_tune: bool = False,
    default_params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Run a single outer fold (for parallel execution).

    Preprocessing is done properly:
    - Outer fold: fit FoldPreprocessor on trainval, transform test
    - Inner folds: each inner fold fits its own FoldPreprocessor (handled by create_regression_objective)
    - Final refit: uses outer fold's preprocessor

    X can be DataFrame (preserves categoricals) or ndarray.
    """
    import time

    t0 = time.time()

    # Handle both DataFrame and ndarray slicing
    if hasattr(X, 'iloc'):
        X_trainval = X.iloc[trainval_idx]
        X_test = X.iloc[test_idx]
    else:
        X_trainval = X[trainval_idx]
        X_test = X[test_idx]
    y_trainval, y_test = y_arr[trainval_idx], y_arr[test_idx]

    # Outer fold preprocessing: fit on trainval only
    outer_prep = FoldPreprocessor(scale=scale)
    X_trainval_t = outer_prep.fit_transform(X_trainval, y_trainval)
    X_test_t = outer_prep.transform(X_test)

    # Inner loop: tune on trainval (inner folds handle their own preprocessing)
    if model_name in NO_TUNING_MODELS:
        best_params = {}
        model = get_default_model(model_name)
        n_pruned = 0
    elif no_tune:
        # Skip tuning, use provided defaults
        best_params = default_params or {}
        model = get_default_model(model_name, best_params)
        n_pruned = 0
    else:
        model_factory = MODEL_FACTORIES[model_name]
        early_stopping = model_name in EARLY_STOPPING_MODELS

        # Inner CV uses raw trainval - each inner fold fits its own preprocessor
        objective = create_regression_objective(
            X_trainval, y_trainval, model_factory,
            n_splits=inner_splits,
            scale=scale,  # Inner folds do their own preprocessing
            random_state=random_state + fold_idx,
            early_stopping=early_stopping
        )

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=random_state + fold_idx, multivariate=True, warn_independent_sampling=False),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_per_fold,
            show_progress_bar=False
        )

        best_params = study.best_params
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        # Create model with best params
        model = get_best_model(model_name, study)

    # Tuning time (includes preprocessing overhead, but that's minimal)
    tune_time = time.time() - t0

    # Refit on full trainval (using outer fold's preprocessed data), evaluate on test
    # Separate timing for train and predict
    t_train_start = time.time()

    # XGBoost: use early stopping on held-out portion to prevent overfitting
    if model_name in EARLY_STOPPING_MODELS:
        from sklearn.model_selection import train_test_split
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_trainval_t, y_trainval, test_size=0.15, random_state=random_state
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False
        )
    else:
        model.fit(X_trainval_t, y_trainval)

    train_time = time.time() - t_train_start

    t_pred_start = time.time()
    y_trainval_pred = model.predict(X_trainval_t)
    y_test_pred = model.predict(X_test_t)
    predict_time = time.time() - t_pred_start

    # Prediction clipping: cap extreme predictions to prevent catastrophic R² scores
    # Use training target range + 3 standard deviations as bounds
    y_std = np.std(y_trainval)
    y_min_clip = y_trainval.min() - 3 * y_std
    y_max_clip = y_trainval.max() + 3 * y_std
    y_trainval_pred = np.clip(y_trainval_pred, y_min_clip, y_max_clip)
    y_test_pred = np.clip(y_test_pred, y_min_clip, y_max_clip)

    # Handle partial predictions (e.g., TabPFN subsampling large test sets)
    # Filter out NaN predictions and compute metrics on valid samples only
    test_valid_mask = ~np.isnan(y_test_pred)
    n_test_valid = np.sum(test_valid_mask)
    if n_test_valid < len(y_test):
        print(f"  [Metrics] Using {n_test_valid}/{len(y_test)} valid predictions", flush=True)
        y_test = y_test[test_valid_mask]
        y_test_pred = y_test_pred[test_valid_mask]

    # Core metrics: R², RMSE, MAE, MAPE
    train_r2 = 1 - np.sum((y_trainval - y_trainval_pred)**2) / np.sum((y_trainval - np.mean(y_trainval))**2)
    test_r2 = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - np.mean(y_test))**2)

    # Use centralized metric functions
    test_rmse = rmse(y_test, y_test_pred)
    test_mae = mae(y_test, y_test_pred)
    test_mape = mape(y_test, y_test_pred)
    train_rmse = rmse(y_trainval, y_trainval_pred)
    train_mae = mae(y_trainval, y_trainval_pred)

    # Extract model complexity info (from benchmark.analysis.model_complexity)
    model_info = extract_model_info(model, model_name)

    # Adjusted R² using dataset dimensionality (from benchmark.evaluation.metrics)
    # NOTE: Use training set size (n_trainval), not test set size, as this determines
    # the effective degrees of freedom used to fit the model
    d = X_trainval_t.shape[1]  # number of input features
    test_r2_adj = adjusted_r2(test_r2, n_samples=len(y_trainval), n_features=d)
    train_r2_adj = adjusted_r2(train_r2, n_samples=len(y_trainval), n_features=d)

    elapsed = time.time() - t0

    result = {
        'fold': fold_idx,
        'best_params': best_params,
        'model_info': model_info,
        # R² metrics
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_r2_adj': test_r2_adj,
        'train_r2_adj': train_r2_adj,
        'gap': train_r2 - test_r2,
        # RMSE/MAE/MAPE
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        # Timing
        'tune_time': tune_time,
        'train_time': train_time,
        'predict_time': predict_time,
        # Tuning info
        'n_pruned': n_pruned,
        'time': elapsed,  # Total fold time (tune + train + predict + metrics)
    }

    # Optionally store fitted model (skip for large models like TabPFN)
    if save_model:
        result['model'] = model

    return result


def nested_cv_tune_and_evaluate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 4,
    outer_repeats: int = 5,
    inner_splits: Optional[int] = None,
    n_trials: int = 30,
    scale: Optional[bool] = None,
    random_state: int = 42,
    timeout_per_fold: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = True,
    save_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    no_tune: bool = False,
    default_params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Full nested CV: tune hyperparameters per outer fold, evaluate on held-out test.

    This is the "gold standard" approach that gives unbiased evaluation without
    requiring separate tuning/evaluation datasets.

    Preprocessing:
    - Dataset-level preprocessing (NA, subsample, prefilter, dim reduction) should be
      done BEFORE calling this function (see batch_runner.preprocess_dataset)
    - Fold-level preprocessing (scaling, encoding) is handled internally via FoldPreprocessor

    Parameters
    ----------
    model_name : str
        Name of model (key in MODEL_FACTORIES)
    X : array-like
        Features
    y : array-like
        Target
    outer_splits : int
        Number of outer CV splits (default: 4)
    outer_repeats : int
        Number of outer CV repeats (default: 5, giving 20 total folds)
    inner_splits : int, optional
        Number of inner CV splits for tuning. Default None = adaptive:
        3-fold for n >= 1000, 5-fold for smaller datasets.
        Rationale: inner CV is for hyperparameter ranking (not estimation),
        so 3-fold suffices for larger datasets. See Cawley & Talbot (2010).
    n_trials : int
        Number of Optuna trials per outer fold (default: 30, reduced for efficiency)
    scale : bool, optional
        Whether to standardize features (default: model-specific)
    random_state : int
        Random seed
    timeout_per_fold : int, optional
        Timeout in seconds per outer fold tuning
    n_jobs : int
        Number of parallel jobs for outer folds (-1 for all cores, default: 1)
    verbose : bool
        Whether to print progress
    save_path : str, optional
        Path to save results (joblib pickle). If None, results not saved.
    dataset_name : str, optional
        Name of dataset (for metadata in saved results)
    no_tune : bool
        Skip hyperparameter tuning, use default_params instead (fast mode)
    default_params : dict, optional
        Parameters to use when no_tune=True
    save_model : bool
        Whether to store fitted models in fold_results. Set False for large
        models like TabPFN to reduce joblib file size. Default True.

    Returns
    -------
    results : dict
        Dictionary with:
        - r2_val: Mean test R² across outer folds
        - r2_val_std: Std of test R²
        - r2_train: Mean train R² (on full train+val after tuning)
        - gap: Mean train-test gap
        - fold_results: List of per-fold results with best_params, model_info, model
        - n_pruned: Total number of pruned trials
    """
    import time
    import pandas as pd
    from sklearn.model_selection import RepeatedKFold

    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FACTORIES.keys())}")

    if scale is None:
        scale = DEFAULT_SCALE_MAP.get(model_name, True)

    y_arr = np.asarray(y)
    n_samples_original = len(y_arr)  # Track original size before any subsampling
    n_samples = n_samples_original
    n_features = X.shape[1]

    # === Sample/feature limits removed (22Jan26) ===
    # Model-specific limits (TabPFN 50K, ERBF n×d budget) are now handled upstream
    # via --max-samples and --max-features in run_benchmark.py.
    # This ensures consistent preprocessing across all models in a benchmark run.
    subsampled_from = None  # No longer subsampling here

    # === Adaptive inner CV splits ===
    # Inner CV is for hyperparameter ranking, not final estimation.
    # 3-fold suffices for n >= 1000; use 5-fold for smaller datasets
    # to reduce variance when train sets are small.
    # Ref: Cawley & Talbot (2010), Auto-sklearn defaults
    if inner_splits is None:
        inner_splits = 3 if n_samples >= 1000 else 5

    # === Dataset-level preprocessing handled BEFORE this function ===
    # All NA cleaning, subsampling, prefiltering, and dim reduction done in batch_runner.preprocess_dataset()
    # This function only handles fold-level preprocessing (scaling, encoding) via FoldPreprocessor
    X_processed = X  # Keep as DataFrame if it was one

    outer_cv = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=random_state)
    n_outer_folds = outer_splits * outer_repeats

    if verbose:
        print(f"\n{'='*60}")
        print(f"Nested CV: {model_name}")
        print(f"Outer: {outer_repeats}x{outer_splits} = {n_outer_folds} folds")
        print(f"Inner: {inner_splits}-fold CV, {n_trials} Optuna trials")
        print(f"Parallel jobs: {n_jobs}")
        print(f"{'='*60}")

    # Collect fold indices (works with both DataFrame and ndarray)
    X_for_split = X_processed.values if hasattr(X_processed, 'values') else X_processed
    fold_data = [
        (fold_idx, trainval_idx, test_idx)
        for fold_idx, (trainval_idx, test_idx) in enumerate(outer_cv.split(X_for_split))
    ]

    t0_total = time.time()

    if n_jobs == 1:
        # Sequential execution with progress
        fold_results = []
        for fold_idx, trainval_idx, test_idx in fold_data:
            if verbose:
                print(f"  Fold {fold_idx+1}/{n_outer_folds}...", end=' ', flush=True)

            result = _run_single_outer_fold(
                fold_idx, trainval_idx, test_idx,
                X_processed, y_arr, model_name,
                inner_splits, n_trials, scale,
                random_state, timeout_per_fold,
                no_tune=no_tune, default_params=default_params,
                save_model=save_model
            )
            fold_results.append(result)

            if verbose:
                print(f"R²={result['test_r2']:.4f}, gap={result['gap']:.4f}, "
                      f"pruned={result['n_pruned']}/{n_trials}, {result['time']:.1f}s")
    else:
        # Parallel execution with sklearn's joblib wrapper
        from sklearn.utils.parallel import Parallel, delayed

        if verbose:
            print(f"  Running {n_outer_folds} folds in parallel...", flush=True)

        fold_results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(_run_single_outer_fold)(
                fold_idx, trainval_idx, test_idx,
                X_processed, y_arr, model_name,
                inner_splits, n_trials, scale,
                random_state, timeout_per_fold,
                no_tune, default_params, save_model
            )
            for fold_idx, trainval_idx, test_idx in fold_data
        )

    elapsed_total = time.time() - t0_total

    # Sort by fold index (parallel may return out of order)
    fold_results = sorted(fold_results, key=lambda x: x['fold'])

    # Aggregate results
    test_scores = [r['test_r2'] for r in fold_results]
    train_scores = [r['train_r2'] for r in fold_results]
    total_pruned = sum(r['n_pruned'] for r in fold_results)

    # Robust aggregation: median and trimmed mean for outlier-resistant metrics
    from scipy.stats import trim_mean
    test_scores_arr = np.array(test_scores)
    r2_val_median = float(np.median(test_scores_arr))
    # Trimmed mean: exclude top/bottom 10% (20% total trimmed)
    r2_val_trimmed = float(trim_mean(test_scores_arr, proportiontocut=0.1))

    # Failed fold tracking: count folds with R² < -1 (catastrophic failure)
    n_failed_folds = int(np.sum(test_scores_arr < -1))

    # Aggregate additional metrics
    test_r2_adj = [r['test_r2_adj'] for r in fold_results]
    train_r2_adj = [r['train_r2_adj'] for r in fold_results]
    test_rmse = [r['test_rmse'] for r in fold_results]
    test_mae = [r['test_mae'] for r in fold_results]
    test_mape = [r['test_mape'] for r in fold_results]
    tune_times = [r['tune_time'] for r in fold_results]
    train_times = [r['train_time'] for r in fold_results]
    predict_times = [r['predict_time'] for r in fold_results]

    results = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        # R² metrics (mean + robust alternatives)
        'r2_val': np.mean(test_scores),
        'r2_val_std': np.std(test_scores),
        'r2_val_median': r2_val_median,
        'r2_val_trimmed': r2_val_trimmed,
        'r2_train': np.mean(train_scores),
        'r2_val_adj': np.nanmean(test_r2_adj),
        'r2_train_adj': np.nanmean(train_r2_adj),
        'gap': np.mean(train_scores) - np.mean(test_scores),
        # Failure tracking
        'n_failed_folds': n_failed_folds,
        # RMSE/MAE/MAPE
        'rmse_val': np.mean(test_rmse),
        'rmse_val_std': np.std(test_rmse),
        'mae_val': np.mean(test_mae),
        'mae_val_std': np.std(test_mae),
        'mape_val': np.nanmean(test_mape),
        # Timing (separate tune/train/predict)
        'tune_time': np.sum(tune_times),
        'train_time': np.sum(train_times),
        'predict_time': np.sum(predict_times),
        # Fold details
        'fold_results': fold_results,
        'n_pruned_total': total_pruned,
        'n_trials_total': n_outer_folds * n_trials,
        'total_time': elapsed_total,
        'n_features_used': X_processed.shape[1],
        # Sample size tracking
        'n_samples_original': n_samples_original,
        'n_samples_used': n_samples,  # After subsampling (if any)
        'subsampled': subsampled_from is not None,
        # Config for reproducibility
        'config': {
            'outer_splits': outer_splits,
            'outer_repeats': outer_repeats,
            'inner_splits': inner_splits,
            'n_trials': n_trials,
            'scale': scale,
            'random_state': random_state,
            'n_jobs': n_jobs,
        },
    }

    if verbose:
        print(f"\n  Summary: R²={results['r2_val']:.4f} ± {results['r2_val_std']:.3f}, "
              f"gap={results['gap']:.4f}, RMSE={results['rmse_val']:.4f}", flush=True)
        if n_failed_folds > 0:
            print(f"  WARNING: {n_failed_folds} failed folds (R²<-1)! "
                  f"Median R²={r2_val_median:.4f}, Trimmed={r2_val_trimmed:.4f}", flush=True)
        if results['n_trials_total'] > 0:
            print(f"  Pruned: {total_pruned}/{results['n_trials_total']} trials "
                  f"({100*total_pruned/results['n_trials_total']:.1f}%)")
        print(f"  Total time: {elapsed_total:.1f}s")

    # Save results if path provided
    if save_path:
        import joblib
        from pathlib import Path
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, save_file)
        if verbose:
            print(f"  Results saved to: {save_file}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from benchmark.data.loader import load_dataset

    print("=" * 60)
    print("Tuning Module - Quick Test")
    print("=" * 60)

    # Load a test dataset
    X, y, meta = load_dataset('friedman1')
    print(f"\nDataset: friedman1 ({meta.n_samples} samples, {meta.n_features} features)")

    # Test tuning Ridge
    print("\n--- Testing Ridge tuning (10 trials) ---")
    study = tune_model('ridge', X, y, n_trials=10, show_progress=True)
    print(f"Best params: {study.best_params}")
    print(f"Best R²: {study.best_value:.4f}")

    # Get best model
    best_model = get_best_model('ridge', study)
    print(f"Best model: {best_model}")
