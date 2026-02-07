"""
Model complexity extraction utilities.

Extract parameter counts and key attributes from fitted models for
complexity analysis and performance comparison.

Created: 23Jan26 (extracted from optuna_cv.py)
"""

from typing import Dict, Any


def extract_model_info(model, model_name: str) -> Dict[str, Any]:
    """
    Extract complexity metrics and key attributes from a fitted model.

    Parameters
    ----------
    model : estimator
        Fitted model
    model_name : str
        Name of model type

    Returns
    -------
    info : dict
        Model complexity metrics and key attributes
    """
    info = {'model_name': model_name}

    # Common: try to get n_features
    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_

    # Model-specific extraction
    if model_name == 'ridge':
        info['n_params'] = len(model.coef_) + 1  # coefs + intercept
        info['alpha'] = model.alpha

    elif model_name == 'dt':
        info['n_leaves'] = model.get_n_leaves()
        info['max_depth_actual'] = model.get_depth()
        info['n_params'] = model.get_n_leaves()  # ~1 param per leaf

    elif model_name == 'rf':
        info['n_estimators'] = model.n_estimators
        info['n_leaves_total'] = sum(t.get_n_leaves() for t in model.estimators_)
        info['n_params'] = info['n_leaves_total']

    elif model_name == 'xgb':
        info['n_estimators'] = model.n_estimators
        if hasattr(model, 'best_iteration'):
            info['best_iteration'] = model.best_iteration

    elif model_name == 'erbf':
        # Note: erbf package uses British spelling 'centres_'
        # Track if 'auto' was used (param value is string 'auto')
        info['n_rbf_auto'] = (model.n_rbf == 'auto')
        info['alpha'] = getattr(model, 'alpha', None)
        info['center_init'] = model.center_init
        info['width_init'] = model.width_init
        if hasattr(model, 'centres_'):
            # Extract actual fitted n_rbf from centres shape
            info['n_rbf'] = len(model.centres_)
            info['n_features'] = model.centres_.shape[1]
            # Params: centres + widths + weights + bias
            n_rbf, n_feat = model.centres_.shape
            if hasattr(model, 'widths_') and model.widths_ is not None:
                if model.widths_.ndim == 2:
                    info['n_width_params'] = model.widths_.size
                else:
                    info['n_width_params'] = len(model.widths_)
            info['n_params'] = n_rbf * n_feat + info.get('n_width_params', n_rbf) + n_rbf + 1

    elif model_name == 'chebypoly':
        info['complexity'] = model.complexity
        info['regressor'] = getattr(model, 'regressor', 'Ridge')
        info['alpha'] = model.alpha
        info['include_interactions'] = getattr(model, 'include_interactions', False)
        # ChebyshevRegressor exposes coef_ directly
        if hasattr(model, 'coef_'):
            info['n_params'] = len(model.coef_) + 1

    elif model_name == 'chebytree':
        info['complexity'] = model.complexity
        info['max_depth'] = model.max_depth
        info['regressor'] = getattr(model, 'regressor', 'Ridge')
        info['alpha'] = model.alpha
        if hasattr(model, 'n_leaves_'):
            info['n_leaves'] = model.n_leaves_
        elif hasattr(model, 'tree_'):
            info['n_leaves'] = model.tree_.get_n_leaves()
        # Get actual n_params from leaf models (dict keyed by leaf ID)
        if hasattr(model, 'leaf_models_') and model.leaf_models_:
            n_params_per_leaf = []
            for leaf_id, lm in model.leaf_models_.items():
                # PolyBasisRegressor has coef_ directly
                if hasattr(lm, 'coef_'):
                    n_params_per_leaf.append(len(lm.coef_) + 1)
            if n_params_per_leaf:
                info['n_params_per_leaf'] = n_params_per_leaf[0]
                info['n_params'] = sum(n_params_per_leaf)

    # TabPFN: foundation model - no conventional params
    # Note: Subsampling is now handled upstream in nested_cv_tune_and_evaluate,
    # so the model itself is a plain TabPFNRegressor without wrapper attributes.
    elif model_name == 'tabpfn':
        info['n_params'] = 'foundation'  # Foundation model - no conventional params

    return info
