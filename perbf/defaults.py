"""
Default hyperparameters for benchmark models.

Sensible defaults derived from analysis of 57-58 datasets (Jan 2026 benchmark).
Used when running with --no-tune flag for quick experiments on large datasets.

See notes-plans/model_defaults_21jan26.md for derivation methodology.
"""

MODEL_DEFAULTS = {
    'ridge': {
        'alpha': 1.0,
    },

    'dt': {
        'max_depth': 10,
        'min_samples_leaf': 0.01,
        'min_samples_split': 0.01,
    },

    'rf': {
        'n_estimators': 200,
        'max_depth': 10,
        'max_features': 0.5,
        'min_samples_leaf': 0.01,
    },

    'xgb': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
    },

    'chebypoly': {
        'complexity': 7,
        'include_interactions': True,
        'max_interaction_complexity': 1,  # 1=raw product, 2=with T_2 expansion
        'alpha': 0.5,
    },

    'chebytree': {
        'complexity': 2,
        'max_depth': 4,
        'min_samples_leaf': 0.05,
        'alpha': 0.1,
    },

    'erbf': {
        'n_rbf': 'auto',
        'center_init': 'lipschitz',
        'width_init': 'local_ridge',
        'alpha': 0.2,
    },

    'tabpfn': {},  # No hyperparameters to tune
}


def get_default_params(model_name: str) -> dict:
    """Get default hyperparameters for a model.

    Parameters
    ----------
    model_name : str
        Model name (ridge, dt, rf, xgb, chebypoly, chebytree, erbf, tabpfn)

    Returns
    -------
    dict
        Default hyperparameters for the model

    Raises
    ------
    ValueError
        If model_name is not recognized
    """
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(MODEL_DEFAULTS.keys())}")
    return MODEL_DEFAULTS[model_name].copy()
