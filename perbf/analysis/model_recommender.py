"""
Model family recommendation based on dataset smoothness characteristics.

Uses discontinuity metrics to suggest whether smooth models (ERBF, Chebyshev)
or tree ensembles (XGBoost, RF) are likely to perform better on a given dataset.

Created: 17Jan26
"""

import numpy as np
from .discontinuity_smoothness import compute_discontinuity_profile


def recommend_model_family(X, y, sample_size=5000, verbose=True):
    """
    Recommend model family based on dataset smoothness characteristics.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values
    sample_size : int
        Sample size for computing discontinuity metrics
    verbose : bool
        Print recommendation reasoning

    Returns
    -------
    dict with:
        - 'recommendation': 'smooth', 'tree', or 'both'
        - 'confidence': 'high', 'medium', 'low'
        - 'smooth_models': list of recommended smooth models
        - 'tree_models': list of recommended tree models
        - 'reasoning': list of reasons for recommendation
        - 'profile': full discontinuity profile
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n, d = X.shape

    # Compute discontinuity profile
    profile = compute_discontinuity_profile(X, y, sample_size=sample_size)

    # Extract key metrics
    disc_score = profile['discontinuity_score']
    cqjs = profile['cqjs_max']
    llqd = profile['llqd_tail_ratio']
    ntj = profile['ntj_spikiness']

    reasoning = []
    smooth_score = 0  # Higher = favor smooth models
    tree_score = 0    # Higher = favor tree models

    # Analyze discontinuity score
    if disc_score < 0.35:
        reasoning.append(f"Low discontinuity score ({disc_score:.2f}): smooth function likely")
        smooth_score += 2
    elif disc_score > 0.55:
        reasoning.append(f"High discontinuity score ({disc_score:.2f}): threshold effects likely")
        tree_score += 2
    else:
        reasoning.append(f"Moderate discontinuity score ({disc_score:.2f}): mixed characteristics")

    # Analyze CQJS (axis-aligned thresholds)
    if cqjs > 1.5:
        reasoning.append(f"Strong axis-aligned jumps (CQJS={cqjs:.2f}): trees excel here")
        tree_score += 2
    elif cqjs < 0.5:
        reasoning.append(f"Weak axis-aligned jumps (CQJS={cqjs:.2f}): smooth models preferred")
        smooth_score += 1

    # Analyze Lipschitz characteristics
    if llqd < 1.0:
        reasoning.append(f"Low Lipschitz tail ratio ({llqd:.2f}): globally smooth")
        smooth_score += 1
    elif llqd > 2.5:
        reasoning.append(f"High Lipschitz tail ratio ({llqd:.2f}): sharp local gradients")
        tree_score += 1

    # Dimensionality considerations
    if d > 50:
        reasoning.append(f"High dimensionality (d={d}): trees handle this well")
        tree_score += 1
    elif d <= 10:
        reasoning.append(f"Low dimensionality (d={d}): smooth models can capture structure")
        smooth_score += 1

    # Sample size considerations
    if n < 1000:
        reasoning.append(f"Small sample (n={n}): smooth models generalize better")
        smooth_score += 1
    elif n > 50000:
        reasoning.append(f"Large sample (n={n}): trees can learn complex patterns")
        tree_score += 1

    # Determine recommendation
    diff = smooth_score - tree_score
    if diff >= 2:
        recommendation = 'smooth'
        confidence = 'high' if diff >= 3 else 'medium'
    elif diff <= -2:
        recommendation = 'tree'
        confidence = 'high' if diff <= -3 else 'medium'
    else:
        recommendation = 'both'
        confidence = 'low'

    # Specific model suggestions
    smooth_models = []
    tree_models = []

    if recommendation in ['smooth', 'both']:
        if d <= 20:
            smooth_models.append('chebypoly')  # Full polynomial for low-d
        smooth_models.append('erbf')  # ERBF works across dimensions
        if d <= 30:
            smooth_models.append('chebytree')  # Piecewise polynomial

    if recommendation in ['tree', 'both']:
        tree_models.append('xgb')  # Generally best tree ensemble
        if n > 5000:
            tree_models.append('rf')  # RF good for larger samples

    # Always consider TabPFN for small/medium datasets
    tabpfn_note = None
    if n <= 10000 and d <= 100:
        tabpfn_note = "TabPFN viable (n<=10K, d<=100): consider as baseline"

    result = {
        'recommendation': recommendation,
        'confidence': confidence,
        'smooth_models': smooth_models,
        'tree_models': tree_models,
        'reasoning': reasoning,
        'profile': profile,
        'scores': {'smooth': smooth_score, 'tree': tree_score},
    }
    if tabpfn_note:
        result['tabpfn_note'] = tabpfn_note

    if verbose:
        print(f"Model Family Recommendation")
        print(f"=" * 40)
        print(f"Dataset: n={n}, d={d}")
        print(f"Discontinuity score: {disc_score:.3f}")
        print()
        print("Analysis:")
        for r in reasoning:
            print(f"  - {r}")
        print()
        print(f"Recommendation: {recommendation.upper()} (confidence: {confidence})")
        if smooth_models:
            print(f"  Smooth models: {', '.join(smooth_models)}")
        if tree_models:
            print(f"  Tree models: {', '.join(tree_models)}")
        if tabpfn_note:
            print(f"  Note: {tabpfn_note}")

    return result


def quick_smoothness_check(X, y, sample_size=2000):
    """
    Quick smoothness check for large datasets.

    Returns a simple score 0-1 where:
    - 0 = highly smooth (favor smooth models)
    - 1 = highly discontinuous (favor trees)
    """
    profile = compute_discontinuity_profile(X, y, sample_size=sample_size)
    return profile['discontinuity_score']
