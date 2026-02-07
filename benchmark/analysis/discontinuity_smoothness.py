"""
Discontinuity and smoothness metrics for dataset and model characterization.

DATASET-LEVEL (characterize data, recommend model family):
- LLQD, NTJ, LLFRR, GTV, CQJS
- compute_discontinuity_profile()

MODEL-LEVEL (characterize predictions, compare model smoothness):
- compute_knn_jump() - average prediction difference between neighbors
- compute_path_metrics() - TV, curvature, jump count along line segments
- compute_violation_probability() - P(big output | small input change)
- compute_local_lipschitz_quantiles() - distribution of local gradients
- compute_model_smoothness_profile() - all combined

Created: 17Jan26 (dataset metrics)
Updated: 22Jan26 (added model-level metrics)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from scipy.stats import median_abs_deviation


# ===========================================================================
# DATASET-LEVEL METRICS (characterize data)
# ===========================================================================

def llqd_distribution(X, y, k=10, sample_size=None, random_state=42):
    """
    Local Lipschitz Quotient Distribution (LLQD).

    For each point, compute max |y_i - y_j| / ||x_i - x_j|| over k-NN.
    Heavy upper tail indicates clean discontinuities.

    Returns dict with llqd_tail_ratio (log q99/q50) as key metric.
    """
    X, y = np.asarray(X), np.asarray(y).ravel()
    n = len(y)

    rng = np.random.RandomState(random_state)
    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, y, n = X[idx], y[idx], sample_size

    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    llqd_values = []
    for i in range(n):
        dists = distances[i, 1:]
        valid = dists > 1e-10
        if not valid.any():
            llqd_values.append(0.0)
            continue
        ratios = np.abs(y[i] - y[indices[i, 1:][valid]]) / dists[valid]
        llqd_values.append(np.max(ratios))

    llqd_values = np.array(llqd_values)
    q50, q99 = np.percentile(llqd_values, [50, 99])

    return {
        'llqd_values': llqd_values,
        'llqd_median': q50,
        'llqd_q99': q99,
        'llqd_tail_ratio': np.log(q99 / q50) if q50 > 1e-10 else np.inf,
    }


def ntj_stats(X, y, k=10, sample_size=None, random_state=42):
    """
    Neighbour Target Jump Statistic (NTJ).

    For each point, compute max |y_i - y_j| within k-NN.
    Returns ntj_spikiness (q95/q50) as key metric.
    """
    X, y = np.asarray(X), np.asarray(y).ravel()
    n = len(y)

    rng = np.random.RandomState(random_state)
    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, y, n = X[idx], y[idx], sample_size

    k_actual = min(k, n - 1)
    _, indices = NearestNeighbors(n_neighbors=k_actual + 1).fit(X).kneighbors(X)

    ntj_values = np.array([np.max(np.abs(y[i] - y[indices[i, 1:]])) for i in range(n)])
    q50, q95 = np.percentile(ntj_values, [50, 95])

    return {
        'ntj_values': ntj_values,
        'ntj_median': q50,
        'ntj_q95': q95,
        'ntj_spikiness': q95 / q50 if q50 > 1e-10 else np.inf,
    }


def llfrr(X, y, k=20, sample_size=None, random_state=42):
    """
    Local Linear Fit Residual Ratio (LLFRR).

    Fit local linear model per neighborhood, compute residual/local_std.
    LLFRR -> 1 near discontinuity boundaries.
    """
    X, y = np.asarray(X), np.asarray(y).ravel()
    n, d = X.shape

    rng = np.random.RandomState(random_state)
    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, y, n = X[idx], y[idx], sample_size

    k_actual = max(min(k, n - 1), d + 2)
    _, indices = NearestNeighbors(n_neighbors=k_actual + 1).fit(X).kneighbors(X)

    llfrr_values = []
    for i in range(n):
        y_local = y[indices[i, 1:]]
        local_std = np.std(y_local)
        if local_std < 1e-10:
            llfrr_values.append(0.0)
            continue
        try:
            lr = LinearRegression().fit(X[indices[i, 1:]], y_local)
            residual = np.abs(y[i] - lr.predict(X[i:i+1])[0])
            # Clip to prevent numerical blow-up (residual >> local_std)
            llfrr_values.append(min(residual / local_std, 10.0))
        except Exception:
            llfrr_values.append(np.nan)

    llfrr_values = np.array(llfrr_values)
    valid = ~np.isnan(llfrr_values)

    return {
        'llfrr_values': llfrr_values,
        'llfrr_median': np.percentile(llfrr_values[valid], 50) if valid.any() else np.nan,
        'llfrr_mean': np.mean(llfrr_values[valid]) if valid.any() else np.nan,
    }


def gtv(X, y, k=10, sample_size=None, random_state=42):
    """
    Graph Total Variation (GTV).

    Sum of edge differences on k-NN graph, normalized by |E| * MAD(y).
    """
    X, y = np.asarray(X), np.asarray(y).ravel()
    n = len(y)

    rng = np.random.RandomState(random_state)
    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, y, n = X[idx], y[idx], sample_size

    k_actual = min(k, n - 1)
    _, indices = NearestNeighbors(n_neighbors=k_actual + 1).fit(X).kneighbors(X)

    gtv_sum = sum(np.sum(np.abs(y[i] - y[indices[i, 1:]])) for i in range(n))
    n_edges = n * k_actual

    mad_y = median_abs_deviation(y)
    if mad_y < 1e-10:
        mad_y = max(np.std(y), 1.0)

    return {
        'gtv_raw': gtv_sum,
        'gtv_normalized': gtv_sum / (n_edges * mad_y) if n_edges > 0 else 0.0,
        'n_edges': n_edges,
    }


def cqjs_binned(X, y, n_bins=20, sample_size=None, random_state=42):
    """
    Conditional Quantile Jump Score (CQJS).

    Per feature: bin values, compute max jump in median y between adjacent bins.
    Returns max across all features - interpretable for axis-aligned thresholds.
    """
    X, y = np.asarray(X), np.asarray(y).ravel()
    n, d = X.shape

    rng = np.random.RandomState(random_state)
    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X, y, n = X[idx], y[idx], sample_size

    y_std = max(np.std(y), 1e-10)
    y_norm = (y - np.mean(y)) / y_std

    cqjs_per_feature = []
    for j in range(d):
        try:
            bins = np.unique(np.percentile(X[:, j], np.linspace(0, 100, n_bins + 1)))
            if len(bins) < 3:
                cqjs_per_feature.append(0.0)
                continue

            bin_idx = np.digitize(X[:, j], bins[1:-1])
            medians = [np.median(y_norm[bin_idx == b]) for b in range(len(bins) - 1)
                       if (bin_idx == b).sum() > 0]

            cqjs_per_feature.append(np.max(np.abs(np.diff(medians))) if len(medians) > 1 else 0.0)
        except Exception:
            cqjs_per_feature.append(0.0)

    cqjs_per_feature = np.array(cqjs_per_feature)

    return {
        'cqjs_per_feature': cqjs_per_feature,
        'cqjs_max': np.max(cqjs_per_feature) if len(cqjs_per_feature) > 0 else 0.0,
        'cqjs_max_feature': int(np.argmax(cqjs_per_feature)) if len(cqjs_per_feature) > 0 else -1,
    }


def compute_discontinuity_profile(X, y, k=10, n_bins=20, sample_size=5000, random_state=42):
    """
    Compute all 5 discontinuity metrics for a dataset.

    Returns dict with summary metrics and composite discontinuity_score.
    Higher scores indicate more discontinuous/threshold-driven data.

    Interpretation:
        - llqd_tail_ratio > 2: likely sharp discontinuities
        - ntj_spikiness > 3: significant local target variation
        - cqjs_max > 1: strong axis-aligned thresholds (1 std jump)
        - discontinuity_score > 0.5: likely S5 (discontinuous stratum)
    """
    llqd_res = llqd_distribution(X, y, k=k, sample_size=sample_size, random_state=random_state)
    ntj_res = ntj_stats(X, y, k=k, sample_size=sample_size, random_state=random_state)
    llf_res = llfrr(X, y, k=max(k, 20), sample_size=sample_size, random_state=random_state)
    gtv_res = gtv(X, y, k=k, sample_size=sample_size, random_state=random_state)
    cqjs_res = cqjs_binned(X, y, n_bins=n_bins, sample_size=sample_size, random_state=random_state)

    profile = {
        'llqd_tail_ratio': llqd_res['llqd_tail_ratio'],
        'ntj_spikiness': ntj_res['ntj_spikiness'],
        'llfrr_mean': llf_res['llfrr_mean'],
        'gtv_normalized': gtv_res['gtv_normalized'],
        'cqjs_max': cqjs_res['cqjs_max'],
    }

    # Composite score via sigmoid normalization
    def sig(x, mid, scale):
        return 1 / (1 + np.exp(-(x - mid) / scale))

    scores = [
        sig(profile['llqd_tail_ratio'], 2.0, 1.0),
        sig(profile['ntj_spikiness'], 3.0, 1.5),
        sig(profile['llfrr_mean'], 0.5, 0.25) if np.isfinite(profile['llfrr_mean']) else 0.5,
        sig(profile['gtv_normalized'], 1.5, 0.5),
        sig(profile['cqjs_max'], 1.0, 0.5),
    ]
    profile['discontinuity_score'] = np.mean([s if np.isfinite(s) else 0.5 for s in scores])

    return profile


# ===========================================================================
# UNIFIED ROUGHNESS METRICS (same formula for data and model comparison)
# ===========================================================================

def compute_knn_roughness(X, values, k=10, n_samples=5000, random_state=42):
    """
    Compute kNN-graph roughness for any values (y or y_pred).

    Uses the same formula for both data targets and model predictions,
    enabling meaningful computation of the roughness amplification ratio ρ.

    Formula:
        R = (1/|E|) × Σ_{(i,j) ∈ E} |v_i - v_j| / ||x_i - x_j||₂

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix
    values : array-like, shape (n,)
        Values to measure roughness of (either y or y_pred)
    k : int
        Number of neighbors for kNN graph
    n_samples : int
        Subsample size for large datasets
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        roughness : float
            Mean |v_i - v_j| / ||x_i - x_j|| over all kNN edges
        roughness_unnorm : float
            Mean |v_i - v_j| without distance normalization
        roughness_q50, _q90, _q99 : float
            Quantiles of per-edge roughness values
        n_edges : int
            Number of edges in kNN graph
    """
    X = np.asarray(X)
    values = np.asarray(values).ravel()
    n = len(values)

    rng = np.random.RandomState(random_state)
    if n > n_samples:
        idx = rng.choice(n, n_samples, replace=False)
        X, values, n = X[idx], values[idx], n_samples

    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    # Compute per-edge roughness (normalized by distance)
    edge_roughness = []
    edge_roughness_unnorm = []

    for i in range(n):
        for j_idx in range(1, k_actual + 1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]

            value_diff = np.abs(values[i] - values[j])
            edge_roughness_unnorm.append(value_diff)

            if dist > 1e-10:
                edge_roughness.append(value_diff / dist)
            # Skip edges with zero distance (duplicates)

    edge_roughness = np.array(edge_roughness)
    edge_roughness_unnorm = np.array(edge_roughness_unnorm)

    if len(edge_roughness) == 0:
        return {
            'roughness': np.nan,
            'roughness_unnorm': np.nan,
            'roughness_q50': np.nan,
            'roughness_q90': np.nan,
            'roughness_q99': np.nan,
            'n_edges': 0,
        }

    return {
        'roughness': np.mean(edge_roughness),
        'roughness_unnorm': np.mean(edge_roughness_unnorm),
        'roughness_q50': np.percentile(edge_roughness, 50),
        'roughness_q90': np.percentile(edge_roughness, 90),
        'roughness_q99': np.percentile(edge_roughness, 99),
        'n_edges': len(edge_roughness),
    }


def compute_roughness_amplification(X, y, y_pred, k=10, n_samples=5000,
                                     multi_scale=False, random_state=42):
    """
    Compute roughness amplification ratio ρ = R_model / R_data.

    This is the key metric for comparing model smoothness: it normalizes
    model roughness by data roughness, making the comparison fair across
    datasets with different intrinsic roughness levels.

    Interpretation:
        ρ ≈ 1  : Model tracks intrinsic data roughness faithfully
        ρ < 1  : Model smooths aggressively (may introduce bias)
        ρ > 1  : Model adds artificial roughness (overfitting artifacts)

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix
    y : array-like, shape (n,)
        True target values
    y_pred : array-like, shape (n,)
        Model predictions
    k : int
        Number of neighbors for kNN graph
    n_samples : int
        Subsample size for large datasets
    multi_scale : bool
        If True, compute ρ at multiple k values (5, 10, 20, 50)
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        rho : float
            Roughness amplification ratio R_model / R_data
        R_data : float
            Data roughness (on y)
        R_model : float
            Model roughness (on y_pred)
        interpretation : str
            Human-readable interpretation of ρ
        rho_unnorm : float
            Same ratio but without distance normalization

    If multi_scale=True, also includes:
        rho_k5, rho_k10, rho_k20, rho_k50 : float
            ρ at different neighborhood sizes
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Subsample consistently for both
    n = len(y)
    rng = np.random.RandomState(random_state)
    if n > n_samples:
        idx = rng.choice(n, n_samples, replace=False)
        X, y, y_pred = X[idx], y[idx], y_pred[idx]

    # Compute roughness for data and model
    R_data_res = compute_knn_roughness(X, y, k=k, n_samples=len(y), random_state=random_state)
    R_model_res = compute_knn_roughness(X, y_pred, k=k, n_samples=len(y_pred), random_state=random_state)

    R_data = R_data_res['roughness']
    R_model = R_model_res['roughness']

    # Compute ratio (handle edge cases)
    if R_data > 1e-10:
        rho = R_model / R_data
    else:
        rho = np.nan  # Data is essentially constant

    # Unnormalized version
    R_data_unnorm = R_data_res['roughness_unnorm']
    R_model_unnorm = R_model_res['roughness_unnorm']
    if R_data_unnorm > 1e-10:
        rho_unnorm = R_model_unnorm / R_data_unnorm
    else:
        rho_unnorm = np.nan

    # Interpretation
    if np.isnan(rho):
        interpretation = "undefined (constant data)"
    elif rho < 0.8:
        interpretation = "smooths aggressively (potential bias)"
    elif rho < 0.95:
        interpretation = "slight smoothing"
    elif rho <= 1.05:
        interpretation = "tracks data roughness well"
    elif rho <= 1.2:
        interpretation = "slight roughness amplification"
    else:
        interpretation = "adds artificial roughness (overfitting artifacts)"

    result = {
        'rho': rho,
        'R_data': R_data,
        'R_model': R_model,
        'R_data_unnorm': R_data_unnorm,
        'R_model_unnorm': R_model_unnorm,
        'rho_unnorm': rho_unnorm,
        'interpretation': interpretation,
        'k': k,
    }

    # Multi-scale analysis
    if multi_scale:
        for k_val in [5, 10, 20, 50]:
            if k_val == k:
                result[f'rho_k{k_val}'] = rho
            else:
                R_data_k = compute_knn_roughness(X, y, k=k_val, n_samples=len(y),
                                                  random_state=random_state)['roughness']
                R_model_k = compute_knn_roughness(X, y_pred, k=k_val, n_samples=len(y_pred),
                                                   random_state=random_state)['roughness']
                if R_data_k > 1e-10:
                    result[f'rho_k{k_val}'] = R_model_k / R_data_k
                else:
                    result[f'rho_k{k_val}'] = np.nan

    return result


# ===========================================================================
# MODEL-LEVEL METRICS (characterize predictions)
# ===========================================================================

def compute_knn_jump(X, y_pred, k=10, n_samples=5000, random_state=42):
    """
    kNN Graph Jump (J) - Primary metric for trees vs smooth.

    Measures average absolute prediction difference between k-nearest neighbors.
    Trees produce high values (piecewise constant), smooth models produce low.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix (same as used for predictions)
    y_pred : array-like, shape (n,)
        Model predictions
    k : int
        Number of neighbors
    n_samples : int
        Subsample size for large datasets
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        knn_jump : float
            Mean absolute prediction difference across all neighbor pairs
        knn_jump_norm : float
            Normalized by MAD(y_pred) for cross-model comparison
        knn_jump_std : float
            Std of per-point mean jump values
        n_edges : int
            Number of edges in kNN graph
    """
    X = np.asarray(X)
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_pred)

    rng = np.random.RandomState(random_state)
    if n > n_samples:
        idx = rng.choice(n, n_samples, replace=False)
        X, y_pred, n = X[idx], y_pred[idx], n_samples

    k_actual = min(k, n - 1)
    _, indices = NearestNeighbors(n_neighbors=k_actual + 1).fit(X).kneighbors(X)

    # Compute per-point mean jump
    per_point_jumps = np.array([
        np.mean(np.abs(y_pred[i] - y_pred[indices[i, 1:]]))
        for i in range(n)
    ])

    knn_jump = np.mean(per_point_jumps)

    # Normalize by MAD for cross-model comparison
    mad_y = median_abs_deviation(y_pred)
    if mad_y < 1e-10:
        mad_y = max(np.std(y_pred), 1e-10)

    return {
        'knn_jump': knn_jump,
        'knn_jump_norm': knn_jump / mad_y,
        'knn_jump_std': np.std(per_point_jumps),
        'n_edges': n * k_actual,
    }


def compute_path_metrics(model, X, n_paths=100, n_steps=50, k_neighbors=10,
                         jump_threshold=None, random_state=42):
    """
    Path-based metrics along line segments between nearby points.

    Creates paths between random pairs of nearby points and measures:
    - Total Variation (TV): sum of absolute differences along path
    - Curvature: sum of second differences (measures non-linearity)
    - Jump count: number of large prediction changes

    Parameters
    ----------
    model : fitted regressor
        Must have predict() method
    X : array-like, shape (n, d)
        Feature matrix to sample paths from
    n_paths : int
        Number of random paths to generate
    n_steps : int
        Number of interpolation steps per path
    k_neighbors : int
        Find paths between k-nearest neighbor pairs
    jump_threshold : float or None
        Threshold for counting jumps. If None, uses 0.5 * std(predictions)
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        path_tv_mean, path_tv_std : float
            Total variation statistics
        path_curvature_mean, path_curvature_std : float
            Curvature statistics
        path_jump_count_mean, path_jump_count_std : float
            Jump count statistics
        paths : list of dicts
            Individual path data (for visualization)
    """
    X = np.asarray(X)
    n, d = X.shape
    rng = np.random.RandomState(random_state)

    # Find nearby pairs using kNN
    k_actual = min(k_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
    _, indices = nn.kneighbors(X)

    # Sample starting points and pick random neighbor as endpoint
    start_indices = rng.choice(n, min(n_paths, n), replace=False)
    paths_data = []

    for start_idx in start_indices:
        # Pick random neighbor (not self)
        neighbor_idx = indices[start_idx, 1 + rng.randint(k_actual)]

        # Create interpolated path
        t = np.linspace(0, 1, n_steps).reshape(-1, 1)
        path_X = X[start_idx] + t * (X[neighbor_idx] - X[start_idx])

        # Get predictions along path
        path_y = model.predict(path_X)

        paths_data.append({
            'start_idx': start_idx,
            'end_idx': neighbor_idx,
            'path_X': path_X,
            'path_y': path_y,
        })

    # Compute metrics
    if jump_threshold is None:
        all_preds = np.concatenate([p['path_y'] for p in paths_data])
        jump_threshold = 0.5 * np.std(all_preds)

    tv_values = []
    curvature_values = []
    jump_counts = []

    for p in paths_data:
        y = p['path_y']
        diffs = np.abs(np.diff(y))

        # Total variation
        tv_values.append(np.sum(diffs))

        # Curvature (second differences)
        if len(y) >= 3:
            second_diffs = np.abs(np.diff(y, n=2))
            curvature_values.append(np.sum(second_diffs))
        else:
            curvature_values.append(0.0)

        # Jump count
        jump_counts.append(np.sum(diffs > jump_threshold))

    return {
        'path_tv_mean': np.mean(tv_values),
        'path_tv_std': np.std(tv_values),
        'path_curvature_mean': np.mean(curvature_values),
        'path_curvature_std': np.std(curvature_values),
        'path_jump_count_mean': np.mean(jump_counts),
        'path_jump_count_std': np.std(jump_counts),
        'jump_threshold': jump_threshold,
        'paths': paths_data,
    }


def compute_violation_probability(X, y_pred, eps_values=None, tau=None,
                                   n_pairs=10000, k_neighbors=20,
                                   random_state=42):
    """
    Violation Probability V(eps, tau) - Local consistency metric.

    Measures P(|f(x) - f(x')| > tau | ||x - x'|| < eps)
    i.e., probability of large output change given small input change.

    Trees have high violation probability (step functions),
    smooth models have low violation probability.

    Uses kNN-based pair sampling to ensure sufficient nearby pairs at small
    epsilon values, supplemented by random pairs for larger epsilon.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix
    y_pred : array-like, shape (n,)
        Model predictions
    eps_values : list of float or None
        Distance thresholds as fractions of data diameter.
        Default: [0.01, 0.05, 0.1]
    tau : float or None
        Output difference threshold. Default: 0.5 * std(y_pred)
    n_pairs : int
        Number of random pairs to sample (for larger epsilon)
    k_neighbors : int
        Number of nearest neighbours per point for small-epsilon pairs
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        violation_prob_{eps} : float
            Violation probabilities at each epsilon (key = eps value as string)
        tau : float
            Threshold used
        n_pairs_evaluated : dict
            Number of pairs within each epsilon ball
    """
    from sklearn.neighbors import NearestNeighbors

    if eps_values is None:
        eps_values = [0.01, 0.05, 0.1]

    X = np.asarray(X)
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_pred)

    rng = np.random.RandomState(random_state)

    # Compute data diameter for relative epsilon
    # Use subset for efficiency
    sample_idx = rng.choice(n, min(1000, n), replace=False)
    X_sample = X[sample_idx]
    dists_sample = np.linalg.norm(X_sample[:, None] - X_sample[None, :], axis=2)
    diameter = np.percentile(dists_sample[dists_sample > 0], 95)  # Robust estimate

    if tau is None:
        tau = 0.5 * np.std(y_pred)

    # --- kNN-based pairs (good for small epsilon) ---
    # Subsample if large, then find k nearest neighbours
    n_sub = min(n, 5000)
    sub_idx = rng.choice(n, n_sub, replace=False) if n > n_sub else np.arange(n)
    X_sub = X[sub_idx]
    y_sub = y_pred[sub_idx]

    k_actual = min(k_neighbors, n_sub - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X_sub)
    nn_dists, nn_indices = nn.kneighbors(X_sub)

    # Build arrays of (distance, |pred_diff|) for all kNN edges
    knn_pair_dists = nn_dists[:, 1:].ravel()  # skip self
    knn_pair_j = nn_indices[:, 1:].ravel()
    knn_pair_i = np.repeat(np.arange(n_sub), k_actual)
    knn_pred_diffs = np.abs(y_sub[knn_pair_i] - y_sub[knn_pair_j])

    # --- Random pairs (good for larger epsilon) ---
    idx1 = rng.randint(0, n, n_pairs)
    idx2 = rng.randint(0, n, n_pairs)
    rand_pair_dists = np.linalg.norm(X[idx1] - X[idx2], axis=1)
    rand_pred_diffs = np.abs(y_pred[idx1] - y_pred[idx2])

    # Combine both pair sources
    all_dists = np.concatenate([knn_pair_dists, rand_pair_dists])
    all_pred_diffs = np.concatenate([knn_pred_diffs, rand_pred_diffs])

    results = {'tau': tau, 'diameter': diameter, 'n_pairs_evaluated': {}}

    for eps in eps_values:
        eps_abs = eps * diameter
        within_eps = all_dists < eps_abs
        n_within = int(np.sum(within_eps))

        if n_within > 0:
            violations = int(np.sum((within_eps) & (all_pred_diffs > tau)))
            prob = violations / n_within
        else:
            prob = np.nan

        key = f"violation_prob_{eps}"
        results[key] = prob
        results['n_pairs_evaluated'][eps] = n_within

    return results


def compute_local_lipschitz_quantiles(X, y_pred, k=10, n_samples=5000,
                                       random_state=42):
    """
    Local Lipschitz quotient distribution from kNN edges.

    For each point, compute |y_pred_i - y_pred_j| / ||x_i - x_j|| over k
    nearest neighbours. Returns quantiles of this distribution.

    Trees produce heavy upper tails (large quotients at split boundaries);
    smooth (C∞) models produce lighter tails.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Feature matrix (test set)
    y_pred : array-like, shape (n,)
        Model predictions
    k : int
        Number of nearest neighbours per point
    n_samples : int
        Subsample size if n > n_samples
    random_state : int
        Random seed

    Returns
    -------
    dict with:
        lipschitz_median, lipschitz_q90, lipschitz_q95, lipschitz_q99 : float
            Quantiles of local Lipschitz quotients
        lipschitz_mean, lipschitz_std : float
            Mean and std
        lipschitz_tail_ratio : float
            log10(q99 / q50) — heavy tail indicator
        n_quotients : int
            Number of quotients computed
    """
    X = np.asarray(X)
    y_pred = np.asarray(y_pred).ravel()
    n, d = X.shape

    rng = np.random.RandomState(random_state)

    # Subsample if needed
    if n > n_samples:
        idx = rng.choice(n, n_samples, replace=False)
        X, y_pred, n = X[idx], y_pred[idx], n_samples

    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
    distances, indices = nn.kneighbors(X)

    # Skip self (column 0), use columns 1..k
    dists = distances[:, 1:].ravel()
    nbr_idx = indices[:, 1:].ravel()
    pt_idx = np.repeat(np.arange(n), k_actual)

    pred_diffs = np.abs(y_pred[pt_idx] - y_pred[nbr_idx])

    # Filter zero distances
    valid = dists > 1e-10
    quotients = pred_diffs[valid] / dists[valid]

    if len(quotients) == 0:
        return {
            'lipschitz_median': np.nan,
            'lipschitz_q90': np.nan,
            'lipschitz_q95': np.nan,
            'lipschitz_q99': np.nan,
            'lipschitz_mean': np.nan,
            'lipschitz_std': np.nan,
            'lipschitz_tail_ratio': np.nan,
            'n_quotients': 0,
        }

    q50 = np.median(quotients)
    q90 = np.percentile(quotients, 90)
    q95 = np.percentile(quotients, 95)
    q99 = np.percentile(quotients, 99)

    tail_ratio = np.log10(q99 / q50) if q50 > 1e-10 else np.nan

    return {
        'lipschitz_median': q50,
        'lipschitz_q90': q90,
        'lipschitz_q95': q95,
        'lipschitz_q99': q99,
        'lipschitz_mean': np.mean(quotients),
        'lipschitz_std': np.std(quotients),
        'lipschitz_tail_ratio': tail_ratio,
        'n_quotients': len(quotients),
    }


def compute_probe_regularity(model, X, y_pred=None, delta_frac=0.1,
                              m=30, k_for_scale=10, clip_range=None,
                              tau_multiplier=0.3, random_state=42):
    """
    Prediction surface regularity via synthetic perturbation probes.

    For each point x_i, generate m synthetic neighbours at distance δ in
    random directions on the unit sphere, predict on them, and compute
    local Lipschitz quotients |Δŷ|/δ. Returns distributional summaries
    and violation probability.

    δ is set as delta_frac × median kNN distance in the model's feature
    space, making it adaptive to scale and dimensionality.

    Parameters
    ----------
    model : fitted regressor with .predict()
    X : array (n, d), test points to probe around
    y_pred : array (n,) or None
        Pre-computed base predictions. If None, computed from model.
    delta_frac : float
        δ = delta_frac × median kNN distance (default 0.1)
    m : int
        Number of probes per point (default 30)
    k_for_scale : int
        k for median kNN distance computation (default 10)
    clip_range : array (d, 2) or None
        Per-feature [min, max] to clip probes. If None, no clipping.
    tau_multiplier : float
        τ = tau_multiplier × std(ŷ) for violation probability
    random_state : int

    Returns
    -------
    dict with:
        lip_tail_ratio : float - log10(q99/q75) of quotient distribution
        lip_q50, lip_q75, lip_q90, lip_q95, lip_q99 : float - quantiles
        lip_mean, lip_std : float
        violation_prob : float - fraction of probes with |Δŷ| > τ
        tau : float - threshold used
        delta : float - perturbation radius used
        median_knn_dist : float - median kNN distance in feature space
        per_point_max_q95 : float - q95 of per-point max quotients
        n_probes : int
    """
    X = np.asarray(X)
    n, d = X.shape
    rng = np.random.RandomState(random_state)

    # Base predictions
    if y_pred is None:
        y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).ravel()

    # Compute δ from median kNN distance
    k_actual = min(k_for_scale, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1).fit(X)
    nn_dists = nn.kneighbors(X, return_distance=True)[0]
    median_knn_dist = np.median(nn_dists[:, 1:])
    delta = delta_frac * median_knn_dist

    if delta < 1e-15:
        # Degenerate case — all points identical
        return {k: np.nan for k in [
            'lip_tail_ratio', 'lip_q50', 'lip_q75', 'lip_q90', 'lip_q95',
            'lip_q99', 'lip_mean', 'lip_std', 'violation_prob', 'tau',
            'delta', 'median_knn_dist', 'per_point_max_q95', 'n_probes',
        ]}

    # Generate probes: random directions on unit sphere
    directions = rng.randn(n, m, d)
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    probes = X[:, None, :] + delta * directions  # (n, m, d)

    # Clip to data support
    if clip_range is not None:
        clip_range = np.asarray(clip_range)
        probes = np.clip(probes, clip_range[:, 0], clip_range[:, 1])

    # Batch predict
    probes_flat = probes.reshape(n * m, d)
    y_probes = model.predict(probes_flat).reshape(n, m)

    # Quotients
    pred_diffs = np.abs(y_probes - y_pred[:, None])
    quotients = pred_diffs / delta
    all_q = quotients.ravel()

    # Quantiles
    q50 = np.median(all_q)
    q75 = np.percentile(all_q, 75)
    q90 = np.percentile(all_q, 90)
    q95 = np.percentile(all_q, 95)
    q99 = np.percentile(all_q, 99)

    # Tail ratios at multiple denominators
    tail_ratio_q75 = np.log10(q99 / q75) if q75 > 1e-10 else np.nan
    tail_ratio_q90 = np.log10(q99 / q90) if q90 > 1e-10 else np.nan

    # Violation probability
    tau = tau_multiplier * np.std(y_pred)
    violation_prob = np.mean(pred_diffs.ravel() > tau)

    # Per-point max quotient
    per_point_max = np.max(quotients, axis=1)

    return {
        'lip_tail_ratio': tail_ratio_q75,
        'lip_tail_ratio_q90': tail_ratio_q90,
        'lip_q50': q50,
        'lip_q75': q75,
        'lip_q90': q90,
        'lip_q95': q95,
        'lip_q99': q99,
        'lip_mean': np.mean(all_q),
        'lip_std': np.std(all_q),
        'violation_prob': violation_prob,
        'tau': tau,
        'delta': delta,
        'median_knn_dist': median_knn_dist,
        'per_point_max_q95': np.percentile(per_point_max, 95),
        'n_probes': n * m,
    }


def compute_model_smoothness_profile(model, X, y_pred=None, include_paths=False,
                                      n_samples=5000, random_state=42):
    """
    Comprehensive smoothness profile for a trained model.

    Combines all model-level metrics into a single profile for comparing
    smoothness characteristics across different model families.

    Parameters
    ----------
    model : fitted regressor or None
        Must have predict() method. If None, only metrics not requiring
        the model will be computed (knn_jump, violation_prob, lipschitz).
    X : array-like, shape (n, d)
        Feature matrix
    y_pred : array-like or None
        Pre-computed predictions. If None and model provided, will compute.
    include_paths : bool
        If True, include detailed path data (larger output)
    n_samples : int
        Subsample size for large datasets
    random_state : int
        Random seed

    Returns
    -------
    dict with all smoothness metrics combined:
        knn_jump, knn_jump_norm : float
        violation_prob_0.01, _0.05, _0.1 : float
        lipschitz_median, lipschitz_q90, lipschitz_q99 : float
        path_tv_mean, path_curvature_mean, path_jump_count_mean : float (if model provided)
        paths : list (if include_paths=True and model provided)
    """
    X = np.asarray(X)

    if y_pred is None and model is not None:
        y_pred = model.predict(X)
    elif y_pred is None:
        raise ValueError("Either model or y_pred must be provided")

    y_pred = np.asarray(y_pred).ravel()

    profile = {}

    # kNN jump metrics
    knn_res = compute_knn_jump(X, y_pred, n_samples=n_samples, random_state=random_state)
    profile.update({
        'knn_jump': knn_res['knn_jump'],
        'knn_jump_norm': knn_res['knn_jump_norm'],
    })

    # Violation probability
    viol_res = compute_violation_probability(X, y_pred, random_state=random_state)
    profile.update({
        'violation_prob_0.01': viol_res.get('violation_prob_0.01', np.nan),
        'violation_prob_0.05': viol_res.get('violation_prob_0.05', np.nan),
        'violation_prob_0.1': viol_res.get('violation_prob_0.1', np.nan),
        'tau': viol_res['tau'],
    })

    # Local Lipschitz
    lip_res = compute_local_lipschitz_quantiles(X, y_pred, n_samples=n_samples,
                                                 random_state=random_state)
    profile.update({
        'lipschitz_median': lip_res['lipschitz_median'],
        'lipschitz_q90': lip_res['lipschitz_q90'],
        'lipschitz_q99': lip_res['lipschitz_q99'],
    })

    # Path metrics (require model for interpolation)
    if model is not None:
        path_res = compute_path_metrics(model, X, random_state=random_state)
        profile.update({
            'path_tv_mean': path_res['path_tv_mean'],
            'path_curvature_mean': path_res['path_curvature_mean'],
            'path_jump_count_mean': path_res['path_jump_count_mean'],
        })
        if include_paths:
            profile['paths'] = path_res['paths']

    return profile
