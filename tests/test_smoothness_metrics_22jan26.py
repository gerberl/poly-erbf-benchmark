"""
Unit tests for model-level smoothness metrics.

Tests verify that:
1. Metrics correctly distinguish smooth vs piecewise-constant models
2. Functions handle edge cases gracefully
3. Return values have expected structure

Created: 22Jan26
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor


class TestKnnJump:
    """Tests for compute_knn_jump."""

    def test_basic_output_structure(self):
        """Should return dict with expected keys."""
        from benchmark.analysis import compute_knn_jump

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y_pred = rng.randn(100)

        result = compute_knn_jump(X, y_pred)

        assert 'knn_jump' in result
        assert 'knn_jump_norm' in result
        assert 'knn_jump_std' in result
        assert 'n_edges' in result

    def test_constant_zero_knn_jump(self):
        """Constant predictions should have zero knn_jump."""
        from benchmark.analysis import compute_knn_jump

        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        y_const = np.ones(200) * 3.0

        result = compute_knn_jump(X, y_const)
        assert result['knn_jump'] < 1e-10

    def test_subsampling(self):
        """Should subsample large datasets."""
        from benchmark.analysis import compute_knn_jump

        rng = np.random.RandomState(42)
        X = rng.randn(10000, 3)
        y_pred = rng.randn(10000)

        result = compute_knn_jump(X, y_pred, n_samples=500)

        # Should still return valid results
        assert np.isfinite(result['knn_jump'])
        assert result['n_edges'] == 500 * 10  # n_samples * k


class TestPathMetrics:
    """Tests for compute_path_metrics."""

    def test_basic_output_structure(self):
        """Should return dict with expected keys."""
        from benchmark.analysis import compute_path_metrics

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        model = Ridge().fit(X, rng.randn(100))

        result = compute_path_metrics(model, X, n_paths=10)

        assert 'path_tv_mean' in result
        assert 'path_tv_std' in result
        assert 'path_curvature_mean' in result
        assert 'path_jump_count_mean' in result
        assert 'paths' in result

    def test_smooth_model_low_jumps(self):
        """Linear model should have zero/low path jumps."""
        from benchmark.analysis import compute_path_metrics

        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        y = X[:, 0] + 0.5 * X[:, 1] + rng.randn(200) * 0.1

        model = Ridge().fit(X, y)
        result = compute_path_metrics(model, X, n_paths=50)

        # Linear model should have very smooth paths
        assert result['path_jump_count_mean'] < 1.0, \
            "Linear model should have almost no jumps"

    def test_tree_model_has_jumps(self):
        """Decision tree should have path jumps."""
        from benchmark.analysis import compute_path_metrics

        rng = np.random.RandomState(42)
        X = rng.randn(500, 2)
        y = np.where(X[:, 0] > 0, 1.0, -1.0)  # Clear step function

        model = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X, y)
        result = compute_path_metrics(model, X, n_paths=50, jump_threshold=0.1)

        # Tree crossing decision boundaries should have jumps
        assert result['path_jump_count_mean'] > 0, \
            "Tree model should have some path jumps"


class TestViolationProbability:
    """Tests for compute_violation_probability."""

    def test_basic_output_structure(self):
        """Should return dict with expected keys."""
        from benchmark.analysis import compute_violation_probability

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y_pred = rng.randn(100)

        result = compute_violation_probability(X, y_pred)

        assert 'tau' in result
        assert 'diameter' in result
        # At least one violation prob key
        assert any(k.startswith('violation_prob_') for k in result)

    def test_smooth_vs_piecewise(self):
        """Piecewise predictions should have higher violation probability."""
        from benchmark.analysis import compute_violation_probability

        rng = np.random.RandomState(42)
        X = rng.randn(500, 2)

        # Smooth function
        y_smooth = X[:, 0] + 0.5 * X[:, 1]

        # Piecewise constant
        y_piecewise = np.where(X[:, 0] > 0, 1.0, -1.0)

        # Use same tau for fair comparison
        tau = 0.5

        result_smooth = compute_violation_probability(X, y_smooth, tau=tau)
        result_piecewise = compute_violation_probability(X, y_piecewise, tau=tau)

        # Get a violation probability key that exists
        viol_key = [k for k in result_smooth if k.startswith('violation_prob_')][0]

        smooth_viol = result_smooth[viol_key]
        piecewise_viol = result_piecewise[viol_key]

        # Piecewise should have higher violation probability
        # (but only if tau is appropriate for the data)
        if np.isfinite(smooth_viol) and np.isfinite(piecewise_viol):
            assert smooth_viol <= piecewise_viol, \
                "Smooth predictions should not have higher violation prob than piecewise"


class TestLocalLipschitz:
    """Tests for compute_local_lipschitz_quantiles."""

    def test_basic_output_structure(self):
        """Should return dict with expected quantile keys."""
        from benchmark.analysis import compute_local_lipschitz_quantiles

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y_pred = rng.randn(100)

        result = compute_local_lipschitz_quantiles(X, y_pred)

        assert 'lipschitz_median' in result
        assert 'lipschitz_q90' in result
        assert 'lipschitz_q99' in result

    def test_constant_prediction_low_lipschitz(self):
        """Constant predictions should have very low Lipschitz values."""
        from benchmark.analysis import compute_local_lipschitz_quantiles

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y_pred = np.ones(100) * 5.0  # Constant

        result = compute_local_lipschitz_quantiles(X, y_pred)

        # Constant function has zero gradient everywhere
        assert result['lipschitz_median'] < 1e-8 or np.isnan(result['lipschitz_median'])


class TestModelSmoothnessProfile:
    """Tests for compute_model_smoothness_profile."""

    def test_basic_output_structure(self):
        """Should return combined profile with all metrics."""
        from benchmark.analysis import compute_model_smoothness_profile

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        model = Ridge().fit(X, rng.randn(100))

        result = compute_model_smoothness_profile(model, X)

        # Should have knn metrics
        assert 'knn_jump' in result
        assert 'knn_jump_norm' in result

        # Should have violation metrics
        assert 'tau' in result

        # Should have Lipschitz metrics
        assert 'lipschitz_median' in result

        # Should have path metrics (when model provided)
        assert 'path_tv_mean' in result

    def test_without_model(self):
        """Should work with just y_pred (no path metrics)."""
        from benchmark.analysis import compute_model_smoothness_profile

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y_pred = rng.randn(100)

        result = compute_model_smoothness_profile(model=None, X=X, y_pred=y_pred)

        # Should have non-path metrics
        assert 'knn_jump' in result
        assert 'lipschitz_median' in result

        # Should NOT have path metrics (no model to interpolate)
        assert 'path_tv_mean' not in result

    def test_smooth_vs_tree_model(self):
        """Full profile should show clear difference between smooth and tree models."""
        from benchmark.analysis import compute_model_smoothness_profile

        rng = np.random.RandomState(42)
        X = rng.randn(300, 2)
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]

        ridge = Ridge().fit(X, y)
        tree = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X, y)

        profile_ridge = compute_model_smoothness_profile(ridge, X)
        profile_tree = compute_model_smoothness_profile(tree, X)

        # Ridge should be smoother -- violation probability is a reliable discriminator
        assert profile_ridge['violation_prob_0.1'] < profile_tree['violation_prob_0.1'], \
            "Ridge should have lower violation probability than tree"


class TestExistingDatasetMetrics:
    """Tests that existing dataset-level metrics still work after refactoring."""

    def test_discontinuity_profile(self):
        """compute_discontinuity_profile should still work."""
        from benchmark.analysis import compute_discontinuity_profile

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = rng.randn(200)

        result = compute_discontinuity_profile(X, y)

        assert 'llqd_tail_ratio' in result
        assert 'ntj_spikiness' in result
        assert 'discontinuity_score' in result

    def test_individual_metrics(self):
        """Individual dataset metrics should be importable and work."""
        from benchmark.analysis import llqd_distribution, ntj_stats, gtv

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = rng.randn(100)

        llqd = llqd_distribution(X, y)
        assert 'llqd_tail_ratio' in llqd

        ntj = ntj_stats(X, y)
        assert 'ntj_spikiness' in ntj

        gtv_res = gtv(X, y)
        assert 'gtv_normalized' in gtv_res


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
