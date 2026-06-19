import numpy as np
import pytest

from mcframework.stats_engine import (
    StatsContext,
    bias_to_target,
    chebyshev_required_n,
    markov_error_prob,
    mse_to_target,
)


class TestDistributionFreeMetrics:
    """[FR-15, FR-16] Test distribution-free statistical bounds."""

    def test_chebyshev_required_n(self, sample_data):
        """[FR-15] Test required sample size calculation."""
        ctx = {"eps": 0.1, "confidence": 0.95}
        result = chebyshev_required_n(sample_data, ctx)
        assert result is not None
        assert isinstance(result, int)
        assert result > 0

    def test_chebyshev_required_n_no_eps(self, sample_data):
        """[FR-15, USA-4] Test raises ValueError when eps not provided."""
        ctx = {"confidence": 0.95}
        with pytest.raises(ValueError, match=r"chebyshev_required_n requires ctx\.eps"):
            chebyshev_required_n(sample_data, ctx)

    def test_markov_error_prob(self):
        """[FR-16] Test Markov inequality error probability."""
        np.random.seed(42)
        data = np.random.normal(3.14159, 0.1, 1000)
        ctx = {"n": 1000, "target": 3.14159, "eps": 0.05}
        result = markov_error_prob(data, ctx)
        assert result is not None
        assert result >= 0  # Markov bound is non-negative (can be > 1)

    def test_markov_error_prob_no_target(self, sample_data):
        """[FR-16, USA-4] Test raises ValueError when target not provided."""
        ctx = {"n": 1000, "eps": 0.05}
        with pytest.raises(ValueError, match=r"markov_error_prob requires ctx\.target"):
            markov_error_prob(sample_data, ctx)

    def test_bias_to_target(self):
        """[FR-16] Test bias calculation to known target."""
        data = np.array([3.2, 3.15, 3.13, 3.16])
        ctx = {"target": np.pi}
        result = bias_to_target(data, ctx)
        expected_bias = np.mean(data) - np.pi
        assert pytest.approx(result) == expected_bias

    def test_mse_to_target(self):
        """[FR-16] Test MSE calculation to known target."""
        np.random.seed(42)
        data = np.random.normal(3.14159, 0.1, 100)
        ctx = {"n": 100, "target": np.pi}
        result = mse_to_target(data, ctx)
        assert result is not None
        assert result >= 0  # MSE is always non-negative

    def test_mse_to_target_is_estimator_mse(self):
        """[FR-16] MSE of the sample mean = s^2/n_eff + bias^2 (not per-observation)."""
        # arr=[1..5], target=3 => mean=3 (zero bias), sample var (ddof=1)=2.5, n=5
        # MSE(X_bar) = 2.5/5 + 0 = 0.5  (the per-observation mean sq dev would be 2.0)
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = StatsContext(n=5, target=3.0)
        assert mse_to_target(arr, ctx) == pytest.approx(0.5)

    def test_mse_to_target_constant_data_is_squared_bias(self):
        """[FR-16] Zero-variance data => MSE collapses to bias^2."""
        arr = np.full(4, 2.0)
        ctx = StatsContext(n=4, target=0.0)
        assert mse_to_target(arr, ctx) == pytest.approx(4.0)

    def test_markov_bound_scales_with_n(self):
        """[FR-16] Markov bound on |X_bar - target| must shrink with n (uses s^2/n)."""
        # Same per-sample spread, different n: the sample-mean error bound must fall ~1/n.
        rng = np.random.default_rng(0)
        small = rng.normal(0.0, 1.0, 100)
        large = rng.normal(0.0, 1.0, 10_000)
        eps = 0.5
        b_small = markov_error_prob(small, StatsContext(n=small.size, target=0.0, eps=eps))
        b_large = markov_error_prob(large, StatsContext(n=large.size, target=0.0, eps=eps))
        # Larger n => tighter (smaller) bound on the mean's error.
        assert b_large < b_small

    def test_markov_equals_mse_over_eps_squared(self):
        """[FR-16] markov_error_prob == mse_to_target / eps^2 (shared definition)."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ctx = StatsContext(n=5, target=3.0, eps=2.0)
        expected = mse_to_target(arr, ctx) / (2.0**2)
        assert markov_error_prob(arr, ctx) == pytest.approx(expected)

    def test_bias_to_target_empty_data_after_cleaning(self):
        """[NFR-4] Test bias_to_target raises InsufficientDataError with empty data after cleaning."""
        from mcframework.stats_engine import InsufficientDataError, NanPolicy, StatsContext

        # All NaN data with omit policy results in empty array
        data = np.array([np.nan, np.nan, np.nan])
        ctx = StatsContext(n=3, target=1.0, nan_policy=NanPolicy.omit)
        with pytest.raises(InsufficientDataError, match="non-empty data"):
            bias_to_target(data, ctx)
