import numpy as np
import pytest

from mcframework.stats_engine import (
    bias_to_target,
    chebyshev_required_n,
    markov_error_prob,
    mse_to_target,
)


class TestDistributionFreeMetrics:
    """Test distribution-free statistical bounds"""

    def test_chebyshev_required_n(self, sample_data):
        """Test required sample size calculation"""
        ctx = {"eps": 0.1, "confidence": 0.95}
        result = chebyshev_required_n(sample_data, ctx)
        assert result is not None
        assert isinstance(result, int)
        assert result > 0

    def test_chebyshev_required_n_no_eps(self, sample_data):
        """Test returns None when eps not provided"""
        ctx = {"confidence": 0.95}
        result = chebyshev_required_n(sample_data, ctx)
        assert result is None

    def test_markov_error_prob(self):
        """Test Markov inequality error probability"""
        np.random.seed(42)
        data = np.random.normal(3.14159, 0.1, 1000)
        ctx = {"n": 1000, "target": 3.14159, "eps": 0.05}
        result = markov_error_prob(data, ctx)
        assert result is not None
        assert 0 <= result <= 1

    def test_markov_error_prob_no_target(self, sample_data):
        """Test returns None when target not provided"""
        ctx = {"n": 1000, "eps": 0.05}
        result = markov_error_prob(sample_data, ctx)
        assert result is None

    def test_bias_to_target(self):
        """Test bias calculation"""
        data = np.array([3.2, 3.15, 3.13, 3.16])
        ctx = {"target": np.pi}
        result = bias_to_target(data, ctx)
        expected_bias = np.mean(data) - np.pi
        assert pytest.approx(result) == expected_bias

    def test_mse_to_target(self):
        """Test MSE calculation"""
        np.random.seed(42)
        data = np.random.normal(3.14159, 0.1, 100)
        ctx = {"n": 100, "target": np.pi}
        result = mse_to_target(data, ctx)
        assert result is not None
        assert result >= 0  # MSE is always non-negative
