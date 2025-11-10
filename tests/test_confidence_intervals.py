import math

import numpy as np

from mcframework.stats_engine import ci_mean, ci_mean_chebyshev


class TestConfidenceIntervals:
    """Test confidence interval calculations"""

    def test_ci_mean_basic(self, sample_data, ctx_basic):
        """Test basic CI calculation for mean"""
        result = ci_mean(sample_data, ctx_basic)
        assert result is not None
        assert "low" in result
        assert "high" in result
        assert "se" in result
        assert "confidence" in result
        assert result["low"] < result["high"]
        assert result["confidence"] == 0.95

    def test_ci_mean_contains_true_mean(self, sample_data, ctx_basic):
        """Test CI contains sample mean"""
        result = ci_mean(sample_data, ctx_basic)
        sample_mean = np.mean(sample_data)
        assert result["low"] < sample_mean < result["high"]

    def test_ci_mean_small_sample(self):
        """Test CI returns None for n < 2"""
        data = np.array([5.0])
        ctx = {"n": 1, "confidence": 0.95, "ci_method": "auto"}
        result = ci_mean(data, ctx)
        assert math.isnan(result["low"])
        assert math.isnan(result["high"])

    def test_ci_mean_chebyshev(self, sample_data):
        """Test Chebyshev CI calculation"""
        ctx = {"n": len(sample_data), "confidence": 0.95, "nan_policy": "propagate"}
        result = ci_mean_chebyshev(sample_data, ctx)
        assert result is not None
        assert result["method"] == "chebyshev"
        assert result["low"] < result["high"]

    def test_ci_mean_chebyshev_wider_than_normal(self, sample_data, ctx_basic):
        """Test Chebyshev CI is wider (distribution-free penalty)"""
        normal_ci = ci_mean(sample_data, ctx_basic)
        cheby_ci = ci_mean_chebyshev(sample_data, ctx_basic)

        normal_width = normal_ci["high"] - normal_ci["low"]
        cheby_width = cheby_ci["high"] - cheby_ci["low"]

        assert cheby_width > normal_width
