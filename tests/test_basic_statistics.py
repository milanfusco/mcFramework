import numpy as np
import pytest

from mcframework.stats_engine import (
    NanPolicy,
    StatsContext,
    kurtosis,
    mean,
    percentiles,
    skew,
    std,
)


class TestBasicStatistics:
    """Test basic statistical functions"""

    def test_mean_simple(self, sample_data):
        """Test mean calculation"""
        ctx = StatsContext(n=len(sample_data))
        result = mean(sample_data, ctx)
        expected = np.mean(sample_data)
        assert pytest.approx(result) == expected

    def test_mean_with_nan_policy_omit(self):
        """Test mean with NaN values and omit policy"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ctx = StatsContext(n=len(data), nan_policy=NanPolicy.omit)
        result = mean(data, ctx)
        assert pytest.approx(result) == 3.0

    def test_std_simple(self, sample_data):
        """Test standard deviation calculation"""
        ctx = StatsContext(n=len(sample_data))
        result = std(sample_data, ctx)
        expected = np.std(sample_data, ddof=1)
        assert pytest.approx(result) == expected

    def test_std_single_value(self):
        """Test std returns None for single value (n_eff <= 1)"""
        data = np.array([5.0])
        ctx = StatsContext(n=len(data))
        result = std(data, ctx)
        assert result is None

    def test_std_empty_array(self):
        """Test std returns None for an empty array"""
        data = np.array([])
        ctx = StatsContext(n=len(data))
        result = std(data, ctx)
        assert result is None

    def test_percentiles_default(self, sample_data):
        """Test percentile calculation with defaults"""
        ctx = StatsContext(n=len(sample_data), percentiles=(25, 50, 75))
        result = percentiles(sample_data, ctx)
        assert 25 in result
        assert 50 in result
        assert 75 in result
        assert result[25] < result[50] < result[75]

    def test_percentiles_with_nan_policy(self):
        """Test percentiles with NaN handling"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ctx = StatsContext(n=len(data), percentiles=(50,), nan_policy=NanPolicy.omit)
        result = percentiles(data, ctx)
        assert result[50] == 3.0

    def test_skew_normal_distribution(self):
        """Test skew on approximately normal data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        ctx = StatsContext(n=len(data))
        result = skew(data, ctx)
        assert abs(result) < 0.1  # Should be close to 0

    def test_skew_small_sample(self):
        """Test skew returns 0 for very small samples"""
        data = np.array([1.0, 2.0])
        ctx = StatsContext(n=len(data))
        result = skew(data, ctx)
        assert result == 0.0

    def test_kurtosis_normal_distribution(self):
        """Test kurtosis on approximately normal data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        ctx = StatsContext(n=len(data))
        result = kurtosis(data, ctx)
        assert abs(result) < 0.2  # Excess kurtosis should be ~0

    def test_kurtosis_small_sample(self):
        """Test kurtosis returns 0 for very small samples"""
        data = np.array([1.0, 2.0, 3.0])
        ctx = StatsContext(n=len(data))
        result = kurtosis(data, ctx)
        assert result == 0.0

