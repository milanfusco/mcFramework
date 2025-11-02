import numpy as np

from mcframework.stats_engine import FnMetric, StatsEngine, build_default_engine, mean, std


class TestStatsEngine:
    """Test StatsEngine class"""

    def test_engine_creation(self):
        """Test creating a stats engine with metrics"""
        metrics = [
            FnMetric("mean", mean),
            FnMetric("std", std),
        ]
        engine = StatsEngine(metrics)
        assert len(engine._metrics) == 2

    def test_engine_compute(self, sample_data, ctx_basic):
        """Test computing all metrics"""
        metrics = [
            FnMetric("mean", mean),
            FnMetric("std", std),
        ]
        engine = StatsEngine(metrics)
        result = engine.compute(sample_data, **ctx_basic)
        assert "mean" in result
        assert "std" in result

    def test_default_engine_build(self):
        """Test building default engine"""
        engine = build_default_engine()
        assert engine is not None
        assert len(engine._metrics) > 0

    def test_default_engine_compute(self, sample_data, ctx_basic):
        """Test default engine computes all metrics"""
        engine = build_default_engine()
        result = engine.compute(sample_data, **ctx_basic)
        assert "mean" in result
        assert "std" in result
        assert "percentiles" in result
        assert "ci_mean" in result

    def test_engine_without_dist_free(self):
        """Test building engine without distribution-free metrics"""
        engine = build_default_engine(include_dist_free=False)
        result = engine.compute(np.array([1, 2, 3]), n=3, confidence=0.95, target=0.0, eps=0.05)
        assert "ci_mean_chebyshev" not in result

    def test_engine_without_target_bounds(self):
        """Test building engine without target bounds"""
        engine = build_default_engine(include_target_bounds=False)
        result = engine.compute(np.array([1, 2, 3]), n=3, confidence=0.95, target=0.0, eps=0.05)
        assert "bias_to_target" not in result

