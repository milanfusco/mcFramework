import numpy as np

from mcframework.stats_engine import ComputeResult, FnMetric, StatsEngine, build_default_engine, mean, std


class TestStatsEngine:
    """[FR-17] Test StatsEngine class."""

    def test_engine_creation(self):
        """[FR-17] Test creating a stats engine with metrics."""
        metrics = [
            FnMetric("mean", mean),
            FnMetric("std", std),
        ]
        engine = StatsEngine(metrics)
        assert len(engine._metrics) == 2

    def test_engine_compute(self, sample_data, ctx_basic):
        """[FR-8, FR-9, FR-17] Test computing all metrics."""
        metrics = [
            FnMetric("mean", mean),
            FnMetric("std", std),
        ]
        engine = StatsEngine(metrics)
        result = engine.compute(sample_data, **ctx_basic)
        assert isinstance(result, ComputeResult)
        assert "mean" in result.metrics
        assert "std" in result.metrics

    def test_default_engine_build(self):
        """[FR-17] Test building default engine."""
        engine = build_default_engine()
        assert engine is not None
        assert len(engine._metrics) > 0

    def test_default_engine_compute(self, sample_data, ctx_basic):
        """[FR-8, FR-9, FR-10, FR-12, FR-17] Test default engine computes all metrics."""
        engine = build_default_engine()
        result = engine.compute(sample_data, **ctx_basic)
        assert isinstance(result, ComputeResult)
        assert "mean" in result.metrics
        assert "std" in result.metrics
        assert "percentiles" in result.metrics
        assert "ci_mean" in result.metrics

    def test_engine_without_dist_free(self):
        """[FR-17] Test building engine without distribution-free metrics."""
        engine = build_default_engine(include_dist_free=False)
        result = engine.compute(np.array([1, 2, 3]), n=3, confidence=0.95, target=0.0, eps=0.05)
        assert isinstance(result, ComputeResult)
        assert "ci_mean_chebyshev" not in result.metrics

    def test_engine_without_target_bounds(self):
        """[FR-17] Test building engine without target bounds."""
        engine = build_default_engine(include_target_bounds=False)
        result = engine.compute(np.array([1, 2, 3]), n=3, confidence=0.95, target=0.0, eps=0.05)
        assert isinstance(result, ComputeResult)
        assert "bias_to_target" not in result.metrics

