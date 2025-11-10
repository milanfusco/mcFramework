from unittest.mock import Mock

import numpy as np
import pytest

from mcframework.sims import PortfolioSimulation
from mcframework.stats_engine import StatsEngine, ci_mean, mean


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_simulation_run(self, simple_simulation):
        """Test running exactly one simulation"""
        result = simple_simulation.run(1, parallel=False, compute_stats=False)
        assert result.n_simulations == 1
        assert len(result.results) == 1
        assert result.std == 0.0  # Only one value

    def test_very_small_confidence_interval(self):
        """Test extreme confidence levels"""
        data = np.random.normal(0, 1, 1000)
        ctx = {"n": 1000, "confidence": 0.5, "ci_method": "z"}
        result = ci_mean(data, ctx)
        assert result is not None
        assert result["confidence"] == 0.5

    def test_very_high_confidence_interval(self):
        """Test extreme confidence levels"""
        data = np.random.normal(0, 1, 1000)
        ctx = {"n": 1000, "confidence": 0.9999, "ci_method": "z"}
        result = ci_mean(data, ctx)
        assert result is not None
        assert (result["high"] - result["low"]) > 0.2

    def test_negative_portfolio_values_possible(self):
        """Test portfolio can theoretically go negative with extreme volatility"""
        sim = PortfolioSimulation()
        sim.set_seed(42)
        # Extreme volatility with negative return
        result = sim.run(
            1000,
            initial_value=10000,
            annual_return=-0.5,
            volatility=1.0,
            years=10,
            parallel=False,
            compute_stats=False
        )
        # Mean should be much lower than initial
        assert result.mean < 10000

    def test_zero_year_portfolio(self):
        """Test portfolio with zero years"""
        sim = PortfolioSimulation()
        # With 0 years, should return approximately initial value
        result = sim.single_simulation(
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=0
        )
        # With 0 time steps, should be close to initial
        assert pytest.approx(result, abs=1) == 10000

    def test_nan_handling_in_stats(self):
        """Test stats engine handles NaN values"""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ctx = {"n": 5, "nan_policy": "omit", "confidence": 0.95, "ci_method": "auto"}

        mean_result = mean(data, ctx)
        assert not np.isnan(mean_result)
        assert pytest.approx(mean_result) == 3.0

    def test_empty_percentiles_list(self, simple_simulation):
        """Test running with no percentiles"""
        result = simple_simulation.run(
            100,
            parallel=False,
            percentiles=[],
            compute_stats=False
        )
        # Should still work, just no percentiles
        assert len(result.percentiles) == 0

    def test_duplicate_percentiles(self, simple_simulation):
        """Test duplicate percentiles are handled"""
        result = simple_simulation.run(
            100,
            parallel=False,
            percentiles=[50, 50, 50],
            compute_stats=False
        )
        # Should deduplicate
        assert 50 in result.percentiles


class TestErrorHandling:
    """Test error handling and validation"""

    def test_invalid_simulation_name_in_framework(self, framework):
        """Test accessing non-existent simulation"""
        with pytest.raises(ValueError, match="not found"):
            framework.run_simulation("DoesNotExist", 100)

    def test_negative_n_simulations(self, simple_simulation):
        """Test negative n_simulations raises error"""
        with pytest.raises(ValueError):
            simple_simulation.run(-10)

    def test_zero_n_simulations(self, simple_simulation):
        """Test zero n_simulations raises error"""
        with pytest.raises(ValueError):
            simple_simulation.run(0)

    def test_invalid_percentile_metric_comparison(self, framework, simple_simulation):
        """Test requesting non-computed percentile"""
        framework.register_simulation(simple_simulation)
        framework.run_simulation("TestSim", 100, percentiles=[25, 75], parallel=False)

        with pytest.raises(ValueError, match="Percentile .* not computed"):
            framework.compare_results(["TestSim"], metric="p50")

    def test_stats_engine_with_empty_metrics(self):
        """Test stats engine with no metrics"""
        engine = StatsEngine([])
        result = engine.compute(np.array([1, 2, 3]), n=3)
        assert len(result) == 0

    def test_simulation_with_failed_stats(self, simple_simulation, monkeypatch):
        """Test simulation continues if stats engine fails"""

        def failing_compute(*args, **kwargs):
            raise RuntimeError("Stats computation failed")

        # Mock the stats engine
        failing_engine = Mock()
        failing_engine.compute = failing_compute

        # Should not crash, falls back to baseline stats
        result = simple_simulation.run(
            100,
            parallel=False,
            compute_stats=True,
            stats_engine=failing_engine
        )

        assert result.n_simulations == 100
        # Should have baseline stats even though engine failed
        assert len(result.stats) > 0
        assert "mean" in result.stats
        assert "ci_mean" in result.stats
