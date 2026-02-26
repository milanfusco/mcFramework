import numpy as np
import pytest

from mcframework.sims import PiEstimationSimulation, PortfolioSimulation


class TestRegression:
    """Tests for known issues or edge cases discovered"""

    def test_seed_reproducibility_across_runs(self):
        """Regression: Ensure seeds produce identical results"""
        sim1 = PiEstimationSimulation()
        sim1.set_seed(12345)
        result1 = sim1.run(100, n_points=5000, backend="sequential", compute_stats=False)

        sim2 = PiEstimationSimulation()
        sim2.set_seed(12345)
        result2 = sim2.run(100, n_points=5000, backend="sequential", compute_stats=False)

        np.testing.assert_array_equal(result1.results, result2.results)

    def test_stats_engine_percentile_merge(self):
        """Regression: Stats engine percentiles merge with requested percentiles"""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            100,
            backend="sequential",
            percentiles=[10,50,90],
            compute_stats=True,
            eps=0.05,
        )

        # Should have requested percentiles
        assert 10 in result.percentiles
        assert 90 in result.percentiles
        assert 50 in result.percentiles

    def test_portfolio_gbm_vs_non_gbm_consistency(self):
        """Regression: GBM and non-GBM should give similar expected values"""
        sim = PortfolioSimulation()

        sim.set_seed(42)
        result_gbm = sim.run(
            500,
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=5,
            use_gbm=True,
            backend="sequential",
            compute_stats=False
        )

        sim.set_seed(42)
        result_non_gbm = sim.run(
            500,
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=5,
            use_gbm=False,
            backend="sequential",
            compute_stats=False
        )

        # Means should be reasonably close
        assert pytest.approx(result_gbm.mean, rel=0.2) == result_non_gbm.mean
