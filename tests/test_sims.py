import numpy as np
import pytest

from mcframework.sims import PiEstimationSimulation, PortfolioSimulation


class TestPiEstimationSimulation:
    """[FR-20] Test Pi estimation simulation."""

    def test_pi_estimation_initialization(self):
        """[FR-20] Test Pi simulation initializes correctly."""
        sim = PiEstimationSimulation()
        assert sim.name == "Pi Estimation"

    def test_pi_estimation_single_run(self):
        """[FR-20] Test single Pi estimation."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        result = sim.single_simulation(n_points=10000)
        assert 2.5 < result < 3.5  # Rough range check

    def test_pi_estimation_convergence(self):
        """[FR-20] Test Pi estimation converges with more points."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result_small = sim.single_simulation(n_points=100)
        sim.set_seed(42)
        result_large = sim.single_simulation(n_points=100000)

        # Larger sample should be closer to pi
        error_small = abs(result_small - np.pi)
        error_large = abs(result_large - np.pi)
        assert error_large < error_small

    def test_pi_estimation_antithetic(self):
        """[FR-20] Test antithetic sampling variance reduction."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        result = sim.single_simulation(n_points=10000, antithetic=True)
        assert 2.5 < result < 3.5

    def test_pi_estimation_full_run(self):
        """[FR-20] Test full Pi estimation run."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)
        result = sim.run(100, backend="sequential", n_points=5000, compute_stats=False)

        # Mean should be close to pi
        assert pytest.approx(result.mean, abs=0.1) == np.pi


class TestPortfolioSimulation:
    """[FR-21] Test Portfolio simulation."""

    def test_portfolio_initialization(self):
        """[FR-21] Test Portfolio simulation initializes correctly."""
        sim = PortfolioSimulation()
        assert sim.name == "Portfolio Simulation"

    def test_portfolio_single_run_basic(self):
        """[FR-21] Test single portfolio simulation."""
        sim = PortfolioSimulation()
        sim.set_seed(42)
        result = sim.single_simulation(
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=10
        )
        assert result > 0  # Value should be positive

    def test_portfolio_positive_return_on_average(self):
        """[FR-21] Test portfolio grows on average with positive return."""
        sim = PortfolioSimulation()
        sim.set_seed(42)
        result = sim.run(
            1000,
            backend="sequential",
            initial_value=10000,
            annual_return=0.10,
            volatility=0.1,
            years=5,
            compute_stats=False
        )
        # Mean should be greater than initial value
        assert result.mean > 10000

    def test_portfolio_non_gbm(self):
        """[FR-21] Test portfolio with non-GBM mode."""
        sim = PortfolioSimulation()
        sim.set_seed(42)
        result = sim.single_simulation(
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=10,
            use_gbm=False
        )
        assert result > 0

    def test_portfolio_zero_volatility(self):
        """[FR-21] Test portfolio with zero volatility."""
        sim = PortfolioSimulation()
        sim.set_seed(42)
        result = sim.run(
            10,
            backend="sequential",
            initial_value=10000,
            annual_return=0.05,
            volatility=0.0,
            years=1,
            compute_stats=False
        )
        # With zero volatility, all outcomes should be similar
        assert result.std < 100  # Very low variance

