import numpy as np
import pytest

from mcframework import MonteCarloFramework, MonteCarloSimulation, PiEstimationSimulation, PortfolioSimulation


class TestIntegration:
    """[FR-6, FR-20, FR-21] Integration tests for complete workflows."""

    def test_end_to_end_pi_estimation(self):
        """[FR-6, FR-20, FR-12] Test complete Pi estimation workflow."""
        fw = MonteCarloFramework()
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        fw.register_simulation(sim)
        result = fw.run_simulation(
            "Pi Estimation",
            1000,
            n_points=10000,
            backend="sequential",
            confidence=0.95,
            extra_context={"target": float(np.pi), "eps": 0.01}
        )

        # Verify result structure
        assert result.n_simulations == 1000
        assert len(result.results) == 1000
        assert "mean" in result.stats
        assert "ci_mean" in result.stats

        # Verify statistical validity
        ci = result.stats["ci_mean"]
        assert ci["low"] < np.pi < ci["high"]  # True value in CI

    def test_end_to_end_portfolio(self):
        """[FR-6, FR-21, FR-10] Test complete Portfolio workflow."""
        fw = MonteCarloFramework()
        sim = PortfolioSimulation()
        sim.set_seed(123)

        fw.register_simulation(sim)
        result = fw.run_simulation(
            "Portfolio Simulation",
            500,
            initial_value=10000,
            annual_return=0.07,
            volatility=0.2,
            years=10,
            backend="sequential",
            percentiles=[5, 95],
            eps=0.05,
        )
        

        assert result.n_simulations == 500
        assert result.mean > 0
        assert 5 in result.percentiles
        assert 95 in result.percentiles

    def test_multiple_simulations_comparison(self):
        """[FR-6, FR-7] Test running and comparing multiple simulations."""
        fw = MonteCarloFramework()

        pi_sim = PiEstimationSimulation()
        pi_sim.set_seed(42)

        port_sim = PortfolioSimulation()
        port_sim.set_seed(43)

        fw.register_simulation(pi_sim)
        fw.register_simulation(port_sim)

        # Run both
        fw.run_simulation("Pi Estimation", 100, n_points=5000, backend="sequential")
        fw.run_simulation("Portfolio Simulation", 100,
                          initial_value=10000, backend="sequential")

        # Compare means
        comparison = fw.compare_results(
            ["Pi Estimation", "Portfolio Simulation"],
            metric="mean"
        )

        assert len(comparison) == 2
        assert comparison["Pi Estimation"] < 10  # Pi ~3.14
        assert comparison["Portfolio Simulation"] > 1000

    def test_parallel_execution_consistency(self):
        """[FR-3, FR-4, NFR-3] Test parallel execution produces consistent results."""
        sim = PiEstimationSimulation()

        results = []
        for _ in range(3):
            sim.set_seed(42)
            result = sim.run(200, backend="auto", n_workers=2,
                             n_points=5000, compute_stats=False)
            results.append(result.mean)

        # All runs with same seed should produce similar results
        assert all(pytest.approx(r, abs=0.01) == results[0] for r in results)

    def test_large_scale_simulation(self):
        """[FR-3, NFR-1] Test framework handles large simulations."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        result = sim.run(
            10000,
            backend="auto",
            n_workers=2,
            n_points=1000,
            compute_stats=True,
            confidence=0.95,
            extra_context={"target": float(np.pi), "eps": 0.01}
        )
        ci = getattr(result, "stats")["ci_mean"]
        low, high = ci["low"], ci["high"]
        ci_width = high - low

        # Should have tight confidence interval with large n
        assert ci_width < 0.02  # Should be quite narrow
        assert result.n_simulations == 10000
        assert len(result.results) == 10000

    def test_custom_simulation_integration(self):
        """[FR-1, FR-6, USA-1] Test custom simulation integration."""

        class ExponentialSim(MonteCarloSimulation):
            def single_simulation(self, rate=1.0, **kwargs):
                return float(np.random.exponential(1.0 / rate))

        fw = MonteCarloFramework()
        sim = ExponentialSim(name="Exponential")
        sim.set_seed(42)

        fw.register_simulation(sim)
        result = fw.run_simulation("Exponential", 1000, rate=2.0, backend="sequential")

        # Exponential with rate=2 has mean=0.5
        assert pytest.approx(result.mean, abs=0.1) == 0.5

    def test_simulation_kwargs_passthrough(self):
        """[FR-1, FR-20] Test simulation kwargs are passed correctly."""
        sim = PiEstimationSimulation()
        sim.set_seed(42)

        # Test with different n_points
        result1 = sim.run(50, n_points=100, backend="sequential", compute_stats=False)
        result2 = sim.run(50, n_points=100000, backend="sequential", compute_stats=False)

        # More points should give better estimate
        error1 = abs(result1.mean - np.pi)
        error2 = abs(result2.mean - np.pi)
        assert error2 < error1
