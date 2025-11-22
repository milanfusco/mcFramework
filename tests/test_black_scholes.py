"""
Tests for Black-Scholes option pricing simulations.

This module contains comprehensive tests for:
- European and American option pricing
- Helper functions (_european_payoff, _simulate_gbm_path, _american_exercise_lsm)
- Greeks calculation
- Path simulation
- Error handling and edge cases
"""

import numpy as np
import pytest

from mcframework.sims import (
    BlackScholesPathSimulation,
    BlackScholesSimulation,
    _american_exercise_lsm,
    _european_payoff,
    _simulate_gbm_path,
)

# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestEuropeanPayoff:
    """Test _european_payoff helper function."""

    def test_call_option_itm(self):
        """Test call option in-the-money."""
        payoff = _european_payoff(S_T=110.0, K=100.0, option_type="call")
        assert payoff == 10.0

    def test_call_option_otm(self):
        """Test call option out-of-the-money."""
        payoff = _european_payoff(S_T=90.0, K=100.0, option_type="call")
        assert payoff == 0.0

    def test_call_option_atm(self):
        """Test call option at-the-money."""
        payoff = _european_payoff(S_T=100.0, K=100.0, option_type="call")
        assert payoff == 0.0

    def test_put_option_itm(self):
        """Test put option in-the-money."""
        payoff = _european_payoff(S_T=90.0, K=100.0, option_type="put")
        assert payoff == 10.0

    def test_put_option_otm(self):
        """Test put option out-of-the-money."""
        payoff = _european_payoff(S_T=110.0, K=100.0, option_type="put")
        assert payoff == 0.0

    def test_put_option_atm(self):
        """Test put option at-the-money."""
        payoff = _european_payoff(S_T=100.0, K=100.0, option_type="put")
        assert payoff == 0.0

    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _european_payoff(S_T=100.0, K=100.0, option_type="invalid")


class TestSimulateGBMPath:
    """Test _simulate_gbm_path helper function."""

    def test_path_starts_at_S0(self):
        """Test that path starts at initial price."""
        rng = np.random.default_rng(42)
        path = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng)
        np.testing.assert_allclose(path[0], 100.0, rtol=1e-10)

    def test_path_length(self):
        """Test that path has correct length."""
        rng = np.random.default_rng(42)
        n_steps = 100
        path = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=n_steps, rng=rng)
        assert len(path) == n_steps + 1

    def test_path_is_positive(self):
        """Test that all prices in path are positive."""
        rng = np.random.default_rng(42)
        path = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng)
        assert np.all(path > 0)

    def test_path_reproducibility(self):
        """Test that same seed produces same path."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        path1 = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng1)
        path2 = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng2)
        np.testing.assert_array_equal(path1, path2)

    def test_path_different_seeds(self):
        """Test that different seeds produce different paths."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        path1 = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng1)
        path2 = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, rng=rng2)
        assert not np.array_equal(path1, path2)


class TestAmericanExerciseLSM:
    """Test _american_exercise_lsm helper function."""

    def test_american_put_pricing(self):
        """Test American put option pricing with LSM."""
        # Generate paths for testing
        rng = np.random.default_rng(42)
        n_paths = 100
        n_steps = 50
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            paths[i] = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=n_steps, rng=rng)

        price = _american_exercise_lsm(paths, K=100.0, r=0.05, dt=1.0 / n_steps, option_type="put")

        # American put should have positive value
        assert price > 0
        # Should be reasonable (between 0 and strike)
        assert 0 < price < 100.0

    def test_american_call_pricing(self):
        """Test American call option pricing with LSM."""
        rng = np.random.default_rng(42)
        n_paths = 100
        n_steps = 50
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            paths[i] = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=n_steps, rng=rng)

        price = _american_exercise_lsm(paths, K=100.0, r=0.05, dt=1.0 / n_steps, option_type="call")

        # American call should have positive value
        assert price >= 0

    def test_invalid_option_type_lsm(self):
        """Test that invalid option type raises ValueError in LSM."""
        rng = np.random.default_rng(42)
        paths = np.zeros((10, 11))
        for i in range(10):
            paths[i] = _simulate_gbm_path(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=10, rng=rng)

        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            _american_exercise_lsm(paths, K=100.0, r=0.05, dt=0.1, option_type="invalid")

    def test_deep_otm_options(self):
        """Test LSM with deep out-of-the-money options."""
        # Create paths that stay well above strike for puts
        n_paths = 50
        n_steps = 20
        paths = np.ones((n_paths, n_steps + 1)) * 200.0  # All paths at $200

        price = _american_exercise_lsm(paths, K=100.0, r=0.05, dt=1.0 / n_steps, option_type="put")

        # Deep OTM put should be worthless
        assert price == 0.0


# =============================================================================
# Tests for BlackScholesSimulation
# =============================================================================


class TestBlackScholesSimulation:
    """Test BlackScholesSimulation class."""

    def test_initialization(self):
        """Test simulation initialization."""
        sim = BlackScholesSimulation()
        assert sim.name == "Black-Scholes Option Pricing"

        sim_custom = BlackScholesSimulation(name="Custom BS")
        assert sim_custom.name == "Custom BS"

    def test_european_call_single_simulation(self):
        """Test single European call option simulation."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        price = sim.single_simulation(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="european"
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_european_put_single_simulation(self):
        """Test single European put option simulation."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        price = sim.single_simulation(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="european"
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_american_call_single_simulation(self):
        """Test single American call option simulation."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        price = sim.single_simulation(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="american"
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_american_put_single_simulation(self):
        """Test single American put option simulation."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        price = sim.single_simulation(
            S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="american"
        )

        assert isinstance(price, float)
        assert price >= 0

    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            sim.single_simulation(
                S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="invalid", exercise_type="european"
            )

    def test_invalid_exercise_type(self):
        """Test that invalid exercise type raises ValueError."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        with pytest.raises(ValueError, match="exercise_type must be 'european' or 'american'"):
            sim.single_simulation(
                S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="invalid"
            )

    def test_european_call_run(self):
        """Test running multiple European call simulations."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        result = sim.run(
            1000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="european"
        )

        assert result.n_simulations == 1000
        assert result.mean > 0
        assert result.std > 0
        assert len(result.results) == 1000

    def test_american_put_run(self):
        """Test running multiple American put simulations."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        result = sim.run(
            1000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="american"
        )

        assert result.n_simulations == 1000
        assert result.mean > 0
        assert len(result.results) == 1000

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        sim1 = BlackScholesSimulation()
        sim1.set_seed(42)
        result1 = sim1.run(
            100, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="european"
        )

        sim2 = BlackScholesSimulation()
        sim2.set_seed(42)
        result2 = sim2.run(
            100, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="european"
        )

        np.testing.assert_array_equal(result1.results, result2.results)


class TestCalculateGreeks:
    """Test Greeks calculation."""

    def test_calculate_greeks_basic(self):
        """Test basic Greeks calculation."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        greeks = sim.calculate_greeks(
            n_simulations=1000,
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.20,
            option_type="call",
            exercise_type="european",
        )

        # Check all Greeks are present
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        # Check reasonable values
        assert greeks["price"] > 0
        assert 0 < greeks["delta"] < 1  # Call delta should be between 0 and 1
        assert greeks["gamma"] > 0  # Gamma should be positive
        assert greeks["vega"] > 0  # Vega should be positive
        # Theta is typically negative for long options
        assert greeks["rho"] > 0  # Rho should be positive for calls

    def test_calculate_greeks_put(self):
        """Test Greeks calculation for put options."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        greeks = sim.calculate_greeks(
            n_simulations=1000,
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.20,
            option_type="put",
            exercise_type="european",
        )

        assert greeks["price"] > 0
        assert -1 < greeks["delta"] < 0  # Put delta should be negative
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_calculate_greeks_custom_bumps(self):
        """Test Greeks with custom bump percentages."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        greeks = sim.calculate_greeks(
            n_simulations=500,
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.20,
            option_type="call",
            exercise_type="european",
            bump_pct=0.02,  # 2% bump
            time_bump_days=7.0,  # 7 days
        )

        assert all(k in greeks for k in ["price", "delta", "gamma", "vega", "theta", "rho"])

    def test_calculate_greeks_parallel(self):
        """Test Greeks calculation with parallel execution."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        greeks = sim.calculate_greeks(
            n_simulations=1000,
            S0=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            sigma=0.20,
            option_type="call",
            exercise_type="european",
            parallel=True,
        )

        assert all(isinstance(greeks[k], float) for k in greeks.keys())

    def test_calculate_greeks_near_expiry(self):
        """Test Greeks calculation near expiry."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        # Use time < 1 day (< 1/365) to trigger theta = 0 case
        greeks = sim.calculate_greeks(
            n_simulations=500,
            S0=100.0,
            K=100.0,
            T=0.002,
            r=0.05,
            sigma=0.20,  # Less than 1 day
            option_type="call",
            exercise_type="european",
        )

        # Theta should be 0 when T < time_bump_days/365
        assert greeks["theta"] == 0.0


# =============================================================================
# Tests for BlackScholesPathSimulation
# =============================================================================


class TestBlackScholesPathSimulation:
    """Test BlackScholesPathSimulation class."""

    def test_initialization(self):
        """Test path simulation initialization."""
        sim = BlackScholesPathSimulation()
        assert sim.name == "Black-Scholes Path Simulation"

        sim_custom = BlackScholesPathSimulation(name="Custom Path")
        assert sim_custom.name == "Custom Path"

    def test_single_simulation(self):
        """Test single path simulation returns final price."""
        sim = BlackScholesPathSimulation()
        sim.set_seed(42)

        final_price = sim.single_simulation(S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252)

        assert isinstance(final_price, float)
        assert final_price > 0

    def test_run_multiple_paths(self):
        """Test running multiple path simulations."""
        sim = BlackScholesPathSimulation()
        sim.set_seed(42)

        result = sim.run(1000, S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252)

        assert result.n_simulations == 1000
        assert result.mean > 0
        assert len(result.results) == 1000
        assert np.all(result.results > 0)

    def test_simulate_paths(self):
        """Test simulate_paths method."""
        sim = BlackScholesPathSimulation()
        sim.set_seed(42)

        paths = sim.simulate_paths(n_paths=50, S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=100)

        assert paths.shape == (50, 101)  # n_paths x (n_steps + 1)
        np.testing.assert_allclose(paths[:, 0], 100.0, rtol=1e-10)  # All start at S0
        assert np.all(paths > 0)  # All prices positive

    def test_simulate_paths_reproducibility(self):
        """Test that simulate_paths is reproducible with same seed."""
        sim1 = BlackScholesPathSimulation()
        sim1.set_seed(42)
        paths1 = sim1.simulate_paths(n_paths=20, S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=50)

        sim2 = BlackScholesPathSimulation()
        sim2.set_seed(42)
        paths2 = sim2.simulate_paths(n_paths=20, S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=50)

        np.testing.assert_array_equal(paths1, paths2)

    def test_simulate_paths_different_parameters(self):
        """Test simulate_paths with different parameters."""
        sim = BlackScholesPathSimulation()
        sim.set_seed(42)

        # High volatility
        paths_high_vol = sim.simulate_paths(n_paths=100, S0=100.0, r=0.05, sigma=0.50, T=1.0, n_steps=252)

        # Low volatility
        sim.set_seed(42)
        paths_low_vol = sim.simulate_paths(n_paths=100, S0=100.0, r=0.05, sigma=0.10, T=1.0, n_steps=252)

        # High volatility should have more spread in final prices
        high_vol_std = np.std(paths_high_vol[:, -1])
        low_vol_std = np.std(paths_low_vol[:, -1])
        assert high_vol_std > low_vol_std

    def test_parallel_execution(self):
        """Test parallel execution of path simulation."""
        sim = BlackScholesPathSimulation()
        sim.set_seed(42)

        result = sim.run(10000, S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252, parallel=True)

        assert result.n_simulations == 10000
        assert result.execution_time > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBlackScholesIntegration:
    """Integration tests for Black-Scholes simulations."""

    def test_put_call_parity_european(self):
        """Test put-call parity for European options."""
        sim = BlackScholesSimulation()
        sim.set_seed(42)

        # Price call
        call_result = sim.run(
            5000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="call", exercise_type="european"
        )

        # Price put with same seed for consistency
        sim.set_seed(42)
        put_result = sim.run(
            5000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="european"
        )

        # Put-call parity: C - P = S0 - K*e^(-rT)
        S0, K, r, T = 100.0, 100.0, 0.05, 1.0
        parity_lhs = call_result.mean - put_result.mean
        parity_rhs = S0 - K * np.exp(-r * T)

        # Allow for Monte Carlo error (within a few standard errors)
        combined_se = np.sqrt(call_result.std**2 + put_result.std**2) / np.sqrt(5000)
        assert abs(parity_lhs - parity_rhs) < 5 * combined_se

    def test_american_vs_european_inequality(self):
        """Test that American option >= European option (early exercise premium)."""
        sim = BlackScholesSimulation()

        # European put
        sim.set_seed(42)
        european_result = sim.run(
            2000, S0=100.0, K=110.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="european"
        )

        # American put
        sim.set_seed(42)
        american_result = sim.run(
            2000, S0=100.0, K=110.0, T=1.0, r=0.05, sigma=0.20, option_type="put", exercise_type="american"
        )

        # American should be >= European (early exercise premium)
        # Allow small Monte Carlo error
        assert american_result.mean >= european_result.mean - 0.5

    def test_increasing_volatility_increases_option_value(self):
        """Test that higher volatility leads to higher option prices."""
        sim = BlackScholesSimulation()

        # Low volatility
        sim.set_seed(42)
        low_vol = sim.run(
            2000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.10, option_type="call", exercise_type="european"
        )

        # High volatility
        sim.set_seed(42)
        high_vol = sim.run(
            2000, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.30, option_type="call", exercise_type="european"
        )

        assert high_vol.mean > low_vol.mean

    def test_option_price_vs_analytical(self):
        """Test that Monte Carlo price is close to analytical Black-Scholes."""
        from scipy.stats import norm

        # Analytical Black-Scholes for European call
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        analytical = black_scholes_call(S0, K, T, r, sigma)

        # Monte Carlo price
        sim = BlackScholesSimulation()
        sim.set_seed(42)
        result = sim.run(
            10000, S0=S0, K=K, T=T, r=r, sigma=sigma, option_type="call", exercise_type="european"
        )

        # Should be within a few standard errors
        se = result.std / np.sqrt(10000)
        assert abs(result.mean - analytical) < 3 * se
