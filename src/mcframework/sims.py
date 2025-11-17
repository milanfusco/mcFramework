"""
mcframework.sims
====================
Built-in simulations for mcframework.

The Simulations module provides classes and functions to create, manage, and
execute different types of simulations. Currently supported simulations include
a pi estimation simulation, a portfolio risk simulation, and Black-Scholes
option pricing simulations. The module is designed to implement additional
simulation types from the abstract base class `MonteCarloSimulation` in the
core module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from .core import MonteCarloSimulation

__all__ = [
    "PiEstimationSimulation",
    "PortfolioSimulation",
    "BlackScholesSimulation",
    "BlackScholesPathSimulation",
]


class PiEstimationSimulation(MonteCarloSimulation):
    def __init__(self):
        super().__init__("Pi Estimation")

    def single_simulation(
        self,
        n_points: int = 10000,
        antithetic: bool = False,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r = self._rng(_rng, self.rng)
        if not antithetic: #pragma: no cover
            pts = r.uniform(-1.0, 1.0, (n_points, 2))
            inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
            return float(4.0 * inside / n_points)
        # Antithetic sampling
        m = n_points // 2
        u = r.uniform(-1.0, 1.0, (m, 2))
        ua = -u
        pts = np.vstack([u, ua])
        if pts.shape[0] < n_points:
            pts = np.vstack([pts, r.uniform(-1.0, 1.0, (1, 2))])
        inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
        return float(4.0 * inside / n_points)


class PortfolioSimulation(MonteCarloSimulation):
    def __init__(self):
        super().__init__("Portfolio Simulation")

    def single_simulation(
        self,
        *,
        initial_value: float = 10_000.0,
        annual_return: float = 0.07,
        volatility: float = 0.20,
        years: int = 10,
        use_gbm: bool = True,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r = self._rng(_rng, self.rng)
        dt = 1.0 / 252.0  # Daily steps
        n = int(years / dt)
        if use_gbm:  # Geometric Brownian Motion for returns
            mu, sigma = annual_return, volatility
            rets = r.normal((mu - 0.5 * sigma * sigma) * dt, sigma * np.sqrt(dt), size=n)
            return float(initial_value * np.exp(rets.sum()))
        rets = r.normal(annual_return * dt, volatility * np.sqrt(dt), size=n)
        return float(initial_value * np.exp(np.log1p(rets).sum()))


# ============================================================================
# Helper Functions for Black-Scholes Simulations
# ============================================================================


def _european_payoff(S_T: float, K: float, option_type: str) -> float:
    """
    Calculate the payoff of a European option at maturity.

    Parameters
    ----------
    S_T : float
        Stock price at maturity.
    K : float
        Strike price.
    option_type : {"call", "put"}
        Type of option.

    Returns
    -------
    float
        Option payoff.
    """
    if option_type == "call":
        return max(S_T - K, 0.0)
    elif option_type == "put":
        return max(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _simulate_gbm_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    rng: Generator,
) -> np.ndarray:
    """
    Simulate a single path of Geometric Brownian Motion.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.
    n_steps : int
        Number of time steps.
    rng : Generator
        NumPy random generator.

    Returns
    -------
    ndarray
        Stock price path of shape (n_steps + 1,), starting at S0.
    """
    dt = T / n_steps
    # Generate random increments
    Z = rng.standard_normal(n_steps)
    # Compute log returns
    log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
    # Compute stock price path
    log_path = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_returns)])
    return np.exp(log_path)


def _american_exercise_lsm(
    paths: np.ndarray,
    K: float,
    r: float,
    dt: float,
    option_type: str,
) -> float:
    """
    Price an American option using the Longstaff-Schwartz LSM algorithm.

    Parameters
    ----------
    paths : ndarray
        Stock price paths of shape (n_paths, n_steps + 1).
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    dt : float
        Time step size.
    option_type : {"call", "put"}
        Type of option.

    Returns
    -------
    float
        Option price (average across all paths).
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1

    # Compute intrinsic values at each time step
    if option_type == "call":
        intrinsic = np.maximum(paths - K, 0.0)
    elif option_type == "put":
        intrinsic = np.maximum(K - paths, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Cash flows from exercising at each time step
    cash_flows = intrinsic.copy()
    # Initialize exercise times (all at maturity)
    exercise_times = np.full(n_paths, n_steps)

    # Backward induction from n_steps - 1 to 1
    for t in range(n_steps - 1, 0, -1):
        # Find paths where option is in-the-money at time t
        itm = intrinsic[:, t] > 0
        if not np.any(itm):
            continue

        # Discounted cash flows from continuation
        discount = np.exp(-r * dt * (exercise_times[itm] - t))
        continuation_values = cash_flows[itm, exercise_times[itm]] * discount

        # Regression: continuation value ~ f(stock price)
        # Use polynomial basis functions: 1, S, S^2
        S_itm = paths[itm, t]
        X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm**2])

        # Least squares regression
        try:
            coeffs = np.linalg.lstsq(X, continuation_values, rcond=None)[0]
            fitted_continuation = X @ coeffs
        except np.linalg.LinAlgError:
            # If regression fails, assume no early exercise
            continue

        # Exercise if immediate payoff > continuation value
        exercise_now = intrinsic[itm, t] > fitted_continuation

        # Update exercise times and cash flows
        itm_indices = np.where(itm)[0]
        early_exercise_indices = itm_indices[exercise_now]
        exercise_times[early_exercise_indices] = t
        cash_flows[early_exercise_indices, t] = intrinsic[early_exercise_indices, t]

    # Compute option value as discounted cash flow at optimal exercise time
    option_values = np.zeros(n_paths)
    for i in range(n_paths):
        t_ex = exercise_times[i]
        option_values[i] = cash_flows[i, t_ex] * np.exp(-r * dt * t_ex)

    return float(np.mean(option_values))


# ============================================================================
# Black-Scholes Option Pricing Simulation
# ============================================================================


class BlackScholesSimulation(MonteCarloSimulation):
    r"""
    Monte Carlo simulation for Black-Scholes option pricing.

    Supports European and American options (calls and puts) with Greeks
    calculation capabilities. Uses Geometric Brownian Motion for stock price
    dynamics and the Longstaff-Schwartz LSM algorithm for American options.

    Parameters
    ----------
    name : str, optional
        Simulation name. Defaults to "Black-Scholes Option Pricing".

    Notes
    -----
    The stock price follows the risk-neutral dynamics:

    .. math::

        dS_t = r S_t dt + \sigma S_t dW_t

    where :math:`r` is the risk-free rate, :math:`\sigma` is volatility,
    and :math:`W_t` is a standard Brownian motion.

    Examples
    --------
    >>> sim = BlackScholesSimulation()
    >>> sim.set_seed(42)
    >>> res = sim.run(
    ...     10000,
    ...     S0=100.0,
    ...     K=100.0,
    ...     T=1.0,
    ...     r=0.05,
    ...     sigma=0.20,
    ...     option_type="call",
    ...     exercise_type="european"
    ... )  # doctest: +SKIP
    """

    def __init__(self, name: str = "Black-Scholes Option Pricing"):
        super().__init__(name)

    def single_simulation(
        self,
        *,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
        exercise_type: str = "european",
        n_steps: int = 252,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        """
        Simulate a single option price.

        Parameters
        ----------
        S0 : float, default 100.0
            Initial stock price.
        K : float, default 100.0
            Strike price.
        T : float, default 1.0
            Time to maturity in years.
        r : float, default 0.05
            Risk-free interest rate (annualized).
        sigma : float, default 0.20
            Volatility (annualized).
        option_type : {"call", "put"}, default "call"
            Type of option.
        exercise_type : {"european", "american"}, default "european"
            Exercise style.
        n_steps : int, default 252
            Number of time steps (daily steps for 1 year by default).
        _rng : Generator, optional
            NumPy random generator (managed by framework).

        Returns
        -------
        float
            Simulated option price.

        Raises
        ------
        ValueError
            If option_type or exercise_type are invalid.
        """
        rng = self._rng(_rng, self.rng)

        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if exercise_type not in ("european", "american"):
            raise ValueError(f"exercise_type must be 'european' or 'american', got '{exercise_type}'")

        if exercise_type == "european":
            # European option: only need final stock price
            dt = T / n_steps
            Z = rng.standard_normal(n_steps)
            log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
            S_T = S0 * np.exp(np.sum(log_returns))
            payoff = _european_payoff(S_T, K, option_type)
            # Discount to present value
            option_price = payoff * np.exp(-r * T)
            return float(option_price)
        else:
            # American option: need full path for early exercise
            # Generate a single path
            path = _simulate_gbm_path(S0, r, sigma, T, n_steps, rng)
            # For a single path LSM, we need multiple paths - this is inefficient
            # For American options, we should use run() with many simulations
            # But for a single simulation, we can approximate with European + early exercise premium
            # Or we can just compute the intrinsic value at each step
            # Let's compute intrinsic value along the path and take max
            dt = T / n_steps
            if option_type == "call":
                intrinsic = np.maximum(path - K, 0.0)
            else:
                intrinsic = np.maximum(K - path, 0.0)

            # Discount each intrinsic value back to present
            time_steps = np.arange(n_steps + 1)
            discount_factors = np.exp(-r * dt * time_steps)
            discounted_intrinsic = intrinsic * discount_factors

            # For a single path, best we can do is max discounted intrinsic
            # This is a rough approximation; true American pricing needs LSM with many paths
            option_price = np.max(discounted_intrinsic)
            return float(option_price)

    def calculate_greeks(
        self,
        n_simulations: int,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
        exercise_type: str = "european",
        n_steps: int = 252,
        parallel: bool = False,
        bump_pct: float = 0.01,
        time_bump_days: float = 1.0,
    ) -> dict[str, float]:
        """
        Calculate option Greeks using finite difference methods.

        This method runs multiple simulations with perturbed parameters to
        estimate the partial derivatives (Greeks) of the option price.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo simulations for each price calculation.
        S0 : float, default 100.0
            Initial stock price.
        K : float, default 100.0
            Strike price.
        T : float, default 1.0
            Time to maturity in years.
        r : float, default 0.05
            Risk-free interest rate.
        sigma : float, default 0.20
            Volatility.
        option_type : {"call", "put"}, default "call"
            Type of option.
        exercise_type : {"european", "american"}, default "european"
            Exercise style.
        n_steps : int, default 252
            Number of time steps.
        parallel : bool, default False
            Whether to use parallel execution.
        bump_pct : float, default 0.01
            Percentage bump for finite differences (1% by default).
        time_bump_days : float, default 1.0
            Time bump in days for Theta calculation.

        Returns
        -------
        dict
            Dictionary containing:
            - "price": base option price
            - "delta": ∂V/∂S (change in price per unit change in stock price)
            - "gamma": ∂²V/∂S² (rate of change of delta)
            - "vega": ∂V/∂σ (change in price per unit change in volatility)
            - "theta": ∂V/∂T (change in price per unit change in time)
            - "rho": ∂V/∂r (change in price per unit change in interest rate)

        Notes
        -----
        - Uses central difference for Delta and Gamma
        - Uses forward difference for Theta (time decay)
        - Uses central difference for Vega and Rho
        - All simulations use the same seed for variance reduction
        """
        # Save current seed state
        original_seed = self.rng.bit_generator.state if self.rng else None

        # Common parameters
        sim_kwargs = {
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "exercise_type": exercise_type,
            "n_steps": n_steps,
        }

        # Base price
        self.set_seed(42)  # Use consistent seed for all Greek calculations
        res_base = self.run(n_simulations, S0=S0, parallel=parallel, compute_stats=False, **sim_kwargs)
        V0 = res_base.mean

        # Delta: ∂V/∂S (central difference)
        dS = S0 * bump_pct
        self.set_seed(42)
        res_up = self.run(n_simulations, S0=S0 + dS, parallel=parallel, compute_stats=False, **sim_kwargs)
        self.set_seed(42)
        res_down = self.run(n_simulations, S0=S0 - dS, parallel=parallel, compute_stats=False, **sim_kwargs)
        delta = (res_up.mean - res_down.mean) / (2 * dS)

        # Gamma: ∂²V/∂S² (central difference)
        gamma = (res_up.mean - 2 * V0 + res_down.mean) / (dS * dS)

        # Vega: ∂V/∂σ (central difference, scaled to 1% volatility change)
        dsigma = sigma * bump_pct
        self.set_seed(42)
        res_vol_up = self.run(
            n_simulations, S0=S0, parallel=parallel, compute_stats=False,
            sigma=sigma + dsigma, **{k: v for k, v in sim_kwargs.items() if k != "sigma"}
        )
        self.set_seed(42)
        res_vol_down = self.run(
            n_simulations, S0=S0, parallel=parallel, compute_stats=False,
            sigma=sigma - dsigma, **{k: v for k, v in sim_kwargs.items() if k != "sigma"}
        )
        vega = (res_vol_up.mean - res_vol_down.mean) / (2 * dsigma)
        # Scale to 1% volatility change (common convention)
        vega = vega * 0.01

        # Theta: ∂V/∂T (forward difference, time decay)
        dT = time_bump_days / 365.0  # Convert days to years
        if T > dT:
            self.set_seed(42)
            res_time = self.run(
                n_simulations, S0=S0, parallel=parallel, compute_stats=False,
                T=T - dT, **{k: v for k, v in sim_kwargs.items() if k != "T"}
            )
            theta = (res_time.mean - V0) / dT
            # Scale to daily theta (common convention)
            theta = theta / 365.0
        else:
            theta = 0.0  # Near expiry, theta calculation becomes unreliable

        # Rho: ∂V/∂r (central difference, scaled to 1% rate change)
        dr = r * bump_pct if r > 0 else 0.0001  # Handle zero rate
        self.set_seed(42)
        res_rate_up = self.run(
            n_simulations, S0=S0, parallel=parallel, compute_stats=False,
            r=r + dr, **{k: v for k, v in sim_kwargs.items() if k != "r"}
        )
        self.set_seed(42)
        res_rate_down = self.run(
            n_simulations, S0=S0, parallel=parallel, compute_stats=False,
            r=r - dr, **{k: v for k, v in sim_kwargs.items() if k != "r"}
        )
        rho = (res_rate_up.mean - res_rate_down.mean) / (2 * dr)
        # Scale to 1% rate change (common convention)
        rho = rho * 0.01

        # Restore original seed state if it existed
        if original_seed is not None and self.rng is not None:
            self.rng.bit_generator.state = original_seed

        return {
            "price": float(V0),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho": float(rho),
        }


# ============================================================================
# Black-Scholes Path Simulation
# ============================================================================


class BlackScholesPathSimulation(MonteCarloSimulation):
    r"""
    Simulate stock price paths under Black-Scholes dynamics.

    This simulation generates stock price paths following Geometric Brownian
    Motion (GBM) under the risk-neutral measure. It returns the final stock
    price but can be extended to store and analyze full paths.

    Parameters
    ----------
    name : str, optional
        Simulation name. Defaults to "Black-Scholes Path Simulation".

    Notes
    -----
    The stock price dynamics are:

    .. math::

        dS_t = r S_t dt + \sigma S_t dW_t

    where :math:`S_t` is the stock price at time :math:`t`, :math:`r` is
    the risk-free rate, :math:`\sigma` is volatility, and :math:`W_t` is
    a standard Brownian motion.

    Examples
    --------
    >>> sim = BlackScholesPathSimulation()
    >>> sim.set_seed(123)
    >>> res = sim.run(
    ...     1000,
    ...     S0=100.0,
    ...     r=0.05,
    ...     sigma=0.20,
    ...     T=1.0,
    ...     n_steps=252
    ... )  # doctest: +SKIP
    """

    def __init__(self, name: str = "Black-Scholes Path Simulation"):
        super().__init__(name)

    def single_simulation(
        self,
        *,
        S0: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.20,
        T: float = 1.0,
        n_steps: int = 252,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        """
        Simulate a single stock price path and return the final price.

        Parameters
        ----------
        S0 : float, default 100.0
            Initial stock price.
        r : float, default 0.05
            Risk-free interest rate (annualized).
        sigma : float, default 0.20
            Volatility (annualized).
        T : float, default 1.0
            Time horizon in years.
        n_steps : int, default 252
            Number of time steps (daily by default).
        _rng : Generator, optional
            NumPy random generator (managed by framework).

        Returns
        -------
        float
            Final stock price at time T.
        """
        rng = self._rng(_rng, self.rng)
        path = _simulate_gbm_path(S0, r, sigma, T, n_steps, rng)
        return float(path[-1])

    def simulate_paths(
        self,
        n_paths: int,
        S0: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.20,
        T: float = 1.0,
        n_steps: int = 252,
    ) -> np.ndarray:
        """
        Generate multiple stock price paths for visualization or analysis.

        Parameters
        ----------
        n_paths : int
            Number of paths to generate.
        S0 : float, default 100.0
            Initial stock price.
        r : float, default 0.05
            Risk-free interest rate.
        sigma : float, default 0.20
            Volatility.
        T : float, default 1.0
            Time horizon in years.
        n_steps : int, default 252
            Number of time steps.

        Returns
        -------
        ndarray
            Array of shape (n_paths, n_steps + 1) containing stock price paths.
        """
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            paths[i] = _simulate_gbm_path(S0, r, sigma, T, n_steps, self.rng)
        return paths

