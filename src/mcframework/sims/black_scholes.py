"""Black-Scholes simulations and helper utilities."""
# pylint: disable=invalid-name
# Finance/math notation (S_T, K, T, S0, Z, X, V0, dS, dT) follows standard conventions

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from ..core import MonteCarloSimulation

__all__ = [
    "_european_payoff",
    "_simulate_gbm_path",
    "_american_exercise_lsm",
    "BlackScholesSimulation",
    "BlackScholesPathSimulation",
]


def _european_payoff(S_T: float, K: float, option_type: str) -> float:
    r"""
    Evaluate the terminal payoff :math:`\Phi(S_T)` of a European option.

    The payoff is given by

    .. math::
       \Phi_{\text{call}}(S_T) = \max(S_T - K, 0), \qquad
       \Phi_{\text{put}}(S_T) = \max(K - S_T, 0).

    Parameters
    ----------
    S_T : float
        Terminal stock price at maturity :math:`T`.
    K : float
        Strike level :math:`K`.
    option_type : {"call", "put"}
        Chooses :math:`\Phi_{\text{call}}` or :math:`\Phi_{\text{put}}`.

    Returns
    -------
    float
        Scalar payoff evaluated at the supplied :math:`S_T`.
    """
    if option_type == "call":
        return max(S_T - K, 0.0)
    if option_type == "put":
        return max(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _simulate_gbm_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    rng: Generator,
) -> np.ndarray:
    r"""
    Simulate a single Geometric Brownian Motion (GBM) path.

    The solution of

    .. math::
       dS_t = r S_t\,dt + \sigma S_t\,dW_t,\qquad S_0 = S_0,

    is

    .. math::
       S_t = S_0 \exp\!\left((r - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right).

    A discrete-time Euler scheme draws :math:`n_{\text{steps}}` increments
    :math:`Z_k \sim \mathcal{N}(0, 1)` and sets

    .. math::
       S_{t_{k+1}} = S_{t_k} \exp\left((r - \tfrac{1}{2}\sigma^2)\Delta t
       + \sigma \sqrt{\Delta t}\,Z_k\right).

    Parameters
    ----------
    S0 : float
        Initial level :math:`S_0`.
    r : float
        Risk-free drift :math:`r`.
    sigma : float
        Volatility :math:`\sigma`.
    T : float
        Horizon in years.
    n_steps : int
        Number of uniform time steps. The spacing is :math:`\Delta t = T / n_{\text{steps}}`.
    rng : numpy.random.Generator
        Source of randomness for :math:`Z_k`.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_steps + 1,)`` containing the path :math:`(S_{t_k})_{k=0}^n`.
    """
    dt = T / n_steps
    Z = rng.standard_normal(n_steps)
    log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
    log_path = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_returns)])
    return np.exp(log_path)


def _american_exercise_lsm(
    paths: np.ndarray,
    K: float,
    r: float,
    dt: float,
    option_type: str,
) -> float:
    r"""
    Apply the Longstaff–Schwartz (LSM) regression algorithm to American options.

    For each simulated path :math:`\{S_{t_k}^{(i)}\}_{k=0}^n` we compute the
    intrinsic value

    .. math::
       C_{t_k}^{(i)} =
       \begin{cases}
            \max(S_{t_k}^{(i)} - K, 0), & \text{call},\\
            \max(K - S_{t_k}^{(i)}, 0), & \text{put},
       \end{cases}

    then regress discounted continuation values onto basis functions
    :math:`\{1, S_{t_k}, S_{t_k}^2\}` to approximate the conditional expectation
    :math:`\mathbb{E}\big[C_{t_{k+1}} \mid S_{t_k}\big]`. Early exercise occurs
    when the intrinsic value exceeds this conditional expectation. The final
    price is the Monte Carlo average of discounted cash flows.

    Parameters
    ----------
    paths : numpy.ndarray
        Array of shape ``(n_paths, n_steps + 1)`` storing simulated price paths.
    K : float
        Strike :math:`K`.
    r : float
        Annualized risk-free rate used for discounting.
    dt : float
        Time-step length :math:`\Delta t`.
    option_type : {"call", "put"}
        Payoff family applied to :math:`C_{t_k}`.

    Returns
    -------
    float
        Estimated arbitrage-free price
        :math:`V_0 = \frac{1}{N}\sum_{i=1}^N e^{-r t_{\tau^{(i)}}} C_{t_{\tau^{(i)}}}^{(i)}`.
    """
    n_paths, n_steps_plus_1 = paths.shape
    n_steps = n_steps_plus_1 - 1

    if option_type == "call":
        intrinsic = np.maximum(paths - K, 0.0)
    elif option_type == "put":
        intrinsic = np.maximum(K - paths, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    cash_flows = intrinsic.copy()
    exercise_times = np.full(n_paths, n_steps)

    for t in range(n_steps - 1, 0, -1):
        itm = intrinsic[:, t] > 0
        if not np.any(itm):
            continue

        discount = np.exp(-r * dt * (exercise_times[itm] - t))
        continuation_values = cash_flows[itm, exercise_times[itm]] * discount
        S_itm = paths[itm, t]
        X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm**2])

        try:
            coeffs = np.linalg.lstsq(X, continuation_values, rcond=None)[0]
            fitted_continuation = X @ coeffs
        except np.linalg.LinAlgError:
            continue

        exercise_now = intrinsic[itm, t] > fitted_continuation
        itm_indices = np.where(itm)[0]
        early_exercise_indices = itm_indices[exercise_now]
        exercise_times[early_exercise_indices] = t
        cash_flows[early_exercise_indices, t] = intrinsic[early_exercise_indices, t]

    option_values = np.zeros(n_paths)
    for i in range(n_paths):
        t_ex = exercise_times[i]
        option_values[i] = cash_flows[i, t_ex] * np.exp(-r * dt * t_ex)
    return float(np.mean(option_values))


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
    """

    def __init__(self, name: str = "Black-Scholes Option Pricing"):
        super().__init__(name)

    def single_simulation(  # pylint: disable=arguments-differ
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
        r"""
        Price a single option instance under Black–Scholes dynamics.
        """
        rng = self._rng(_rng, self.rng)

        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if exercise_type not in ("european", "american"):
            raise ValueError(f"exercise_type must be 'european' or 'american', got '{exercise_type}'")

        if exercise_type == "european":
            dt = T / n_steps
            Z = rng.standard_normal(n_steps)
            log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
            S_T = S0 * np.exp(np.sum(log_returns))
            payoff = _european_payoff(S_T, K, option_type)
            return float(payoff * np.exp(-r * T))

        path = _simulate_gbm_path(S0, r, sigma, T, n_steps, rng)
        dt = T / n_steps
        if option_type == "call":
            intrinsic = np.maximum(path - K, 0.0)
        else:
            intrinsic = np.maximum(K - path, 0.0)

        time_steps = np.arange(n_steps + 1)
        discount_factors = np.exp(-r * dt * time_steps)
        discounted_intrinsic = intrinsic * discount_factors
        return float(np.max(discounted_intrinsic))

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
        r"""
        Estimate primary Greeks via finite differences.
        """
        original_seed = self.rng.bit_generator.state if self.rng else None
        sim_kwargs = {
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "option_type": option_type,
            "exercise_type": exercise_type,
            "n_steps": n_steps,
        }

        self.set_seed(42)
        res_base = self.run(n_simulations, S0=S0, parallel=parallel, compute_stats=False, **sim_kwargs)
        V0 = res_base.mean

        dS = S0 * bump_pct
        self.set_seed(42)
        res_up = self.run(n_simulations, S0=S0 + dS, parallel=parallel, compute_stats=False, **sim_kwargs)
        self.set_seed(42)
        res_down = self.run(n_simulations, S0=S0 - dS, parallel=parallel, compute_stats=False, **sim_kwargs)
        delta = (res_up.mean - res_down.mean) / (2 * dS)
        gamma = (res_up.mean - 2 * V0 + res_down.mean) / (dS * dS)

        dsigma = sigma * bump_pct
        self.set_seed(42)
        res_vol_up = self.run(
            n_simulations,
            S0=S0,
            parallel=parallel,
            compute_stats=False,
            sigma=sigma + dsigma,
            **{k: v for k, v in sim_kwargs.items() if k != "sigma"},
        )
        self.set_seed(42)
        res_vol_down = self.run(
            n_simulations,
            S0=S0,
            parallel=parallel,
            compute_stats=False,
            sigma=sigma - dsigma,
            **{k: v for k, v in sim_kwargs.items() if k != "sigma"},
        )
        vega = (res_vol_up.mean - res_vol_down.mean) / (2 * dsigma) * 0.01

        dT = time_bump_days / 365.0
        if T > dT:
            self.set_seed(42)
            res_time = self.run(
                n_simulations,
                S0=S0,
                parallel=parallel,
                compute_stats=False,
                T=T - dT,
                **{k: v for k, v in sim_kwargs.items() if k != "T"},
            )
            theta = (res_time.mean - V0) / dT / 365.0
        else:
            theta = 0.0

        dr = r * bump_pct if r > 0 else 0.0001
        self.set_seed(42)
        res_rate_up = self.run(
            n_simulations,
            S0=S0,
            parallel=parallel,
            compute_stats=False,
            r=r + dr,
            **{k: v for k, v in sim_kwargs.items() if k != "r"},
        )
        self.set_seed(42)
        res_rate_down = self.run(
            n_simulations,
            S0=S0,
            parallel=parallel,
            compute_stats=False,
            r=r - dr,
            **{k: v for k, v in sim_kwargs.items() if k != "r"},
        )
        rho = (res_rate_up.mean - res_rate_down.mean) / (2 * dr) * 0.01

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


class BlackScholesPathSimulation(MonteCarloSimulation):
    r"""
    Simulate stock price paths under Black-Scholes dynamics.
    """

    def __init__(self, name: str = "Black-Scholes Path Simulation"):
        super().__init__(name)

    def single_simulation(  # pylint: disable=arguments-differ
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
        r"""
        Draw a GBM path and return the terminal value :math:`S_T`.
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
        r"""
        Generate :math:`n_{\text{paths}}` independent GBM paths.
        """
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            paths[i] = _simulate_gbm_path(S0, r, sigma, T, n_steps, self.rng)
        return paths
