"""Black-Scholes simulations and helper utilities."""
# pylint: disable=invalid-name
# Finance/math notation (S_T, K, T, S0, Z, X, V0, dS, dT) follows standard conventions

from __future__ import annotations

import logging

import numpy as np
from numpy.random import Generator
from scipy.stats import norm  # type: ignore[import-untyped]

from ..core import MonteCarloSimulation

logger = logging.getLogger(__name__)

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

    path_indices = np.arange(n_paths)
    option_values = (
        cash_flows[path_indices, exercise_times]
        * np.exp(-r * dt * exercise_times)
    )
    return float(np.mean(option_values))


class BlackScholesSimulation(MonteCarloSimulation):
    r"""
    Monte Carlo simulation for Black-Scholes option pricing.

    Uses Geometric Brownian Motion for stock price dynamics and supports
    European and American options (calls and puts) with Greeks calculation.

    Exercise handling
    -----------------
    - **European** (``exercise_type="european"``): unbiased discounted terminal
      payoff via :meth:`single_simulation`.
    - **American**: there are two paths, and they are *not* equivalent:

      * :meth:`single_simulation` with ``exercise_type="american"`` returns a
        **high-biased upper bound** (perfect-foresight maximum of discounted
        intrinsic value along a single path). It assumes knowledge of the whole
        path when choosing the exercise time, so it systematically overprices.
        It is cheap and useful as a sanity ceiling, not as a fair price.
      * :meth:`price_american` returns a proper, look-ahead-free price using the
        Longstaff-Schwartz (LSM) regression across a batch of paths. Prefer this
        for actual American option valuation.

    Parameters
    ----------
    name : str, optional
        Simulation name. Defaults to "Black-Scholes Option Pricing".
    """

    reference_source = "Black-Scholes-Merton (1973), closed-form European price"

    def __init__(self, name: str = "Black-Scholes Option Pricing"):
        super().__init__(name)
        self._american_bias_warned = False

    def analytic_reference(
        self,
        *,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
        exercise_type: str = "european",
        **params,
    ) -> float | None:
        r"""
        Closed-form Black-Scholes-Merton price (the oracle for European exercise).

        .. math::
           C = S_0\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2), \qquad
           P = K e^{-rT}\,\Phi(-d_2) - S_0\,\Phi(-d_1),

        with :math:`d_1 = \frac{\ln(S_0/K) + (r + \tfrac12\sigma^2)T}{\sigma\sqrt T}`
        and :math:`d_2 = d_1 - \sigma\sqrt T`.

        Returns ``None`` for ``exercise_type="american"``: an American option has no
        closed-form price, so it is deliberately not convergence-validated. That
        absence is itself the governance signal.
        """
        if exercise_type != "european":
            return None
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sqrt_t
        d2 = d1 - sqrt_t
        disc_k = K * np.exp(-r * T)
        if option_type == "call":
            return float(S0 * norm.cdf(d1) - disc_k * norm.cdf(d2))
        return float(disc_k * norm.cdf(-d2) - S0 * norm.cdf(-d1))

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
        _rng: Generator | None = None,
        **kwargs,
    ) -> float:
        r"""
        Price a single option instance under Black–Scholes dynamics.

        Notes
        -----
        For ``exercise_type="american"`` this returns the **perfect-foresight
        upper bound** (the maximum discounted intrinsic value along the path),
        which is a high-biased estimator, not a fair price. Use
        :meth:`price_american` for a look-ahead-free Longstaff-Schwartz price.
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

        # American via single_simulation is a perfect-foresight UPPER BOUND, not a
        # fair price (it picks the best exercise time with full knowledge of the
        # path). Warn once and steer callers to price_american() for true LSM.
        if not self._american_bias_warned:
            logger.warning(
                "%s: single_simulation(exercise_type='american') returns a high-biased "
                "perfect-foresight upper bound, not a fair price. Use price_american() "
                "for a look-ahead-free Longstaff-Schwartz estimate.",
                self.name,
            )
            self._american_bias_warned = True

        path = _simulate_gbm_path(S0, r, sigma, T, n_steps, rng)
        dt = T / n_steps
        intrinsic = np.maximum(path - K, 0.0) if option_type == "call" else np.maximum(K - path, 0.0)

        time_steps = np.arange(n_steps + 1)
        discount_factors = np.exp(-r * dt * time_steps)
        discounted_intrinsic = intrinsic * discount_factors
        return float(np.max(discounted_intrinsic))

    def price_american(
        self,
        n_paths: int,
        *,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.20,
        option_type: str = "call",
        n_steps: int = 50,
        _rng: Generator | None = None,
    ) -> float:
        r"""
        Price an American option with the Longstaff-Schwartz (LSM) algorithm.

        Unlike :meth:`single_simulation` with ``exercise_type="american"`` (which
        is a high-biased perfect-foresight bound), this simulates a *batch* of GBM
        paths jointly and regresses continuation values across paths, so the
        exercise decision uses no future information. This is the estimator to use
        for actual American option valuation.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to simulate. LSM regression quality
            improves with more paths.
        S0, K, T, r, sigma : float
            Standard Black-Scholes parameters (spot, strike, maturity in years,
            risk-free rate, volatility).
        option_type : {"call", "put"}, default ``"call"``
            Payoff family. Early exercise matters mainly for puts (and dividend
            -paying calls, not modeled here).
        n_steps : int, default 50
            Number of exercise opportunities (time steps) along each path.
        _rng : numpy.random.Generator, optional
            Explicit generator. Falls back to ``self.rng`` (set via
            :meth:`~mcframework.simulation.MonteCarloSimulation.set_seed`).

        Returns
        -------
        float
            Estimated arbitrage-free American option price.
        """
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        rng = self._rng(_rng, self.rng)
        dt = T / n_steps

        # Vectorized batch of GBM paths, shape (n_paths, n_steps + 1).
        Z = rng.standard_normal((n_paths, n_steps))
        log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
        log_S0 = np.log(S0)
        log_paths = np.concatenate(
            [np.full((n_paths, 1), log_S0), log_S0 + np.cumsum(log_returns, axis=1)],
            axis=1,
        )
        paths = np.exp(log_paths)

        return _american_exercise_lsm(paths, K, r, dt, option_type)

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
        backend: str = "sequential",
        bump_pct: float = 0.01,
        time_bump_days: float = 1.0,
    ) -> dict[str, float]:
        r"""
        Estimate primary Greeks via finite differences.
        """
        # Preserve the caller's RNG *and* seed sequence: the finite-difference
        # bumps below repeatedly call set_seed(42) for common random numbers,
        # which overwrites both self.rng and self.seed_seq. Restore both at the
        # end so the simulation is left exactly as the caller had it.
        original_seed = self.rng.bit_generator.state if self.rng else None
        original_seed_seq = self.seed_seq
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
        res_base = self.run(
            n_simulations, S0=S0, backend=backend, compute_stats=False,
            **sim_kwargs,  # type: ignore[arg-type]
        )
        V0 = res_base.mean

        dS = S0 * bump_pct
        self.set_seed(42)
        res_up = self.run(
            n_simulations, S0=S0 + dS, backend=backend, compute_stats=False,
            **sim_kwargs,  # type: ignore[arg-type]
        )
        self.set_seed(42)
        res_down = self.run(
            n_simulations, S0=S0 - dS, backend=backend, compute_stats=False,
            **sim_kwargs,  # type: ignore[arg-type]
        )
        delta = (res_up.mean - res_down.mean) / (2 * dS)
        gamma = (res_up.mean - 2 * V0 + res_down.mean) / (dS * dS)

        dsigma = sigma * bump_pct
        self.set_seed(42)
        res_vol_up = self.run(
            n_simulations,
            S0=S0,
            backend=backend,
            compute_stats=False,
            sigma=sigma + dsigma,
            **{k: v for k, v in sim_kwargs.items() if k != "sigma"},  # type: ignore[arg-type]
        )
        self.set_seed(42)
        res_vol_down = self.run(
            n_simulations,
            S0=S0,
            backend=backend,
            compute_stats=False,
            sigma=sigma - dsigma,
            **{k: v for k, v in sim_kwargs.items() if k != "sigma"},  # type: ignore[arg-type]
        )
        vega = (res_vol_up.mean - res_vol_down.mean) / (2 * dsigma) * 0.01

        dT = time_bump_days / 365.0
        if dT < T:
            self.set_seed(42)
            res_time = self.run(
                n_simulations,
                S0=S0,
                backend=backend,
                compute_stats=False,
                T=T - dT,
                **{k: v for k, v in sim_kwargs.items() if k != "T"},  # type: ignore[arg-type]
            )
            theta = (res_time.mean - V0) / dT / 365.0
        else:
            theta = 0.0

        dr = r * bump_pct if r > 0 else 0.0001
        self.set_seed(42)
        res_rate_up = self.run(
            n_simulations,
            S0=S0,
            backend=backend,
            compute_stats=False,
            r=r + dr,
            **{k: v for k, v in sim_kwargs.items() if k != "r"},  # type: ignore[arg-type]
        )
        self.set_seed(42)
        res_rate_down = self.run(
            n_simulations,
            S0=S0,
            backend=backend,
            compute_stats=False,
            r=r - dr,
            **{k: v for k, v in sim_kwargs.items() if k != "r"},  # type: ignore[arg-type]
        )
        rho = (res_rate_up.mean - res_rate_down.mean) / (2 * dr) * 0.01

        self.seed_seq = original_seed_seq
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

    reference_source = "Risk-neutral GBM: E[S_T] = S0 * exp(r * T)"

    def __init__(self, name: str = "Black-Scholes Path Simulation"):
        super().__init__(name)

    def analytic_reference(
        self,
        *,
        S0: float = 100.0,
        r: float = 0.05,
        T: float = 1.0,
        **params,
    ) -> float:
        r"""Oracle: under the risk-neutral measure :math:`\mathbb{E}[S_T] = S_0 e^{rT}`."""
        return float(S0 * np.exp(r * T))

    def single_simulation(  # pylint: disable=arguments-differ
        self,
        *,
        S0: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.20,
        T: float = 1.0,
        n_steps: int = 252,
        _rng: Generator | None = None,
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
        dt = T / n_steps
        Z = self.rng.standard_normal((n_paths, n_steps))
        log_returns = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z
        log_S0 = np.log(S0)
        log_paths = np.concatenate(
            [np.full((n_paths, 1), log_S0), log_S0 + np.cumsum(log_returns, axis=1)],
            axis=1,
        )
        return np.exp(log_paths)
