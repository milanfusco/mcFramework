"""mcframework.sims
====================

Canonical, reference-quality simulation classes used throughout the framework.
Each simulation exposes a :class:`~mcframework.core.MonteCarloSimulation`
interface together with mathematical derivations of the underlying models so
that the generated documentation reads as a miniature lecture note.

The module currently contains three families of examples:

* Pi estimation via geometric probability (:class:`PiEstimationSimulation`).
* Geometric Brownian Motion (GBM) wealth evolution (:class:`PortfolioSimulation`).
* Black–Scholes pricing and path sampling
  (:class:`BlackScholesSimulation`, :class:`BlackScholesPathSimulation`).
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
    r"""
    Estimate :math:`\pi` by geometric probability on the unit disk.

    The simulation throws :math:`n` i.i.d. points :math:`(X_i, Y_i)` uniformly on
    :math:`[-1, 1]^2` and uses the identity

    .. math::
       \pi = 4 \,\Pr\!\left(X^2 + Y^2 \le 1\right),

    to form the Monte Carlo estimator

    .. math::
       \widehat{\pi}_n = \frac{4}{n} \sum_{i=1}^n \mathbf{1}\{X_i^2 + Y_i^2 \le 1\}.

    Attributes
    ----------
    name : str
        Human-readable label registered with :class:`~mcframework.core.MonteCarloFramework`.
    """

    def __init__(self):
        super().__init__("Pi Estimation")

    def single_simulation(
        self,
        n_points: int = 10_000,
        antithetic: bool = False,
        _rng: Optional[Generator] = None,
        **kwargs,
    ) -> float:
        r"""
        Throw :math:`n_{\text{points}}` darts at :math:`[-1, 1]^2` and return the
        single-run estimator :math:`\widehat{\pi}`.

        Parameters
        ----------
        n_points : int, default ``10_000``
            Number of uniformly distributed points to simulate. The Monte Carlo
            variance decays as :math:`\mathcal{O}(n_{\text{points}}^{-1})`.
        antithetic : bool, default ``False``
            Whether to pair each point :math:`(x, y)` with its reflection
            :math:`(-x, -y)` to achieve first-order variance cancellation.
        _rng : numpy.random.Generator, optional
            Worker-local generator injected by :class:`~mcframework.core.MonteCarloSimulation`.
        ``**kwargs`` :
            Ignored. Present for compatibility with the base signature.

        Returns
        -------
        float
            Estimate of :math:`\pi` computed via
            :math:`\widehat{\pi} = 4 \,\widehat{p}`, where
            :math:`\widehat{p}` is the observed fraction of darts that land inside
            the unit disk.
        """
        rng = self._rng(_rng, self.rng)
        if not antithetic:  # pragma: no cover
            pts = rng.uniform(-1.0, 1.0, (n_points, 2))
            inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
            return float(4.0 * inside / n_points)
        # Antithetic sampling mirrors each draw (x, y) with (-x, -y)
        m = n_points // 2
        u = rng.uniform(-1.0, 1.0, (m, 2))
        ua = -u
        pts = np.vstack([u, ua])
        if pts.shape[0] < n_points:
            pts = np.vstack([pts, rng.uniform(-1.0, 1.0, (1, 2))])
        inside = np.sum(np.sum(pts * pts, axis=1) <= 1.0)
        return float(4.0 * inside / n_points)


class PortfolioSimulation(MonteCarloSimulation):
    r"""
    Compound an initial wealth under log-normal or arithmetic return models.

    Let :math:`V_0` be the initial value. Under GBM dynamics the terminal value
    after :math:`T` years with :math:`n = 252T` daily steps is

    .. math::
       V_T = V_0 \exp\left(\sum_{k=1}^n \Big[(\mu - \tfrac{1}{2}\sigma^2)\Delta t
       + \sigma \sqrt{\Delta t}\,Z_k\Big]\right),

    where :math:`Z_k \sim \mathcal{N}(0, 1)` i.i.d. The alternative branch
    integrates arithmetic returns via :math:`\log(1 + R_k)`.

    Attributes
    ----------
    name : str
        Default registry label ``\"Portfolio Simulation\"``.
    """

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
        r"""
        Simulate the terminal portfolio value under discrete compounding.

        Parameters
        ----------
        initial_value : float, default ``10_000``
            Starting wealth :math:`V_0` expressed in currency units.
        annual_return : float, default ``0.07``
            Drift :math:`\mu` expressed as an annualized continuously compounded
            rate.
        volatility : float, default ``0.20``
            Annualized diffusion coefficient :math:`\sigma`.
        years : int, default ``10``
            Investment horizon :math:`T` in years. The simulation uses daily
            steps :math:`\Delta t = 1/252`.
        use_gbm : bool, default ``True``
            If ``True`` evolve log returns via GBM; otherwise simulate simple
            returns and compose them multiplicatively.
        _rng : numpy.random.Generator, optional
            Worker-local RNG supplied by the framework.
        ``**kwargs`` :
            Unused placeholder to maintain parity with the abstract signature.

        Returns
        -------
        float
            Terminal value :math:`V_T`. Under GBM the logarithm follows
            :math:`\log V_T \sim \mathcal{N}\big(\log V_0 + (\mu - \tfrac{1}{2}\sigma^2)T,\;\sigma^2 T\big)`.
        """
        rng = self._rng(_rng, self.rng)
        dt = 1.0 / 252.0  # Daily steps
        n = int(years / dt)
        if use_gbm:  # Geometric Brownian Motion for returns
            mu, sigma = annual_return, volatility
            rets = rng.normal((mu - 0.5 * sigma * sigma) * dt, sigma * np.sqrt(dt), size=n)
            return float(initial_value * np.exp(rets.sum()))
        rets = rng.normal(annual_return * dt, volatility * np.sqrt(dt), size=n)
        return float(initial_value * np.exp(np.log1p(rets).sum()))


# ============================================================================
# Helper Functions for Black-Scholes Simulations
# ============================================================================


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
    option_type : {\"call\", \"put\"}
        Chooses :math:`\Phi_{\text{call}}` or :math:`\Phi_{\text{put}}`.

    Returns
    -------
    float
        Scalar payoff evaluated at the supplied :math:`S_T`.
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
    option_type : {\"call\", \"put\"}
        Payoff family applied to :math:`C_{t_k}`.

    Returns
    -------
    float
        Estimated arbitrage-free price
        :math:`V_0 = \frac{1}{N}\sum_{i=1}^N e^{-r t_{\\tau^{(i)}}} C_{t_{\\tau^{(i)}}}^{(i)}`.
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
    ...     T=.25,
    ...     r=0.05,
    ...     sigma=0.20,
    ...     option_type="call",
    ...     exercise_type="european"
    ... )  # doctest: +SKIP

    Attributes
    ----------
    name : str
        Descriptive identifier, default ``\"Black-Scholes Option Pricing\"``.
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
        r"""
        Price a single option instance under Black–Scholes dynamics.

        Parameters
        ----------
        S0 : float, default ``100``
            Spot :math:`S_0`.
        K : float, default ``100``
            Strike.
        T : float, default ``1``
            Maturity in years.
        r : float, default ``0.05``
            Risk-free rate :math:`r`.
        sigma : float, default ``0.20``
            Volatility :math:`\sigma`.
        option_type : {\"call\", \"put\"}, default ``\"call\"``
            Payoff family.
        exercise_type : {\"european\", \"american\"}, default ``\"european\"``
            Exercise style; the American branch uses a heuristic early-exercise
            proxy described below.
        n_steps : int, default ``252``
            Discrete grid size for the GBM path.
        _rng : numpy.random.Generator, optional
            RNG managed by the framework.
        ``**kwargs`` :
            Additional keyword arguments ignored by the implementation (slot for extensions).

        Returns
        -------
        float
            Discounted payoff sample :math:`e^{-rT}\Phi(S_T)` (European) or the
            maximal discounted intrinsic value along the path (American proxy).

        Raises
        ------
        ValueError
            If ``option_type`` or ``exercise_type`` is invalid.

        Notes
        -----
        * For European contracts the terminal price is simulated via
          :math:`S_T = S_0 \exp((r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z)`
          with :math:`Z \sim \mathcal{N}(0, 1)`; the payoff is then discounted.
        * For American contracts a single path is generated and the intrinsic
          values :math:`\Phi(S_{t_k})` are discounted and maximized. This is a
          coarse approximation; for production quality use many paths with
          :func:`~mcframework.sims._american_exercise_lsm`.
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
        r"""
        Estimate primary Greeks via finite differences.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo draws per pricing run.
        S0, K, T, r, sigma, option_type, exercise_type, n_steps :
            Passed directly to :meth:`single_simulation`.
        parallel : bool, default ``False``
            Reuse parallel execution for each pricing call.
        bump_pct : float, default ``0.01``
            Relative bump :math:`\epsilon` applied to :math:`S_0`, :math:`\sigma`,
            and :math:`r` (i.e. :math:`S_0(1 \pm \epsilon)`).
        time_bump_days : float, default ``1``
            Converts to :math:`\Delta T = \text{time\_bump\_days}/365` for the
            forward-difference theta.

        Returns
        -------
        dict[str, float]
            Mapping with keys

            ``price`` :
                Baseline estimator :math:`V(S_0)`.
            ``delta`` :
                Central difference
                :math:`\frac{V(S_0 + \epsilon S_0) - V(S_0 - \epsilon S_0)}{2 \epsilon S_0}`.
            ``gamma`` :
                Discrete Laplacian
                :math:`\frac{V(S_0 + \epsilon S_0) - 2V(S_0) + V(S_0 - \epsilon S_0)}{(\epsilon S_0)^2}`.
            ``vega`` :
                Central difference in :math:`\sigma`, scaled to a 1% move.
            ``theta`` :
                Forward difference
                :math:`\frac{V(T - \Delta T) - V(T)}{\Delta T}` expressed per day.
            ``rho`` :
                Central difference in :math:`r`, scaled to a 1% rate move.

        Notes
        -----
        All perturbations reuse the same RNG seed (:math:`42`) for variance
        reduction via common random numbers.
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
        vega = (res_vol_up.mean - res_vol_down.mean) / (2 * dsigma)
        # Scale to 1% volatility change (common convention)
        vega = vega * 0.01

        # Theta: ∂V/∂T (forward difference, time decay)
        dT = time_bump_days / 365.0  # Convert days to years
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
            theta = (res_time.mean - V0) / dT
            # Scale to daily theta (common convention)
            theta = theta / 365.0
        else:
            theta = 0.0  # Near expiry, theta calculation becomes unreliable

        # Rho: ∂V/∂r (central difference, scaled to 1% rate change)
        dr = r * bump_pct if r > 0 else 0.0001  # Handle zero rate
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

    Attributes
    ----------
    name : str
        Friendly display name, default ``\"Black-Scholes Path Simulation\"``.
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
        r"""
        Draw a GBM path and return the terminal value :math:`S_T`.

        Parameters
        ----------
        S0, r, sigma, T, n_steps :
            See :func:`~mcframework.sims._simulate_gbm_path`.
        _rng : numpy.random.Generator, optional
            RNG wrapper provided by the framework.
        ``**kwargs`` :
            Extra keyword arguments ignored.

        Returns
        -------
        float
            Final level ``path[-1]`` corresponding to
            :math:`S_T = S_0 \exp((r - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z)`.
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

        Parameters
        ----------
        n_paths : int
            Number of trajectories to sample.
        S0, r, sigma, T, n_steps :
            Same semantics as in :func:`~mcframework.sims._simulate_gbm_path`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_paths, n_steps + 1)`` where the :math:`i`-th row
            stores :math:`\{S_{t_k}^{(i)}\}_{k=0}^n`.
        """
        paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            paths[i] = _simulate_gbm_path(S0, r, sigma, T, n_steps, self.rng)
        return paths
