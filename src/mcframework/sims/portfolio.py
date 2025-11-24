"""Portfolio wealth simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from ..core import MonteCarloSimulation

__all__ = ["PortfolioSimulation"]


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
        Default registry label ``"Portfolio Simulation"``.
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

