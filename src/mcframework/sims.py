"""
mcframework.sims
====================
Built-in simulations for mcframework.

The Simulations module provides classes and functions to create, manage, and
execute different types of simulations. Currently supported simulations include
a pi estimation simulation and a portfolio risk simulation. The module is
designed to implement additional simulation types from the abstract base class
`MonteCarloSimulation` in the core module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from .core import MonteCarloSimulation

__all__ = ["PiEstimationSimulation", "PortfolioSimulation"]


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
