"""Pi estimation simulation."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.random import Generator

from ..core import MonteCarloSimulation

__all__ = ["PiEstimationSimulation"]


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

