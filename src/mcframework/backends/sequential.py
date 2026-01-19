r"""
Sequential execution backend for Monte Carlo simulations.

This module provides a single-threaded execution strategy that runs
simulations sequentially with optional progress reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from ..simulation import MonteCarloSimulation

__all__ = ["SequentialBackend"]


class SequentialBackend:
    r"""
    Sequential (single-threaded) execution backend.

    Executes simulation draws one at a time on the main thread.
    Suitable for small simulations or debugging.

    Examples
    --------
    >>> backend = SequentialBackend()
    >>> results = backend.run(sim, n_simulations=1000, seed_seq=None, progress_callback=None)
    """

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        _seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations sequentially on a single thread.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run.
        n_simulations : int
            Number of simulation draws to perform.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **simulation_kwargs : Any
            Additional keyword arguments passed to ``single_simulation``.

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations,)``.
        """
        results = np.empty(n_simulations, dtype=float)
        # Report progress every 1% of simulations
        step = max(1, n_simulations // 100)

        for i in range(n_simulations):
            results[i] = float(sim.single_simulation(**simulation_kwargs))
            if progress_callback and (((i + 1) % step == 0) or (i + 1 == n_simulations)):
                progress_callback(i + 1, n_simulations)

        return results
