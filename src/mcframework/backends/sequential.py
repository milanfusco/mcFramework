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
        seed_seq: np.random.SeedSequence | None,
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
        seed_seq : SeedSequence or None
            Seed sequence for creating a deterministic RNG stream.
            When provided, a dedicated RNG is spawned and passed to each
            ``single_simulation`` call via the ``_rng`` keyword, matching
            the reproducibility semantics of parallel backends.
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
        step = max(1, n_simulations // 100)

        if seed_seq is not None:
            child_seq = seed_seq.spawn(1)[0]
            local_rng = np.random.Generator(np.random.Philox(child_seq))
            for i in range(n_simulations):
                results[i] = float(sim.single_simulation(_rng=local_rng, **simulation_kwargs))
                if progress_callback and (((i + 1) % step == 0) or (i + 1 == n_simulations)):
                    progress_callback(i + 1, n_simulations)
        else:
            for i in range(n_simulations):
                results[i] = float(sim.single_simulation(**simulation_kwargs))
                if progress_callback and (((i + 1) % step == 0) or (i + 1 == n_simulations)):
                    progress_callback(i + 1, n_simulations)

        return results
