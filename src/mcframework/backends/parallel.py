r"""
Parallel execution backends for Monte Carlo simulations.

This module provides:

Classes
    :class:`ThreadBackend` — Thread-based parallelism using ThreadPoolExecutor
    :class:`ProcessBackend` — Process-based parallelism using ProcessPoolExecutor
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .base import make_blocks, worker_run_chunk

if TYPE_CHECKING:
    from ..simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = [
    "ThreadBackend",
    "ProcessBackend",
]

# Default configuration constants
_CHUNKS_PER_WORKER = 8  # Number of chunks per worker for load balancing


class ThreadBackend:
    r"""
    Thread-based parallel execution backend.

    Uses :class:`concurrent.futures.ThreadPoolExecutor` for parallel execution.
    Effective when NumPy releases the GIL (most numerical operations).

    Parameters
    ----------
    n_workers : int
        Number of worker threads to use.
    chunks_per_worker : int, default 8
        Number of work chunks per worker for load balancing.

    Examples
    --------
    >>> backend = ThreadBackend(n_workers=4)
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq, progress_callback=None)
    """

    def __init__(self, n_workers: int, chunks_per_worker: int = _CHUNKS_PER_WORKER):
        self.n_workers = n_workers
        self.chunks_per_worker = chunks_per_worker

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations in parallel using threads.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run.
        n_simulations : int
            Number of simulation draws to perform.
        seed_seq : SeedSequence or None
            Seed sequence for spawning independent RNG streams per chunk.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **simulation_kwargs : Any
            Additional keyword arguments passed to ``single_simulation``.

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations,)``.
        """
        blocks, child_seqs = self._prepare_blocks(n_simulations, seed_seq)
        results = np.empty(n_simulations, dtype=float)
        completed = 0
        max_workers = min(self.n_workers, len(blocks))

        def _work(args):
            (a, b), ss = args
            rng = np.random.Generator(np.random.Philox(ss))
            out = np.empty(b - a, dtype=float)
            for k in range(out.size):
                out[k] = float(sim.single_simulation(_rng=rng, **simulation_kwargs))
            return (a, b), out

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_work, (blk, ss)) for blk, ss in zip(blocks, child_seqs)]
            for f in as_completed(futs):
                (i, j), arr = f.result()
                results[i:j] = arr
                completed += j - i
                if progress_callback:
                    progress_callback(completed, n_simulations)  # pragma: no cover

        return results

    def _prepare_blocks(
        self, n_simulations: int, seed_seq: np.random.SeedSequence | None
    ) -> tuple[list[tuple[int, int]], list[np.random.SeedSequence]]:
        """Prepare work blocks and independent random seeds."""
        block_size = max(1, n_simulations // (self.n_workers * self.chunks_per_worker))
        blocks = make_blocks(n_simulations, block_size)

        if seed_seq is not None:
            child_seqs = seed_seq.spawn(len(blocks))
        else:
            child_seqs = [np.random.SeedSequence() for _ in range(len(blocks))]

        return blocks, child_seqs


class ProcessBackend:
    r"""
    Process-based parallel execution backend.

    Uses :class:`concurrent.futures.ProcessPoolExecutor` with spawn context
    for parallel execution. Required on Windows or when thread-safety is a concern.

    Parameters
    ----------
    n_workers : int
        Number of worker processes to use.
    chunks_per_worker : int, default 8
        Number of work chunks per worker for load balancing.

    Notes
    -----
    The simulation instance must be pickleable for process-based execution.

    Examples
    --------
    >>> backend = ProcessBackend(n_workers=4)
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq, progress_callback=None)
    """

    def __init__(self, n_workers: int, chunks_per_worker: int = _CHUNKS_PER_WORKER):
        self.n_workers = n_workers
        self.chunks_per_worker = chunks_per_worker

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations in parallel using processes.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run. Must be pickleable.
        n_simulations : int
            Number of simulation draws to perform.
        seed_seq : SeedSequence or None
            Seed sequence for spawning independent RNG streams per chunk.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **simulation_kwargs : Any
            Additional keyword arguments passed to ``single_simulation``.

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations,)``.
        """
        blocks, child_seqs = self._prepare_blocks(n_simulations, seed_seq)
        results = np.empty(n_simulations, dtype=float)
        completed = 0
        max_workers = min(self.n_workers, len(blocks))

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as ex:
            futs = []
            for (i, j), ss in zip(blocks, child_seqs):
                f = ex.submit(worker_run_chunk, sim, j - i, ss, dict(simulation_kwargs))
                f.blk = (i, j)  # type: ignore[attr-defined]
                futs.append(f)
            try:
                for f in as_completed(futs):
                    i, j = f.blk  # type: ignore[attr-defined]
                    chunk = f.result()
                    results[i:j] = chunk
                    completed += j - i
                    if progress_callback:
                        progress_callback(completed, n_simulations)  # pragma: no cover
            except KeyboardInterrupt:  # pragma: no cover
                for f in futs:
                    f.cancel()
                raise

        return results

    def _prepare_blocks(
        self, n_simulations: int, seed_seq: np.random.SeedSequence | None
    ) -> tuple[list[tuple[int, int]], list[np.random.SeedSequence]]:
        """Prepare work blocks and independent random seeds."""
        block_size = max(1, n_simulations // (self.n_workers * self.chunks_per_worker))
        blocks = make_blocks(n_simulations, block_size)

        if seed_seq is not None:
            child_seqs = seed_seq.spawn(len(blocks))
        else:
            child_seqs = [np.random.SeedSequence() for _ in range(len(blocks))]

        return blocks, child_seqs
