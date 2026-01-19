r"""
Base classes and utilities for execution backends.

This module provides:

Protocol
    :class:`ExecutionBackend` — Interface for simulation execution strategies

Functions
    :func:`make_blocks` — Chunking helper for parallel work distribution
    :func:`worker_run_chunk` — Top-level worker for process-based parallelism

Helpers
    :func:`is_windows_platform` — Platform detection for backend selection
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..simulation import MonteCarloSimulation

__all__ = [
    "ExecutionBackend",
    "make_blocks",
    "worker_run_chunk",
    "is_windows_platform",
]


def is_windows_platform() -> bool:
    """Return True when running on a Windows platform."""
    return sys.platform.startswith("win") or (sys.platform == "cli")


def make_blocks(n: int, block_size: int = 10_000) -> list[tuple[int, int]]:
    r"""
    Partition an integer range :math:`[0, n)` into half-open blocks :math:`(i, j)`.

    Parameters
    ----------
    n : int
        Total number of items.
    block_size : int, default: 10_000
        Target block length.

    Returns
    -------
    list of tuple[int, int]
        List of ``(i, j)`` index pairs covering ``[0, n)``.

    Examples
    --------
    >>> make_blocks(5, block_size=2)
    [(0, 2), (2, 4), (4, 5)]
    """
    blocks = []
    i = 0
    while i < n:
        j = min(i + block_size, n)
        blocks.append((i, j))
        i = j
    return blocks


def worker_run_chunk(
    sim: "MonteCarloSimulation",
    chunk_size: int,
    seed_seq: np.random.SeedSequence,
    simulation_kwargs: dict[str, Any],
) -> list[float]:
    r"""
    Execute a small batch of single simulations in a **separate worker**.

    Parameters
    ----------
    sim :
        Simulation instance to call (:meth:`MonteCarloSimulation.single_simulation`).
        Must be pickleable when used with a process backend.
    chunk_size : int
        Number of draws to compute in this worker.
    seed_seq : :class:`numpy.random.SeedSequence`
        Seed sequence for creating an **independent** RNG stream in the worker.
    simulation_kwargs : dict
        Keyword arguments forwarded to :meth:`MonteCarloSimulation.single_simulation`.

    Returns
    -------
    list[float]
        The simulated values.

    Notes
    -----
    Uses :class:`numpy.random.Philox` to spawn a deterministic, independent stream per
    worker chunk.
    """
    bitgen = np.random.Philox(seed_seq)
    local_rng = np.random.Generator(bitgen)
    return [float(sim.single_simulation(_rng=local_rng, **simulation_kwargs)) for _ in range(chunk_size)]


class ExecutionBackend(Protocol):
    r"""
    Protocol defining the interface for execution backends.

    Backends are responsible for executing simulation draws and returning results.
    They handle the details of sequential vs parallel execution, thread vs process
    pools, and progress reporting.
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
        Run simulation draws and return results.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run.
        n_simulations : int
            Number of simulation draws to perform.
        seed_seq : SeedSequence or None
            Seed sequence for reproducible random streams.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **simulation_kwargs : Any
            Additional keyword arguments passed to ``single_simulation``.

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations,)``.
        """
