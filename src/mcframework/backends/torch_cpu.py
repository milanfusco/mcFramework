r"""
Torch CPU execution backend for Monte Carlo simulations.

This module provides:

Classes
    :class:`TorchCPUBackend` — Torch-based batch execution on CPU

The CPU backend enables vectorized execution using PyTorch on CPU,
providing a good balance of speed and compatibility.

Notes
-----
**When to use CPU backend:**

- Baseline testing before GPU deployment
- Systems without GPU acceleration
- Debugging and validation
- Small to medium simulation sizes

**RNG discipline.** Uses explicit :class:`torch.Generator` objects seeded from
:class:`numpy.random.SeedSequence`. Fully deterministic with same seed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .torch_base import import_torch, make_torch_generator

if TYPE_CHECKING:
    from ..simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = ["TorchCPUBackend"]


class TorchCPUBackend:
    r"""
    Torch CPU batch execution backend.

    Uses PyTorch for vectorized execution on CPU. Requires simulations to
    implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` and
    set :attr:`~mcframework.simulation.MonteCarloSimulation.supports_batch` to ``True``.

    Notes
    -----
    **RNG architecture.** Uses explicit :class:`torch.Generator` objects seeded from
    :meth:`numpy.random.SeedSequence.spawn`. This preserves:

    - Deterministic parallel streams
    - Counter-based RNG (Philox) semantics
    - Identical statistical structure across backends

    **Never uses** :func:`torch.manual_seed` (global state).

    Examples
    --------
    >>> backend = TorchCPUBackend()
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq)  # doctest: +SKIP
    """

    device_type: str = "cpu"

    _MAX_BATCH: int = 10_000_000

    def __init__(self, max_batch_size: int | None = None):
        """
        Initialize Torch CPU backend.

        Parameters
        ----------
        max_batch_size : int or None, default None
            Maximum number of simulations per batch.  If *None* the
            class default ``_MAX_BATCH`` (10 M) is used.  Workloads larger
            than this are split into batches to keep memory bounded.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        th = import_torch()
        self.device = th.device("cpu")
        if max_batch_size is None:
            self._max_batch = self._MAX_BATCH
        elif max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        else:
            self._max_batch = max_batch_size

    def run(
        self,
        sim: MonteCarloSimulation,
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **_simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations using Torch CPU batch execution.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run. Must have
            :attr:`~mcframework.simulation.MonteCarloSimulation.supports_batch` = ``True``
            and implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.
        n_simulations : int
            Number of simulation draws to perform.
        seed_seq : SeedSequence or None
            Seed sequence for reproducible random streams.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **_simulation_kwargs : Any
            Ignored for Torch backend (batch method handles all parameters).

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations, ...)``.

        Raises
        ------
        ValueError
            If the simulation does not support batch execution.
        NotImplementedError
            If the simulation does not implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.
        """
        th = import_torch()

        if not getattr(sim, "supports_batch", False):
            raise ValueError(
                f"Simulation '{sim.name}' does not support Torch batch execution. "
                "Set supports_batch = True and implement torch_batch()."
            )

        logger.info(
            "Computing %d simulations using Torch CPU batch...",
            n_simulations,
        )

        if n_simulations <= self._max_batch:
            generator = make_torch_generator(self.device, seed_seq)
            samples = sim.torch_batch(n_simulations, device=self.device, generator=generator)
            samples = samples.detach().to(th.float64)
            if progress_callback:
                progress_callback(n_simulations, n_simulations)
            return samples.numpy()

        n_batches = (n_simulations + self._max_batch - 1) // self._max_batch
        batch_seeds = seed_seq.spawn(n_batches) if seed_seq else [None] * n_batches
        parts: list[np.ndarray] = []
        completed = 0
        for bs in batch_seeds:
            batch_n = min(self._max_batch, n_simulations - completed)
            generator = make_torch_generator(self.device, bs)
            samples = sim.torch_batch(batch_n, device=self.device, generator=generator)
            parts.append(samples.detach().to(th.float64).numpy())
            completed += batch_n
            if progress_callback:
                progress_callback(completed, n_simulations)
        return np.concatenate(parts)
