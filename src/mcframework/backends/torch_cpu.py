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

**RNG discipline.** Uses explicit ``torch.Generator`` objects seeded from
:class:`numpy.random.SeedSequence`. Fully deterministic with same seed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

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
    implement :meth:`torch_batch` and set ``supports_batch = True``.

    Notes
    -----
    **RNG architecture.** Uses explicit ``torch.Generator`` objects seeded from
    :class:`numpy.random.SeedSequence` via ``spawn()``. This preserves:

    - Deterministic parallel streams
    - Counter-based RNG (Philox) semantics
    - Identical statistical structure across backends

    **Never uses** ``torch.manual_seed()`` (global state).

    Examples
    --------
    >>> backend = TorchCPUBackend()
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq)  # doctest: +SKIP
    """

    device_type: str = "cpu"

    def __init__(self):
        """
        Initialize Torch CPU backend.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        th = import_torch()
        self.device = th.device("cpu")

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations using Torch CPU batch execution.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run. Must have ``supports_batch = True``
            and implement :meth:`torch_batch`.
        n_simulations : int
            Number of simulation draws to perform.
        seed_seq : SeedSequence or None
            Seed sequence for reproducible random streams.
        progress_callback : callable or None
            Optional callback ``f(completed, total)`` for progress reporting.
        **simulation_kwargs : Any
            Ignored for Torch backend (batch method handles all parameters).

        Returns
        -------
        np.ndarray
            Array of simulation results with shape ``(n_simulations,)``.

        Raises
        ------
        ValueError
            If the simulation does not support batch execution.
        NotImplementedError
            If the simulation does not implement :meth:`torch_batch`.
        """
        th = import_torch()

        # Validate simulation supports batch execution
        if not getattr(sim, "supports_batch", False):
            raise ValueError(
                f"Simulation '{sim.name}' does not support Torch batch execution. "
                "Set supports_batch = True and implement torch_batch()."
            )

        # Create explicit generator from SeedSequence (never use global RNG)
        generator = make_torch_generator(self.device, seed_seq)

        logger.info(
            "Computing %d simulations using Torch CPU batch...",
            n_simulations
        )

        # Execute the vectorized batch with explicit generator
        samples = sim.torch_batch(n_simulations, device=self.device, generator=generator)

        # Ensure float64 for stats engine precision
        samples = samples.detach().to(th.float64)

        # Report completion (batch execution is atomic)
        if progress_callback:
            progress_callback(n_simulations, n_simulations)

        # Convert to NumPy for stats engine compatibility
        return samples.numpy()

