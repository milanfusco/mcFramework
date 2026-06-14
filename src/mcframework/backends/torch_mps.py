r"""
Torch MPS (Metal Performance Shaders) backend for Apple Silicon.

This module provides:

Classes
    :class:`TorchMPSBackend` — GPU-accelerated batch execution on Apple Silicon

Functions
    :func:`is_mps_available` — Check MPS availability
    :func:`validate_mps_device` — Validate MPS is usable

The MPS backend enables GPU-accelerated Monte Carlo simulations on
Apple Silicon Macs (M1/M2/M3/M4) using Metal Performance Shaders.

Notes
-----
**MPS determinism caveat.** Torch MPS preserves RNG stream structure but does
not guarantee bitwise reproducibility due to Metal backend scheduling and
float32 arithmetic. Statistical properties (mean, variance, CI coverage)
remain correct.

**Dtype policy.** MPS performs best with float32. Sampling uses float32,
but results are promoted to float64 on CPU before returning to ensure
stats engine precision.

**System requirements:**
- macOS 12.3 (Monterey) or later
- Apple Silicon (M1, M2, M3, M4 series)
- PyTorch built with MPS support
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

__all__ = [
    "TorchMPSBackend",
    "is_mps_available",
    "validate_mps_device",
]


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns
    -------
    bool
        True if MPS is available and PyTorch was built with MPS support.

    Examples
    --------
    >>> if is_mps_available():
    ...     backend = TorchMPSBackend()  # doctest: +SKIP
    """
    try:
        th = import_torch()
        return (
            hasattr(th.backends, "mps")
            and th.backends.mps.is_available()
            and th.backends.mps.is_built()
        )
    except ImportError:
        return False


def validate_mps_device() -> None:
    """
    Validate that MPS device is available and usable.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    RuntimeError
        If MPS is not available or not built into PyTorch.

    Examples
    --------
    >>> validate_mps_device()  # doctest: +SKIP
    """
    th = import_torch()

    if not th.backends.mps.is_available():
        raise RuntimeError(
            "MPS device requested but not available. "
            "MPS requires macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)."
        )
    if not th.backends.mps.is_built():
        raise RuntimeError(
            "MPS device requested but PyTorch was not built with MPS support. "
            "Reinstall PyTorch with MPS support enabled."
        )


class TorchMPSBackend:
    r"""
    Torch MPS batch execution backend for Apple Silicon GPUs.

    Uses PyTorch with MPS (Metal Performance Shaders) backend for GPU-accelerated
    execution on Apple Silicon Macs and leverage unified memory architecture. 
    Requires simulations to implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` and 
    set :attr:`~mcframework.simulation.MonteCarloSimulation.supports_batch` to ``True`` to 
    enable Metal Performance Shaders GPU-accelerated batch execution.

    Notes
    -----
    **RNG architecture.** Uses explicit :class:`~torch.Generator` objects seeded from
    :class:`~numpy.random.SeedSequence` via :meth:`~numpy.random.SeedSequence.spawn`. This preserves:

    - Deterministic parallel streams (best-effort on MPS)
    - Counter-based RNG (Philox) semantics
    - Correct statistical structure

    **Never uses** :meth:`~torch.Generator.manual_seed` (global state).

    **Dtype policy.** MPS performs best with :meth:`~torch.Tensor.float` (float32):

    - Sampling uses :meth:`~torch.Tensor.float` (float32) on device
    - Results moved to CPU and promoted to :meth:`~torch.Tensor.double` (float64). 
    - The framework converts the results to :class:`numpy.ndarray` of :class:`numpy.double` (float64)
    for stats engine compatibility.

    **MPS determinism caveat.** Torch MPS preserves RNG stream structure but
    does not guarantee bitwise reproducibility due to:

    - Metal backend scheduling variations
    - float32 arithmetic rounding
    - GPU kernel execution order

    Statistical properties (mean, variance, CI coverage) remain correct
    despite potential bitwise differences between runs (see ``tests/test_torch_mps.py``)

    Examples
    --------
    >>> if is_mps_available():
    ...     backend = TorchMPSBackend()
    ...     results = backend.run(sim, n_simulations=1_000_000, seed_seq=seed_seq)
    ... # doctest: +SKIP

    See Also
    --------
    :func:`is_mps_available` : Check MPS availability before instantiation.
    :class:`TorchCPUBackend` : Fallback for non-Apple systems.
    """

    device_type: str = "mps"

    _MAX_BATCH: int = 10_000_000

    def __init__(self, max_batch_size: int | None = None):
        """
        Initialize Torch MPS backend.

        Parameters
        ----------
        max_batch_size : int or None, default None
            Maximum number of simulations per batch.  If *None* the
            class default ``_MAX_BATCH`` (10 M) is used.  Workloads larger
            than this are split into batches to keep GPU memory bounded.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        RuntimeError
            If MPS is not available on this system.
        """
        validate_mps_device()
        th = import_torch()
        self.device = th.device("mps")
        if max_batch_size is None:
            self._max_batch = self._MAX_BATCH
        elif max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        else:
            self._max_batch = max_batch_size

    def _run_batch(self, sim, n, seed_seq):
        """Execute a single batch, returning a float64 NumPy array."""
        th = import_torch()
        generator = make_torch_generator(self.device, seed_seq)
        samples = sim.torch_batch(n, device=self.device, generator=generator)
        return samples.detach().cpu().to(th.float64).numpy()

    def run(
        self,
        sim: MonteCarloSimulation,
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **_simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations using Torch MPS batch execution.

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
            Array of simulation results with shape ``(n_simulations,)``.
            Results are float64 despite MPS using float32 internally.

        Raises
        ------
        ValueError
            If the simulation does not support batch execution.
        NotImplementedError
            If the simulation does not implement
            :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.
        """
        if not getattr(sim, "supports_batch", False):
            raise ValueError(
                f"Simulation '{sim.name}' does not support Torch batch execution. "
                "Set supports_batch = True and implement torch_batch()."
            )

        logger.info(
            "Computing %d simulations using Torch MPS (Apple Silicon GPU)...",
            n_simulations,
        )

        if n_simulations <= self._max_batch:
            result = self._run_batch(sim, n_simulations, seed_seq)
            if progress_callback:
                progress_callback(n_simulations, n_simulations)
            return result

        n_batches = (n_simulations + self._max_batch - 1) // self._max_batch
        batch_seeds = seed_seq.spawn(n_batches) if seed_seq else [None] * n_batches
        parts: list[np.ndarray] = []
        completed = 0
        for bs in batch_seeds:
            batch_n = min(self._max_batch, n_simulations - completed)
            parts.append(self._run_batch(sim, batch_n, bs))
            completed += batch_n
            if progress_callback:
                progress_callback(completed, n_simulations)
        return np.concatenate(parts)
