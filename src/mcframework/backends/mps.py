r"""
Apple Metal Performance Shaders (MPS) backend for Monte Carlo simulations.

This module provides GPU-accelerated execution on Apple Silicon (M1/M2/M3/M4 Macs)
using PyTorch's MPS backend.

Classes
    :class:`MPSBackend` — Apple Silicon GPU execution via Metal

Functions
    :func:`is_mps_available` — Check if MPS backend is available
    :func:`validate_mps_device` — Validate MPS device availability

Notes
-----
**MPS dtype policy.** MPS performs best with float32. Sampling uses float32
internally, but results are promoted to float64 after moving to CPU to ensure
stats engine precision. This is transparent to the user.

**MPS determinism caveat.** Torch MPS preserves RNG stream structure but does
not guarantee bitwise reproducibility due to Metal backend scheduling and
float32 arithmetic. Statistical properties (mean, variance, CI coverage)
remain correct.

**System Requirements:**
    - macOS 12.3 or later
    - Apple Silicon (M1, M2, M3, M4 series)
    - PyTorch with MPS support (2.0+)

Examples
--------
>>> from mcframework.backends import MPSBackend
>>> backend = MPSBackend()
>>> results = backend.run(sim, 1_000_000, seed_seq)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .torch_base import TorchBackend, require_torch

if TYPE_CHECKING:
    import torch
    from ..simulation import MonteCarloSimulation


logger = logging.getLogger(__name__)

__all__ = [
    "MPSBackend",
    "is_mps_available",
    "validate_mps_device",
]


def is_mps_available() -> bool:
    """
    Check if the MPS backend is available on this system.

    Returns
    -------
    bool
        True if MPS is available and built into PyTorch.

    Examples
    --------
    >>> if is_mps_available():
    ...     backend = MPSBackend()  # doctest: +SKIP
    """
    try:
        th = require_torch()
        return (
            hasattr(th.backends, "mps")
            and th.backends.mps.is_available()
            and th.backends.mps.is_built()
        )
    except ImportError:
        return False


def validate_mps_device() -> None:
    """
    Validate that the MPS device is available.

    Raises
    ------
    RuntimeError
        If MPS is not available or not built into PyTorch.

    Examples
    --------
    >>> validate_mps_device()  # Raises on non-Apple Silicon  # doctest: +SKIP
    """
    th = require_torch()

    if not th.backends.mps.is_available():
        raise RuntimeError(
            "MPS device requested but not available. "
            "MPS requires macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)."
        )
    if not th.backends.mps.is_built():
        raise RuntimeError(
            "MPS device requested but PyTorch was not built with MPS support. "
            "Reinstall PyTorch with MPS support: pip install torch"
        )


class MPSBackend(TorchBackend):
    r"""
    Apple Silicon GPU backend using Metal Performance Shaders.

    Provides GPU-accelerated batch execution on M1/M2/M3/M4 Macs via
    PyTorch's MPS backend. Requires simulations to implement :meth:`torch_batch`
    and set ``supports_batch = True``.

    Notes
    -----
    **Float32 execution.** MPS performs best with float32 arithmetic. The backend
    automatically handles dtype conversion:

    1. Simulation runs in float32 on the GPU
    2. Results are moved to CPU
    3. Results are promoted to float64 for stats engine precision

    **Determinism caveat.** MPS preserves RNG stream structure but does not
    guarantee bitwise reproducibility across runs due to:

    - Metal backend scheduling variations
    - Float32 arithmetic precision
    - GPU kernel execution ordering

    Statistical properties (mean, variance, confidence intervals) remain correct.

    **Performance tips:**

    - Use large batch sizes (100K+) to amortize GPU transfer overhead
    - Ensure simulations return float32 tensors (not float64)
    - Avoid frequent small allocations in ``torch_batch``

    Examples
    --------
    >>> backend = MPSBackend()
    >>> results = backend.run(sim, n_simulations=1_000_000, seed_seq=seed_seq)  # doctest: +SKIP

    See Also
    --------
    :class:`TorchBackend` : Base Torch backend for CPU execution
    :class:`CUDABackend` : NVIDIA GPU backend
    """

    device_type: str = "mps"

    def __init__(self):
        """
        Initialize MPS backend for Apple Silicon GPU execution.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        RuntimeError
            If MPS is not available on this system.
        """
        th = require_torch()

        # Validate MPS availability
        validate_mps_device()

        self.device_type = "mps"
        self.device = th.device("mps")

        logger.debug("Initialized MPS backend on Apple Silicon GPU")

    def _validate_device(self) -> None:
        """Validate that MPS device is still available."""
        validate_mps_device()

    def _post_process(self, samples: "torch.Tensor") -> np.ndarray:
        """
        Post-process MPS samples with float32 → float64 promotion.

        MPS operates in float32 for performance. This method:
        1. Moves samples to CPU (required before dtype conversion)
        2. Promotes to float64 for stats engine precision
        3. Converts to NumPy array

        Parameters
        ----------
        samples : torch.Tensor
            Raw float32 samples from MPS execution.

        Returns
        -------
        np.ndarray
            Float64 samples as NumPy array.
        """
        th = require_torch()

        # Move to CPU first (required before float64 conversion on MPS)
        samples = samples.detach().cpu()

        # Promote to float64 for stats engine precision
        # MPS returns float32; stats computations need float64 accuracy
        samples = samples.to(th.float64)

        return samples.numpy()

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations on Apple Silicon GPU via MPS.

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
            Ignored for MPS backend.

        Returns
        -------
        np.ndarray
            Array of float64 simulation results with shape ``(n_simulations,)``.

        Notes
        -----
        For best performance, use batch sizes of 100K+ to amortize GPU transfer
        overhead. Small batches may be slower than CPU due to memory transfer costs.
        """
        logger.info(
            "Running %d simulations on Apple Silicon GPU (MPS)...",
            n_simulations
        )
        return super().run(sim, n_simulations, seed_seq, progress_callback, **simulation_kwargs)

