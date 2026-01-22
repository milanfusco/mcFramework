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
from typing import TYPE_CHECKING, Any, Callable

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
    despite potential bitwise differences between runs. (see ``TestMPSDeterminism`` 
    in ``tests/test_torch_backend.py`` for actual tests)

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

    def __init__(self):
        """
        Initialize Torch MPS backend.

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

    def run(
        self,
        sim: "MonteCarloSimulation",
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
            If the simulation does not implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.

        Notes
        -----
        The dtype conversion flow is:

        1. :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` returns :meth:`~torch.Tensor.float` (float32) on MPS device.
        2. :class:`~torch.Tensor` moved to CPU via :meth:`~torch.Tensor.detach` and :meth:`~torch.Tensor.cpu`
        3. Promoted to :meth:`~torch.Tensor.double` (float64) via :meth:`~torch.Tensor.to`
        4. Converted to :class:`~numpy.ndarray` of :class:`~numpy.double` (float64) via :meth:`~torch.Tensor.numpy`

        This ensures stats engine precision while maximizing MPS performance.
        """# noqa: E501 pylint: disable=line-too-long
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
            "Computing %d simulations using Torch MPS (Apple Silicon GPU)...",
            n_simulations
        )

        # Execute the vectorized batch with explicit generator
        # torch_batch should return float32 for MPS compatibility
        samples = sim.torch_batch(n_simulations, device=self.device, generator=generator)

        # Move to CPU first (required before float64 conversion for MPS)
        samples = samples.detach().cpu()

        # Promote to float64 for stats engine precision
        samples = samples.to(th.float64)

        # Report completion (batch execution is atomic)
        if progress_callback:
            progress_callback(n_simulations, n_simulations)

        # Convert to NumPy for stats engine compatibility
        return samples.numpy()
