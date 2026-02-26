r"""
Torch execution backend for GPU-accelerated Monte Carlo simulations.

This module provides a unified interface for Torch-based backends:

Classes
    :class:`TorchBackend` — Factory that selects appropriate device backend

Device-Specific Backends
    :class:`TorchCPUBackend` — CPU execution (torch_cpu.py)
    :class:`TorchMPSBackend` — Apple Silicon GPU (torch_mps.py)
    :class:`TorchCUDABackend` — NVIDIA GPU (torch_cuda.py, stub)

Utilities
    :func:`validate_torch_device` — Validate device availability
    :func:`make_torch_generator` — Create explicit RNG generators
    :data:`VALID_TORCH_DEVICES` — Supported device types

Device Support
    - ``cpu`` — Safe default, works everywhere
    - ``mps`` — Apple Metal Performance Shaders (M1/M2/M3/M4 Macs)
    - ``cuda`` — NVIDIA Compute Unified Device Architecture (CUDA 12.x with CuPy for CuRAND)

Notes
-----
Use :class:`TorchBackend` as the main entry point—it automatically
selects the appropriate device-specific backend based on the ``device``
parameter.

Example
-------
>>> from mcframework.backends import TorchBackend
>>> backend = TorchBackend(device="mps")  # Auto-selects TorchMPSBackend
>>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq)  # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

# Import from submodules
from .torch_base import (
    VALID_TORCH_DEVICES,
    make_torch_generator,
)
from .torch_cpu import TorchCPUBackend
from .torch_cuda import TorchCUDABackend, is_cuda_available, validate_cuda_device
from .torch_mps import TorchMPSBackend, is_mps_available, validate_mps_device

if TYPE_CHECKING:
    from ..simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = [
    # Main backend class
    "TorchBackend",
    # Device-specific backends
    "TorchCPUBackend",
    "TorchMPSBackend",
    "TorchCUDABackend",
    # Validation functions
    "validate_torch_device",
    "is_mps_available",
    "is_cuda_available",
    "validate_mps_device",
    "validate_cuda_device",
    # Utilities
    "make_torch_generator",
    "VALID_TORCH_DEVICES",
]


def validate_torch_device(device_type: str) -> None:
    r"""
    Validate that the requested Torch device is available.

    Parameters
    ----------
    device_type : str
        Device type to validate (``"cpu"``, ``"mps"``, ``"cuda"``).

    Raises
    ------
    ValueError
        If the device type is not recognized.
    RuntimeError
        If the device is not available on this system.

    Examples
    --------
    >>> validate_torch_device("cpu")  # Always succeeds
    >>> validate_torch_device("mps")  # Succeeds on Apple Silicon  # doctest: +SKIP
    """
    if device_type not in VALID_TORCH_DEVICES:
        raise ValueError(
            f"torch_device must be one of {VALID_TORCH_DEVICES}, got '{device_type}'"
        )

    if device_type == "cpu":
        return  # Always available

    if device_type == "mps":
        validate_mps_device()
        return

    if device_type == "cuda":
        validate_cuda_device()
        return


class TorchBackend:
    r"""
    Factory class that creates and wraps the appropriate device-specific backend.

    This is a factory class that creates and wraps the appropriate
    device-specific backend (:class:`TorchCPUBackend`, :class:`TorchMPSBackend`,
    or :class:`TorchCUDABackend`) based on the ``device`` parameter.

    Parameters
    ----------
    device : {"cpu", "mps", "cuda"}, default ``"cpu"``
        Torch device for computation:

        - ``"cpu"`` — Uses :class:`TorchCPUBackend`
        - ``"mps"`` — Uses :class:`TorchMPSBackend` (Apple Silicon)
        - ``"cuda"`` — Uses :class:`TorchCUDABackend` (NVIDIA, stub)

    Notes
    -----
    **Delegation model.** This class delegates all execution to the
    device-specific backend. It exists to provide a unified interface
    and for backward compatibility.

    **Device selection.** The backend is selected at construction time
    based on the ``device`` parameter. Device availability is validated
    during construction.

    Examples
    --------
    >>> # CPU execution
    >>> backend = TorchBackend(device="cpu")
    >>> results = backend.run(sim, n_simulations=100000, seed_seq=seed_seq)  # doctest: +SKIP

    >>> # Apple Silicon GPU
    >>> backend = TorchBackend(device="mps")  # doctest: +SKIP
    >>> results = backend.run(sim, n_simulations=1000000, seed_seq=seed_seq)  # doctest: +SKIP

    >>> # NVIDIA GPU (CUDA 12.x with CuPy for CuRAND)
    >>> backend = TorchBackend(device="cuda")  # doctest: +SKIP

    See Also
    --------
    :class:`TorchCPUBackend` : Direct CPU backend access.
    :class:`TorchMPSBackend` : Direct MPS backend access.
    :class:`TorchCUDABackend` : Direct CUDA backend access.
    """

    def __init__(self, device: str = "cpu", **device_kwargs: Any):
        """
        Initialize Torch backend with specified device.

        Parameters
        ----------
        device : {"cpu", "mps", "cuda"}, default ``"cpu"``
            Torch device for computation.
        **device_kwargs : Any
            Device-specific configuration options:

            **CUDA options** (ignored for cpu/mps):

            - ``device_id`` : int, default 0 — CUDA device index
            - ``use_curand`` : bool, default False — Use cuRAND via CuPy
            - ``batch_size`` : int or None — Fixed batch size (None = adaptive)
            - ``use_streams`` : bool, default True — Enable CUDA streams

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If the device type is not recognized.
        RuntimeError
            If the requested device is not available.

        Examples
        --------
        >>> # CPU (no kwargs needed)
        >>> backend = TorchBackend(device="cpu")  # doctest: +SKIP

        >>> # MPS (no kwargs needed)
        >>> backend = TorchBackend(device="mps")  # doctest: +SKIP

        >>> # CUDA with default settings
        >>> backend = TorchBackend(device="cuda")  # doctest: +SKIP

        >>> # CUDA with custom settings
        >>> backend = TorchBackend(
        ...     device="cuda",
        ...     device_id=0,
        ...     use_curand=True,
        ...     batch_size=100_000,
        ...     use_streams=True,
        ... )  # doctest: +SKIP
        """
        # Check torch availability (optional dependency)
        import importlib.util  # pylint: disable=import-outside-toplevel

        if importlib.util.find_spec("torch") is None:
            raise ImportError(
                "Torch backend requires PyTorch. Install with: pip install mcframework[gpu]"
            )

        # Validate device type and availability
        validate_torch_device(device)

        self.device_type = device

        # Create device-specific backend with appropriate kwargs
        if device == "cpu":
            if device_kwargs:
                logger.warning(
                    "CPU backend ignores device_kwargs: %s", list(device_kwargs.keys())
                )
            self._backend = TorchCPUBackend()

        elif device == "mps":
            if device_kwargs:
                logger.warning(
                    "MPS backend ignores device_kwargs: %s", list(device_kwargs.keys())
                )
            self._backend = TorchMPSBackend()

        elif device == "cuda":
            # Extract CUDA-specific kwargs
            cuda_kwargs = {
                k: v for k, v in device_kwargs.items()
                if k in ("device_id", "use_curand", "batch_size", "use_streams")
            }
            extra_kwargs = set(device_kwargs.keys()) - set(cuda_kwargs.keys())
            if extra_kwargs:
                logger.warning(
                    "CUDA backend ignores unknown kwargs: %s", list(extra_kwargs)
                )
            self._backend = TorchCUDABackend(**cuda_kwargs)

        else:  # pragma: no cover
            # Should not reach here due to validation
            raise ValueError(f"Unknown device: {device}")

        # Expose device from underlying backend
        self.device = self._backend.device

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations using the device-specific Torch backend.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run. Must have ``supports_batch = True``
            and implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.
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
            Array of simulation results with shape ``(n_simulations, ...)``.

        Raises
        ------
        ValueError
            If the simulation does not support batch execution.
        NotImplementedError
            If the simulation does not implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`.
        """
        return self._backend.run(
            sim, n_simulations, seed_seq, progress_callback, **simulation_kwargs
        )
