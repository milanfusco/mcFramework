r"""
NVIDIA CUDA backend for Monte Carlo simulations.

This module provides GPU-accelerated execution on NVIDIA GPUs using
PyTorch's CUDA backend.

Classes
    :class:`CUDABackend` — NVIDIA GPU execution via CUDA

Functions
    :func:`is_cuda_available` — Check if CUDA backend is available
    :func:`validate_cuda_device` — Validate CUDA device availability
    :func:`get_cuda_device_info` — Get information about available CUDA devices

Notes
-----
**CUDA determinism.** With explicit ``torch.Generator`` objects, CUDA execution
is fully deterministic when using the same seed. Unlike MPS, CUDA supports
both float32 and float64 natively.

**Multi-GPU support.** By default, uses ``cuda:0``. For multi-GPU setups,
specify the device index in the constructor.

**System Requirements:**
    - NVIDIA GPU with CUDA capability
    - CUDA toolkit installed
    - PyTorch with CUDA support

Examples
--------
>>> from mcframework.backends import CUDABackend
>>> backend = CUDABackend()  # Uses cuda:0
>>> results = backend.run(sim, 1_000_000, seed_seq)  # doctest: +SKIP

>>> # Multi-GPU: use specific device
>>> backend = CUDABackend(device_index=1)  # Uses cuda:1  # doctest: +SKIP
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
    "CUDABackend",
    "is_cuda_available",
    "validate_cuda_device",
    "get_cuda_device_info",
]


def is_cuda_available() -> bool:
    """
    Check if CUDA backend is available on this system.

    Returns
    -------
    bool
        True if CUDA is available and functional.

    Examples
    --------
    >>> if is_cuda_available():
    ...     backend = CUDABackend()  # doctest: +SKIP
    """
    try:
        th = require_torch()
        return th.cuda.is_available()
    except ImportError:
        return False


def validate_cuda_device(device_index: int = 0) -> None:
    """
    Validate that the CUDA device is available.

    Parameters
    ----------
    device_index : int, default 0
        CUDA device index to validate.

    Raises
    ------
    RuntimeError
        If CUDA is not available or device index is invalid.

    Examples
    --------
    >>> validate_cuda_device()  # Raises on systems without NVIDIA GPU  # doctest: +SKIP
    >>> validate_cuda_device(1)  # Validate second GPU  # doctest: +SKIP
    """
    th = require_torch()

    if not th.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Ensure NVIDIA drivers and CUDA toolkit are installed."
        )

    device_count = th.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            f"CUDA device {device_index} requested but only {device_count} "
            f"device(s) available. Valid indices: 0-{device_count - 1}."
        )


def get_cuda_device_info(device_index: int = 0) -> dict[str, Any]:
    """
    Get information about a CUDA device.

    Parameters
    ----------
    device_index : int, default 0
        CUDA device index.

    Returns
    -------
    dict
        Device information including name, memory, and compute capability.

    Examples
    --------
    >>> info = get_cuda_device_info()  # doctest: +SKIP
    >>> print(info['name'])  # e.g., 'NVIDIA GeForce RTX 4090'  # doctest: +SKIP
    """
    th = require_torch()

    if not th.cuda.is_available():
        return {"available": False}

    props = th.cuda.get_device_properties(device_index)
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024 ** 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "device_index": device_index,
    }


class CUDABackend(TorchBackend):
    r"""
    NVIDIA GPU backend using CUDA for high-performance batch execution.

    Provides GPU-accelerated batch execution on NVIDIA GPUs via PyTorch's
    CUDA backend. Requires simulations to implement :meth:`torch_batch`
    and set ``supports_batch = True``.

    Parameters
    ----------
    device_index : int, default 0
        CUDA device index for multi-GPU systems. Use ``cuda:0`` for the
        first GPU, ``cuda:1`` for the second, etc.

    Notes
    -----
    **Full determinism.** Unlike MPS, CUDA with explicit generators provides
    bitwise reproducibility across runs with the same seed.

    **Float64 support.** CUDA natively supports float64 arithmetic. For
    maximum precision, ensure ``torch_batch`` returns float64 tensors.
    For performance, float32 is often sufficient and faster.

    **Multi-GPU execution.** Each CUDABackend instance is bound to a single
    GPU. For multi-GPU parallelism, create multiple backend instances with
    different ``device_index`` values.

    **Performance tips:**

    - Use large batch sizes (100K+) to maximize GPU utilization
    - Minimize CPU-GPU memory transfers
    - Consider float32 for compute-bound simulations

    Examples
    --------
    >>> backend = CUDABackend()  # Use first GPU (cuda:0)
    >>> results = backend.run(sim, n_simulations=1_000_000, seed_seq=seed_seq)  # doctest: +SKIP

    >>> # Multi-GPU: distribute across devices
    >>> backend_gpu0 = CUDABackend(device_index=0)  # doctest: +SKIP
    >>> backend_gpu1 = CUDABackend(device_index=1)  # doctest: +SKIP

    See Also
    --------
    :class:`TorchBackend` : Base Torch backend for CPU execution
    :class:`MPSBackend` : Apple Silicon GPU backend
    """

    device_type: str = "cuda"

    def __init__(self, device_index: int = 0):
        """
        Initialize CUDA backend for NVIDIA GPU execution.

        Parameters
        ----------
        device_index : int, default 0
            CUDA device index (0 for first GPU, 1 for second, etc.).

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        RuntimeError
            If CUDA is not available or device index is invalid.
        """
        th = require_torch()

        # Validate CUDA availability and device index
        validate_cuda_device(device_index)

        self.device_index = device_index
        self.device_type = "cuda"
        self.device = th.device(f"cuda:{device_index}")

        # Log device info
        info = get_cuda_device_info(device_index)
        logger.debug(
            "Initialized CUDA backend on %s (%.1f GB, compute %s)",
            info.get("name", "Unknown"),
            info.get("total_memory_gb", 0),
            info.get("compute_capability", "Unknown"),
        )

    def _validate_device(self) -> None:
        """Validate that CUDA device is still available."""
        validate_cuda_device(self.device_index)

    def _post_process(self, samples: "torch.Tensor") -> np.ndarray:
        """
        Post-process CUDA samples.

        CUDA supports both float32 and float64 natively. This method:
        1. Moves samples to CPU
        2. Ensures float64 for stats engine precision
        3. Converts to NumPy array

        Parameters
        ----------
        samples : torch.Tensor
            Raw samples from CUDA execution.

        Returns
        -------
        np.ndarray
            Float64 samples as NumPy array.
        """
        th = require_torch()

        # Move to CPU
        samples = samples.detach().cpu()

        # Ensure float64 for stats engine precision
        if samples.dtype != th.float64:
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
        Run simulations on NVIDIA GPU via CUDA.

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
            Ignored for CUDA backend.

        Returns
        -------
        np.ndarray
            Array of float64 simulation results with shape ``(n_simulations,)``.

        Notes
        -----
        CUDA execution is fully deterministic with explicit generators, providing
        bitwise reproducibility across runs with the same seed.
        """
        logger.info(
            "Running %d simulations on NVIDIA GPU (cuda:%d)...",
            n_simulations, self.device_index
        )
        return super().run(sim, n_simulations, seed_seq, progress_callback, **simulation_kwargs)

