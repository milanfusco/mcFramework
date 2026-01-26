"""
Execution backends for Monte Carlo simulations.

This subpackage provides pluggable execution strategies:

CPU Backends
    :class:`SequentialBackend` — Single-threaded execution
    :class:`ThreadBackend` — Thread-based parallelism
    :class:`ProcessBackend` — Process-based parallelism

Torch Backends (GPU-accelerated)
    :class:`TorchBackend` — Unified Torch backend (auto-selects device)
    :class:`TorchCPUBackend` — Torch CPU execution
    :class:`TorchMPSBackend` — Apple Silicon GPU (Metal Performance Shaders)
    :class:`TorchCUDABackend` — NVIDIA GPU (CUDA 12.x with adaptive batching)

Utilities
    :func:`make_blocks` — Chunking helper for parallel work distribution
    :func:`worker_run_chunk` — Top-level worker for process pools
    :func:`is_windows_platform` — Platform detection helper

Torch Utilities
    :func:`validate_torch_device` — Check Torch device availability
    :func:`make_torch_generator` — Create explicit Torch RNG generators
    :func:`make_curand_generator` — Create explicit cuRAND RNG generators
    :func:`is_mps_available` — Check Apple MPS availability
    :func:`is_cuda_available` — Check NVIDIA CUDA availability
    :data:`VALID_TORCH_DEVICES` — Supported Torch device types

Protocol
    :class:`ExecutionBackend` — Interface for custom backends
"""

from .base import ExecutionBackend, is_windows_platform, make_blocks, worker_run_chunk
from .parallel import ProcessBackend, ThreadBackend
from .sequential import SequentialBackend

# Torch-related names for lazy import
_TORCH_NAMES = (
    # Main backend
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
    "make_curand_generator",
    "VALID_TORCH_DEVICES",
)


def __getattr__(name: str):
    """Lazy import Torch backends and utilities to avoid hard torch dependency."""
    if name in _TORCH_NAMES:
        from . import torch as torch_backend  # pylint: disable=import-outside-toplevel
        return getattr(torch_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Lazily-imported Torch names are available via __getattr__ but not defined at module level.
# This is intentional to avoid requiring torch as a hard dependency.
# pylint: disable=undefined-all-variable
__all__ = [
    # Protocol
    "ExecutionBackend",
    # CPU Backends
    "SequentialBackend",
    "ThreadBackend",
    "ProcessBackend",
    # Torch Backends (lazily imported via __getattr__)
    "TorchBackend",
    "TorchCPUBackend",
    "TorchMPSBackend",
    "TorchCUDABackend",
    # Utility Functions
    "make_blocks",
    "worker_run_chunk",
    "is_windows_platform",
    # Torch Utilities (lazily imported via __getattr__)
    "validate_torch_device",
    "make_torch_generator",
    "make_curand_generator",
    "is_mps_available",
    "is_cuda_available",
    "VALID_TORCH_DEVICES",
]
# pylint: enable=undefined-all-variable
