"""
Execution backends for Monte Carlo simulations.

This subpackage provides pluggable execution strategies:

Backends
    :class:`SequentialBackend` — Single-threaded execution
    :class:`ThreadBackend` — Thread-based parallelism
    :class:`ProcessBackend` — Process-based parallelism

Utilities
    :func:`make_blocks` — Chunking helper for parallel work distribution
    :func:`worker_run_chunk` — Top-level worker for process pools
    :func:`is_windows_platform` — Platform detection helper

Protocol
    :class:`ExecutionBackend` — Interface for custom backends
"""

from .base import ExecutionBackend, is_windows_platform, make_blocks, worker_run_chunk
from .parallel import ProcessBackend, ThreadBackend
from .sequential import SequentialBackend

__all__ = [
    # Protocol
    "ExecutionBackend",
    # Backends
    "SequentialBackend",
    "ThreadBackend",
    "ProcessBackend",
    # Functions
    "make_blocks",
    "worker_run_chunk",
    "is_windows_platform",
]
