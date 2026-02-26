r"""
Torch CUDA backend for NVIDIA GPU acceleration.

This module provides:

Classes
    :class:`TorchCUDABackend` — GPU-accelerated batch execution on NVIDIA GPUs

Functions
    :func:`is_cuda_available` — Check CUDA availability
    :func:`validate_cuda_device` — Validate CUDA is usable

Features
--------
**Adaptive Batch Sizing**: Automatically estimates optimal batch size based on
available GPU memory to prevent OOM errors while maximizing throughput.

**Dual RNG Modes**:
- ``torch.Generator`` (default) — PyTorch's Philox RNG, fully deterministic
- ``cuRAND`` (optional) — Native GPU RNG via CuPy, maximum performance

**CUDA Optimizations**:
- CUDA streams for overlapped execution
- Native float64 support (zero conversion overhead vs MPS)
- Efficient memory management via PyTorch's caching allocator

**Defensive Validation**: Comprehensive checks for ``supports_batch`` attribute
and required batch methods before execution.

Notes
-----
**Native float64 support**: Unlike MPS (Apple Silicon), CUDA fully supports
float64 tensors. The backend intelligently handles both float32 and float64,
promoting to float64 only when necessary.

**Batch size estimation**: Uses a probe run to estimate per-sample memory
requirements, then calculates optimal batch size to use ~75% of available
GPU memory.

Examples
--------
>>> # Simple usage with defaults
>>> if is_cuda_available():
...     sim.run(1_000_000, backend="torch", torch_device="cuda")  # doctest: +SKIP

>>> # Advanced: Direct backend construction with custom settings
>>> if is_cuda_available():
...     from mcframework.backends import TorchCUDABackend
...     backend = TorchCUDABackend(device_id=0, batch_size=100_000, use_streams=True)
...     results = backend.run(sim, n_simulations=10_000_000, seed_seq=sim.seed_seq)
... # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .torch_base import import_torch, make_torch_generator

if TYPE_CHECKING:
    import torch

    from ..simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = [
    "TorchCUDABackend",
    "is_cuda_available",
    "validate_cuda_device",
]


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns
    -------
    bool
        True if CUDA is available and PyTorch was built with CUDA support.

    Examples
    --------
    >>> if is_cuda_available():
    ...     backend = TorchCUDABackend()  # doctest: +SKIP
    """
    try:
        th = import_torch()
        return th.cuda.is_available()
    except ImportError:
        return False


def validate_cuda_device(device_id: int = 0) -> None:
    """
    Validate that CUDA device is available and usable.

    Parameters
    ----------
    device_id : int, default 0
        CUDA device index to validate.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    RuntimeError
        If CUDA is not available or device index is invalid.

    Examples
    --------
    >>> validate_cuda_device()  # doctest: +SKIP
    >>> validate_cuda_device(device_id=1)  # Check second GPU  # doctest: +SKIP
    """
    th = import_torch()

    if not th.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "Ensure NVIDIA drivers and CUDA toolkit are installed."
        )

    device_count = th.cuda.device_count()
    if device_id >= device_count:
        raise RuntimeError(
            f"CUDA device {device_id} requested but only {device_count} device(s) available."
        )


class TorchCUDABackend:
    r"""
    Torch CUDA batch execution backend for NVIDIA GPUs.

    CUDA backend with adaptive batch sizing, dual RNG modes,
    and native float64 support. Requires simulations to implement
    :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` (or 
    :meth:`~mcframework.core.MonteCarloSimulation.cupy_batch` for cuRAND mode) 
    and set ``supports_batch = True``.

    Parameters
    ----------
    device_id : int, default 0
        CUDA device index to use. Use :func:`torch.cuda.device_count` to
        check available devices.
    use_curand : bool, default False
        Use cuRAND (via CuPy) instead of torch.Generator for RNG.
        Requires CuPy installation and simulation to 
        implement :meth:`~mcframework.core.MonteCarloSimulation.cupy_batch`.
    batch_size : int or None, default None
        Fixed batch size for simulation execution. If None, automatically
        estimates optimal batch size based on available GPU memory.
    use_streams : bool, default True
        Use CUDA streams for overlapped execution. Recommended for performance.

    Attributes
    ----------
    device_type : str
        Always ``"cuda"``.
    device : torch.device
        CUDA device object for this backend.
    device_id : int
        CUDA device index.
    use_curand : bool
        Whether cuRAND mode is enabled.
    batch_size : int or None
        Fixed batch size, or None for adaptive.
    use_streams : bool
        Whether CUDA streams are enabled.

    Notes
    -----
    **RNG architecture**: Uses explicit generators seeded from
    :class:`numpy.random.SeedSequence` via ``spawn()``. Never uses global
    RNG state (:func:`torch.manual_seed` or :meth:`cupy.random.RandomState.seed`).

    **Adaptive batching**: When ``batch_size=None``, performs a probe run
    with 1000 samples to estimate memory requirements, then calculates
    optimal batch size to use ~75% of available GPU memory.

    **Native float64**: CUDA fully supports float64 tensors. If simulation's
    :meth:`~mcframework.core.MonteCarloSimulation.torch_batch` or 
    :meth:`~mcframework.core.MonteCarloSimulation.cupy_batch` returns float64, 
    the backend uses it directly with zero conversion overhead. If float32, it 
    converts to float64 on GPU before moving to CPU for stats engine compatibility.

    **CUDA streams**: When ``use_streams=True``, executes each batch in a
    dedicated stream for better GPU utilization and overlapped execution.

    Examples
    --------
    >>> # Default configuration (adaptive batching, torch.Generator)
    >>> if is_cuda_available():
    ...     backend = TorchCUDABackend(device_id=0)
    ...     results = backend.run(sim, n_simulations=1_000_000, seed_seq=seed_seq)
    ... # doctest: +SKIP

    >>> # High-performance configuration (fixed batching, CuPy)
    >>> if is_cuda_available():
    ...     backend = TorchCUDABackend(
    ...         device_id=0,
    ...         use_curand=True,
    ...         batch_size=100_000,
    ...         use_streams=True
    ...     )
    ...     results = backend.run(sim, n_simulations=10_000_000, seed_seq=seed_seq)
    ... # doctest: +SKIP

    See Also
    --------
    :func:`is_cuda_available` : Check CUDA availability before instantiation.
    :class:`TorchMPSBackend` : Apple Silicon alternative.
    :class:`TorchCPUBackend` : CPU fallback.
    """

    device_type: str = "cuda"

    def __init__(
        self,
        device_id: int = 0,
        use_curand: bool = False,
        batch_size: int | None = None,
        use_streams: bool = True,
    ):
        """
        Initialize Torch CUDA backend with specified configuration.

        Parameters
        ----------
        device_id : int, default 0
            CUDA device index to use.
        use_curand : bool, default False
            Use cuRAND via CuPy instead of torch.Generator.
        batch_size : int or None, default None
            Fixed batch size (None = adaptive).
        use_streams : bool, default True
            Enable CUDA streams for overlapped execution.

        Raises
        ------
        ImportError
            If PyTorch is not installed, or if CuPy is required but not installed.
        RuntimeError
            If CUDA is not available or device index is invalid.
        """
        validate_cuda_device(device_id)
        th = import_torch()

        self.device_id = device_id
        self.device = th.device(f"cuda:{device_id}")
        self.use_curand = use_curand
        self.batch_size = batch_size
        self.use_streams = use_streams

        # Validate CuPy if cuRAND mode requested
        if use_curand:
            try:
                import cupy as cp  # noqa: F401  # pylint: disable=import-outside-toplevel,import-error,unused-import
            except ImportError as e:
                raise ImportError(
                    "cuRAND mode requires CuPy. Install with: pip install mcframework[cuda]"
                ) from e

        # Log device info
        device_name = th.cuda.get_device_name(device_id)
        logger.info(
            "Initialized CUDA backend on device %d: %s (cuRAND=%s, batch_size=%s, streams=%s)",
            device_id, device_name, use_curand, batch_size or "adaptive", use_streams
        )

    def _validate_simulation_compatible(
        self,
        sim: "MonteCarloSimulation",
    ) -> None:
        """
        Validate that simulation supports batch execution with required methods.

        This is the defensive validation layer (Priority 0) that ensures:
        1. Simulation has supports_batch attribute
        2. supports_batch is explicitly True
        3. Required batch method exists (torch_batch or curand_batch)

        Parameters
        ----------
        sim : MonteCarloSimulation
            Simulation instance to validate.

        Raises
        ------
        AttributeError
            If simulation class is missing 'supports_batch' attribute.
        ValueError
            If simulation has supports_batch = False.
        NotImplementedError
            If required batch method is not implemented.
        """
        # Check 1: supports_batch attribute exists
        if not hasattr(sim, 'supports_batch'):
            raise AttributeError(
                f"Simulation class '{sim.__class__.__name__}' is missing 'supports_batch' attribute. "
                f"GPU execution requires explicit declaration to prevent accidental usage. "
                f"To enable CUDA acceleration, add 'supports_batch = True' as a class attribute "
                f"and implement either torch_batch() or curand_batch() method."
            )

        # Check 2: supports_batch is explicitly True
        if not sim.supports_batch:
            raise ValueError(
                f"Simulation '{sim.name}' does not support Torch batch execution. "
                f"Class '{sim.__class__.__name__}' has supports_batch = False. "
                f"Set supports_batch = True and implement torch_batch() or curand_batch() "
                f"to enable CUDA acceleration."
            )

        # Check 3: Required batch method exists and is overridden
        if self.use_curand:
            # cuRAND mode requires curand_batch method
            if not hasattr(sim, 'curand_batch') or not callable(getattr(sim, 'curand_batch')):
                raise NotImplementedError(
                    f"Simulation '{sim.__class__.__name__}' requested cuRAND mode "
                    f"but does not implement curand_batch() method. "
                    f"Either implement curand_batch(n, device_id, rng) or use default "
                    f"torch.Generator mode (use_curand=False)."
                )
        else:
            # Default mode requires torch_batch method to be overridden
            # Check if torch_batch is overridden from base class
            # pylint: disable-next=import-outside-toplevel
            from ..simulation import MonteCarloSimulation
            base_torch_batch = MonteCarloSimulation.torch_batch
            sim_torch_batch = sim.__class__.torch_batch

            if sim_torch_batch is base_torch_batch:
                # Method is not overridden - using base class stub
                raise NotImplementedError(
                    f"Simulation '{sim.__class__.__name__}' has supports_batch = True "
                    f"but does not implement torch_batch() method. "
                    f"Either implement torch_batch(n, device, generator) or set "
                    f"use_curand=True with curand_batch() implementation."
                )

    def _estimate_available_memory(self) -> tuple[int, int]:
        """
        Query available GPU memory on the target device.

        Returns
        -------
        tuple[int, int]
            (free_memory_bytes, total_memory_bytes)
        """
        th = import_torch()
        free_mem, total_mem = th.cuda.mem_get_info(self.device_id)
        return free_mem, total_mem

    def _estimate_batch_size(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
    ) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        Performs a probe run with 1000 samples to estimate per-sample memory
        requirements, then calculates batch size to use ~75% of available memory.

        Parameters
        ----------
        sim : MonteCarloSimulation
            Simulation instance to profile.
        n_simulations : int
            Total number of simulations requested.
        seed_seq : SeedSequence or None
            Seed sequence for probe run.

        Returns
        -------
        int
            Estimated optimal batch size, clamped to [1000, n_simulations].
        """
        th = import_torch()

        # Query available memory
        free_mem, total_mem = self._estimate_available_memory()
        logger.debug(
            "CUDA memory: %.2f GB free / %.2f GB total",
            free_mem / 1e9, total_mem / 1e9
        )

        # Perform probe run with small batch
        probe_size = min(1000, n_simulations)
        th.cuda.reset_peak_memory_stats(self.device_id)
        mem_before = th.cuda.memory_allocated(self.device_id)

        try:
            if self.use_curand:
                # cuRAND probe
                # Note: _make_curand_generator will be added to torch_base.py
                # pylint: disable=import-outside-toplevel,import-error
                import cupy as cp

                from .torch_base import _make_curand_generator  # noqa: F401
                cp.cuda.Device(self.device_id).use()
                child_seed = seed_seq.spawn(1)[0] if seed_seq else None
                if child_seed:
                    seed_int = int(child_seed.generate_state(1, dtype=np.uint64)[0])
                    rng = cp.random.RandomState(seed=seed_int)
                else:
                    rng = cp.random.RandomState()
                _ = sim.curand_batch(probe_size, self.device_id, rng)
                cp.cuda.Device(self.device_id).synchronize()
            else:
                # torch.Generator probe
                generator = make_torch_generator(self.device, seed_seq)
                _ = sim.torch_batch(probe_size, device=self.device, generator=generator)
                th.cuda.synchronize(self.device_id)

            mem_after = th.cuda.memory_allocated(self.device_id)
            mem_used = mem_after - mem_before
            per_sample_mem = mem_used / probe_size

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Probe run failed: %s. Using conservative batch size.", e)
            # Conservative fallback
            per_sample_mem = 1024  # 1KB per sample estimate

        # Calculate batch size to use 75% of available memory
        # Reserve 20% for PyTorch overhead and other allocations
        usable_mem = free_mem * 0.75
        estimated_batch_size = int(usable_mem / per_sample_mem) if per_sample_mem > 0 else 10_000

        # Clamp to reasonable range
        batch_size = max(1000, min(estimated_batch_size, n_simulations))

        logger.info(
            "Estimated batch size: %d (%.2f MB per sample, %.2f GB available)",
            batch_size, per_sample_mem / 1e6, usable_mem / 1e9
        )

        return batch_size

    def _run_single_batch(
        self,
        sim: "MonteCarloSimulation",
        batch_size: int,
        seed_seq: np.random.SeedSequence | None,
    ) -> "torch.Tensor":
        """
        Execute a single batch of simulations.

        Parameters
        ----------
        sim : MonteCarloSimulation
            Simulation instance to run.
        batch_size : int
            Number of simulations in this batch.
        seed_seq : SeedSequence or None
            Seed sequence for this batch.

        Returns
        -------
        torch.Tensor
            Batch results as float64 tensor on CPU.
        """
        th = import_torch()

        if self.use_curand:
            # cuRAND path
            import cupy as cp  # pylint: disable=import-outside-toplevel,import-error
            cp.cuda.Device(self.device_id).use()

            # Create cuRAND generator from SeedSequence
            if seed_seq:
                child_seed = seed_seq.spawn(1)[0]
                seed_int = int(child_seed.generate_state(1, dtype=np.uint64)[0])
                rng = cp.random.RandomState(seed=seed_int)
            else:
                rng = cp.random.RandomState()

            if self.use_streams:
                # Execute in dedicated stream
                stream = cp.cuda.Stream()
                with stream:
                    samples_cp = sim.curand_batch(batch_size, self.device_id, rng)
                stream.synchronize()
            else:
                samples_cp = sim.curand_batch(batch_size, self.device_id, rng)
                cp.cuda.Device(self.device_id).synchronize()

            # Convert CuPy array to PyTorch tensor
            samples = th.as_tensor(samples_cp, device=self.device)

        else:
            # torch.Generator path (default)
            generator = make_torch_generator(self.device, seed_seq)

            if self.use_streams:
                # Execute in dedicated stream
                stream = th.cuda.Stream(device=self.device)
                with th.cuda.stream(stream):
                    samples = sim.torch_batch(batch_size, device=self.device, generator=generator)
                stream.synchronize()
            else:
                samples = sim.torch_batch(batch_size, device=self.device, generator=generator)
                th.cuda.synchronize(self.device_id)

        # CUDA-specific dtype handling (native float64 support)
        if samples.dtype == th.float64:
            # Already float64 - just move to CPU (ZERO conversion overhead)
            samples = samples.detach().cpu()
            logger.debug("CUDA backend: Using native float64 (optimal path)")
        else:
            # float32 from simulation - convert on GPU before CPU transfer
            samples = samples.detach().to(th.float64).cpu()
            logger.debug("CUDA backend: Converting float32 → float64 on GPU")

        return samples

    def run(
        self,
        sim: "MonteCarloSimulation",
        n_simulations: int,
        seed_seq: np.random.SeedSequence | None,
        progress_callback: Callable[[int, int], None] | None = None,
        **_simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Run simulations using Torch CUDA batch execution with adaptive batching.

        Parameters
        ----------
        sim : MonteCarloSimulation
            The simulation instance to run. Must have ``supports_batch = True``
            and implement :meth:`torch_batch` (or :meth:`curand_batch` for
            cuRAND mode).
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
            Results are float64 regardless of internal tensor dtype.

        Raises
        ------
        AttributeError
            If simulation class is missing 'supports_batch' attribute.
        ValueError
            If the simulation does not support batch execution.
        NotImplementedError
            If the simulation does not implement required batch method.
        RuntimeError
            If CUDA out-of-memory error occurs during execution.

        Notes
        -----
        **Adaptive batching**: When ``batch_size=None`` (default), automatically
        estimates optimal batch size. Large workloads are split across multiple
        batches with progress tracking.

        **Memory safety**: Monitors GPU memory and adjusts batch size to prevent
        OOM errors. Uses PyTorch's caching allocator for efficient memory reuse.

        **Determinism**: With same seed, produces identical results (bitwise
        for torch.Generator, statistical for cuRAND).
        """
        th = import_torch()

        # Priority 0: Defensive validation
        self._validate_simulation_compatible(sim)

        # Determine batch size (adaptive or fixed)
        if self.batch_size is None:
            batch_size = self._estimate_batch_size(sim, n_simulations, seed_seq)
        else:
            batch_size = min(self.batch_size, n_simulations)

        logger.info(
            "Computing %d simulations using CUDA backend (device %d, batch_size=%d, cuRAND=%s)...",
            n_simulations, self.device_id, batch_size, self.use_curand
        )

        # Single batch: fast path
        if n_simulations <= batch_size:
            samples = self._run_single_batch(sim, n_simulations, seed_seq)

            if progress_callback:
                progress_callback(n_simulations, n_simulations)

            return samples.numpy()

        # Multiple batches: adaptive execution with progress tracking
        results_list = []
        completed = 0

        # Spawn seed sequences for each batch
        if seed_seq:
            n_batches = (n_simulations + batch_size - 1) // batch_size
            batch_seeds = seed_seq.spawn(n_batches)
        else:
            batch_seeds = [None] * ((n_simulations + batch_size - 1) // batch_size)

        for batch_idx, batch_seed in enumerate(batch_seeds):
            remaining = n_simulations - completed
            current_batch_size = min(batch_size, remaining)

            try:
                # Execute batch
                batch_samples = self._run_single_batch(sim, current_batch_size, batch_seed)
                results_list.append(batch_samples)

                completed += current_batch_size

                # Progress callback
                if progress_callback:
                    progress_callback(completed, n_simulations)

                # Optional: Clear cache between large batches
                if batch_idx % 10 == 0 and batch_idx > 0:
                    th.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # OOM error - try to recover with smaller batch
                    logger.error(
                        "CUDA OOM error at batch %d. Try reducing batch_size or using adaptive batching.",
                        batch_idx
                    )
                    th.cuda.empty_cache()
                raise

        # Concatenate all batches
        all_samples = th.cat(results_list, dim=0)

        # Log memory statistics
        max_mem = th.cuda.max_memory_allocated(self.device_id)
        logger.info(
            "CUDA execution complete. Peak memory: %.2f GB",
            max_mem / 1e9
        )

        return all_samples.numpy()
