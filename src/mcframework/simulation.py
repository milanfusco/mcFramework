r"""
Monte Carlo simulation base class and orchestration logic.

This module provides:

Classes
    :class:`MonteCarloSimulation` — Abstract base class for defining simulations

The simulation class handles:
- Reproducible seeding via :class:`numpy.random.SeedSequence`
- Sequential and parallel execution (delegated to backends)
- Statistics computation via the stats engine
- Result assembly and percentile handling

Example
-------
>>> from mcframework.simulation import MonteCarloSimulation
>>> class DiceSim(MonteCarloSimulation):
...     def single_simulation(self, _rng=None):
...         rng = self._rng(_rng, self.rng)
...         return float(rng.integers(1, 7, size=2).sum())
>>> sim = DiceSim(name="2d6")
>>> sim.set_seed(42)
>>> result = sim.run(10_000)  # doctest: +SKIP

See Also
--------
mcframework.backends
    Execution backends for sequential and parallel execution.
mcframework.stats_engine
    Statistical metrics and confidence intervals.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .backends import ProcessBackend, SequentialBackend, ThreadBackend, is_windows_platform
from .stats_engine import (
    _PCTS as _ENGINE_PCTS,
)
from .stats_engine import (
    DEFAULT_ENGINE,
    CIMethod,
    StatsContext,
    StatsEngine,
    _ensure_ctx,
    ci_mean,
    mean,
    std,
)

if TYPE_CHECKING:
    import cupy
    import torch

    from .core import SimulationResult

logger = logging.getLogger(__name__)

__all__ = ["MonteCarloSimulation"]

class MonteCarloSimulation(ABC):
    r"""
    Abstract base class for Monte Carlo simulations.

    Subclass this and implement :meth:`single_simulation`. The framework takes care of
    reproducible seeding, (optional) parallel execution, statistics, and percentiles.

    Examples
    --------
    >>> from mcframework.simulation import MonteCarloSimulation
    >>> class PiSim(MonteCarloSimulation):
    ...     def single_simulation(self, _rng=None, n_points: int = 10_000):
    ...         rng = self._rng(_rng, self.rng)
    ...         x, y = rng.random(n_points), rng.random(n_points)
    ...         return 4.0 * ((x*x + y*y) <= 1.0).mean()
    ...
    >>> sim = PiSim()
    >>> sim.set_seed(42)
    >>> res = sim.run(10_000, parallel=True, compute_stats=True)  # doctest: +SKIP

    Notes
    -----
    **Parallel backend.**
    The ``parallel_backend`` attribute can be ``"auto"``, ``"thread"``, or ``"process"``.
    With NumPy RNGs (which release the GIL), threads are usually faster and avoid
    process-spawn overhead.

    **Percentiles.**
    If ``compute_stats=True``, the stats engine computes defaults
    ``_PCTS`` = ``(5, 25, 50, 75, 95)`` and merges them with user-requested
    percentiles. The original user request is preserved in
    ``result.metadata["requested_percentiles"]`` and enforced by
    :meth:`MonteCarloFramework.compare_results` for percentile metrics.
    """
    _PCTS = _ENGINE_PCTS
    # Minimum simulations to use parallel execution (soft limit)
    _PARALLEL_THRESHOLD = 20_000
    # Number of chunks per worker for load balancing (ensures dynamic work distribution)
    _CHUNKS_PER_WORKER = 8

    #: Whether this simulation supports batch GPU execution.
    #: Override in subclass by setting ``supports_batch = True``.
    _supports_batch: bool = False

    #: Citation for the simulation's analytic reference / oracle, e.g.
    #: ``"Black-Scholes-Merton (1973)"``. Empty when no oracle is declared.
    reference_source: str = ""

    #: Kind of reference returned by :meth:`analytic_reference`. One of
    #: ``"closed-form"`` (an exact analytic answer), ``"benchmark"`` (a published,
    #: cited reference value), or ``"limit"`` (a known asymptotic/limiting value).
    #: Empty when no oracle is declared. Used by the validation layer for reporting
    #: and as the governance signal for promoting a simulation out of draft.
    reference_kind: str = ""

    @property
    def supports_batch(self) -> bool:
        """
        Whether this simulation supports batch GPU execution.

        Subclasses that implement :meth:`~mcframework.core.MonteCarloSimulation.torch_batch`
        or :meth:`~mcframework.core.MonteCarloSimulation.curand_batch` should set this to
        ``True`` either as a class attribute or by setting ``self.supports_batch = True``.

        Returns
        -------
        bool
            ``True`` if the simulation supports vectorized Torch execution.

        Examples
        --------
        >>> class MySim(MonteCarloSimulation):
        ...     supports_batch = True  # Class-level override
        ...     def torch_batch(self, n, *, device, generator):
        ...         ...
        """
        # Check instance attribute first, then class attribute
        return getattr(self, "_supports_batch", False)

    @supports_batch.setter
    def supports_batch(self, value: bool) -> None:
        self._supports_batch = value

    @staticmethod
    def _rng(
        rng: np.random.Generator | None,
        default: np.random.Generator | None = None,
    ) -> np.random.Generator:
        r"""
        Choose the RNG to use inside :meth:`single_simulation`.

        Parameters
        ----------
        rng : :class:`numpy.random.Generator` or None
            RNG passed down by the framework (per-worker/per-chunk stream).
        default : :class:`numpy.random.Generator` or None
            Fallback RNG, typically ``self.rng``.

        Returns
        -------
        :class:`numpy.random.Generator`
            The generator to use.

        Notes
        -----
        This helper makes subclass code concise:

        >>> def single_simulation(self, _rng=None):
        ...     rng = self._rng(_rng, self.rng)
        ...     return float(rng.normal())
        """
        if rng is not None:
            return rng
        if default is not None:
            return default
        raise ValueError(
            "No RNG available: both the per-worker generator and the "
            "simulation-level fallback are None. Call set_seed() first."
        )

    def __init__(self, name: str = "Simulation"):
        self.name = name
        self.seed_seq: np.random.SeedSequence | None = None
        self.rng = np.random.default_rng()
        self.backend: str = "auto"

    def __getstate__(self):
        """Avoid pickling the RNG (not pickleable)."""
        state = self.__dict__.copy()
        state["rng"] = None
        return state

    def __setstate__(self, state):
        """Recreate the RNG after unpickling."""
        self.__dict__.update(state)
        if self.seed_seq is not None:
            self.rng = np.random.default_rng(self.seed_seq)
        else:
            self.rng = np.random.default_rng()

    @abstractmethod
    def single_simulation(self, *args, **kwargs) -> float:
        r"""
        Perform a single simulation run.

        Notes
        -----
        Subclasses must implement this method.

        Returns
        -------
        float
            The result of the simulation run.
        """
        raise NotImplementedError  # pragma: no cover

    def torch_batch(
        self,
        n: int,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Optional vectorized Torch implementation.

        Override this method in subclasses to enable GPU-accelerated batch execution.
        When implemented alongside ``supports_batch = True``, the framework will use
        this method instead of repeated ``single_simulation`` calls.

        Parameters
        ----------
        n : int
            Number of simulation draws.
        device : torch.device
            Device to use for the simulation (``"cpu"``, ``"mps"``, or ``"cuda"``).
        generator : torch.Generator
            Explicit Torch generator for reproducible random sampling. This generator
            is seeded from :class:`numpy.random.SeedSequence` to maintain the same
            spawning semantics as the NumPy backend.

        Returns
        -------
        torch.Tensor
            A 1D tensor of length ``n`` containing simulation results. Use float32
            for MPS compatibility; the framework promotes to float64 after moving
            to CPU.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        Notes
        -----
        **RNG discipline.** All random sampling must use the provided ``generator``
        explicitly. Never use global Torch RNG (``torch.manual_seed``).

        **Dtype policy (device-specific):**

        - **MPS (Apple Silicon):** Must return float32 (Metal doesn't support float64).
          Framework promotes to float64 on CPU.
        - **CUDA (NVIDIA):** Can return float32 or float64. Float64 preferred for zero
          conversion overhead and full precision.
        - **CPU:** Can return float32 or float64. Float64 preferred for consistency
          with framework precision.

        This method is optional and must be implemented by subclasses that support
        the Torch backend. If not implemented, the framework will fall back to the
        NumPy backend.

        Examples
        --------
        >>> class PiSim(MonteCarloSimulation):
        ...     supports_batch = True
        ...     def torch_batch(self, n, *, device, generator):
        ...         import torch
        ...         x = torch.rand(n, device=device, generator=generator)
        ...         y = torch.rand(n, device=device, generator=generator)
        ...         inside = (x * x + y * y) <= 1.0
        ...         return 4.0 * inside.float()  # float32 for MPS compatibility
        """
        raise NotImplementedError

    def curand_batch(
        self, n: int, device_id: int, rng: cupy.random.RandomState) -> cupy.ndarray:
        """
        Optional vectorized cuRAND implementation using CuPy.

        Override this method in subclasses to enable GPU-accelerated batch execution
        via cuRAND. When implemented alongside ``supports_batch = True`` and
        ``use_curand=True``, the CUDA backend will call this method instead of
        :meth:`torch_batch`.

        Parameters
        ----------
        n : int
            Number of simulation draws.
        device_id : int
            CUDA device index.
        rng : cupy.random.RandomState
            cuRAND generator for reproducible random sampling.

        Returns
        -------
        cupy.ndarray
            A 1D array of length ``n`` containing simulation results.
        """
        raise NotImplementedError

    def analytic_reference(self, **params: Any) -> float | None:
        r"""
        Known expected value of :meth:`single_simulation` ("oracle"), if one exists.

        Override in subclasses that have a closed-form or otherwise-known answer for
        :math:`\mathbb{E}[\texttt{single\_simulation}]` under the given parameters.
        The value is consumed by :func:`mcframework.validation.validate_convergence`
        to assert that the Monte Carlo estimate converges to ground truth.

        Parameters
        ----------
        **params : Any
            The same keyword parameters passed to :meth:`single_simulation` / :meth:`run`,
            so the reference can depend on the configuration.

        Returns
        -------
        float or None
            The oracle value, or ``None`` (default) when no reference exists. Returning
            ``None`` signals that the simulation has no oracle and cannot be convergence
            -validated (it is research, not a verifiable demo).

        Examples
        --------
        >>> import math
        >>> class PiSim(MonteCarloSimulation):
        ...     reference_source = "Closed form: pi"
        ...     def single_simulation(self, _rng=None):
        ...         ...
        ...     def analytic_reference(self, **params):
        ...         return math.pi
        """
        return None

    def set_seed(self, seed: int | None) -> None:
        r"""
        Set the random seed for reproducible experiments.

        Parameters
        ----------
        seed : int or None
            Seed for :class:`numpy.random.SeedSequence`. :data:`None` chooses entropy
            from the OS.

        Notes
        -----
        The framework spawns independent child sequences per worker/chunk via
        :meth:`numpy.random.SeedSequence.spawn`, ensuring deterministic parallel
        streams given the same ``seed`` *and the same block layout*.

        .. warning::

           Reproducibility is per-(backend, block-layout), **not** absolute. The
           number of spawned child streams depends on the resolved backend and
           the work partition:

           - The sequential backend spawns **one** child stream and draws every
             sample from it.
           - The thread/process backends spawn **one child stream per block**,
             where the block count depends on ``n_workers``, ``n_simulations``,
             and ``_CHUNKS_PER_WORKER``.

           Because ``backend="auto"`` switches from sequential to parallel once
           ``n_simulations`` crosses :attr:`_PARALLEL_THRESHOLD`
           (``20_000``), the *same* ``seed`` can produce different draws above vs.
           below that threshold, and sequential vs. parallel results are not
           bitwise identical. For run-to-run reproducible numbers, pin the
           backend and ``n_workers`` explicitly. Statistical properties (mean,
           variance, CI coverage) are unaffected.
        """
        self.seed_seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(self.seed_seq)

    # Valid backend values for execution (torch added for GPU-ready API)
    _VALID_BACKENDS = ("auto", "sequential", "thread", "process", "torch")

    def _validate_run_params(
        self,
        n_simulations: int,
        n_workers: int | None,
        confidence: float,
        ci_method: str,
        backend: str = "auto",
    ) -> None:
        """Validate parameters for run() method."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in the interval (0, 1)")
        if ci_method not in ("auto", "z", "t", "bootstrap"):
            raise ValueError(f"ci_method must be one of 'auto', 'z', 't', 'bootstrap', got '{ci_method}'")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"backend must be one of {self._VALID_BACKENDS}, got '{backend}'")

    def _compute_stats_with_engine(
        self,
        results: np.ndarray,
        n_simulations: int,
        confidence: float,
        ci_method: str,
        stats_engine: StatsEngine | None,
        extra_context: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[int, float]]:
        """
        Compute statistics using the stats engine.

        Returns
        -------
        tuple[dict[str, Any], dict[int, float]]
            (stats dict, percentiles dict)
        """
        eng = stats_engine or DEFAULT_ENGINE
        if eng is None:
            return {}, {}

        engine_defaults = self._PCTS

        # Convert string ci_method to enum
        ci_method_enum = CIMethod(ci_method)

        # Create StatsContext object
        try:
            ctx = StatsContext(
                n=n_simulations,
                percentiles=engine_defaults,
                confidence=confidence,
                ci_method=ci_method_enum,
                **(dict(extra_context) if extra_context else {}),
            )
        except (TypeError, ValueError) as e:
            logger.warning("Invalid context parameters: %s. Using defaults.", e)
            ctx = StatsContext(
                n=n_simulations,
                percentiles=engine_defaults,
                confidence=confidence,
                ci_method=ci_method_enum,
            )

        try:
            result = eng.compute(results, ctx)
            stats = result.metrics if hasattr(result, "metrics") else {}
        except (ValueError, TypeError, ArithmeticError, RuntimeError) as e:
            logger.error("Stats engine failed: %s", e)
            stats = {}

        # Merge engine stats with baseline (engine wins on collisions)
        baseline = self._compute_stats_block(results, ctx)
        merged_stats = dict(baseline)
        merged_stats.update(stats if isinstance(stats, dict) else {})
        stats = merged_stats

        # Pull percentiles returned by the engine (if any)
        engine_perc: dict[int, float] = {}
        if isinstance(stats, dict) and "percentiles" in stats:
            engine_perc = stats.pop("percentiles") or {}

        percentile_map = {int(k): float(v) for k, v in engine_perc.items()}

        return stats, percentile_map

    def _handle_percentiles(
        self,
        results: np.ndarray,
        percentiles: Iterable[int] | None,
        compute_stats: bool,
        percentile_map: dict[int, float],
    ) -> tuple[dict[int, float], list[int], bool]:
        """
        Handle percentile computation and tracking.

        Returns
        -------
        tuple[dict[int, float], list[int], bool]
            (final percentile_map, requested_percentiles list, engine_defaults_used flag)
        """
        user_percentiles_provided = percentiles is not None
        user_pcts: tuple[int, ...] = tuple(int(p) for p in (percentiles or ()))

        if not compute_stats:
            # No stats engine: only compute user-requested percentiles
            if not user_percentiles_provided:
                final_map = {}
            else:
                final_map = self._percentiles(results, user_pcts) if user_pcts else {}
            requested_percentiles = list(user_pcts) if user_percentiles_provided else []
            return final_map, requested_percentiles, False

        # If the user requested extra percentiles beyond engine defaults, compute & merge them
        if user_pcts:
            percentile_map.update(self._percentiles(results, user_pcts))

        requested_percentiles = list(user_pcts)
        return percentile_map, requested_percentiles, True

    def run(  # pylint: disable=too-many-arguments
        self,
        n_simulations: int,
        *,
        backend: str = "auto",
        torch_device: str = "cpu",
        cuda_device_id: int = 0,
        cuda_use_curand: bool = False,
        cuda_batch_size: int | None = None,
        cuda_use_streams: bool = True,
        n_workers: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        percentiles: Iterable[int] | None = None,
        compute_stats: bool = True,
        stats_engine: StatsEngine | None = None,
        confidence: float = 0.95,
        ci_method: str = "auto",
        extra_context: Mapping[str, Any] | None = None,
        **simulation_kwargs: Any,
    ) -> SimulationResult:
        r"""
        Run the Monte Carlo simulation.

        Parameters
        ----------
        n_simulations : int
            Number of simulation draws.
        backend : {"auto", "sequential", "thread", "process", "torch"}, default ``"auto"``
            Execution backend to use:

            - ``"auto"`` — Sequential for small jobs, parallel (thread/process) for large jobs
            - ``"sequential"`` — Single-threaded execution
            - ``"thread"`` — Thread-based parallelism (best when NumPy releases GIL)
            - ``"process"`` — Process-based parallelism (required on Windows for true parallelism)
            - ``"torch"`` — Torch batch execution (requires ``supports_batch = True``)

        torch_device : {"cpu", "mps", "cuda"}, default ``"cpu"``
            Torch device for ``backend="torch"``. Ignored for other backends.

            - ``"cpu"`` — Safe default, works everywhere
            - ``"mps"`` — Apple Metal Performance Shaders (M1/M2/M3 Macs)
            - ``"cuda"`` — NVIDIA GPU acceleration

        cuda_device_id : int, default 0
            CUDA device index for multi-GPU systems. Only used when
            ``backend="torch"`` and ``torch_device="cuda"``.
        cuda_use_curand : bool, default False
            Use cuRAND (via CuPy) instead of torch.Generator for maximum
            GPU performance. Requires CuPy and ``curand_batch()`` implementation.
        cuda_batch_size : int or None, default None
            Fixed batch size for CUDA execution. If None, automatically
            estimates optimal batch size based on available GPU memory.
        cuda_use_streams : bool, default True
            Use CUDA streams for overlapped execution. Recommended for performance.
        n_workers : int, optional
            Worker count for parallel backends. Defaults to CPU count.
        progress_callback : callable, optional
            A function ``f(completed: int, total: int)`` called periodically.
        percentiles : iterable of int, optional
            Percentiles to compute from raw results. If ``None`` and
            ``compute_stats=True``, the stats engine's defaults (``_PCTS``)
            are used; if ``compute_stats=False``, **no** percentiles are computed
            unless explicitly provided.
        compute_stats : bool, default ``True``
            Compute additional metrics via a :class:`~mcframework.stats_engine.StatsEngine`.
        stats_engine : StatsEngine, optional
            Custom engine (defaults to ``mcframework.stats_engine.DEFAULT_ENGINE``).
        confidence : float, default ``0.95``
            Confidence level for CI-related metrics.
        ci_method : {"auto","z","t"}, default ``"auto"``
            Which critical values the stats engine should use.
        extra_context : mapping, optional
            Extra context forwarded to the stats engine.
        **simulation_kwargs : Any
            Keyword arguments forwarded to :meth:`single_simulation`.

        Returns
        -------
        SimulationResult
            See :class:`~mcframework.core.SimulationResult`.

        Notes
        -----
        **MPS determinism caveat.** When using ``torch_device="mps"``, the framework
        preserves RNG stream structure but does not guarantee bitwise reproducibility
        due to Metal backend scheduling and float32 arithmetic. Statistical properties
        (mean, variance, CI coverage) remain correct.

        See Also
        --------
        :meth:`~mcframework.core.MonteCarloFramework.run_simulation` : Run a registered simulation by name.
        """
        # Validate parameters
        self._validate_run_params(n_simulations, n_workers, confidence, ci_method, backend)

        # Execute simulation using appropriate backend
        t0 = time.perf_counter()
        results = self._execute_with_backend(
            backend, n_simulations, n_workers, progress_callback,
            torch_device=torch_device,
            cuda_device_id=cuda_device_id,
            cuda_use_curand=cuda_use_curand,
            cuda_batch_size=cuda_batch_size,
            cuda_use_streams=cuda_use_streams,
            **simulation_kwargs,
        )

        exec_time = time.perf_counter() - t0

        # Compute stats and percentiles
        stats: dict[str, Any] = {}
        percentile_map: dict[int, float] = {}

        if compute_stats:
            stats, percentile_map = self._compute_stats_with_engine(
                results, n_simulations, confidence, ci_method, stats_engine, extra_context
            )

        percentile_map, requested_percentiles, engine_defaults_used = self._handle_percentiles(
            results, percentiles, compute_stats, percentile_map
        )

        return self._create_result(
            results,
            n_simulations,
            exec_time,
            percentile_map,
            stats,
            requested_percentiles,
            engine_defaults_used,
        )

    def _resolve_backend_type(self, requested: str | None = None) -> str:
        """
        Resolve the effective parallel backend type.

        This method only resolves *parallel* backends (``"thread"`` or ``"process"``).
        It does not handle ``"sequential"`` — that is handled by :meth:`_execute_with_backend`.

        Parameters
        ----------
        requested : str or None
            Explicitly requested backend type. If None, uses ``self.backend``.

        Returns
        -------
        str
            Resolved parallel backend type: ``"thread"`` or ``"process"``.

        Notes
        -----
        ``"auto"`` maps to:
        * ``"thread"`` on POSIX-like platforms where NumPy releases the GIL.
        * ``"process"`` on Windows where threads tend to serialize under the GIL.
        Invalid values fall back to ``"auto"`` and are then resolved.
        """
        backend = requested or self.backend
        if backend not in self._VALID_BACKENDS:
            logger.warning(
                "backend must be one of %s, got '%s'. Defaulting to 'auto'.",
                self._VALID_BACKENDS,
                backend,
            )
            backend = "auto"

        if backend == "auto":
            on_windows = is_windows_platform()
            resolved = "process" if on_windows else "thread"
            if on_windows:
                logger.info("Parallel backend 'auto' resolved to 'process' on Windows platform.")
            return resolved

        return backend

    def _create_backend(
        self, backend: str, n_workers: int | None
    ) -> SequentialBackend | ThreadBackend | ProcessBackend:
        r"""
        Create and instantiate the appropriate execution backend.

        Parameters
        ----------
        backend : str
            Backend type: ``"sequential"``, ``"thread"``, or ``"process"``.
        n_workers : int or None
            Number of workers for parallel backends.

        Returns
        -------
        SequentialBackend, ThreadBackend, or ProcessBackend
            Configured backend instance.
        """
        if backend == "sequential":
            return SequentialBackend()

        if backend == "torch":
            # Torch backend is handled separately via _run_torch_batch
            raise RuntimeError(
                "Torch backend should be dispatched via _run_torch_batch, not _create_backend."
            )

        # Parallel backends need n_workers
        if n_workers is None:
            n_workers = mp.cpu_count()  # pragma: no cover

        if backend == "thread":
            return ThreadBackend(n_workers=n_workers)
        return ProcessBackend(n_workers=n_workers)

    def _execute_with_backend(
        self,
        backend: str,
        n_simulations: int,
        n_workers: int | None,
        progress_callback: Callable[[int, int], None] | None,
        *,
        torch_device: str = "cpu",
        cuda_device_id: int = 0,
        cuda_use_curand: bool = False,
        cuda_batch_size: int | None = None,
        cuda_use_streams: bool = True,
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Execute simulation draws using the specified backend.

        Parameters
        ----------
        backend : str
            Backend type: ``"auto"``, ``"sequential"``, ``"thread"``, ``"process"``, or ``"torch"``.
        n_simulations : int
            Number of simulation draws.
        n_workers : int or None
            Number of workers for parallel backends.
        progress_callback : callable or None
            Progress reporting callback.
        torch_device : str, default ``"cpu"``
            Torch device type (``"cpu"``, ``"mps"``, ``"cuda"``). Only used for ``backend="torch"``.
        cuda_device_id : int, default 0
            CUDA device index. Only used when ``torch_device="cuda"``.
        cuda_use_curand : bool, default False
            Use cuRAND via CuPy. Only used when ``torch_device="cuda"``.
        cuda_batch_size : int or None, default None
            Fixed batch size for CUDA (None = adaptive). Only used when ``torch_device="cuda"``.
        cuda_use_streams : bool, default True
            Enable CUDA streams. Only used when ``torch_device="cuda"``.
        **simulation_kwargs : Any
            Arguments forwarded to ``single_simulation``.

        Returns
        -------
        np.ndarray
            Array of simulation results.

        Notes
        -----
        For ``"auto"`` backend:
        - Small jobs (< ``_PARALLEL_THRESHOLD``) use sequential execution
        - Large jobs resolve to thread/process based on platform

        For ``"torch"`` backend:
        - Requires ``supports_batch = True`` and :meth:`torch_batch` implementation
        - Ignores ``simulation_kwargs`` (batch method handles all parameters)
        """
        # Early dispatch to Torch if explicitly requested
        if backend == "torch":
            from .backends import TorchBackend  # pylint: disable=import-outside-toplevel

            # Build device kwargs for CUDA
            device_kwargs: dict[str, Any] = {}
            if torch_device == "cuda":
                device_kwargs = {
                    "device_id": cuda_device_id,
                    "use_curand": cuda_use_curand,
                    "batch_size": cuda_batch_size,
                    "use_streams": cuda_use_streams,
                }

            torch_backend = TorchBackend(device=torch_device, **device_kwargs)
            return torch_backend.run(self, n_simulations, self.seed_seq, progress_callback)

        # Resolve "auto" backend
        if backend == "auto":
            if n_workers is None:
                n_workers = mp.cpu_count()  # pragma: no cover

            # Small job fallback to sequential
            if n_workers <= 1 or n_simulations < self._PARALLEL_THRESHOLD:
                backend = "sequential"
            else:
                backend = self._resolve_backend_type()

        # Log execution info
        if backend == "sequential":
            logger.info("Computing %d simulations sequentially...", n_simulations)
        else:
            if n_workers is None:
                n_workers = mp.cpu_count()  # pragma: no cover
            logger.info(
                "Computing %d simulations in parallel using %s backend with %d workers...",
                n_simulations, backend, n_workers
            )

        # Create and run with backend
        backend_instance = self._create_backend(backend, n_workers)
        return backend_instance.run(
            self, n_simulations, self.seed_seq, progress_callback, **simulation_kwargs
        )

    @staticmethod
    def _percentiles(arr: np.ndarray, ps: Iterable[int]) -> dict[int, float]:
        """Return a ``{percentile: value}`` map computed via :func:`numpy.percentile`."""
        return {int(p): float(np.percentile(arr, int(p))) for p in ps}

    @staticmethod
    def _compute_stats_block(results: np.ndarray, ctx) -> dict[str, object]:
        """
        Build the stats dict expected by tests:
        - 'mean': float
        - 'std' : float
        - 'ci_mean' : (low, high)
        """
        ctx = _ensure_ctx(ctx, results)
        results = np.asarray(results, dtype=float).ravel()
        if results.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "ci_mean": (float("nan"), float("nan"))}

        m = mean(results, ctx)
        s = std(results, ctx)
        ci = ci_mean(results, ctx)
        return {
            "mean": float(m) if m is not None else float("nan"),
            "std": float(s) if s is not None else float("nan"),
            "ci_mean": (float(ci["low"]), float(ci["high"])),
            "confidence": float(ci["confidence"]),
            "method": ci["method"],
            "se": float(ci["se"]),
            "crit": float(ci["crit"]),
        }

    def _create_result(
        self,
        results: np.ndarray,
        n_simulations: int,
        execution_time: float,
        percentiles: dict[int, float],
        stats: dict[str, Any],
        requested_percentiles: list[int],
        engine_defaults_used: bool,
    ) -> SimulationResult:
        r"""
        Assemble a :class:`SimulationResult` and merge any stats-engine percentiles.

        Notes
        -----
        Preserves the user's requested percentiles in
        ``metadata["requested_percentiles"]`` and whether engine defaults were used
        in ``metadata["engine_defaults_used"]``.
        """
        # Import here to avoid circular dependency
        from .core import SimulationResult  # pylint: disable=import-outside-toplevel

        stats = dict(stats) if stats else {}

        # Reuse the mean/std the stats engine already computed (it runs the same
        # ddof=1 sample estimators); only fall back to recomputing when stats are
        # absent (compute_stats=False) or non-finite.
        mean_val = stats.get("mean")
        if mean_val is None or not np.isfinite(mean_val):
            mean_val = float(np.mean(results)) if results.size else float("nan")
        else:
            mean_val = float(mean_val)

        std_sample = stats.get("std")
        if std_sample is None or not np.isfinite(std_sample):
            std_sample = float(np.std(results, ddof=1)) if results.size > 1 else 0.0
        else:
            std_sample = float(std_sample)

        # Defensive: if a caller hands us stats that still carry a "percentiles"
        # entry (the run() path strips it upstream, but _create_result is also a
        # public entry point), merge it into the percentile map exactly once.
        if "percentiles" in stats:
            stats_percentiles = stats.pop("percentiles")
            for k, v in stats_percentiles.items():
                percentiles.setdefault(int(k), float(v))

        # Gather metadata and include user-requested percentiles
        meta = {
            "simulation_name": self.name,
            "timestamp": time.time(),
            "n": n_simulations,
            "seed_entropy": self.seed_seq.entropy if self.seed_seq else None,
            "requested_percentiles": requested_percentiles,
            "engine_defaults_used": engine_defaults_used,
        }

        return SimulationResult(
            results=results,
            n_simulations=n_simulations,
            execution_time=execution_time,
            mean=mean_val,
            std=std_sample,
            percentiles=percentiles,
            stats=stats,
            metadata=meta,
        )
