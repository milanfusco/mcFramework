r"""

mcframework.core
================

Core primitives for building and running Monte Carlo simulations.

The core module of the Monte Carlo Framework provides essential classes and functions to set up and
run Monte Carlo simulations. It includes tools for defining simulation parameters, executing
simulations, and managing results.

This module provides:

* :class:`~mcframework.core.MonteCarloSimulation` – abstract base for simulations.
* :class:`~mcframework.core.SimulationResult` – a lightweight container for outputs.
* :class:`~mcframework.core.MonteCarloFramework` – registry + convenience runner.
* :func:`~mcframework.core.make_blocks` – chunking helper for parallel runs.

Parallel backends
----------------

``MonteCarloSimulation.run(..., parallel=True)`` can use threads or processes.
By default, (:attr:`~mcframework.core.MonteCarloSimulation.parallel_backend` = ``"auto"``),
the framework **prefers threads** because NumPy's random generators release the Global
Interpreter Lock (GIL), avoiding the heavy spawn/pickle cost of processes on macOS.

Set ``parallel_backend = "process"`` for heavy, Python-bound work that does **not**
release the GIL.

Confidence intervals
--------------------

By default a 95% confidence interval for the mean uses the usual formula

.. math::

   \bar{X} \pm z_{\alpha/2}\,\frac{s}{\sqrt{n}}

or a t–critical value if requested via the stats engine.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

from .stats_engine import DEFAULT_ENGINE, StatsContext, StatsEngine
from .utils import autocrit

logger = logging.getLogger(__name__)  # pragma: no cover
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _worker_run_chunk(
    sim: "MonteCarloSimulation",
    chunk_size: int,
    seed_seq: np.random.SeedSequence,
    simulation_kwargs: dict[str, Any],
) -> list[float]:
    r"""
    Execute a small batch of single simulations in a **separate worker**.

    Parameters
    ----------
    sim :
        Simulation instance to call (:meth:`MonteCarloSimulation.single_simulation`).
        Must be pickleable when used with a process backend.
    chunk_size : int
        Number of draws to compute in this worker.
    seed_seq : :class:`numpy.random.SeedSequence`
        Seed sequence for creating an **independent** RNG stream in the worker.
    simulation_kwargs : dict
        Keyword arguments forwarded to :meth:`MonteCarloSimulation.single_simulation`.

    Returns
    -------
    list of float
        The simulated values.

    Notes
    -----
    Uses :class:`numpy.random.Philox` to spawn a deterministic, independent stream per
    worker chunk.
    """
    bitgen = np.random.Philox(seed_seq)
    local_rng = np.random.Generator(bitgen)
    return [float(sim.single_simulation(_rng=local_rng, **simulation_kwargs)) for _ in range(chunk_size)]


def make_blocks(n, block_size=10_000):
    r"""
    Partition an integer range ``[0, n)`` into half-open blocks ``(i, j)``.

    Parameters
    ----------
    n : int
        Total number of items.
    block_size : int, default ``10_000``
        Target block length.

    Returns
    -------
    list of tuple[int, int]
        List of ``(i, j)`` index pairs covering ``[0, n)``.

    Examples
    --------
    >>> make_blocks(5, block_size=2)
    [(0, 2), (2, 4), (4, 5)]
    """
    blocks = []
    i = 0
    while i < n:
        j = min(i + block_size, n)
        blocks.append((i, j))
        i = j
    return blocks


@dataclass
class SimulationResult:
    r"""
    Container for the outcome of a Monte Carlo run.

    Attributes
    ----------
    results : ndarray of float
        Raw simulation values of length :attr:`n_simulations`.
    n_simulations : int
        Number of simulations performed.
    execution_time : float
        Wall-clock time in seconds.
    mean : float
        Sample mean :math:`\bar X`.
    std : float
        Sample standard deviation with ``ddof=1``.
    percentiles : dict[int, float]
        Map of computed percentiles, e.g. ``{5: ..., 50: ..., 95: ...}``.
    stats : dict
        Additional statistics from the stats engine (e.g. ``"ci_mean"``).
    metadata : dict
        Freeform metadata. Includes ``"simulation_name"``, ``"timestamp"``,
        ``"seed_entropy"``, ``"requested_percentiles"`` and ``"engine_defaults_used"``.
    """

    results: np.ndarray
    n_simulations: int
    execution_time: float
    mean: float
    std: float
    percentiles: dict[int, float]
    stats: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def result_to_string(
        self,
        confidence: float = 0.95,
        method: str = "auto",
    ) -> str:
        r"""
         Pretty, human-readable summary of the result.

        Prints the dictionary attributes in a readable format.

        Parameters
        ----------
        confidence : float, default ``0.95``
            Confidence level for the displayed CI.
        method : {"auto", "z", "t"}, default ``"auto"`` Which critical value to use (``"auto"``
            chooses based on ``n``).

        Returns
        -------
        str
            Multiline textual summary.

        Notes
        -----
        The CI in the string is

        .. math::

            \bar{X} \pm c \frac{s}{\sqrt{n}}

        where :math:`c` is either a z or t critical value depending on ``method``.
        """
        print("=" * 20 + " SIM RESULTS " + "=" * 20)
        if simulation_name := self.metadata.get("simulation_name"):
            title = f"Results for simulation '{simulation_name}':"
        else:
            title = "Results for simulation:"
        n = int(self.n_simulations)
        crit, kind = autocrit(confidence, n, method)
        se = self.std / np.sqrt(max(1, n))
        lo = self.mean - crit * se
        hi = self.mean + crit * se
        lines = [
            title,
            f"  Number of simulations: {self.n_simulations}",
            f"  Execution time: {self.execution_time:.2f} seconds",
            f"  Mean: {self.mean:.5f}   (SE: {se:.5f}, "
            f"{int(confidence * 100)}% {kind}-CI: [{lo:.5f}, {hi:.5f}])",
            f"  Std Dev (sample): {self.std:.5f}",
            "  Percentiles:",
        ]
        for p in sorted(self.percentiles):
            lines.append(f"    {p}th: {self.percentiles[p]:.5f}")
        ci = self.stats.get("ci_mean")
        if isinstance(ci, dict) and ci.get("confidence") == confidence:
            lines.append(
                f"  (engine) CI: [{ci['low']:.5f}, {ci['high']:.5f}] via {ci['method']}-crit={ci['crit']:.5f}"
            )
        if self.stats:
            lines.append("Additional Stats:")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")

        if self.metadata:
            lines.append("Metadata:")
        for k, v in self.metadata.items():
            lines.append(f"    {k}: {v}")
        lines.append("=" * 20 + " END " + "=" * 20)
        return "\n".join(lines)


class MonteCarloSimulation(ABC):
    r"""
    Abstract base class for Monte Carlo simulations.

    Subclass this and implement :meth:`single_simulation`. The framework takes care of
    reproducible seeding, (optional) parallel execution, statistics, and percentiles.

    Quick example
    -------------
    >>> from mcframework.core import MonteCarloSimulation
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
    :attr:`parallel_backend` can be ``"auto"``, ``"thread"``, or ``"process"``.
    With NumPy RNGs (which release the GIL), threads are usually faster and avoid
    process-spawn overhead.

    **Percentiles.**
    If ``compute_stats=True``, the stats engine computes defaults
    :attr:`_PCTS` = ``(5, 25, 50, 75, 95)`` and merges them with user-requested
    percentiles. The original user request is preserved in
    ``result.metadata["requested_percentiles"]`` and enforced by
    :meth:`MonteCarloFramework.compare_results` for percentile metrics.
    """

    _PCTS = (5, 25, 50, 75, 95)  # Default percentiles for stats engine
    _PARALLEL_THRESHOLD = 20_000  # Minimum simulations to use parallel execution
    _CHUNKS_PER_WORKER = 8  # Number of chunks per worker for load balancing

    @staticmethod
    def _rng(
        rng: Optional[np.random.Generator],
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
        return rng if rng is not None else default  # type: ignore[return-value]

    def __init__(self, name: str = "Simulation"):
        self.name = name
        self.seed_seq: Optional[np.random.SeedSequence] = None
        self.rng = np.random.default_rng()
        self.parallel_backend: str = "auto"  # "auto" | "thread" | "process"

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
        the result of the simulation run.
        """

    def set_seed(self, seed: int | None) -> None:
        r"""
        Set the random seed for reproducible experiments.

        Parameters
        ----------
        seed : int or None
            Seed for :class:`numpy.random.SeedSequence`. ``None`` chooses entropy
            from the OS.

        Notes
        -----
        The framework spawns independent child sequences per worker/chunk via
        :meth:`numpy.random.SeedSequence.spawn`, ensuring deterministic parallel
        streams given the same ``seed`` and block layout.
        """
        self.seed_seq = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(self.seed_seq)

    def run(
        self,
        n_simulations: int,
        *,
        parallel: bool = False,
        n_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        percentiles: Optional[Iterable[int]] = None,
        compute_stats: bool = True,
        stats_engine: Optional[StatsEngine] = None,
        confidence: float = 0.95,
        ci_method: str = "auto",
        extra_context: Optional[Mapping[str, Any]] = None,
        **simulation_kwargs: Any,
    ) -> SimulationResult:
        r"""
        Run the Monte Carlo simulation.

        Parameters
        ----------
        n_simulations : int
            Number of simulation draws.
        parallel : bool, default ``False``
            Use :meth:`_run_parallel` if ``True``; otherwise run sequentially.
        n_workers : int, optional
            Worker count. Defaults to CPU count when needed.
        progress_callback : callable, optional
            A function ``f(completed: int, total: int)`` called periodically.
        percentiles : iterable of int, optional
            Percentiles to compute from raw results. If ``None`` and
            ``compute_stats=True``, the stats engine's defaults (:attr:`_PCTS`)
            are used; if ``compute_stats=False``, **no** percentiles are computed
            unless explicitly provided.
        compute_stats : bool, default ``True``
            Compute additional metrics via a :class:`~mcframework.stats_engine.StatsEngine`.
        stats_engine : StatsEngine, optional
            Custom engine (defaults to :data:`mcframework.stats_engine.DEFAULT_ENGINE`).
        confidence : float, default ``0.95``
            Confidence level for CI-related metrics.
        ci_method : {"auto","z","t"}, default ``"auto"``
            Which critical values the stats engine should use.
        extra_context : mapping, optional
            Extra context forwarded to the stats engine.
        **simulation_kwargs :
            Forwarded to :meth:`single_simulation`.

        Returns
        -------
        SimulationResult
            See :class:`~mcframework.core.SimulationResult`.

        See Also
        --------
        mcframework.core.MonteCarloFramework.run_simulation : Run a registered simulation by name.
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in the interval (0, 1)")
        if ci_method not in ("auto", "z", "t", "bootstrap"):
            raise ValueError(f"ci_method must be one of 'auto', 'z', 't', 'bootstrap', got '{ci_method}'")
        t0 = time.time()
        if parallel:
            if n_workers is None:
                n_workers = mp.cpu_count()  # pragma: no cover
            logger.info(f"Computing {n_simulations} simulations in parallel using {n_workers} workers...")
            results = self._run_parallel(n_simulations, n_workers, progress_callback, **simulation_kwargs)
        else:
            logger.info(f"Computing {n_simulations} simulations sequentially...")
            results = self._run_sequential(n_simulations, progress_callback, **simulation_kwargs)

        exec_time = time.time() - t0

        # === Percentile and Stats handling ===
        stats: dict[str, Any] = {}
        percentile_map: dict[int, float] = {}
        user_percentiles_provided = percentiles is not None
        user_pcts: tuple[int, ...] = tuple(int(p) for p in (percentiles or ()))
        if not compute_stats:
            if not user_percentiles_provided:
                percentile_map = {}
            else:
                percentile_map = self._percentiles(results, user_pcts) if user_pcts else {}

            requested_percentiles = list(user_pcts) if user_percentiles_provided else []
            return self._create_result(
                results, n_simulations, exec_time, percentile_map, stats,
                requested_percentiles, engine_defaults_used=False
            )
        
        # Compute stats with engine
        eng = stats_engine or DEFAULT_ENGINE
        if eng is not None:
            engine_defaults = self._PCTS
            
            # Build context dictionary
            ctx_dict = {
                "n": n_simulations,
                "percentiles": engine_defaults,
                "confidence": confidence,
                "ci_method": ci_method,
            }
            
            # Merge extra_context if provided
            if extra_context:
                ctx_dict.update(dict(extra_context))
            
            # Create StatsContext object
            try:
                ctx = StatsContext(**ctx_dict)
            except (TypeError, ValueError) as e:
                # Handle case where extra_context has invalid fields
                logger.warning(f"Invalid context parameters: {e}. Using defaults.")
                ctx = StatsContext(
                    n=n_simulations,
                    percentiles=engine_defaults,
                    confidence=confidence,
                    ci_method=ci_method,  # Pass as string; StatsContext will validate
                )
            
            # Compute stats - pass StatsContext object
            try:
                stats = eng.compute(results, ctx)
            except Exception as e:
                logger.error(f"Stats engine failed: {e}")
                stats = {}
            
            # Start with engine-provided percentiles if present
            engine_perc = {}
            if isinstance(stats, dict) and "percentiles" in stats:
                engine_perc = stats.pop("percentiles") or {}
            
            percentile_map = dict((int(k), float(v)) for k, v in engine_perc.items())
            if user_pcts:
                user_map = self._percentiles(results, user_pcts)
                percentile_map.update(user_map)
        
        requested_percentiles = list(user_pcts)
        return self._create_result(
            results, n_simulations, exec_time, percentile_map, stats,
            requested_percentiles, engine_defaults_used=True
        )

    def _run_sequential(
        self,
        n_simulations: int,
        progress_callback: Optional[Callable[[int, int], None]],
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        """Compute ``n_simulations`` draws on a single thread, with optional progress."""
        results = np.empty(n_simulations, dtype=float)
        step = max(1, n_simulations // 100)
        for i in range(n_simulations):
            results[i] = float(self.single_simulation(**simulation_kwargs))
            if progress_callback and (((i + 1) % step == 0) or (i + 1 == n_simulations)):
                progress_callback(i + 1, n_simulations)
        return results

    def _run_parallel(
        self,
        n_simulations: int,
        n_workers: Optional[int],
        progress_callback: Optional[Callable[[int, int], None]],
        **simulation_kwargs: Any,
    ) -> np.ndarray:
        r"""
        Compute draws in parallel using threads (default) or processes.

        Notes
        -----
        * Threads path uses a local worker function (no pickling).
        * Process path submits a top-level worker (:func:`_worker_run_chunk`) so it is
          pickleable under the ``spawn`` start method (macOS/Windows).
        * For small jobs (``n_simulations < 20_000``) a sequential fallback avoids
          parallel overhead.
        """
        if n_workers is None:
            n_workers = mp.cpu_count()  # pragma: no cover

        # ---- Short-job fallback (cheap runs shouldn't pay parallel overhead) ----
        if n_workers <= 1 or n_simulations < self._PARALLEL_THRESHOLD:
            return self._run_sequential(n_simulations, progress_callback, **simulation_kwargs)

        block_size = max(1, n_simulations // (n_workers * self._CHUNKS_PER_WORKER)) # floor division
        blocks = make_blocks(n_simulations, block_size)
        if self.seed_seq is not None:
            child_seqs = self.seed_seq.spawn(len(blocks))
        else:
            child_seqs = [np.random.SeedSequence() for _ in range(len(blocks))]

        # ---- Choose backend (prefer threads) ----
        backend = getattr(self, "parallel_backend", "auto")
        use_threads = backend in ("thread", "auto")
        results = np.empty(n_simulations, dtype=float)
        completed = 0
        max_workers = min(n_workers, len(blocks))

        if use_threads:
            # Threads: inline worker is fine (no pickling)
            def _work(args):
                (a, b), seed_seq = args
                rng = np.random.Generator(np.random.Philox(seed_seq))
                out = np.empty(b - a, dtype=float)
                for k in range(out.size):
                    out[k] = float(self.single_simulation(_rng=rng, **simulation_kwargs))
                return (a, b), out

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_work, (blk, ss)) for blk, ss in zip(blocks, child_seqs)]
                for f in as_completed(futs):
                    (i, j), arr = f.result()
                    results[i:j] = arr
                    completed += j - i
                    if progress_callback:
                        progress_callback(completed, n_simulations)  # pragma: no cover
            return results

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as ex:
            futs = []
            for (i, j), ss in zip(blocks, child_seqs):
                f = ex.submit(_worker_run_chunk, self, j - i, ss, dict(simulation_kwargs))
                f.blk = (i, j)  # type: ignore[attr-defined]
                futs.append(f)
            try:
                for f in as_completed(futs):
                    i, j = f.blk  # type: ignore[attr-defined]
                    chunk = f.result()
                    results[i:j] = chunk
                    completed += j - i
                    if progress_callback:
                        progress_callback(completed, n_simulations)  # pragma: no cover
            except KeyboardInterrupt:
                for f in futs:
                    f.cancel()
                raise

        return results

    @staticmethod
    def _percentiles(arr: np.ndarray, ps: Iterable[int]) -> dict[int, float]:
        """Return a ``{percentile: value}`` map computed via :func:`numpy.percentile`."""
        return {int(p): float(np.percentile(arr, int(p))) for p in ps}

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
        mean = float(np.mean(results))
        std_sample = float(np.std(results, ddof=1)) if results.size > 1 else 0.0
        stats = dict(stats) if stats else {}

        # If the stats engine also returned percentiles merge once.
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
            mean=mean,
            std=std_sample,
            percentiles=percentiles,
            stats=stats,
            metadata=meta,
        )


class MonteCarloFramework:
    r"""
    Registry for named simulations that runs and compares results.

    Orchestrates multiple :class:`MonteCarloSimulation` instances.

    Examples
    --------
    >>> from mcframework.core import MonteCarloFramework, MonteCarloSimulation
    >>> class MySim(MonteCarloSimulation):
    ...     def single_simulation(self, _rng=None):
    ...         rng = self._rng(_rng, self.rng)
    ...         return float(rng.normal())
    ...
    >>> sim1 = MySim(name="NormalSim")
    >>> sim2 = MySim(name="AnotherSim")
    >>> framework = MonteCarloFramework()
    >>> framework.register_simulation(sim1)
    >>> framework.register_simulation(sim2)
    >>> res1 = framework.run_simulation("NormalSim", 10000, parallel=True)  # doctest: +SKIP
    >>> res2 = framework.run_simulation("AnotherSim", 10000)  # doctest: +SKIP
    >>> comparison = framework.compare_results(["NormalSim", "AnotherSim"], metric="mean")  # doctest: +SKIP
    >>> print(comparison)  # doctest: +SKIP
    {'NormalSim': 0.01234, 'AnotherSim': -0.05678}

    """

    def __init__(self):
        self.simulations: dict[str, MonteCarloSimulation] = {}
        self.results: dict[str, SimulationResult] = {}

    def register_simulation(
        self,
        simulation: MonteCarloSimulation,
        name: Optional[str] = None,
    ):
        r"""
        Register a simulation instance under a name.

        Parameters
        ----------
        simulation : MonteCarloSimulation
            The simulation instance to register.
        name : str, optional
            If omitted, :attr:`MonteCarloSimulation.name` is used.
        """
        sim_name = name or simulation.name
        self.simulations[sim_name] = simulation

    def run_simulation(
        self,
        name: str,
        n_simulations: int,
        **kwargs,
    ) -> SimulationResult:
        r"""
        Run a registered simulation by name.

        Parameters
        ----------
        name : str
            Key used in :meth:`register_simulation`.
        n_simulations : int
            Number of draws.
        **kwargs :
            Forwarded to :meth:`MonteCarloSimulation.run`.

        Returns
        -------
        SimulationResult
        """
        if name not in self.simulations:
            raise ValueError(f"Simulation '{name}' not found")
        sim = self.simulations[name]
        res = sim.run(n_simulations, **kwargs)
        self.results[name] = res
        return res

    def compare_results(
        self,
        names: list[str],
        metric: str = "mean",
    ) -> dict[str, float]:
        r"""
        Compare a metric across previously run simulations.

        Parameters
        ----------
        names : list of str
            Simulation names (must exist in :attr:`results`).
        metric : {"mean","std","var","se","pX"}, default ``"mean"``
            Metric to extract. ``"pX"`` requests the X-th percentile (e.g. ``"p95"``).

        Returns
        -------
        dict
            ``{name: value}`` pairs.

        Raises
        ------
        ValueError
            If a requested percentile was **not** part of the user's requested
            set at run time (enforced via ``result.metadata["requested_percentiles"]``),
            or if the metric name is unknown.
        """
        out: dict[str, float] = {}
        for name in names:
            if name not in self.results:
                raise ValueError(f"No results found for simulation '{name}'")
            r = self.results[name]
            if metric == "mean":
                out[name] = r.mean
            elif metric == "std":
                out[name] = r.std
            elif metric == "var":
                out[name] = r.std**2
            elif metric == "se":
                out[name] = r.std / np.sqrt(max(1, r.n_simulations))
            elif metric.lower().startswith("p") and metric[1:].isdigit():
                p = int(metric[1:])
                requested = r.metadata.get("requested_percentiles")
                if requested:
                    requested_set = {int(x) for x in requested}
                    if p not in requested_set:
                        raise ValueError(f"Percentile {p} not computed")
                if p in r.percentiles:
                    out[name] = r.percentiles[p]
                else:
                    raise ValueError(f"Percentile {p} not computed")
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return out


__all__ = [
    "SimulationResult",
    "MonteCarloSimulation",
    "MonteCarloFramework",
    "make_blocks",
    "_worker_run_chunk",
]
