r"""
Core primitives for building and running Monte Carlo simulations.

This module provides:

Classes
    :class:`MonteCarloSimulation` — Abstract base class for defining simulations
    :class:`SimulationResult` — Container for simulation outputs and statistics
    :class:`MonteCarloFramework` — Registry and runner for multiple simulations

Functions
    :func:`make_blocks` — Chunking helper for parallel work distribution

Parallel Backends
    - ``"thread"`` — ThreadPoolExecutor (default on POSIX, NumPy releases GIL)
    - ``"process"`` — ProcessPoolExecutor with spawn (default on Windows)
    - ``"auto"`` — Platform-appropriate selection

Example
-------
>>> from mcframework.core import MonteCarloSimulation
>>> class DiceSim(MonteCarloSimulation):
...     def single_simulation(self, _rng=None):
...         rng = self._rng(_rng, self.rng)
...         return float(rng.integers(1, 7, size=2).sum())
>>> sim = DiceSim(name="2d6")
>>> sim.set_seed(42)
>>> result = sim.run(10_000)  # doctest: +SKIP
>>> result.mean  # doctest: +SKIP
7.0

See Also
--------
mcframework.stats_engine
    Statistical metrics and confidence intervals.
mcframework.sims
    Built-in simulations (Pi, Portfolio, Black-Scholes).
mcframework.backends
    Execution backends for sequential and parallel execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Re-export backend utilities for backward compatibility
from .backends import is_windows_platform, make_blocks
from .backends.base import worker_run_chunk as _worker_run_chunk

# Re-export MonteCarloSimulation from simulation module for backward compatibility
from .simulation import MonteCarloSimulation

# Re-export stats_engine symbols for backward compatibility with tests
from .utils import autocrit

# Backward compatibility alias
_is_windows_platform = is_windows_platform


__all__ = [
    "SimulationResult",
    "MonteCarloSimulation",
    "MonteCarloFramework",
    "make_blocks",
    "_worker_run_chunk",
]


@dataclass
class SimulationResult:
    r"""
    Container for the outcome of a Monte Carlo run.

    Attributes
    ----------
    results : :class:`numpy.ndarray`
        Float array of raw simulation values of length :attr:`n_simulations`.
    n_simulations : int
        Number of simulations performed.
    execution_time : float
        Time taken to execute the simulations in seconds.
    mean : float
        Sample mean :math:`\bar X`.
    std : float
        Sample standard deviation with ``ddof=1`` (default for NumPy's :func:`numpy.std`).
    percentiles : dict[int, float]
        Dictionary of computed percentiles, e.g. ``{5: 0.05, 50: 0.50, 95: 0.95}``.
    stats : dict
        Additional statistics from the stats engine (e.g. ``"ci_mean"``, ``"skew"``, etc.).
    metadata : dict
        Freeform metadata. Includes ``"simulation_name"``, ``"timestamp"``, ``"seed_entropy"``,
        ``"requested_percentiles"``, and ``"engine_defaults_used"``.
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
        confidence : float
            Confidence level for the displayed CI. (default ``0.95``)
        method : str
            Which critical value to use (``"auto"`` chooses based on ``n``). (default ``"auto"``)

        Returns
        -------
        str
            Multiline textual summary.

        Notes
        -----
        The parametric CI method for the mean is given by:

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
        if isinstance(ci, (tuple, list)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
            lines.append(f"  (engine) CI: [{ci[0]:.5f}, {ci[1]:.5f}]")
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
    >>> res1 = framework.run_simulation("NormalSim", 10000, backend="auto")  # doctest: +SKIP
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
        name: str | None = None,
    ):
        r"""
        Register a simulation instance under a name.

        Parameters
        ----------
        simulation : MonteCarloSimulation
            The simulation instance to register.
        name : str, optional
            If omitted, ``simulation.name`` is used.
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
        **kwargs : Any
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
            Simulation names (must exist in ``self.results``).
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
