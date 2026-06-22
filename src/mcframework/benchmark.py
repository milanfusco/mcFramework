r"""
Backend benchmarking for Monte Carlo simulations.

A simulation framework is only as compelling as the speedups it can demonstrate.
This module turns benchmarking into a first-class, repeatable operation: given a
simulation, it times the same workload across every available execution backend
(sequential, thread, process, Torch CPU, Apple-Silicon MPS) and returns a
structured, plottable, JSON-serializable report.

The measurement core is dependency-light (NumPy only). Plotting lives in
:func:`plot_benchmarks`, which imports :mod:`matplotlib` lazily so the library
never carries a hard plotting dependency (install ``mcframework[viz]`` to plot).

Design
------
Backends are described by :class:`BackendSpec`, so the matrix is data, not code.
Adding NVIDIA CUDA later is a single extra spec plus a device gate in
:func:`default_backends` -- no new benchmark script required.

Example
-------
>>> from mcframework import PiEstimationSimulation
>>> from mcframework.benchmark import run_suite, default_backends
>>> sim = PiEstimationSimulation()
>>> report = run_suite(sim, [1_000, 10_000], default_backends())  # doctest: +SKIP
>>> print(report.summary_table())  # doctest: +SKIP

See Also
--------
mcframework.profiling
    Operator-level PyTorch profiling (per-kernel timings, traces).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backends import is_cuda_available, is_mps_available

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .simulation import MonteCarloSimulation

logger = logging.getLogger(__name__)

__all__ = [
    "BackendSpec",
    "BenchmarkResult",
    "BenchmarkReport",
    "system_info",
    "default_backends",
    "benchmark_run",
    "run_suite",
    "plot_benchmarks",
    "main",
]

# Smallest positive time we trust for a throughput/speedup ratio (avoids div-by-zero
# on absurdly fast vectorized runs that round to ~0s).
_TIME_EPS = 1e-12
_DEFAULT_MAX_SEQUENTIAL_SIZE = 1_000_000
_DEFAULT_SIZES = (1_000, 10_000, 100_000, 1_000_000)
# Trimmed sweep so --quick (and the demo smoke test) finish in well under a second.
_QUICK_SIZES = (1_000, 10_000, 100_000)


def _torch_version() -> str | None:
    """Return the installed torch version, or ``None`` when torch is unavailable."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    return str(torch.__version__)


@dataclass(frozen=True)
class BackendSpec:
    r"""
    Description of one execution backend to benchmark.

    Attributes
    ----------
    label : str
        Display name / result key (e.g. ``"torch-mps"``).
    backend : str
        The ``backend`` value passed to :meth:`MonteCarloSimulation.run`
        (``"sequential"``, ``"thread"``, ``"process"``, or ``"torch"``).
    torch_device : str
        Device for ``backend="torch"`` (``"cpu"``, ``"mps"``, ``"cuda"``).
        Ignored by non-Torch backends.
    """

    label: str
    backend: str
    torch_device: str = "cpu"


@dataclass(frozen=True)
class BenchmarkResult:
    r"""
    Timing outcome for one (backend, size) cell.

    Attributes
    ----------
    backend : str
        Backend label (matches :attr:`BackendSpec.label`).
    device : str
        Device the work ran on (``"cpu"``, ``"mps"``, ``"cuda"``).
    n_simulations : int
        Number of simulation draws timed.
    execution_time : float
        Best (minimum) wall-clock time across repeats, in seconds.
    throughput : float
        ``n_simulations / execution_time`` (simulations per second).
    mean_estimate : float
        The simulation's mean estimate (a sanity signal, not a timing metric).
    peak_memory_mb : float or None
        Peak device memory in MB when available (reserved for GPU backends).
    """

    backend: str
    device: str
    n_simulations: int
    execution_time: float
    throughput: float
    mean_estimate: float
    peak_memory_mb: float | None = None


@dataclass
class BenchmarkReport:
    r"""
    Collection of :class:`BenchmarkResult` rows plus run context.

    Attributes
    ----------
    results : list[BenchmarkResult]
        All successfully measured cells.
    sizes : list[int]
        Simulation sizes that were attempted.
    system : dict
        Output of :func:`system_info` captured at run time.
    """

    results: list[BenchmarkResult] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    system: dict[str, Any] = field(default_factory=dict)

    def by_backend(self) -> dict[str, list[BenchmarkResult]]:
        """Group results by backend label, each list sorted by ``n_simulations``."""
        grouped: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            grouped.setdefault(r.backend, []).append(r)
        for rows in grouped.values():
            rows.sort(key=lambda r: r.n_simulations)
        return grouped

    def all_sizes(self) -> list[int]:
        """Sorted unique sizes that actually produced a result."""
        return sorted({r.n_simulations for r in self.results})

    def speedups(self, baseline: str = "sequential") -> dict[str, list[tuple[int, float]]]:
        r"""
        Speedup factors relative to ``baseline``, as ``(n, factor)`` pairs.

        Parameters
        ----------
        baseline : str, default "sequential"
            Backend label whose times define ``speedup = 1.0``.

        Returns
        -------
        dict[str, list[tuple[int, float]]]
            Per-backend speedups at each size where both the backend and the
            baseline produced a usable (non-zero) time.
        """
        grouped = self.by_backend()
        if baseline not in grouped:
            return {}
        base_time = {
            r.n_simulations: r.execution_time
            for r in grouped[baseline]
            if r.execution_time > _TIME_EPS
        }
        out: dict[str, list[tuple[int, float]]] = {}
        for label, rows in grouped.items():
            out[label] = [
                (r.n_simulations, base_time[r.n_simulations] / r.execution_time)
                for r in rows
                if r.n_simulations in base_time and r.execution_time > _TIME_EPS
            ]
        return out

    def best(self, n: int | None = None) -> BenchmarkResult | None:
        r"""
        Fastest result at size ``n`` (default: the largest measured size).

        Parameters
        ----------
        n : int, optional
            Size to inspect. Defaults to the largest size with any result.

        Returns
        -------
        BenchmarkResult or None
            The fastest cell at that size, or ``None`` if there are no results.
        """
        sizes = self.all_sizes()
        if not sizes:
            return None
        target = sizes[-1] if n is None else n
        candidates = [r for r in self.results if r.n_simulations == target]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.execution_time)

    def summary_table(self) -> str:
        """Render a fixed-width execution-time + speedup summary (``N/A`` for gaps)."""
        grouped = self.by_backend()
        sizes = self.all_sizes()
        speedups = self.speedups()

        lines = ["", "=" * 80, "BENCHMARK SUMMARY", "=" * 80]
        header = f"{'Backend':<15}" + "".join(f" | {s:>12,}" for s in sizes)
        lines.append(header)
        lines.append("-" * 80)

        lines.append("Execution Time (seconds):")
        for label, rows in grouped.items():
            time_at = {r.n_simulations: r.execution_time for r in rows}
            row = f"  {label:<13}"
            for s in sizes:
                row += f" | {time_at[s]:>12.4f}" if s in time_at else f" | {'N/A':>12}"
            lines.append(row)

        if speedups:
            lines.append("")
            lines.append("Speedup vs Sequential:")
            for label in grouped:
                factor_at = dict(speedups.get(label, []))
                row = f"  {label:<13}"
                for s in sizes:
                    row += f" | {factor_at[s]:>11.2f}x" if s in factor_at else f" | {'N/A':>12}"
                lines.append(row)

        best = self.best()
        if best is not None:
            lines.append("")
            lines.append(f"Best performer (n={best.n_simulations:,}): {best.backend}")
            lines.append(f"  Execution time: {best.execution_time:.4f}s")
            lines.append(f"  Throughput: {best.throughput:,.0f} simulations/sec")
            factor_at = dict(speedups.get(best.backend, []))
            if best.n_simulations in factor_at:
                lines.append(f"  Speedup: {factor_at[best.n_simulations]:.1f}x faster than sequential")
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (system info, sizes, and result rows)."""
        return {
            "system": dict(self.system),
            "sizes": list(self.sizes),
            "results": [asdict(r) for r in self.results],
        }


def system_info() -> dict[str, Any]:
    r"""
    Capture host + accelerator details for benchmark provenance.

    Returns
    -------
    dict
        Platform, machine, processor, Python version, CPU count, and Torch /
        MPS / CUDA availability. JSON-serializable.
    """
    torch_version = _torch_version()
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "cpu_count": mp.cpu_count(),
        "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "torch_version": torch_version,
        "torch_available": torch_version is not None,
        "mps_available": is_mps_available(),
        "cuda_available": is_cuda_available(),
    }


def default_backends(
    *,
    include_torch: bool = True,
    include_mps: bool = True,
    include_cuda: bool = False,
) -> list[BackendSpec]:
    r"""
    Build the device-aware default backend matrix.

    The CPU backends are always included. Torch backends are added only when
    their device is actually available, so a run never advertises a backend it
    cannot execute.

    Parameters
    ----------
    include_torch : bool, default True
        Include the Torch CPU backend when Torch is installed.
    include_mps : bool, default True
        Include the Apple-Silicon MPS backend when MPS is available.
    include_cuda : bool, default False
        Include the NVIDIA CUDA backend when CUDA is available. Off by default;
        CUDA support is the next milestone.

    Returns
    -------
    list[BackendSpec]
    """
    specs = [
        BackendSpec("sequential", "sequential"),
        BackendSpec("thread", "thread"),
        BackendSpec("process", "process"),
    ]
    torch_available = _torch_version() is not None
    if include_torch and torch_available:
        specs.append(BackendSpec("torch-cpu", "torch", "cpu"))
        if include_mps and is_mps_available():
            specs.append(BackendSpec("torch-mps", "torch", "mps"))
        if include_cuda and is_cuda_available():
            specs.append(BackendSpec("torch-cuda", "torch", "cuda"))
    return specs


def benchmark_run(
    sim: MonteCarloSimulation,
    n: int,
    spec: BackendSpec,
    *,
    n_workers: int | None = None,
    repeats: int = 1,
    seed: int = 42,
    warmup: bool = False,
) -> BenchmarkResult | None:
    r"""
    Time one (backend, size) cell.

    Runs the workload ``repeats`` times and keeps the best (minimum) wall-clock
    time, which is the standard way to suppress scheduler/GC noise.

    Parameters
    ----------
    sim : MonteCarloSimulation
        Simulation to benchmark.
    n : int
        Number of draws.
    spec : BackendSpec
        Backend/device to use.
    n_workers : int, optional
        Worker count for parallel backends.
    repeats : int, default 1
        Number of timed repetitions; the minimum time is reported.
    seed : int, default 42
        Seed applied before each repetition for reproducibility.
    warmup : bool, default False
        If True, run once and return ``None`` (used to prime pools/JIT/caches).

    Returns
    -------
    BenchmarkResult or None
        ``None`` for warmup runs or if the backend raised (e.g. Torch missing).

    Notes
    -----
    ``ponytail:`` best-of-N is the simple denoiser here; it does not capture
    variance or per-operator cost. Upgrade path for kernel-level detail is
    :mod:`mcframework.profiling`.
    """
    times: list[float] = []
    mean_estimate = float("nan")
    n_repeats = 1 if warmup else max(1, repeats)
    try:
        for _ in range(n_repeats):
            sim.set_seed(seed)
            t0 = time.perf_counter()
            result = sim.run(
                n,
                backend=spec.backend,
                torch_device=spec.torch_device,
                n_workers=n_workers,
                compute_stats=False,
            )
            times.append(time.perf_counter() - t0)
            mean_estimate = float(result.mean)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Intentionally broad: a single backend failing (e.g. torch not installed,
        # MPS unsupported) must not abort the whole suite.
        if not warmup:
            logger.warning("Backend %s failed at n=%d: %s", spec.label, n, exc)
        return None

    if warmup:
        return None

    best = min(times)
    throughput = n / best if best > _TIME_EPS else float("inf")
    return BenchmarkResult(
        backend=spec.label,
        device=spec.torch_device if spec.backend == "torch" else "cpu",
        n_simulations=n,
        execution_time=best,
        throughput=throughput,
        mean_estimate=mean_estimate,
    )


def run_suite(
    sim: MonteCarloSimulation,
    sizes: Sequence[int],
    specs: Sequence[BackendSpec],
    *,
    n_workers: int | None = None,
    repeats: int = 1,
    quick: bool = False,
    max_sequential_size: int = _DEFAULT_MAX_SEQUENTIAL_SIZE,
    progress: Callable[[str, int, int], None] | None = None,
) -> BenchmarkReport:
    r"""
    Benchmark ``sim`` across the ``sizes`` x ``specs`` matrix.

    Parameters
    ----------
    sim : MonteCarloSimulation
        Simulation to benchmark.
    sizes : sequence of int
        Simulation sizes to time.
    specs : sequence of BackendSpec
        Backends to time.
    n_workers : int, optional
        Worker count for parallel backends (defaults to CPU count downstream).
    repeats : int, default 1
        Timed repetitions per cell (best time kept).
    quick : bool, default False
        Skip the sequential backend above ``max_sequential_size`` to keep the
        run fast (sequential is O(n) and dominates wall time at large sizes).
    max_sequential_size : int, default 1_000_000
        Sequential cutoff used when ``quick`` is enabled.
    progress : callable, optional
        Called as ``progress(label, n, total)`` after each measured cell.

    Returns
    -------
    BenchmarkReport
    """
    report = BenchmarkReport(sizes=list(sizes), system=system_info())

    # Warm up each backend once at the smallest size to prime pools / JIT / caches.
    warm_n = min(sizes) if sizes else 1_000
    for spec in specs:
        benchmark_run(sim, warm_n, spec, n_workers=n_workers, warmup=True)

    def _skip(spec: BackendSpec, n: int) -> bool:
        return quick and spec.backend == "sequential" and n > max_sequential_size

    total = sum(1 for n in sizes for spec in specs if not _skip(spec, n))
    done = 0
    for n in sizes:
        for spec in specs:
            if _skip(spec, n):
                continue
            result = benchmark_run(sim, n, spec, n_workers=n_workers, repeats=repeats)
            done += 1
            if result is not None:
                report.results.append(result)
            if progress is not None:
                progress(spec.label, done, total)
    return report


# Stable, readable colors per backend (Tableau-ish); unknown backends fall back
# to the default matplotlib cycle.
_BACKEND_COLORS = {
    "sequential": "#6b7280",
    "thread": "#2563eb",
    "process": "#7c3aed",
    "torch-cpu": "#d97706",
    "torch-mps": "#059669",
    "torch-cuda": "#dc2626",
}


def _color_map(labels: Sequence[str]) -> dict[str, str]:
    """Assign a stable color to each backend label."""
    fallback = ["#0891b2", "#db2777", "#65a30d", "#ea580c"]
    out: dict[str, str] = {}
    k = 0
    for label in labels:
        if label in _BACKEND_COLORS:
            out[label] = _BACKEND_COLORS[label]
        else:
            out[label] = fallback[k % len(fallback)]
            k += 1
    return out


def plot_benchmarks(
    report: BenchmarkReport,
    *,
    title: str = "McFramework Backend Performance",
    subtitle: str | None = None,
) -> Figure:
    r"""
    Render a clean four-panel benchmark figure.

    Panels: execution time vs n (log-log), throughput vs n (log-log), speedup
    bars at the largest common size, and speedup scaling vs n.

    Parameters
    ----------
    report : BenchmarkReport
        A populated report from :func:`run_suite`.
    title : str
        Figure title.
    subtitle : str, optional
        Secondary line under the title. Defaults to a host summary.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If matplotlib is not installed (``pip install mcframework[viz]``).
    ValueError
        If ``report`` contains no results to plot.
    """
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "plot_benchmarks requires matplotlib. Install with: pip install mcframework[viz]"
        ) from exc

    grouped = report.by_backend()
    if not grouped:
        raise ValueError("BenchmarkReport has no results to plot")

    colors = _color_map(list(grouped))
    speedups = report.speedups()
    if subtitle is None:
        sysinfo = report.system
        subtitle = (
            f"{sysinfo.get('machine', '?')} | {sysinfo.get('cpu_count', '?')} cores | "
            f"Python {sysinfo.get('python', '?')}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10, color="#555555")
    (ax_time, ax_tput), (ax_bar, ax_scale) = axes

    # --- Panel 1: execution time (log-log) ---
    for label, rows in grouped.items():
        ns = [r.n_simulations for r in rows]
        ts = [r.execution_time for r in rows]
        ax_time.loglog(ns, ts, "o-", color=colors[label], label=label, linewidth=2, markersize=6)
    ax_time.set_title("Execution time vs simulation count")
    ax_time.set_xlabel("n (simulations)")
    ax_time.set_ylabel("execution time (s)")
    ax_time.grid(True, which="both", ls=":", alpha=0.4)
    ax_time.legend(fontsize=8)

    # --- Panel 2: throughput (log-log) ---
    for label, rows in grouped.items():
        ns = [r.n_simulations for r in rows]
        tp = [r.throughput for r in rows]
        ax_tput.loglog(ns, tp, "s-", color=colors[label], label=label, linewidth=2, markersize=6)
    ax_tput.set_title("Throughput vs simulation count")
    ax_tput.set_xlabel("n (simulations)")
    ax_tput.set_ylabel("throughput (sims/sec)")
    ax_tput.grid(True, which="both", ls=":", alpha=0.4)
    ax_tput.legend(fontsize=8)

    # --- Panel 3: speedup bars at the largest size common to non-baseline backends ---
    non_baseline = [lbl for lbl in grouped if lbl != "sequential" and speedups.get(lbl)]
    common: set[int] = set()
    for i, lbl in enumerate(non_baseline):
        sizes_for = {n for n, _ in speedups[lbl]}
        common = sizes_for if i == 0 else (common & sizes_for)
    if non_baseline and common:
        at = max(common)
        labels_at = [lbl for lbl in non_baseline if at in dict(speedups[lbl])]
        vals = [dict(speedups[lbl])[at] for lbl in labels_at]
        bars = ax_bar.bar(labels_at, vals, color=[colors[lbl] for lbl in labels_at], alpha=0.9)
        for rect, val in zip(bars, vals, strict=True):
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2, rect.get_height(),
                f"{val:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax_bar.axhline(1.0, color="#dc2626", ls="--", linewidth=1.5, label="sequential baseline")
        ax_bar.set_title(f"Speedup vs sequential (n={at:,})")
        ax_bar.set_ylabel("speedup factor")
        ax_bar.tick_params(axis="x", labelrotation=15)
        ax_bar.legend(fontsize=8)
    else:
        ax_bar.set_visible(False)
    ax_bar.grid(True, axis="y", ls=":", alpha=0.4)

    # --- Panel 4: speedup scaling with problem size ---
    plotted = False
    for label in non_baseline:
        pts = speedups[label]
        if len(pts) > 1:
            ns = [n for n, _ in pts]
            vals = [v for _, v in pts]
            ax_scale.semilogx(ns, vals, "^-", color=colors[label], label=label,
                              linewidth=2, markersize=6)
            plotted = True
    if plotted:
        ax_scale.axhline(1.0, color="#dc2626", ls="--", linewidth=1.5, label="sequential baseline")
        ax_scale.set_title("Speedup scaling with problem size")
        ax_scale.set_xlabel("n (simulations)")
        ax_scale.set_ylabel("speedup factor")
        ax_scale.legend(fontsize=8)
    else:
        ax_scale.set_visible(False)
    ax_scale.grid(True, which="both", ls=":", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def _format_system_info(info: dict[str, Any]) -> str:
    """Human-readable banner of host + accelerator details."""
    torch = info.get("torch_version") or "not installed"
    accel = []
    if info.get("mps_available"):
        accel.append("MPS")
    if info.get("cuda_available"):
        accel.append("CUDA")
    accel_str = ", ".join(accel) if accel else "none"
    return (
        "\n".join(
            [
                "=" * 80,
                "SYSTEM INFO",
                "=" * 80,
                f"  Platform     : {info.get('platform')}",
                f"  Machine      : {info.get('machine')}",
                f"  Processor    : {info.get('processor')}",
                f"  Python       : {info.get('python')}",
                f"  CPU cores    : {info.get('cpu_count')}",
                f"  Torch        : {torch}",
                f"  Accelerators : {accel_str}",
                "=" * 80,
            ]
        )
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mcframework-benchmark",
        description="Benchmark Monte Carlo execution backends and plot the results.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Trim sizes and skip the slow sequential backend at large n.",
    )
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated simulation sizes (default: 1000,10000,100000,1000000).",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Timed repetitions per cell; the best (min) time is kept.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Worker count for parallel backends (default: CPU count).",
    )
    parser.add_argument(
        "--max-sequential-size", type=int, default=_DEFAULT_MAX_SEQUENTIAL_SIZE,
        help="With --quick, skip the sequential backend above this size.",
    )
    parser.add_argument(
        "--no-mps", action="store_true", help="Exclude the Apple-Silicon MPS backend."
    )
    parser.add_argument("--save", type=str, default=None, help="Save the figure to this path.")
    parser.add_argument("--json", type=str, default=None, help="Write the report JSON to this path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    r"""
    Command-line entry point for ``mcframework-benchmark``.

    Runs the default backend suite against :class:`PiEstimationSimulation`,
    prints a summary table, and optionally saves a figure / JSON artifact.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse (defaults to ``sys.argv``).

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    from .sims import PiEstimationSimulation  # pylint: disable=import-outside-toplevel

    args = _parse_args(argv)
    sim = PiEstimationSimulation()

    if args.sizes:
        sizes: list[int] = [int(s) for s in args.sizes.split(",") if s.strip()]
    elif args.quick:
        sizes = list(_QUICK_SIZES)
    else:
        sizes = list(_DEFAULT_SIZES)

    specs = default_backends(include_mps=not args.no_mps)
    info = system_info()
    print(_format_system_info(info))
    print(f"\nBenchmarking {len(specs)} backend(s) across {len(sizes)} size(s)...\n")

    def _progress(label: str, done: int, total: int) -> None:
        print(f"  [{done:>3}/{total}] {label}")

    report = run_suite(
        sim, sizes, specs,
        n_workers=args.n_workers, repeats=args.repeats,
        quick=args.quick, max_sequential_size=args.max_sequential_size,
        progress=_progress,
    )
    print(report.summary_table())

    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json}")

    if args.save or not args.no_show:
        try:
            fig = plot_benchmarks(report)
        except ImportError as exc:
            print(f"\n[plot skipped] {exc}")
            return 0
        if args.save:
            fig.savefig(args.save, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {args.save}")
        if not args.no_show:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

            plt.show()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
