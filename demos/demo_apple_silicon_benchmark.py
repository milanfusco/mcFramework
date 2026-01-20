#!/usr/bin/env python3
"""
Apple Silicon Performance Benchmark Demo

This demo compares the execution speed of different backends on Apple Silicon:
  - Sequential (single-threaded)
  - Thread (ThreadPoolExecutor)
  - Process (ProcessPoolExecutor with spawn)
  - Torch CPU (vectorized batch on CPU)
  - Torch MPS (Metal Performance Shaders - Apple GPU)

Requirements:
  - Apple Silicon Mac (M1/M2/M3/M4)
  - mcframework with GPU extras: pip install mcframework[gpu]

Usage:
  python demo_apple_silicon_benchmark.py
"""

from __future__ import annotations

import multiprocessing as mp
import platform
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# Check for Apple Silicon
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    backend: str
    n_simulations: int
    execution_time: float
    mean_estimate: float
    throughput: float  # simulations per second

    @property
    def speedup_vs(self) -> float:
        """Placeholder for relative speedup calculation."""
        return 1.0


def check_torch_availability() -> tuple[bool, bool]:
    """
    Check if Torch is available and if MPS backend is supported.
    
    Returns
    -------
    tuple[bool, bool]
        (torch_available, mps_available)
    """
    try:
        import torch
        torch_available = True
        mps_available = (
            hasattr(torch.backends, "mps") 
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
    except ImportError:
        torch_available = False
        mps_available = False
    
    return torch_available, mps_available


def print_system_info():
    """Print system information for benchmark context."""
    torch_available, mps_available = check_torch_availability()
    
    print("=" * 70)
    print("APPLE SILICON BENCHMARK - System Information")
    print("=" * 70)
    print(f"  Platform:      {platform.system()} {platform.release()}")
    print(f"  Machine:       {platform.machine()}")
    print(f"  Processor:     {platform.processor() or 'Apple Silicon'}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  CPU Cores:     {mp.cpu_count()}")
    print(f"  Apple Silicon: {'Yes' if IS_APPLE_SILICON else 'No'}")
    print(f"  PyTorch:       {'Installed' if torch_available else 'Not installed'}")
    
    if torch_available:
        import torch
        print(f"  PyTorch Ver:   {torch.__version__}")
        print(f"  MPS Backend:   {'Available' if mps_available else 'Not available'}")
    
    print("=" * 70)
    print()
    
    if not IS_APPLE_SILICON:
        print("Warning: This benchmark is designed for Apple Silicon Macs.")
        print("   MPS acceleration will not be available on this system.\n")
    
    return torch_available, mps_available


def run_benchmark(
    sim,
    n_simulations: int,
    backend: str,
    torch_device: str = "cpu",
    n_workers: int | None = None,
    warmup: bool = False,
) -> BenchmarkResult | None:
    """
    Run a single benchmark with the specified backend.
    
    Parameters
    ----------
    sim : MonteCarloSimulation
        The simulation instance to benchmark.
    n_simulations : int
        Number of simulation draws.
    backend : str
        Backend to use: "sequential", "thread", "process", or "torch".
    torch_device : str
        Device for torch backend: "cpu" or "mps".
    n_workers : int, optional
        Number of workers for parallel backends.
    warmup : bool
        If True, this is a warmup run (don't return results).
    
    Returns
    -------
    BenchmarkResult or None
        Benchmark results, or None if warmup run.
    """
    sim.set_seed(42)  # Ensure reproducibility
    
    try:
        result = sim.run(
            n_simulations,
            backend=backend,
            torch_device=torch_device,
            n_workers=n_workers,
            compute_stats=False,  # Skip stats for pure timing
        )
        
        if warmup:
            return None
        
        return BenchmarkResult(
            backend=f"{backend}" if backend != "torch" else f"torch-{torch_device}",
            n_simulations=n_simulations,
            execution_time=result.execution_time,
            mean_estimate=result.mean,
            throughput=n_simulations / result.execution_time,
        )
    except Exception as e:
        if not warmup:
            print(f"  {backend} ({torch_device}): {e}")
        return None


def run_benchmark_suite(
    sim,
    simulation_sizes: list[int],
    backends: list[tuple[str, str]],  # (backend, torch_device)
    n_workers: int | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run full benchmark suite across multiple simulation sizes and backends.
    
    Parameters
    ----------
    sim : MonteCarloSimulation
        The simulation instance to benchmark.
    simulation_sizes : list[int]
        List of simulation counts to test.
    backends : list[tuple[str, str]]
        List of (backend, torch_device) tuples.
    n_workers : int, optional
        Number of workers for parallel backends.
    
    Returns
    -------
    dict[str, list[BenchmarkResult]]
        Results keyed by backend name.
    """
    results: dict[str, list[BenchmarkResult]] = {}
    
    # Warmup run for each backend
    print("Warming up backends...")
    for backend, device in backends:
        run_benchmark(sim, 1000, backend, device, n_workers, warmup=True)
    print()
    
    total_runs = len(simulation_sizes) * len(backends)
    current_run = 0
    
    for n_sims in simulation_sizes:
        print(f"Running benchmarks for n={n_sims:,} simulations...")
        
        for backend, device in backends:
            current_run += 1
            backend_name = f"{backend}" if backend != "torch" else f"torch-{device}"
            
            result = run_benchmark(sim, n_sims, backend, device, n_workers)
            
            if result:
                if backend_name not in results:
                    results[backend_name] = []
                results[backend_name].append(result)
                
                print(
                    f"  [{current_run}/{total_runs}] {backend_name:15s}: "
                    f"{result.execution_time:8.4f}s "
                    f"({result.throughput:,.0f} sims/sec)"
                )
        print()
    
    return results


def calculate_speedups(
    results: dict[str, list[BenchmarkResult]],
    baseline: str = "sequential",
) -> dict[str, list[float]]:
    """
    Calculate speedup factors relative to a baseline backend.
    
    Parameters
    ----------
    results : dict[str, list[BenchmarkResult]]
        Benchmark results by backend name.
    baseline : str
        Backend name to use as baseline (speedup = 1.0).
    
    Returns
    -------
    dict[str, list[float]]
        Speedup factors by backend name.
    """
    if baseline not in results:
        print(f"Warning: Baseline '{baseline}' not found in results.")
        return {}
    
    baseline_times = [r.execution_time for r in results[baseline]]
    speedups: dict[str, list[float]] = {}
    
    for backend, backend_results in results.items():
        speedups[backend] = [
            bt / r.execution_time 
            for bt, r in zip(baseline_times, backend_results)
        ]
    
    return speedups


def create_benchmark_visualizations(
    results: dict[str, list[BenchmarkResult]],
    simulation_sizes: list[int],
) -> plt.Figure:
    """
    Create comprehensive benchmark visualization charts.
    
    Parameters
    ----------
    results : dict[str, list[BenchmarkResult]]
        Benchmark results by backend name.
    simulation_sizes : list[int]
        List of simulation counts tested.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing benchmark charts.
    """
    # Define color palette - distinctive colors for each backend
    colors = {
        "sequential": "#6B7280",    # Gray
        "thread": "#3B82F6",        # Blue
        "process": "#8B5CF6",       # Purple
        "torch-cpu": "#F59E0B",     # Amber
        "torch-mps": "#10B981",     # Emerald (hero color for Apple Silicon)
    }
    
    # Fallback for unknown backends
    default_colors = ["#EF4444", "#EC4899", "#14B8A6", "#F97316"]
    color_idx = 0
    for backend in results.keys():
        if backend not in colors:
            colors[backend] = default_colors[color_idx % len(default_colors)]
            color_idx += 1
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0F172A")  # Dark slate background
    
    # Main title
    fig.suptitle(
        "McFramework Backend Performance Comparison",
        fontsize=20,
        fontweight="bold",
        color="#F8FAFC",
        y=0.98,
    )
    fig.text(
        0.5, 0.94,
        "Apple Silicon (MPS) vs CPU Backends",
        ha="center",
        fontsize=12,
        color="#94A3B8",
    )
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, 
                          left=0.08, right=0.95, top=0.88, bottom=0.08)
    
    # --- Chart 1: Execution Time (Log Scale) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1E293B")
    
    for backend, backend_results in results.items():
        times = [r.execution_time for r in backend_results]
        ax1.plot(
            simulation_sizes, times,
            marker="o", markersize=8, linewidth=2.5,
            color=colors[backend], label=backend,
        )
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Simulations", fontsize=11, color="#E2E8F0")
    ax1.set_ylabel("Execution Time (seconds)", fontsize=11, color="#E2E8F0")
    ax1.set_title("Execution Time vs Simulation Count", fontsize=13, 
                  fontweight="bold", color="#F8FAFC", pad=10)
    ax1.legend(loc="upper left", facecolor="#334155", edgecolor="#475569",
               labelcolor="#E2E8F0", fontsize=9)
    ax1.grid(True, alpha=0.3, color="#475569")
    ax1.tick_params(colors="#94A3B8")
    for spine in ax1.spines.values():
        spine.set_color("#475569")
    
    # --- Chart 2: Throughput (Simulations per Second) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1E293B")
    
    for backend, backend_results in results.items():
        throughputs = [r.throughput for r in backend_results]
        ax2.plot(
            simulation_sizes, throughputs,
            marker="s", markersize=8, linewidth=2.5,
            color=colors[backend], label=backend,
        )
    
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Simulations", fontsize=11, color="#E2E8F0")
    ax2.set_ylabel("Throughput (simulations/sec)", fontsize=11, color="#E2E8F0")
    ax2.set_title("Throughput vs Simulation Count", fontsize=13,
                  fontweight="bold", color="#F8FAFC", pad=10)
    ax2.legend(loc="lower right", facecolor="#334155", edgecolor="#475569",
               labelcolor="#E2E8F0", fontsize=9)
    ax2.grid(True, alpha=0.3, color="#475569")
    ax2.tick_params(colors="#94A3B8")
    for spine in ax2.spines.values():
        spine.set_color("#475569")
    
    # --- Chart 3: Speedup vs Sequential (Bar Chart) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    
    speedups = calculate_speedups(results, baseline="sequential")
    
    if speedups:
        # Use largest simulation size for the comparison
        largest_idx = -1
        backends_list = [b for b in results.keys() if b != "sequential"]
        speedup_values = [speedups[b][largest_idx] for b in backends_list]
        bar_colors = [colors[b] for b in backends_list]
        
        x_pos = np.arange(len(backends_list))
        bars = ax3.bar(x_pos, speedup_values, color=bar_colors, 
                       edgecolor="#1E293B", linewidth=1.5, alpha=0.9)
        
        # Add value labels on bars
        for bar, val in zip(bars, speedup_values):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2., height + 0.1,
                f"{val:.1f}×",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#F8FAFC",
            )
        
        ax3.axhline(1.0, color="#EF4444", linestyle="--", linewidth=2,
                    label="Sequential baseline")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(backends_list, fontsize=10, color="#E2E8F0")
        ax3.set_ylabel("Speedup Factor", fontsize=11, color="#E2E8F0")
        ax3.set_title(
            f"Speedup vs Sequential (n={simulation_sizes[-1]:,})",
            fontsize=13, fontweight="bold", color="#F8FAFC", pad=10,
        )
        ax3.legend(loc="upper right", facecolor="#334155", edgecolor="#475569",
                   labelcolor="#E2E8F0", fontsize=9)
    
    ax3.grid(True, alpha=0.3, color="#475569", axis="y")
    ax3.tick_params(colors="#94A3B8")
    for spine in ax3.spines.values():
        spine.set_color("#475569")
    
    # --- Chart 4: Efficiency Scaling ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    
    # Show how speedup changes with problem size
    if speedups and len(simulation_sizes) > 1:
        for backend in results.keys():
            if backend == "sequential":
                continue
            ax4.plot(
                simulation_sizes, speedups[backend],
                marker="^", markersize=8, linewidth=2.5,
                color=colors[backend], label=backend,
            )
        
        ax4.axhline(1.0, color="#EF4444", linestyle="--", linewidth=2,
                    label="Sequential baseline")
        ax4.set_xscale("log")
        ax4.set_xlabel("Number of Simulations", fontsize=11, color="#E2E8F0")
        ax4.set_ylabel("Speedup Factor", fontsize=11, color="#E2E8F0")
        ax4.set_title("Speedup Scaling with Problem Size", fontsize=13,
                      fontweight="bold", color="#F8FAFC", pad=10)
        ax4.legend(loc="lower right", facecolor="#334155", edgecolor="#475569",
                   labelcolor="#E2E8F0", fontsize=9)
    
    ax4.grid(True, alpha=0.3, color="#475569")
    ax4.tick_params(colors="#94A3B8")
    for spine in ax4.spines.values():
        spine.set_color("#475569")
    
    return fig


def create_summary_table(
    results: dict[str, list[BenchmarkResult]],
    simulation_sizes: list[int],
) -> str:
    """
    Create a text summary table of benchmark results.
    
    Parameters
    ----------
    results : dict[str, list[BenchmarkResult]]
        Benchmark results by backend name.
    simulation_sizes : list[int]
        List of simulation counts tested.
    
    Returns
    -------
    str
        Formatted summary table.
    """
    speedups = calculate_speedups(results, baseline="sequential")
    
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 80)
    
    # Header
    header = f"{'Backend':<15}"
    for n in simulation_sizes:
        header += f" | {n:>12,}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Execution times
    lines.append("Execution Time (seconds):")
    for backend, backend_results in results.items():
        row = f"  {backend:<13}"
        for r in backend_results:
            row += f" | {r.execution_time:>12.4f}"
        lines.append(row)
    
    lines.append("")
    
    # Speedups
    if speedups:
        lines.append("Speedup vs Sequential:")
        for backend in results.keys():
            row = f"  {backend:<13}"
            for s in speedups[backend]:
                row += f" | {s:>11.2f}×"
            lines.append(row)
    
    lines.append("")
    
    # Best performer for largest size
    if results:
        largest_idx = -1
        best_backend = min(results.keys(), 
                          key=lambda b: results[b][largest_idx].execution_time)
        best_result = results[best_backend][largest_idx]
        
        lines.append(f"🏆 Best performer (n={simulation_sizes[-1]:,}): {best_backend}")
        lines.append(f"   Execution time: {best_result.execution_time:.4f}s")
        lines.append(f"   Throughput: {best_result.throughput:,.0f} simulations/sec")
        
        if best_backend in speedups:
            lines.append(f"   Speedup: {speedups[best_backend][largest_idx]:.1f}× faster than sequential")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Run the Apple Silicon benchmark demo."""
    # Print system info and check capabilities
    torch_available, mps_available = print_system_info()
    
    # Import mcframework
    try:
        from mcframework import PiEstimationSimulation
    except ImportError:
        print("Error: mcframework not installed.")
        print("Install with: pip install mcframework")
        sys.exit(1)
    
    # Create simulation instance
    sim = PiEstimationSimulation()
    
    # Define simulation sizes to test (log scale)
    simulation_sizes = [
        1_000,
        10_000,
        100_000,
        1_000_000,
        5_000_000,
    ]
    
    # Define backends to test
    backends: list[tuple[str, str]] = [
        ("sequential", "cpu"),
        ("thread", "cpu"),
        ("process", "cpu"),
    ]
    
    # Add torch backends if available
    if torch_available:
        backends.append(("torch", "cpu"))
        if mps_available:
            backends.append(("torch", "mps"))
        else:
            print("MPS not available - skipping torch-mps benchmark\n")
    else:
        print("PyTorch not installed - skipping torch benchmarks")
        print("   Install with: pip install mcframework[gpu]\n")
    
    # Run benchmark suite
    print("\n" + "=" * 70)
    print("STARTING BENCHMARK SUITE")
    print("=" * 70 + "\n")
    
    n_workers = mp.cpu_count()
    print(f"Using {n_workers} workers for parallel backends\n")
    
    results = run_benchmark_suite(
        sim,
        simulation_sizes,
        backends,
        n_workers=n_workers,
    )
    
    if not results:
        print("No benchmark results collected. Exiting.")
        sys.exit(1)
    
    # Print summary table
    summary = create_summary_table(results, simulation_sizes)
    print(summary)
    
    # Create visualizations
    print("\nGenerating benchmark visualizations...")
    
    fig = create_benchmark_visualizations(results, simulation_sizes)
    
    # Show the plot
    plt.show()
    
    # Optionally save
    try:
        save = input("\nSave benchmark chart to file? (y/N): ").lower().strip()
        if save == "y":
            filename = "apple_silicon_benchmark.png"
            fig.savefig(filename, dpi=300, facecolor=fig.get_facecolor(),
                       bbox_inches="tight")
            print(f"Saved to {filename}")
    except EOFError:
        # Non-interactive mode
        pass


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    main()

