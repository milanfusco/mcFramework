#!/usr/bin/env python3
"""
CUDA Performance Benchmark Demo

This demo compares the execution speed of different backends on NVIDIA GPUs:
  - Sequential (single-threaded CPU)
  - Thread (ThreadPoolExecutor)
  - Process (ProcessPoolExecutor with spawn)
  - Torch CPU (vectorized batch on CPU)
  - Torch CUDA (NVIDIA GPU acceleration)
  - Torch CUDA + cuRAND (native GPU RNG, requires CuPy)

Requirements:
  - NVIDIA GPU with CUDA support
  - PyTorch with CUDA: pip install torch
  - mcframework with GPU extras: pip install mcframework[gpu]
  - Optional: CuPy for cuRAND mode: pip install mcframework[cuda]

Usage:
  python demo_cuda_benchmark.py
"""

from __future__ import annotations

import multiprocessing as mp
import platform
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    backend: str
    n_simulations: int
    execution_time: float
    mean_estimate: float
    throughput: float  # simulations per second
    peak_memory_mb: float = 0.0  # Peak GPU memory in MB

    @property
    def speedup_vs(self) -> float:
        """Placeholder for relative speedup calculation."""
        return 1.0


def check_cuda_availability() -> tuple[bool, bool, int]:
    """
    Check if CUDA is available and if CuPy is installed.
    
    Returns
    -------
    tuple[bool, bool, int]
        (torch_available, cuda_available, num_gpus)
    """
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if cuda_available else 0
    except ImportError:
        torch_available = False
        cuda_available = False
        num_gpus = 0
    
    return torch_available, cuda_available, num_gpus


def check_cupy_availability() -> bool:
    """Check if CuPy is installed for cuRAND support."""
    try:
        import cupy as cp  # noqa: F401
        return True
    except ImportError:
        return False


def print_system_info():
    """Print system information for benchmark context."""
    torch_available, cuda_available, num_gpus = check_cuda_availability()
    cupy_available = check_cupy_availability()
    
    print("=" * 70)
    print("CUDA BENCHMARK - System Information")
    print("=" * 70)
    print(f"  Platform:      {platform.system()} {platform.release()}")
    print(f"  Machine:       {platform.machine()}")
    print(f"  Processor:     {platform.processor()}")
    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  CPU Cores:     {mp.cpu_count()}")
    print(f"  PyTorch:       {'Installed' if torch_available else 'Not installed'}")
    
    if torch_available:
        import torch
        print(f"  PyTorch Ver:   {torch.__version__}")
        print(f"  CUDA Available: {'Yes' if cuda_available else 'No'}")
        
        if cuda_available:
            print(f"  CUDA Version:  {torch.version.cuda}")
            print(f"  GPU Count:     {num_gpus}")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                print(f"    [{i}] {gpu_name} ({total_memory:.1f} GB)")
        
        print(f"  CuPy (cuRAND): {'Installed' if cupy_available else 'Not installed'}")
    
    print("=" * 70)
    print()
    
    if not cuda_available:
        print("Warning: CUDA not available on this system.")
        print("   GPU acceleration will not be available.\n")
    elif not cupy_available:
        print("Info: CuPy not installed - cuRAND benchmarks will be skipped.")
        print("   Install with: pip install mcframework[cuda]\n")
    
    return torch_available, cuda_available, cupy_available, num_gpus


def run_benchmark(
    sim,
    n_simulations: int,
    backend: str,
    torch_device: str = "cpu",
    n_workers: int | None = None,
    warmup: bool = False,
    device_id: int = 0,
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
        Device for torch backend: "cpu" or "cuda".
    n_workers : int, optional
        Number of workers for parallel backends.
    warmup : bool
        If True, this is a warmup run (don't return results).
    device_id : int
        CUDA device ID to use.
    
    Returns
    -------
    BenchmarkResult or None
        Benchmark results, or None if warmup run.
    """
    sim.set_seed(42)  # Ensure reproducibility
    
    # Track GPU memory if using CUDA
    peak_memory = 0.0
    if backend == "torch" and torch_device == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats(device_id)
    
    try:
        result = sim.run(
            n_simulations,
            backend=backend,
            torch_device=torch_device,
            n_workers=n_workers,
            compute_stats=False,  # Skip stats for pure timing
        )
        
        # Capture peak memory usage for CUDA
        if backend == "torch" and torch_device == "cuda":
            import torch
            peak_memory = torch.cuda.max_memory_allocated(device_id) / (1024**2)  # MB
        
        if warmup:
            return None
        
        return BenchmarkResult(
            backend=f"{backend}" if backend != "torch" else f"torch-{torch_device}",
            n_simulations=n_simulations,
            execution_time=result.execution_time,
            mean_estimate=result.mean,
            throughput=n_simulations / result.execution_time,
            peak_memory_mb=peak_memory,
        )
    except Exception as e:
        if not warmup:
            print(f"  {backend} ({torch_device}): ERROR - {e}")
        return None


def run_benchmark_suite(
    sim,
    cpu_sizes: list[int],
    gpu_sizes: list[int],
    backends: list[tuple[str, str]],  # (backend, torch_device)
    n_workers: int | None = None,
    device_id: int = 0,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run full benchmark suite across multiple simulation sizes and backends.
    
    CPU backends (sequential, thread, process) use smaller cpu_sizes to avoid long waits.
    Torch backends (torch-cpu, torch-cuda) use full gpu_sizes range.
    
    Parameters
    ----------
    sim : MonteCarloSimulation
        The simulation instance to benchmark.
    cpu_sizes : list[int]
        Simulation counts for pure CPU backends (smaller to avoid long waits).
    gpu_sizes : list[int]
        Simulation counts for Torch backends (can be much larger).
    backends : list[tuple[str, str]]
        List of (backend, torch_device) tuples.
    n_workers : int, optional
        Number of workers for parallel backends.
    device_id : int
        CUDA device ID to use.
    
    Returns
    -------
    dict[str, list[BenchmarkResult]]
        Results keyed by backend name.
    """
    results: dict[str, list[BenchmarkResult]] = {}
    
    # Warmup run for each backend
    print("Warming up backends...")
    for backend, device in backends:
        run_benchmark(sim, 1000, backend, device, n_workers, warmup=True, device_id=device_id)
    print()
    
    # Determine which sizes each backend should use
    backend_sizes = {}
    for backend, device in backends:
        backend_name = f"{backend}" if backend != "torch" else f"torch-{device}"
        
        # Torch backends get full GPU range
        if backend == "torch":
            backend_sizes[backend_name] = gpu_sizes
        # Pure CPU backends get smaller sizes only (avoid long waits)
        else:
            backend_sizes[backend_name] = cpu_sizes
    
    # Count total runs
    total_runs = sum(len(sizes) for sizes in backend_sizes.values())
    current_run = 0
    
    # Get all unique sizes in sorted order
    all_sizes = sorted(set(cpu_sizes + gpu_sizes))
    
    for n_sims in all_sizes:
        # Check which backends should run at this size
        backends_to_run = [
            (backend, device, backend_name)
            for backend, device in backends
            for backend_name in [f"{backend}" if backend != "torch" else f"torch-{device}"]
            if n_sims in backend_sizes[backend_name]
        ]
        
        if not backends_to_run:
            continue
        
        print(f"Running benchmarks for n={n_sims:,} simulations...")
        
        for backend, device, backend_name in backends_to_run:
            current_run += 1
            
            result = run_benchmark(sim, n_sims, backend, device, n_workers, device_id=device_id)
            
            if result:
                if backend_name not in results:
                    results[backend_name] = []
                results[backend_name].append(result)
                
                memory_str = f" [{result.peak_memory_mb:.0f} MB]" if result.peak_memory_mb > 0 else ""
                print(
                    f"  [{current_run}/{total_runs}] {backend_name:15s}: "
                    f"{result.execution_time:8.4f}s "
                    f"({result.throughput:,.0f} sims/sec){memory_str}"
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
) -> plt.Figure:
    """
    Create comprehensive benchmark visualization charts.
    
    Handles backends with different simulation sizes (CPU vs Torch backends).
    
    Parameters
    ----------
    results : dict[str, list[BenchmarkResult]]
        Benchmark results by backend name.
    
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
        "torch-cuda": "#10B981",    # Emerald (hero color for CUDA)
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
        "CUDA (NVIDIA GPU) vs CPU Backends",
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
        sizes = [r.n_simulations for r in backend_results]
        times = [r.execution_time for r in backend_results]
        ax1.plot(
            sizes, times,
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
        sizes = [r.n_simulations for r in backend_results]
        throughputs = [r.throughput for r in backend_results]
        ax2.plot(
            sizes, throughputs,
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
    
    # Find largest common size for fair comparison
    if "sequential" in results:
        seq_sizes = [r.n_simulations for r in results["sequential"]]
        largest_common_size = max(seq_sizes)
        
        # Calculate speedups at largest common size
        seq_time_at_size = next(
            (r.execution_time for r in results["sequential"] if r.n_simulations == largest_common_size),
            None
        )
        
        if seq_time_at_size:
            backends_list = []
            speedup_values = []
            bar_colors = []
            
            for backend, backend_results in results.items():
                if backend == "sequential":
                    continue
                
                # Find result at largest common size
                result_at_size = next(
                    (r for r in backend_results if r.n_simulations == largest_common_size),
                    None
                )
                
                if result_at_size:
                    backends_list.append(backend)
                    speedup_values.append(seq_time_at_size / result_at_size.execution_time)
                    bar_colors.append(colors[backend])
            
            if backends_list:
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
                ax3.set_xticklabels(backends_list, fontsize=10, color="#E2E8F0", rotation=15)
                ax3.set_ylabel("Speedup Factor", fontsize=11, color="#E2E8F0")
                ax3.set_title(
                    f"Speedup vs Sequential (n={largest_common_size:,})",
                    fontsize=13, fontweight="bold", color="#F8FAFC", pad=10,
                )
                ax3.legend(loc="upper right", facecolor="#334155", edgecolor="#475569",
                           labelcolor="#E2E8F0", fontsize=9)
    
    ax3.grid(True, alpha=0.3, color="#475569", axis="y")
    ax3.tick_params(colors="#94A3B8")
    for spine in ax3.spines.values():
        spine.set_color("#475569")
    
    # --- Chart 4: GPU Memory Usage (if CUDA data available) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    
    # Check if we have CUDA memory data
    cuda_backends = [b for b in results.keys() if "cuda" in b.lower()]
    if cuda_backends:
        for backend in cuda_backends:
            sizes = [r.n_simulations for r in results[backend]]
            memory_usage = [r.peak_memory_mb for r in results[backend]]
            if any(m > 0 for m in memory_usage):
                ax4.plot(
                    sizes, memory_usage,
                    marker="^", markersize=8, linewidth=2.5,
                    color=colors[backend], label=backend,
                )
        
        ax4.set_xscale("log")
        ax4.set_xlabel("Number of Simulations", fontsize=11, color="#E2E8F0")
        ax4.set_ylabel("Peak GPU Memory (MB)", fontsize=11, color="#E2E8F0")
        ax4.set_title("GPU Memory Usage", fontsize=13,
                      fontweight="bold", color="#F8FAFC", pad=10)
        ax4.legend(loc="upper left", facecolor="#334155", edgecolor="#475569",
                   labelcolor="#E2E8F0", fontsize=9)
    
    ax4.grid(True, alpha=0.3, color="#475569")
    ax4.tick_params(colors="#94A3B8")
    for spine in ax4.spines.values():
        spine.set_color("#475569")
    
    return fig


def create_summary_table(
    results: dict[str, list[BenchmarkResult]],
) -> str:
    """
    Create a text summary table of benchmark results.
    
    Handles backends with different simulation sizes.
    
    Parameters
    ----------
    results : dict[str, list[BenchmarkResult]]
        Benchmark results by backend name.
    
    Returns
    -------
    str
        Formatted summary table.
    """
    # Get all unique sizes tested across all backends
    all_sizes = sorted(set(
        r.n_simulations 
        for backend_results in results.values() 
        for r in backend_results
    ))
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 100)
    
    # Header
    header = f"{'Backend':<15}"
    for n in all_sizes:
        header += f" | {n:>12,}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Execution times
    lines.append("Execution Time (seconds):")
    for backend, backend_results in results.items():
        row = f"  {backend:<13}"
        # Create a map of size -> result
        size_to_result = {r.n_simulations: r for r in backend_results}
        for size in all_sizes:
            if size in size_to_result:
                row += f" | {size_to_result[size].execution_time:>12.4f}"
            else:
                row += f" | {'—':>12}"  # Em dash for missing data
        lines.append(row)
    
    lines.append("")
    
    # Speedups (only for sizes where sequential was tested)
    if "sequential" in results:
        seq_sizes = [r.n_simulations for r in results["sequential"]]
        seq_times = {r.n_simulations: r.execution_time for r in results["sequential"]}
        
        lines.append("Speedup vs Sequential:")
        for backend, backend_results in results.items():
            row = f"  {backend:<13}"
            size_to_result = {r.n_simulations: r for r in backend_results}
            
            for size in all_sizes:
                if size in seq_times and size in size_to_result:
                    speedup = seq_times[size] / size_to_result[size].execution_time
                    row += f" | {speedup:>11.2f}×"
                else:
                    row += f" | {'—':>12}"
            lines.append(row)
    
    lines.append("")
    
    # GPU Memory (if available)
    cuda_backends = [b for b in results.keys() if "cuda" in b.lower()]
    if cuda_backends:
        lines.append("Peak GPU Memory (MB):")
        for backend in cuda_backends:
            row = f"  {backend:<13}"
            size_to_result = {r.n_simulations: r for r in results[backend]}
            for size in all_sizes:
                if size in size_to_result and size_to_result[size].peak_memory_mb > 0:
                    row += f" | {size_to_result[size].peak_memory_mb:>11.0f}M"
                else:
                    row += f" | {'—':>12}"
            lines.append(row)
        lines.append("")
    
    # Best performer for largest tested size overall
    if results:
        largest_size = max(all_sizes)
        
        # Find all backends that tested the largest size
        backends_at_largest = {
            backend: next((r for r in backend_results if r.n_simulations == largest_size), None)
            for backend, backend_results in results.items()
        }
        backends_at_largest = {k: v for k, v in backends_at_largest.items() if v is not None}
        
        if backends_at_largest:
            best_backend = min(backends_at_largest.keys(), 
                              key=lambda b: backends_at_largest[b].execution_time)
            best_result = backends_at_largest[best_backend]
            
            lines.append(f"🏆 Best performer (n={largest_size:,}): {best_backend}")
            lines.append(f"   Execution time: {best_result.execution_time:.4f}s")
            lines.append(f"   Throughput: {best_result.throughput:,.0f} simulations/sec")
            
            # Calculate speedup if sequential was tested at this size
            if "sequential" in results:
                seq_result_at_size = next(
                    (r for r in results["sequential"] if r.n_simulations == largest_size),
                    None
                )
                if seq_result_at_size:
                    speedup = seq_result_at_size.execution_time / best_result.execution_time
                    lines.append(f"   Speedup: {speedup:.1f}× faster than sequential")
            
            if best_result.peak_memory_mb > 0:
                lines.append(f"   Peak GPU memory: {best_result.peak_memory_mb:.0f} MB")
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def main():
    """Run the CUDA benchmark demo."""
    # Print system info and check capabilities
    torch_available, cuda_available, cupy_available, num_gpus = print_system_info()
    
    if not cuda_available:
        print("Error: CUDA not available on this system.")
        print("This benchmark requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)
    
    # Import mcframework
    try:
        from mcframework import PiEstimationSimulation
    except ImportError:
        print("Error: mcframework not installed.")
        print("Install with: pip install mcframework")
        sys.exit(1)
    
    # Create simulation instance
    sim = PiEstimationSimulation()
    
    # Define simulation sizes to test
    # CPU backends (sequential, thread, process): smaller sizes to avoid long waits
    cpu_sizes = [
        10_000,
        100_000,  # Max for CPU backends (sequential ~3min on Apple Silicon)
    ]
    
    # Torch backends (torch-cpu, torch-cuda): full range including large sizes
    gpu_sizes = [
        10_000,      # Common with CPU for comparison
        100_000,     # Common with CPU for comparison
        1_000_000,   # Torch only
        5_000_000,   # Torch only
        10_000_000,  # Torch only
    ]
    
    # Define backends to test
    # Note: Only torch backends will be tested (sequential/thread/process
    # included for comparison at smaller sizes only)
    backends: list[tuple[str, str]] = []
    
    # Add CPU backends (will only test on cpu_sizes)
    backends.extend([
        ("sequential", ""),
        ("thread", ""),
        ("process", ""),
    ])
    
    # Add torch backends (will test on full gpu_sizes)
    if torch_available:
        backends.append(("torch", "cpu"))
        if cuda_available:
            backends.append(("torch", "cuda"))
    
    # Select GPU device
    device_id = 0
    if num_gpus > 1:
        print(f"Multiple GPUs detected ({num_gpus} devices)")
        try:
            device_input = input(f"Select GPU device (0-{num_gpus-1}, default 0): ").strip()
            if device_input:
                device_id = int(device_input)
                if device_id >= num_gpus:
                    print(f"Invalid device ID. Using device 0.")
                    device_id = 0
        except (ValueError, EOFError):
            device_id = 0
        print(f"Using GPU device {device_id}\n")
    
    # Run benchmark suite
    print("\n" + "=" * 70)
    print("STARTING BENCHMARK SUITE")
    print("=" * 70 + "\n")
    
    n_workers = mp.cpu_count()
    print(f"Using {n_workers} workers for parallel backends")
    print(f"CPU backends testing: {', '.join(f'{s:,}' for s in cpu_sizes)} simulations")
    print(f"Torch backends testing: {', '.join(f'{s:,}' for s in gpu_sizes)} simulations\n")
    
    results = run_benchmark_suite(
        sim,
        cpu_sizes=cpu_sizes,
        gpu_sizes=gpu_sizes,
        backends=backends,
        n_workers=n_workers,
        device_id=device_id,
    )
    
    if not results:
        print("No benchmark results collected. Exiting.")
        sys.exit(1)
    
    # Print summary table
    summary = create_summary_table(results)
    print(summary)
    
    # Create visualizations
    print("\nGenerating benchmark visualizations...")
    
    fig = create_benchmark_visualizations(results)
    
    # Show the plot
    plt.show()
    
    # Optionally save
    try:
        save = input("\nSave benchmark chart to file? (y/N): ").lower().strip()
        if save == "y":
            filename = "cuda_benchmark.png"
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

