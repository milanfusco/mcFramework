"""
Black-Scholes Monte Carlo Simulation Demo
==========================================

This module demonstrates the Black-Scholes option pricing and path simulation
capabilities of the mcframework package. It generates comprehensive visualizations
including static plots and animations of stock price paths and distributions.

Features:
    - European and American option pricing
    - Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
    - Stock price path simulation
    - Animated visualizations
    - Convergence analysis

Example:
    python demoBlackScholes.py
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.typing import NDArray
from scipy.stats import norm

from mcframework.core import MonteCarloFramework, SimulationResult
from mcframework.sims import BlackScholesPathSimulation, BlackScholesSimulation

# =============================================================================
# Configuration Constants
# =============================================================================

# Output Configuration
OUTPUT_DIR = Path("img/black_scholes")
DPI = 150
ANIMATION_FPS_PATHS = 30
ANIMATION_FPS_DIST = 20
ANIMATION_BITRATE = 1800

# Visualization Parameters
N_PATHS_TO_DISPLAY = 20
FIGURE_SIZE_LARGE = (10, 6)
FIGURE_SIZE_MEDIUM = (8, 5)
ALPHA_INDIVIDUAL_PATHS = 0.6
ALPHA_HISTOGRAM = 0.7
LINEWIDTH_PATHS = 0.8
LINEWIDTH_MEAN = 2.5
GRID_ALPHA = 0.3

# Simulation Parameters
SEED = 999
N_SIMULATIONS_PRICING = 10_000
N_SIMULATIONS_GREEKS = 5_000
N_SIMULATIONS_PATHS = 1_000
N_PATHS_VISUALIZATION = 100

# Color Scheme
COLOR_PATHS = 'steelblue'
COLOR_MEAN = 'red'
COLOR_MEDIAN = 'orange'
COLOR_HISTOGRAM = 'steelblue'
GREEKS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save a matplotlib figure to the output directory.
    
    Parameters
    ----------
    fig : Figure
        The matplotlib figure to save.
    filename : str
        Name of the output file.
    """
    filepath = OUTPUT_DIR / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', pad_inches=0.5)
    print(f"Saved plot to {filepath}")


# =============================================================================
# Static Plotting Functions
# =============================================================================

def plot_paths(
    paths: NDArray[np.float64],
    T: float = 1.0,
    show_mean: bool = True,
    show_median: bool = False,
    filename: str = "bs_paths.png"
) -> None:
    """
    Plot simulated stock price paths with optional mean/median trend lines.
    
    Parameters
    ----------
    paths : ndarray
        Array of shape (n_paths, n_steps + 1) containing stock price paths.
    T : float, default 1.0
        Time horizon in years.
    show_mean : bool, default True
        If True, plot the mean path as a trend line.
    show_median : bool, default False
        If True, plot the median path as a trend line.
    filename : str, default "bs_paths.png"
        Output filename.
    """
    n_paths, n_steps_plus_1 = paths.shape
    t = np.linspace(0, T, n_steps_plus_1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    
    # Plot individual paths
    n_display = min(N_PATHS_TO_DISPLAY, n_paths)
    for i in range(n_display):
        ax.plot(t, paths[i], alpha=ALPHA_INDIVIDUAL_PATHS, 
                linewidth=LINEWIDTH_PATHS, color=COLOR_PATHS)
    
    # Plot mean path if requested
    if show_mean:
        mean_path = np.mean(paths, axis=0)
        ax.plot(t, mean_path, color=COLOR_MEAN, linewidth=LINEWIDTH_MEAN, 
                label=f'Mean Path (${mean_path[-1]:.2f})', linestyle='-')
    
    # Plot median path if requested
    if show_median:
        median_path = np.median(paths, axis=0)
        ax.plot(t, median_path, color=COLOR_MEDIAN, linewidth=LINEWIDTH_MEAN, 
                label=f'Median Path (${median_path[-1]:.2f})', linestyle='--')
    
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price ($)")
    ax.set_title("Simulated Black–Scholes Price Paths")
    ax.legend(loc='best')
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_figure(fig, filename)
    plt.close(fig)


def plot_final_price_distribution(
    paths: NDArray[np.float64],
    filename: str = "bs_final_distribution.png"
) -> None:
    """
    Plot histogram of final stock prices at maturity.
    
    Parameters
    ----------
    paths : ndarray
        Array of shape (n_paths, n_steps + 1) containing stock price paths.
    filename : str, default "bs_final_distribution.png"
        Output filename.
    """
    final_prices = paths[:, -1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    ax.hist(final_prices, bins=30, edgecolor="black", 
            alpha=ALPHA_HISTOGRAM, color=COLOR_HISTOGRAM)
    ax.set_xlabel("Final Stock Price ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Final Prices at Maturity")
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_figure(fig, filename)
    plt.close(fig)


def plot_convergence(
    estimates: NDArray[np.float64],
    filename: str = "bs_convergence.png"
) -> None:
    """
    Plot convergence of Monte Carlo estimates over simulation runs.
    
    Parameters
    ----------
    estimates : ndarray
        Array of individual simulation estimates.
    filename : str, default "bs_convergence.png"
        Output filename.
    """
    cumulative_mean = np.cumsum(estimates) / np.arange(1, len(estimates) + 1)
    sample_indices = np.arange(1, len(estimates) + 1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    ax.plot(sample_indices, cumulative_mean, color="blue", 
            alpha=0.8, linewidth=1.5)
    ax.axhline(cumulative_mean[-1], color=COLOR_MEAN, linestyle="--", 
               linewidth=2, label=f"Final Mean: ${cumulative_mean[-1]:.4f}")
    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Cumulative Mean Option Price ($)")
    ax.set_title("Monte Carlo Option Price Convergence")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    
    save_figure(fig, filename)
    plt.close(fig)


def plot_greeks(
    greeks: dict[str, float],
    filename: str = "bs_greeks.png"
) -> None:
    """
    Plot option Greeks as a bar chart.
    
    Parameters
    ----------
    greeks : dict
        Dictionary containing Greeks values with keys: 
        'delta', 'gamma', 'vega', 'theta', 'rho'.
    filename : str, default "bs_greeks.png"
        Output filename.
    """
    keys = ["delta", "gamma", "vega", "theta", "rho"]
    values = [greeks[k] for k in keys]
    labels = [k.capitalize() for k in keys]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    ax.bar(labels, values, color=GREEKS_COLORS, 
           alpha=ALPHA_HISTOGRAM, edgecolor='black')
    ax.set_title("Greeks for European Call Option")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    
    save_figure(fig, filename)
    plt.close(fig)


def black_scholes_call_price(S, K, r, sigma, tau):
    """
    Vectorized Black–Scholes formula for call options.
    Works with scalars, 1D arrays, and 2D arrays (for surface grids).
    """
    S = np.asarray(S)
    tau = np.asarray(tau)

    # Payoff at maturity (tau = 0)
    intrinsic = np.maximum(S - K, 0.0)

    # Where tau == 0, return intrinsic value
    zero_mask = tau <= 0

    # Allocate result array
    C = np.zeros_like(S, dtype=float)
    C[zero_mask] = intrinsic[zero_mask]

    # Normal BS formula where tau > 0
    positive_mask = ~zero_mask
    if np.any(positive_mask):
        tau_pos = tau[positive_mask]
        S_pos = S[positive_mask]

        d1 = (np.log(S_pos / K) + (r + 0.5 * sigma**2) * tau_pos) / (sigma * np.sqrt(tau_pos))
        d2 = d1 - sigma * np.sqrt(tau_pos)

        C[positive_mask] = (
            S_pos * norm.cdf(d1) -
            K * np.exp(-r * tau_pos) * norm.cdf(d2)
        )

    return C


def plot_price_surface_3d(paths, T=1.0, K=100, r=0.05, sigma=0.20,
                           filename="bs_surface.png"):
    """
    Create a 3D plot of option price surface + simulated stock paths.
    """
    n_paths, n_steps = paths.shape
    t = np.linspace(0, T, n_steps)

    # ===== 1) Create Stock Price Grid =====
    S_min = np.min(paths)
    S_max = np.max(paths)
    S_vals = np.linspace(S_min, S_max, 120)

    T_vals = np.linspace(0, T, 120)
    T_grid, S_grid = np.meshgrid(T_vals, S_vals)

    # Time to maturity tau = T - t
    tau_grid = (T - T_grid)

    # ===== 2) Compute Option Price Surface =====
    C_grid = black_scholes_call_price(S_grid, K, r, sigma, tau_grid)

    # ===== 3) Compute Option Value Along Simulated Paths =====
    C_paths = np.zeros_like(paths)
    for i in range(n_steps):
        tau = T - t[i]
        C_paths[:, i] = black_scholes_call_price(paths[:, i], K, r, sigma, tau)

    # ===== 4) Plot Everything =====
    fig = plt.figure(figsize=(24, 10)) 
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1, 0.8])

    # Surface
    ax.plot_surface(
        T_grid, S_grid, C_grid,
        cmap="viridis", alpha=0.4, rstride=4, cstride=4, linewidth=0
    )

    # Path cloud
    for i in range(min(80, n_paths)):
        ax.plot(t, paths[i], C_paths[i], color="black", alpha=0.05)

    # Highlight mean path in red
    mean_path = np.mean(paths, axis=0)
    tau_mean = T - t  # Time to maturity for each time point along the mean path
    mean_price = black_scholes_call_price(mean_path, K, r, sigma, tau_mean)
    ax.plot(t, mean_path, mean_price, color="red", linewidth=2)

    ax.set_xlabel("Time t")
    ax.set_ylabel("Stock Price Sₜ")
    ax.set_zlabel("Call Price Cₜ")
    ax.set_title("Black–Scholes Call Price Surface with Simulated Paths")

    save_figure(fig, filename)
    plt.close(fig)


# =============================================================================
# Animation Functions
# =============================================================================

def animate_paths(
    paths: NDArray[np.float64],
    T: float = 1.0,
    filename: str = "bs_paths_animation.mp4"
) -> None:
    """
    Create animated visualization of Black-Scholes price paths with mean line.
    
    Parameters
    ----------
    paths : ndarray
        Array of shape (n_paths, n_steps + 1) containing stock price paths.
    T : float, default 1.0
        Time horizon in years.
    filename : str, default "bs_paths_animation.mp4"
        Output filename.
    """
    n_paths, n_steps_plus_1 = paths.shape
    t = np.linspace(0, T, n_steps_plus_1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    ax.set_xlim(0, T)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price ($)")
    ax.set_title("Black-Scholes Price Paths, first 20 paths")
    ax.grid(True, alpha=GRID_ALPHA)
    
    # Start with tight y-limits that will expand
    initial_price = paths[0, 0]
    ax.set_ylim(initial_price * 0.95, initial_price * 1.05)

    # Initialize line objects
    num_display = min(N_PATHS_TO_DISPLAY, n_paths)
    lines = [ax.plot([], [], alpha=ALPHA_INDIVIDUAL_PATHS, 
                     linewidth=LINEWIDTH_PATHS, color=COLOR_PATHS)[0] 
             for _ in range(num_display)]
    
    mean_line, = ax.plot([], [], color=COLOR_MEAN, linewidth=2.0, 
                         label='Mean Path', linestyle='-')
    ax.legend(loc='best')

    def update(frame: int) -> list:
        """Update function for animation."""
        # Skip initial frames to avoid empty arrays
        if frame < 2:
            frame = 2
        
        # Update individual path lines
        for i, line in enumerate(lines):
            line.set_data(t[:frame], paths[i, :frame])

        # Update mean trend line
        mean_path = np.mean(paths, axis=0)
        mean_line.set_data(t[:frame], mean_path[:frame])

        # Adjust y-limits dynamically
        current_data = paths[:num_display, :frame]
        current_mean = mean_path[:frame]
        combined_data = np.concatenate([current_data.flatten(), current_mean])
        y_min, y_max = np.min(combined_data), np.max(combined_data)
        padding = 0.05
        ax.set_ylim(y_min * (1 - padding), y_max * (1 + padding))

        return lines + [mean_line]

    anim = FuncAnimation(fig, update, frames=n_steps_plus_1,
                        interval=20, blit=False)

    writer = FFMpegWriter(fps=ANIMATION_FPS_PATHS, bitrate=ANIMATION_BITRATE)
    output_path = OUTPUT_DIR / filename
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def animate_distribution(
    paths: NDArray[np.float64],
    T: float = 1.0,
    filename: str = "bs_distribution_animation.mp4"
) -> None:
    """
    Create animated histogram of evolving price distribution.
    
    Parameters
    ----------
    paths : ndarray
        Array of shape (n_paths, n_steps + 1) containing stock price paths.
    T : float, default 1.0
        Time horizon in years.
    filename : str, default "bs_distribution_animation.mp4"
        Output filename.
    """
    n_paths, n_steps_plus_1 = paths.shape
    t = np.linspace(0, T, n_steps_plus_1)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    
    # Set consistent limits
    x_min, x_max = np.min(paths) * 0.9, np.max(paths) * 1.1
    
    # Estimate max frequency for y-limits
    mid_frame = n_steps_plus_1 // 2
    sample_hist, _ = np.histogram(paths[:, mid_frame], bins=30)
    y_max = int(np.max(sample_hist) * 1.3)

    def update(frame: int) -> None:
        """Update function for distribution animation."""
        ax.cla()
        
        # Create histogram
        ax.hist(paths[:, frame], bins=30, range=(x_min, x_max),
                edgecolor="black", alpha=ALPHA_HISTOGRAM, color=COLOR_HISTOGRAM)
        
        # Set limits and labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Stock Price ($)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Price Distribution at t = {t[frame]:.3f} years")
        ax.grid(True, alpha=GRID_ALPHA, axis='y')
        
        # Add mean and median indicators
        mean_price = np.mean(paths[:, frame])
        median_price = np.median(paths[:, frame])
        ax.axvline(mean_price, color=COLOR_MEAN, linestyle='--', 
                   linewidth=2, label=f'Mean: ${mean_price:.2f}')
        ax.axvline(median_price, color=COLOR_MEDIAN, linestyle='--',
                   linewidth=2, label=f'Median: ${median_price:.2f}')
        ax.legend()

    anim = FuncAnimation(fig, update, frames=n_steps_plus_1, 
                        interval=40, blit=False)

    writer = FFMpegWriter(fps=ANIMATION_FPS_DIST, bitrate=ANIMATION_BITRATE)
    output_path = OUTPUT_DIR / filename
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


# =============================================================================
# Simulation Functions
# =============================================================================

def run_option_pricing(
    fw: MonteCarloFramework,
    seed: int
) -> tuple[SimulationResult, SimulationResult, BlackScholesSimulation]:
    """
    Run European and American option pricing simulations.
    
    Parameters
    ----------
    fw : MonteCarloFramework
        Framework instance for managing simulations.
    
    Returns
    -------
    tuple
        (european_result, american_result, european_sim) containing 
        simulation results and the European simulation instance.
    """
    # European Call Option
    european_sim = BlackScholesSimulation(name="European Call Option")
    european_sim.set_seed(seed)
    fw.register_simulation(european_sim)
    
    european_result = fw.run_simulation(
        "European Call Option",
        N_SIMULATIONS_PRICING,
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20,
        option_type="call", exercise_type="european"
    )
    
    # American Put Option
    american_sim = BlackScholesSimulation(name="American Put Option")
    american_sim.set_seed(seed)
    fw.register_simulation(american_sim)
    
    american_result = fw.run_simulation(
        "American Put Option",
        N_SIMULATIONS_PRICING,
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20,
        option_type="put", exercise_type="american"
    )
    
    return european_result, american_result, european_sim


def calculate_greeks(sim: BlackScholesSimulation) -> dict[str, float]:
    """
    Calculate option Greeks using finite differences.
    
    Parameters
    ----------
    sim : BlackScholesSimulation
        Simulation instance to use for Greeks calculation.
    
    Returns
    -------
    dict
        Dictionary containing Greeks: price, delta, gamma, vega, theta, rho.
    """
    return sim.calculate_greeks(
        n_simulations=N_SIMULATIONS_GREEKS,
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20,
        option_type="call", exercise_type="european",
        parallel=True
    )


def simulate_paths() -> NDArray[np.float64]:
    """
    Simulate stock price paths for visualization.
    
    Returns
    -------
    ndarray
        Array of shape (n_paths, n_steps + 1) containing price paths.
    """
    path_sim = BlackScholesPathSimulation(name="Black-Scholes Path Simulation")
    path_sim.set_seed(SEED)
    
    # Get statistics on final prices
    result_paths = path_sim.run(
        N_SIMULATIONS_PATHS,
        S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252
    )
    print(f"Average final price: ${result_paths.mean:.2f}")

    # Generate paths for visualization
    path_sim.set_seed(SEED)
    paths = path_sim.simulate_paths(
        n_paths=N_PATHS_VISUALIZATION,
        S0=100.0, r=0.05, sigma=0.20, T=1.0, n_steps=252
    )
    print(f"Generated paths shape: {paths.shape}")
    
    return paths


def print_comparison(fw: MonteCarloFramework) -> None:
    """Print comparison metrics between European and American options."""
    print("\n" + "=" * 50)
    print("COMPARISON METRICS:")
    comparison = fw.compare_results(["European Call Option", "American Put Option"])
    for name, value in comparison.items():
        print(f"  {name}: {value:.5f}")
    print("=" * 50 + "\n")


def print_greeks(greeks: dict[str, float]) -> None:
    """Print Greeks values."""
    print("\n" + "=" * 50)
    print("GREEKS:")
    print(f"  Option Price: ${greeks['price']:.4f}")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega: {greeks['vega']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Rho: {greeks['rho']:.4f}")
    print("=" * 50)


def generate_visualizations(
    paths: NDArray[np.float64],
    european_result: SimulationResult,
    greeks: dict[str, float]
) -> None:
    """
    Generate all static plots and animations.
    
    Parameters
    ----------
    paths : ndarray
        Stock price paths for visualization.
    european_result : SimulationResult
        European option simulation results.
    greeks : dict
        Calculated Greeks values.
    """
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS:")
    
    # Static plots
    plot_paths(paths)
    plot_final_price_distribution(paths)
    plot_convergence(european_result.results)
    plot_greeks(greeks)
    plot_price_surface_3d(paths)
    # Animations
    animate_paths(paths)
    animate_distribution(paths)
    
    print("=" * 50)


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    """
    Main execution function for Black-Scholes demo.
    
    Runs option pricing simulations, calculates Greeks, generates stock price
    paths, and creates comprehensive visualizations including animations.
    """
    # Setup
    ensure_output_directory()
    fw = MonteCarloFramework()
    
    # Run simulations
    print("Running option pricing simulations...")
    european_result, american_result, european_sim = run_option_pricing(fw, SEED)
    
    # Display results
    print_comparison(fw)
    print(european_result.result_to_string())
    print("\n")
    print(american_result.result_to_string())
    
    # Calculate Greeks
    print("\nCalculating Greeks...")
    greeks = calculate_greeks(european_sim)
    print_greeks(greeks)
    
    # Simulate paths
    print("\nSimulating stock price paths...")
    paths = simulate_paths()
    
    # Generate visualizations
    generate_visualizations(paths, european_result, greeks)
    
    print("\nBlack-Scholes demo complete.\n\n")
    print(f"Plots saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
