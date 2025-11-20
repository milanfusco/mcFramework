"""
Ticker Black-Scholes Analysis
===================================

This module fetches stock market data and performs
Black-Scholes Monte Carlo simulations and analysis using market parameters.

Features:
    - Fetch historical stock data from Yahoo Finance
    - Estimate volatility and drift from market data
    - Price options using calibrated parameters
    - Generate forecasts and visualizations
    - Compare simulated paths with historical data

Usage:
    python demoTickerBlackScholes.py AAPL
    python demoTickerBlackScholes.py MSFT --days 252
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from mcframework.sims import BlackScholesPathSimulation, BlackScholesSimulation

# Try to import yfinance, provide helpful message if not available
try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance package is required for this demo.")
    print("Install it with: pip install yfinance")
    sys.exit(1)


# =============================================================================
# Configuration Constants
# =============================================================================

# Risk-free rate (approximate US Treasury rate)
RISK_FREE_RATE = 0.05  # 5% annual

# Visualization Parameters
DPI = 150
FIGURE_SIZE_LARGE = (14, 8)
FIGURE_SIZE_MEDIUM = (10, 6)
ALPHA_HISTORICAL = 0.8
ALPHA_SIMULATED = 0.3
COLOR_HISTORICAL = 'blue'
COLOR_SIMULATED = 'gray'
COLOR_MEAN = 'red'
GRID_ALPHA = 0.3


# =============================================================================
# Data Fetching and Parameter Estimation
# =============================================================================


def fetch_ticker_data(ticker: str, days: int = 252) -> tuple[np.ndarray, datetime, datetime]:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT').
    days : int, default 252
        Number of trading days to fetch (252 ≈ 1 year).
    
    Returns
    -------
    tuple
        (prices, start_date, end_date) where prices is array of closing prices.
    
    Raises
    ------
    ValueError
        If ticker is invalid or no data is available.
    """
    print(f"\nFetching data for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(days * 1.5))  # Fetch extra to ensure enough trading days
    
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data found for ticker '{ticker}'")
        
        # Get closing prices
        prices = hist['Close'].values
        
        # Take last 'days' data points
        if len(prices) > days:
            prices = prices[-days:]
        
        actual_start = hist.index[0].to_pydatetime()
        actual_end = hist.index[-1].to_pydatetime()
        
        print(f"✓ Fetched {len(prices)} data points from {actual_start.date()} to {actual_end.date()}")
        print(f"  Current price: ${prices[-1]:.2f}")
        
        return prices, actual_start, actual_end
        
    except Exception as e:
        raise ValueError(f"Failed to fetch data for '{ticker}': {str(e)}")


def estimate_parameters(prices: np.ndarray) -> dict[str, float]:
    """
    Estimate Black-Scholes parameters from historical price data.
    
    Parameters
    ----------
    prices : ndarray
        Array of historical closing prices.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'S0': Current stock price (last price)
        - 'mu': Drift (annualized return)
        - 'sigma': Volatility (annualized)
        - 'daily_return_mean': Mean daily return
        - 'daily_return_std': Daily return standard deviation
    """
    # Calculate daily returns
    returns = np.diff(np.log(prices))
    
    # Estimate parameters
    daily_mean = np.mean(returns)
    daily_std = np.std(returns, ddof=1)
    
    # Annualize (252 trading days per year)
    mu = daily_mean * 252
    sigma = daily_std * np.sqrt(252)
    
    S0 = float(prices[-1])
    
    print("\n" + "=" * 60)
    print("ESTIMATED PARAMETERS FROM HISTORICAL DATA")
    print("=" * 60)
    print(f"Current Price (S₀):     ${S0:.2f}")
    print(f"Drift (μ):              {mu:.4f} ({mu*100:.2f}% annual)")
    print(f"Volatility (σ):         {sigma:.4f} ({sigma*100:.2f}% annual)")
    print(f"Daily Return Mean:      {daily_mean:.6f}")
    print(f"Daily Return Std:       {daily_std:.6f}")
    print("=" * 60 + "\n")
    
    return {
        'S0': S0,
        'mu': mu,
        'sigma': sigma,
        'daily_return_mean': daily_mean,
        'daily_return_std': daily_std,
    }


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_historical_with_simulations(
    historical_prices: np.ndarray,
    simulated_paths: np.ndarray,
    ticker: str,
    T: float,
    output_dir: Path,
    filename: str = "historical_vs_simulated.png"
) -> None:
    """
    Plot historical price data alongside simulated future paths.
    
    Parameters
    ----------
    historical_prices : ndarray
        Historical closing prices.
    simulated_paths : ndarray
        Simulated future price paths (n_paths, n_steps + 1).
    ticker : str
        Stock ticker symbol.
    T : float
        Forecast horizon in years.
    output_dir : Path
        Output directory for saving plot.
    filename : str
        Output filename.
    """
    n_hist = len(historical_prices)
    n_paths, n_steps = simulated_paths.shape
    
    # Time arrays
    hist_time = np.arange(n_hist)
    future_time = np.arange(n_hist, n_hist + n_steps)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    
    # Plot historical data
    ax.plot(hist_time, historical_prices, color=COLOR_HISTORICAL, 
            linewidth=2, label='Historical', alpha=ALPHA_HISTORICAL)
    
    # Plot simulated paths
    for i in range(min(50, n_paths)):
        ax.plot(future_time, simulated_paths[i], color=COLOR_SIMULATED, 
                alpha=ALPHA_SIMULATED, linewidth=0.5)
    
    # Plot mean forecast
    mean_forecast = np.mean(simulated_paths, axis=0)
    ax.plot(future_time, mean_forecast, color=COLOR_MEAN, 
            linewidth=2.5, label='Mean Forecast', linestyle='--')
    
    # Add confidence bands
    lower_bound = np.percentile(simulated_paths, 5, axis=0)
    upper_bound = np.percentile(simulated_paths, 95, axis=0)
    ax.fill_between(future_time, lower_bound, upper_bound, 
                     color=COLOR_MEAN, alpha=0.1, label='90% Confidence')
    
    # Styling
    ax.axvline(x=n_hist-1, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.text(n_hist-1, ax.get_ylim()[1]*0.95, 'Today', ha='right', va='top')
    
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} - Historical Data & {T:.1f}-Year Monte Carlo Forecast')
    ax.legend(loc='best')
    ax.grid(True, alpha=GRID_ALPHA)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def plot_return_distribution(
    historical_prices: np.ndarray,
    simulated_paths: np.ndarray,
    ticker: str,
    output_dir: Path,
    filename: str = "return_distribution.png"
) -> None:
    """
    Compare historical vs simulated return distributions.
    
    Parameters
    ----------
    historical_prices : ndarray
        Historical closing prices.
    simulated_paths : ndarray
        Simulated price paths.
    ticker : str
        Stock ticker symbol.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.
    """
    # Calculate returns
    hist_returns = np.diff(np.log(historical_prices))
    
    # Calculate returns for all simulated paths
    sim_returns = []
    for path in simulated_paths:
        sim_returns.extend(np.diff(np.log(path)))
    sim_returns = np.array(sim_returns)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
    
    # Historical returns histogram
    ax1.hist(hist_returns, bins=50, alpha=0.7, color=COLOR_HISTORICAL, edgecolor='black')
    ax1.axvline(np.mean(hist_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(hist_returns):.6f}')
    ax1.set_xlabel('Log Returns')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{ticker} - Historical Returns')
    ax1.legend()
    ax1.grid(True, alpha=GRID_ALPHA)
    
    # Simulated returns histogram
    ax2.hist(sim_returns, bins=50, alpha=0.7, color=COLOR_SIMULATED, edgecolor='black')
    ax2.axvline(np.mean(sim_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sim_returns):.6f}')
    ax2.set_xlabel('Log Returns')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{ticker} - Simulated Returns')
    ax2.legend()
    ax2.grid(True, alpha=GRID_ALPHA)
    
    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def plot_forecast_distribution(
    simulated_paths: np.ndarray,
    S0: float,
    ticker: str,
    T: float,
    output_dir: Path,
    filename: str = "forecast_distribution.png"
) -> None:
    """
    Plot distribution of forecasted final prices.
    
    Parameters
    ----------
    simulated_paths : ndarray
        Simulated price paths.
    S0 : float
        Current stock price.
    ticker : str
        Stock ticker symbol.
    T : float
        Forecast horizon in years.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.
    """
    final_prices = simulated_paths[:, -1]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_MEDIUM)
    
    # Histogram
    ax.hist(final_prices, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add statistics
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    
    ax.axvline(S0, color='blue', linestyle='--', linewidth=2, label=f'Current: ${S0:.2f}')
    ax.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_price:.2f}')
    ax.axvline(median_price, color='orange', linestyle='--', linewidth=2, label=f'Median: ${median_price:.2f}')
    
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{ticker} - Price Distribution in {T:.1f} Years\n(5th: ${p5:.2f}, 95th: ${p95:.2f})')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA, axis='y')
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def black_scholes_call_price(S, K, r, sigma, tau):
    """
    Vectorized Black-Scholes formula for call options.
    Works with scalars, 1D arrays, and 2D arrays (for surface grids).
    
    Parameters
    ----------
    S : float or ndarray
        Stock price(s).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    tau : float or ndarray
        Time to maturity.
    
    Returns
    -------
    float or ndarray
        Call option price(s).
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


def black_scholes_put_price(S, K, r, sigma, tau):
    """
    Vectorized Black-Scholes formula for put options.
    Works with scalars, 1D arrays, and 2D arrays (for surface grids).
    
    Parameters
    ----------
    S : float or ndarray
        Stock price(s).
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    tau : float or ndarray
        Time to maturity.
    
    Returns
    -------
    float or ndarray
        Put option price(s).
    """
    S = np.asarray(S)
    tau = np.asarray(tau)

    # Payoff at maturity (tau = 0)
    intrinsic = np.maximum(K - S, 0.0)

    # Where tau == 0, return intrinsic value
    zero_mask = tau <= 0

    # Allocate result array
    P = np.zeros_like(S, dtype=float)
    P[zero_mask] = intrinsic[zero_mask]

    # Normal BS formula where tau > 0
    positive_mask = ~zero_mask
    if np.any(positive_mask):
        tau_pos = tau[positive_mask]
        S_pos = S[positive_mask]

        d1 = (np.log(S_pos / K) + (r + 0.5 * sigma**2) * tau_pos) / (sigma * np.sqrt(tau_pos))
        d2 = d1 - sigma * np.sqrt(tau_pos)

        P[positive_mask] = (
            K * np.exp(-r * tau_pos) * norm.cdf(-d2) -
            S_pos * norm.cdf(-d1)
        )

    return P


def plot_call_surface_3d(
    paths: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    ticker: str,
    T: float,
    output_dir: Path,
    filename: str = "call_surface_3d.png"
) -> None:
    """
    Create a 3D plot of call option price surface with simulated stock paths.
    
    Parameters
    ----------
    paths : ndarray
        Simulated stock price paths.
    S0 : float
        Current stock price (used as strike).
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    ticker : str
        Stock ticker symbol.
    T : float
        Time horizon in years.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.
    """
    n_paths, n_steps = paths.shape
    t = np.linspace(0, T, n_steps)
    K = S0  # ATM option

    # Create Stock Price Grid
    S_min = np.min(paths)
    S_max = np.max(paths)
    S_vals = np.linspace(S_min, S_max, 120)

    T_vals = np.linspace(0, T, 120)
    T_grid, S_grid = np.meshgrid(T_vals, S_vals)

    # Time to maturity tau = T - t
    tau_grid = (T - T_grid)

    # Compute Option Price Surface
    C_grid = black_scholes_call_price(S_grid, K, r, sigma, tau_grid)

    # Compute Option Value Along Simulated Paths
    C_paths = np.zeros_like(paths)
    for i in range(n_steps):
        tau = T - t[i]
        C_paths[:, i] = black_scholes_call_price(paths[:, i], K, r, sigma, tau)

    # Plot Everything
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1, 0.8])

    # Surface
    ax.plot_surface(
        T_grid, S_grid, C_grid,
        cmap="viridis", alpha=0.4, rstride=4, cstride=4, linewidth=0
    )

    # Path cloud
    for i in range(min(50, n_paths)):
        ax.plot(t, paths[i], C_paths[i], color="black", alpha=0.05)

    # Highlight mean path in red
    mean_path = np.mean(paths, axis=0)
    tau_mean = T - t
    mean_price = black_scholes_call_price(mean_path, K, r, sigma, tau_mean)
    ax.plot(t, mean_path, mean_price, color="red", linewidth=2.5, label='Mean Path')

    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Stock Price ($)", fontsize=10)
    ax.set_zlabel("Call Price ($)", fontsize=10)
    ax.set_title(f"{ticker} - Call Option Price Surface (K=${K:.2f})", fontsize=12)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def plot_put_surface_3d(
    paths: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    ticker: str,
    T: float,
    output_dir: Path,
    filename: str = "put_surface_3d.png"
) -> None:
    """
    Create a 3D plot of put option price surface with simulated stock paths.
    
    Parameters
    ----------
    paths : ndarray
        Simulated stock price paths.
    S0 : float
        Current stock price (used as strike).
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    ticker : str
        Stock ticker symbol.
    T : float
        Time horizon in years.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.
    """
    n_paths, n_steps = paths.shape
    t = np.linspace(0, T, n_steps)
    K = S0  # ATM option

    # Create Stock Price Grid
    S_min = np.min(paths)
    S_max = np.max(paths)
    S_vals = np.linspace(S_min, S_max, 120)

    T_vals = np.linspace(0, T, 120)
    T_grid, S_grid = np.meshgrid(T_vals, S_vals)

    # Time to maturity tau = T - t
    tau_grid = (T - T_grid)

    # Compute Option Price Surface
    P_grid = black_scholes_put_price(S_grid, K, r, sigma, tau_grid)

    # Compute Option Value Along Simulated Paths
    P_paths = np.zeros_like(paths)
    for i in range(n_steps):
        tau = T - t[i]
        P_paths[:, i] = black_scholes_put_price(paths[:, i], K, r, sigma, tau)

    # Plot Everything
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1.5, 1, 0.8])

    # Surface
    ax.plot_surface(
        T_grid, S_grid, P_grid,
        cmap="plasma", alpha=0.4, rstride=4, cstride=4, linewidth=0
    )

    # Path cloud
    for i in range(min(50, n_paths)):
        ax.plot(t, paths[i], P_paths[i], color="darkred", alpha=0.05)

    # Highlight mean path in blue
    mean_path = np.mean(paths, axis=0)
    tau_mean = T - t
    mean_price = black_scholes_put_price(mean_path, K, r, sigma, tau_mean)
    ax.plot(t, mean_path, mean_price, color="blue", linewidth=2.5, label='Mean Path')

    ax.set_xlabel("Time (years)", fontsize=10)
    ax.set_ylabel("Stock Price ($)", fontsize=10)
    ax.set_zlabel("Put Price ($)", fontsize=10)
    ax.set_title(f"{ticker} - Put Option Price Surface (K=${K:.2f})", fontsize=12)

    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def plot_option_analysis(
    S0: float,
    sigma: float,
    r: float,
    ticker: str,
    greeks: dict[str, float],
    output_dir: Path,
    filename: str = "option_analysis.png"
) -> None:
    """
    Plot option price sensitivity analysis and Greeks.
    
    Parameters
    ----------
    S0 : float
        Current stock price.
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    ticker : str
        Stock ticker symbol.
    greeks : dict
        Dictionary of Greeks values.
    output_dir : Path
        Output directory.
    filename : str
        Output filename.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
    
    T = 0.25  # 3 months
    K = S0  # ATM option
    
    # 1. Option price vs Stock price
    S_range = np.linspace(S0 * 0.7, S0 * 1.3, 100)
    call_prices = []
    put_prices = []
    
    for S in S_range:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        call_prices.append(call)
        put_prices.append(put)
    
    ax1.plot(S_range, call_prices, label='Call', color='green', linewidth=2)
    ax1.plot(S_range, put_prices, label='Put', color='red', linewidth=2)
    ax1.axvline(S0, color='black', linestyle='--', alpha=0.5, label=f'Current: ${S0:.2f}')
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title('Option Price vs Stock Price (3M ATM)')
    ax1.legend()
    ax1.grid(True, alpha=GRID_ALPHA)
    
    # 2. Greeks bar chart
    greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
    greek_values = [greeks['delta'], greeks['gamma'], greeks['vega'], 
                    greeks['theta'], greeks['rho']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax2.bar(greek_names, greek_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{ticker} - Option Greeks (3M ATM Call)')
    ax2.grid(True, alpha=GRID_ALPHA, axis='y')
    
    # 3. Option price vs Volatility
    sigma_range = np.linspace(sigma * 0.5, sigma * 1.5, 100)
    call_vol = []
    
    for s in sigma_range:
        d1 = (np.log(S0 / K) + (r + 0.5 * s**2) * T) / (s * np.sqrt(T))
        d2 = d1 - s * np.sqrt(T)
        call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        call_vol.append(call)
    
    ax3.plot(sigma_range * 100, call_vol, color='purple', linewidth=2)
    ax3.axvline(sigma * 100, color='black', linestyle='--', alpha=0.5, 
                label=f'Current: {sigma*100:.1f}%')
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Call Price ($)')
    ax3.set_title('Option Price vs Volatility')
    ax3.legend()
    ax3.grid(True, alpha=GRID_ALPHA)
    
    # 4. Option price vs Time to maturity
    T_range = np.linspace(0.01, 2, 100)
    call_time = []
    
    for t in T_range:
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        call = S0 * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        call_time.append(call)
    
    ax4.plot(T_range, call_time, color='teal', linewidth=2)
    ax4.set_xlabel('Time to Maturity (years)')
    ax4.set_ylabel('Call Price ($)')
    ax4.set_title('Option Price vs Time to Maturity')
    ax4.grid(True, alpha=GRID_ALPHA)
    
    plt.tight_layout()
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filepath}")


# =============================================================================
# Main Analysis Function
# =============================================================================


def analyze_ticker(
    ticker: str,
    historical_days: int = 252,
    forecast_horizon: float = 1.0,
    n_simulations: int = 1000,
    n_paths_viz: int = 100,
    seed: int = 42
) -> None:
    """
    Perform complete Black-Scholes analysis on a stock ticker.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    historical_days : int
        Number of historical trading days to fetch.
    forecast_horizon : float
        Forecast horizon in years.
    n_simulations : int
        Number of Monte Carlo simulations.
    n_paths_viz : int
        Number of paths to visualize.
    seed : int
        Random seed for reproducibility.
    """
    # Create output directory
    output_dir = Path(f"img/{ticker.upper()}_BlackScholes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"BLACK-SCHOLES ANALYSIS FOR {ticker.upper()}")
    print("=" * 60)
    
    # 1. Fetch historical data
    try:
        prices, start_date, end_date = fetch_ticker_data(ticker, historical_days)
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # 2. Estimate parameters
    params = estimate_parameters(prices)
    S0 = params['S0']
    mu = params['mu']
    sigma = params['sigma']
    
    # 3. Run Monte Carlo simulations
    print("Running Monte Carlo simulations...")
    path_sim = BlackScholesPathSimulation(name=f"{ticker} Path Simulation")
    path_sim.set_seed(seed)
    
    # Simulate paths for visualization
    n_steps = int(forecast_horizon * 252)  # Daily steps
    paths = path_sim.simulate_paths(
        n_paths=n_paths_viz,
        S0=S0,
        r=mu,  # Use estimated drift as risk-neutral drift
        sigma=sigma,
        T=forecast_horizon,
        n_steps=n_steps
    )
    
    print(f"✓ Generated {n_paths_viz} simulated paths")
    
    # 4. Price options
    print("\nPricing options...")
    option_sim = BlackScholesSimulation(name=f"{ticker} Option Pricing")
    option_sim.set_seed(seed)
    
    # ATM Call option (3 months)
    K = S0
    T_option = 0.25
    
    call_result = option_sim.run(
        n_simulations,
        S0=S0, K=K, T=T_option, r=RISK_FREE_RATE, sigma=sigma,
        option_type="call", exercise_type="european"
    )
    
    put_result = option_sim.run(
        n_simulations,
        S0=S0, K=K, T=T_option, r=RISK_FREE_RATE, sigma=sigma,
        option_type="put", exercise_type="european"
    )
    
    print(f"\n3-Month ATM Options (K=${K:.2f}):")
    print(f"  Call Price: ${call_result.mean:.2f} ± ${call_result.std:.2f}")
    print(f"  Put Price:  ${put_result.mean:.2f} ± ${put_result.std:.2f}")
    
    # 5. Calculate Greeks
    print("\nCalculating Greeks...")
    greeks = option_sim.calculate_greeks(
        n_simulations=n_simulations,
        S0=S0, K=K, T=T_option, r=RISK_FREE_RATE, sigma=sigma,
        option_type="call", exercise_type="european",
        parallel=True
    )
    
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega:  {greeks['vega']:.4f}")
    print(f"  Theta: {greeks['theta']:.4f}")
    print(f"  Rho:   {greeks['rho']:.4f}")
    
    # 6. Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_historical_with_simulations(prices, paths, ticker.upper(), 
                                    forecast_horizon, output_dir)
    plot_return_distribution(prices, paths, ticker.upper(), output_dir)
    plot_forecast_distribution(paths, S0, ticker.upper(), forecast_horizon, output_dir)
    plot_option_analysis(S0, sigma, RISK_FREE_RATE, ticker.upper(), greeks, output_dir)
    
    # 3D Surface plots for call and put options
    print("\nGenerating 3D surface plots...")
    plot_call_surface_3d(paths, S0, sigma, RISK_FREE_RATE, ticker.upper(), 
                         forecast_horizon, output_dir)
    plot_put_surface_3d(paths, S0, sigma, RISK_FREE_RATE, ticker.upper(), 
                        forecast_horizon, output_dir)
    
    # 7. Summary statistics
    final_prices = paths[:, -1]
    print("\n" + "=" * 60)
    print(f"FORECAST SUMMARY ({forecast_horizon:.1f}-YEAR HORIZON)")
    print("=" * 60)
    print(f"Current Price:        ${S0:.2f}")
    print(f"Mean Forecast:        ${np.mean(final_prices):.2f}")
    print(f"Median Forecast:      ${np.median(final_prices):.2f}")
    print(f"5th Percentile:       ${np.percentile(final_prices, 5):.2f}")
    print(f"95th Percentile:      ${np.percentile(final_prices, 95):.2f}")
    print(f"Probability > ${S0:.2f}:   {100 * np.mean(final_prices > S0):.1f}%")
    print("=" * 60)
    
    print(f"\n✓ Analysis complete! Charts saved to: {output_dir}/")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for ticker analysis."""
    parser = argparse.ArgumentParser(
        description='Black-Scholes Monte Carlo Analysis for Real Stock Tickers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demoTickerBlackScholes.py AAPL
  python demoTickerBlackScholes.py MSFT --days 500 --horizon 2.0
  python demoTickerBlackScholes.py TSLA --simulations 5000 --seed 999
        """
    )
    
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)')
    parser.add_argument('--days', type=int, default=252, 
                       help='Number of historical trading days to fetch (default: 252)')
    parser.add_argument('--horizon', type=float, default=1.0,
                       help='Forecast horizon in years (default: 1.0)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of Monte Carlo simulations (default: 1000)')
    parser.add_argument('--paths', type=int, default=100,
                       help='Number of paths to visualize (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Run analysis
    try:
        analyze_ticker(
            ticker=args.ticker.upper(),
            historical_days=args.days,
            forecast_horizon=args.horizon,
            n_simulations=args.simulations,
            n_paths_viz=args.paths,
            seed=args.seed
        )
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

