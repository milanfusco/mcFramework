"""
State model for ticker analysis data.

This module defines the central state container that holds all data
related to a ticker analysis session, following the Single Responsibility
Principle by focusing solely on state representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mcframework.core import SimulationResult


@dataclass
class MarketParameters:
    """
    Estimated market parameters from historical data.
    
    Attributes:
        spot_price: Current stock price (S0)
        drift: Annualized drift (mu)
        volatility: Annualized volatility (sigma)
        daily_return_mean: Mean of daily log returns
        daily_return_std: Standard deviation of daily log returns
    """
    spot_price: float = 0.0
    drift: float = 0.0
    volatility: float = 0.0
    daily_return_mean: float = 0.0
    daily_return_std: float = 0.0

    @classmethod
    def from_dict(cls, params: dict) -> "MarketParameters":
        """Create MarketParameters from estimate_parameters() output."""
        return cls(
            spot_price=params.get("S0", 0.0),
            drift=params.get("mu", 0.0),
            volatility=params.get("sigma", 0.0),
            daily_return_mean=params.get("daily_return_mean", 0.0),
            daily_return_std=params.get("daily_return_std", 0.0),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format for compatibility."""
        return {
            "S0": self.spot_price,
            "mu": self.drift,
            "sigma": self.volatility,
            "daily_return_mean": self.daily_return_mean,
            "daily_return_std": self.daily_return_std,
        }


@dataclass
class OptionPricingResult:
    """
    Results from option pricing simulation.
    
    Attributes:
        price: Estimated option price
        std_error: Standard error of the estimate
        confidence_interval: Tuple of (lower, upper) CI bounds
    """
    price: float = 0.0
    std_error: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_simulation_result(cls, result: "SimulationResult") -> "OptionPricingResult":
        """Create from a SimulationResult object."""
        ci = result.stats.get("ci_mean")
        
        # Handle different CI formats:
        # - tuple/list: (low, high)
        # - dict: {"low": ..., "high": ...}
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            ci_tuple = (float(ci[0]), float(ci[1]))
        elif isinstance(ci, dict) and "low" in ci and "high" in ci:
            ci_tuple = (float(ci["low"]), float(ci["high"]))
        else:
            ci_tuple = (0.0, 0.0)
        
        return cls(
            price=result.mean,
            std_error=result.std / np.sqrt(max(1, result.n_simulations)),
            confidence_interval=ci_tuple,
        )


@dataclass
class GreeksResult:
    """
    Option Greeks from sensitivity analysis.
    
    Attributes:
        delta: Rate of change of option price with respect to underlying
        gamma: Rate of change of delta with respect to underlying
        vega: Sensitivity to volatility (per 1% change)
        theta: Time decay (per day)
        rho: Sensitivity to interest rate (per 1% change)
        price: Option price at current parameters
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    price: float = 0.0

    @classmethod
    def from_dict(cls, greeks: dict) -> "GreeksResult":
        """Create GreeksResult from calculate_greeks() output."""
        return cls(
            delta=greeks.get("delta", 0.0),
            gamma=greeks.get("gamma", 0.0),
            vega=greeks.get("vega", 0.0),
            theta=greeks.get("theta", 0.0),
            rho=greeks.get("rho", 0.0),
            price=greeks.get("price", 0.0),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "price": self.price,
        }


@dataclass
class SimulationConfig:
    """
    Configuration for running a simulation.
    
    Attributes:
        ticker: Stock ticker symbol
        historical_days: Number of historical days to fetch
        forecast_horizon: Forecast horizon in years
        n_simulations: Number of Monte Carlo simulations
        n_paths_viz: Number of paths for visualization
        seed: Random seed for reproducibility
        compute_greeks: Whether to calculate Greeks
        generate_3d_plots: Whether to generate 3D surface plots
        risk_free_rate: Risk-free interest rate
    """
    ticker: str = "AAPL"
    historical_days: int = 252
    forecast_horizon: float = 1.0
    n_simulations: int = 1000
    n_paths_viz: int = 100
    seed: int = 42
    compute_greeks: bool = True
    generate_3d_plots: bool = True
    risk_free_rate: float = 0.05


@dataclass
class TickerAnalysisState:
    """
    Central state container for ticker analysis.
    
    This dataclass holds all data associated with a single ticker analysis
    session. It acts as the single source of truth for the application state,
    enabling clean separation between UI components and business logic.
    
    Attributes:
        config: Current simulation configuration
        prices: Historical closing prices array
        start_date: Start date of historical data
        end_date: End date of historical data
        parameters: Estimated market parameters
        simulated_paths: Monte Carlo simulated price paths
        call_result: Call option pricing result
        put_result: Put option pricing result
        call_greeks: Greeks for call option
        put_greeks: Greeks for put option
        chart_paths: Dictionary mapping chart names to file paths
        output_directory: Directory for saving outputs
        is_running: Whether a simulation is currently running
        last_error: Last error message if any
        last_updated: Timestamp of last state update
    """
    config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Historical data
    prices: np.ndarray | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    
    # Estimated parameters
    parameters: MarketParameters | None = None
    
    # Simulation results
    simulated_paths: np.ndarray | None = None
    call_result: OptionPricingResult | None = None
    put_result: OptionPricingResult | None = None
    call_greeks: GreeksResult | None = None
    put_greeks: GreeksResult | None = None
    
    # Output paths
    chart_paths: dict[str, Path] = field(default_factory=dict)
    output_directory: Path | None = None
    
    # Status
    is_running: bool = False
    last_error: str | None = None
    last_updated: datetime | None = None

    def reset(self) -> None:
        """Reset all results while preserving configuration."""
        self.prices = None
        self.start_date = None
        self.end_date = None
        self.parameters = None
        self.simulated_paths = None
        self.call_result = None
        self.put_result = None
        self.call_greeks = None
        self.put_greeks = None
        self.chart_paths.clear()
        self.last_error = None
        self.last_updated = None

    def has_market_data(self) -> bool:
        """Check if market data has been fetched."""
        return self.prices is not None and len(self.prices) > 0

    def has_simulation_results(self) -> bool:
        """Check if simulation has been run."""
        return self.simulated_paths is not None

    def has_option_pricing(self) -> bool:
        """Check if options have been priced."""
        return self.call_result is not None and self.put_result is not None

    def get_current_price(self) -> float:
        """Get the current (latest) stock price."""
        if self.prices is not None and len(self.prices) > 0:
            return float(self.prices[-1])
        return 0.0

    def update_timestamp(self) -> None:
        """Update the last_updated timestamp to now."""
        self.last_updated = datetime.now()

