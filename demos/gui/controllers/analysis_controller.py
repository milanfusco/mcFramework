"""
Controller for ticker analysis operations.

This module provides the TickerAnalysisController that orchestrates
data fetching, parameter estimation, and simulation execution.
It follows the Dependency Inversion Principle by depending on
abstractions (signals) rather than concrete UI components.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from mcframework.sims import BlackScholesPathSimulation, BlackScholesSimulation

from ..models.state import (
    GreeksResult,
    MarketParameters,
    OptionPricingResult,
    SimulationConfig,
    TickerAnalysisState,
)


# Risk-free rate constant (approximate US Treasury rate)
DEFAULT_RISK_FREE_RATE = 0.05


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    prices: np.ndarray
    start_date: datetime
    end_date: datetime


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""
    paths: np.ndarray
    call_pricing: OptionPricingResult
    put_pricing: OptionPricingResult
    call_greeks: GreeksResult | None
    put_greeks: GreeksResult | None
    chart_paths: dict[str, Path]


class TickerAnalysisController(QObject):
    """
    Controller for coordinating ticker analysis operations.
    
    This class wraps the procedural functions from demoTickerBlackScholes.py
    and provides a clean interface with Qt signals for UI updates. It follows
    the Single Responsibility Principle by focusing on orchestration logic.
    
    Signals:
        log_message: Emitted when a log message should be displayed
        progress_updated: Emitted with (current, total) during simulation
        data_fetched: Emitted when market data is successfully fetched
        parameters_estimated: Emitted with estimated parameters
        simulation_started: Emitted when simulation begins
        simulation_progress: Emitted with (phase, current, total)
        simulation_complete: Emitted when all simulations finish
        charts_generated: Emitted with chart paths dictionary
        error_occurred: Emitted when an error occurs
    """
    
    # Logging signals
    log_message = Signal(str)
    
    # Progress signals
    progress_updated = Signal(int, int)  # current, total
    simulation_progress = Signal(str, int, int)  # phase, current, total
    
    # Data signals
    data_fetched = Signal(object)  # FetchResult
    parameters_estimated = Signal(object)  # MarketParameters
    
    # Simulation signals
    simulation_started = Signal()
    simulation_complete = Signal(object)  # SimulationResult
    charts_generated = Signal(dict)  # chart_paths
    
    # Error handling
    error_occurred = Signal(str)

    def __init__(self, state: TickerAnalysisState, parent: QObject | None = None):
        """
        Initialize the controller.
        
        Args:
            state: Shared application state
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._state = state
        self._option_sim: BlackScholesSimulation | None = None
        self._path_sim: BlackScholesPathSimulation | None = None

    @property
    def state(self) -> TickerAnalysisState:
        """Access the current application state."""
        return self._state

    def _log(self, message: str) -> None:
        """Emit a log message."""
        self.log_message.emit(message)

    def _create_progress_callback(self, phase: str) -> Callable[[int, int], None]:
        """Create a progress callback for a simulation phase."""
        def callback(current: int, total: int) -> None:
            self.simulation_progress.emit(phase, current, total)
            self.progress_updated.emit(current, total)
        return callback

    def fetch_ticker_data(self, ticker: str, days: int = 252) -> FetchResult | None:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of trading days to fetch
            
        Returns:
            FetchResult if successful, None otherwise
        """
        # Import here to avoid loading yfinance at module level
        try:
            import yfinance as yf
        except ImportError:
            self.error_occurred.emit(
                "yfinance package is required. Install with: pip install yfinance"
            )
            return None

        self._log(f"Fetching data for {ticker}...")
        
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                self.error_occurred.emit(f"No data found for ticker '{ticker}'")
                return None
            
            prices = hist['Close'].values
            
            # Take last 'days' data points
            if len(prices) > days:
                prices = prices[-days:]
            
            actual_start = hist.index[0].to_pydatetime()
            actual_end = hist.index[-1].to_pydatetime()
            
            self._log(
                f"✓ Fetched {len(prices)} data points from "
                f"{actual_start.date()} to {actual_end.date()}"
            )
            self._log(f"  Current price: ${prices[-1]:.2f}")
            
            result = FetchResult(
                prices=prices,
                start_date=actual_start,
                end_date=actual_end,
            )
            
            # Update state
            self._state.prices = prices
            self._state.start_date = actual_start
            self._state.end_date = actual_end
            self._state.update_timestamp()
            
            self.data_fetched.emit(result)
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to fetch data for '{ticker}': {str(e)}")
            return None

    def estimate_parameters(self, prices: np.ndarray) -> MarketParameters | None:
        """
        Estimate Black-Scholes parameters from historical prices.
        
        Args:
            prices: Array of historical closing prices
            
        Returns:
            MarketParameters if successful, None otherwise
        """
        if prices is None or len(prices) < 2:
            self.error_occurred.emit("Insufficient price data for parameter estimation")
            return None

        try:
            # Calculate daily log returns
            returns = np.diff(np.log(prices))
            
            # Estimate parameters
            daily_mean = float(np.mean(returns))
            daily_std = float(np.std(returns, ddof=1))
            
            # Annualize (252 trading days per year)
            mu = daily_mean * 252
            sigma = daily_std * np.sqrt(252)
            spot = float(prices[-1])
            
            params = MarketParameters(
                spot_price=spot,
                drift=mu,
                volatility=sigma,
                daily_return_mean=daily_mean,
                daily_return_std=daily_std,
            )
            
            self._log("\n" + "=" * 50)
            self._log("ESTIMATED PARAMETERS FROM HISTORICAL DATA")
            self._log("=" * 50)
            self._log(f"Current Price (S₀):     ${spot:.2f}")
            self._log(f"Drift (μ):              {mu:.4f} ({mu*100:.2f}% annual)")
            self._log(f"Volatility (σ):         {sigma:.4f} ({sigma*100:.2f}% annual)")
            self._log("=" * 50 + "\n")
            
            # Update state
            self._state.parameters = params
            self._state.update_timestamp()
            
            self.parameters_estimated.emit(params)
            return params
            
        except Exception as e:
            self.error_occurred.emit(f"Parameter estimation failed: {str(e)}")
            return None

    def run_path_simulation(
        self,
        config: SimulationConfig,
        params: MarketParameters,
    ) -> np.ndarray | None:
        """
        Run Monte Carlo path simulation.
        
        Args:
            config: Simulation configuration
            params: Market parameters
            
        Returns:
            Simulated paths array if successful, None otherwise
        """
        try:
            self._log("Running path simulations...")
            
            self._path_sim = BlackScholesPathSimulation(
                name=f"{config.ticker} Path Simulation"
            )
            self._path_sim.set_seed(config.seed)
            
            n_steps = int(config.forecast_horizon * 252)
            
            paths = self._path_sim.simulate_paths(
                n_paths=config.n_paths_viz,
                S0=params.spot_price,
                r=params.drift,  # Use estimated drift
                sigma=params.volatility,
                T=config.forecast_horizon,
                n_steps=n_steps,
            )
            
            self._log(f"✓ Generated {config.n_paths_viz} simulated paths")
            
            # Update state
            self._state.simulated_paths = paths
            self._state.update_timestamp()
            
            return paths
            
        except Exception as e:
            self.error_occurred.emit(f"Path simulation failed: {str(e)}")
            return None

    def price_options(
        self,
        config: SimulationConfig,
        params: MarketParameters,
        option_maturity: float = 0.25,
    ) -> tuple[OptionPricingResult, OptionPricingResult] | None:
        """
        Price call and put options using Monte Carlo.
        
        Args:
            config: Simulation configuration
            params: Market parameters
            option_maturity: Time to maturity in years (default 3 months)
            
        Returns:
            Tuple of (call_result, put_result) if successful, None otherwise
        """
        try:
            self._log("Pricing options...")
            
            self._option_sim = BlackScholesSimulation(
                name=f"{config.ticker} Option Pricing"
            )
            self._option_sim.set_seed(config.seed)
            
            strike = params.spot_price  # ATM option
            
            # Price call option
            call_mc_result = self._option_sim.run(
                config.n_simulations,
                S0=params.spot_price,
                K=strike,
                T=option_maturity,
                r=config.risk_free_rate,
                sigma=params.volatility,
                option_type="call",
                exercise_type="european",
                progress_callback=self._create_progress_callback("Pricing call"),
            )
            
            # Price put option
            put_mc_result = self._option_sim.run(
                config.n_simulations,
                S0=params.spot_price,
                K=strike,
                T=option_maturity,
                r=config.risk_free_rate,
                sigma=params.volatility,
                option_type="put",
                exercise_type="european",
                progress_callback=self._create_progress_callback("Pricing put"),
            )
            
            call_result = OptionPricingResult.from_simulation_result(call_mc_result)
            put_result = OptionPricingResult.from_simulation_result(put_mc_result)
            
            self._log(f"\n3-Month ATM Options (K=${strike:.2f}):")
            self._log(f"  Call Price: ${call_result.price:.2f} ± ${call_result.std_error:.2f}")
            self._log(f"  Put Price:  ${put_result.price:.2f} ± ${put_result.std_error:.2f}")
            
            # Update state
            self._state.call_result = call_result
            self._state.put_result = put_result
            self._state.update_timestamp()
            
            return call_result, put_result
            
        except Exception as e:
            self.error_occurred.emit(f"Option pricing failed: {str(e)}")
            return None

    def calculate_greeks(
        self,
        config: SimulationConfig,
        params: MarketParameters,
        option_maturity: float = 0.25,
    ) -> tuple[GreeksResult, GreeksResult] | None:
        """
        Calculate option Greeks using finite differences.
        
        Args:
            config: Simulation configuration
            params: Market parameters
            option_maturity: Time to maturity in years
            
        Returns:
            Tuple of (call_greeks, put_greeks) if successful, None otherwise
        """
        if not config.compute_greeks:
            return None

        try:
            self._log("Calculating Greeks...")
            
            if self._option_sim is None:
                self._option_sim = BlackScholesSimulation(
                    name=f"{config.ticker} Greeks"
                )
            self._option_sim.set_seed(config.seed)
            
            strike = params.spot_price
            
            # Calculate call Greeks
            call_greeks_dict = self._option_sim.calculate_greeks(
                n_simulations=config.n_simulations,
                S0=params.spot_price,
                K=strike,
                T=option_maturity,
                r=config.risk_free_rate,
                sigma=params.volatility,
                option_type="call",
                exercise_type="european",
                parallel=True,
            )
            
            # Calculate put Greeks
            put_greeks_dict = self._option_sim.calculate_greeks(
                n_simulations=config.n_simulations,
                S0=params.spot_price,
                K=strike,
                T=option_maturity,
                r=config.risk_free_rate,
                sigma=params.volatility,
                option_type="put",
                exercise_type="european",
                parallel=True,
            )
            
            call_greeks = GreeksResult.from_dict(call_greeks_dict)
            put_greeks = GreeksResult.from_dict(put_greeks_dict)
            
            self._log(f"  Call Delta: {call_greeks.delta:.4f}")
            self._log(f"  Call Gamma: {call_greeks.gamma:.6f}")
            self._log(f"  Call Vega:  {call_greeks.vega:.4f}")
            self._log(f"  Call Theta: {call_greeks.theta:.4f}")
            self._log(f"  Call Rho:   {call_greeks.rho:.4f}")
            
            # Update state
            self._state.call_greeks = call_greeks
            self._state.put_greeks = put_greeks
            self._state.update_timestamp()
            
            return call_greeks, put_greeks
            
        except Exception as e:
            self.error_occurred.emit(f"Greeks calculation failed: {str(e)}")
            return None

    def get_output_directory(self, ticker: str) -> Path:
        """
        Get or create the output directory for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Path to the output directory
        """
        output_dir = Path(f"img/{ticker.upper()}_BlackScholes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._state.output_directory = output_dir
        return output_dir

    @staticmethod
    def calculate_bs_price(
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call",
    ) -> float:
        """
        Calculate Black-Scholes option price analytically.
        
        This is a static method for the Option Calculator dialog.
        
        Args:
            spot: Current stock price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility (annualized)
            option_type: "call" or "put"
            
        Returns:
            Theoretical option price
        """
        from scipy.stats import norm
        
        if time_to_maturity <= 0:
            # At expiry
            if option_type == "call":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)
        
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
        return float(price)

    @staticmethod
    def calculate_bs_greeks(
        spot: float,
        strike: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = "call",
    ) -> GreeksResult:
        """
        Calculate Black-Scholes Greeks analytically.
        
        Args:
            spot: Current stock price
            strike: Strike price
            time_to_maturity: Time to maturity in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility (annualized)
            option_type: "call" or "put"
            
        Returns:
            GreeksResult with all Greeks
        """
        from scipy.stats import norm
        
        if time_to_maturity <= 0:
            # At expiry, Greeks are degenerate
            return GreeksResult()
        
        sqrt_t = np.sqrt(time_to_maturity)
        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        # Common terms
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        pdf_d1 = norm.pdf(d1)
        discount = np.exp(-risk_free_rate * time_to_maturity)
        
        # Delta
        if option_type == "call":
            delta = nd1
        else:
            delta = nd1 - 1.0
        
        # Gamma (same for call and put)
        gamma = pdf_d1 / (spot * volatility * sqrt_t)
        
        # Vega (same for call and put, per 1% move)
        vega = spot * pdf_d1 * sqrt_t * 0.01
        
        # Theta (per day)
        theta_term1 = -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
        if option_type == "call":
            theta_term2 = -risk_free_rate * strike * discount * nd2
            theta = (theta_term1 + theta_term2) / 365
        else:
            theta_term2 = risk_free_rate * strike * discount * norm.cdf(-d2)
            theta = (theta_term1 + theta_term2) / 365
        
        # Rho (per 1% move)
        if option_type == "call":
            rho = strike * time_to_maturity * discount * nd2 * 0.01
        else:
            rho = -strike * time_to_maturity * discount * norm.cdf(-d2) * 0.01
        
        # Price
        price = TickerAnalysisController.calculate_bs_price(
            spot, strike, time_to_maturity, risk_free_rate, volatility, option_type
        )
        
        return GreeksResult(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            price=float(price),
        )

