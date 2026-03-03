"""
Controller for ticker analysis operations.

This module provides the TickerAnalysisController that orchestrates
data fetching, parameter estimation, and simulation execution.
It follows the Dependency Inversion Principle by depending on
abstractions (signals) rather than concrete UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
    opens: np.ndarray | None
    highs: np.ndarray | None
    lows: np.ndarray | None
    start_date: datetime
    end_date: datetime
    volumes: np.ndarray | None = None
    dates: list[datetime] | None = None
    fast_info: dict[str, Any] = field(default_factory=dict)
    history_metadata: dict[str, Any] = field(default_factory=dict)
    dividends: list[dict[str, Any]] = field(default_factory=list)
    splits: list[dict[str, Any]] = field(default_factory=list)
    recommendations: dict[str, Any] | None = None
    price_targets: dict[str, Any] | None = None


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

    @staticmethod
    def _extract_fast_info(stock: Any) -> dict[str, Any]:
        """Return a serializable snapshot of yfinance fast_info."""
        fast_info: dict[str, Any] = {}
        raw = getattr(stock, "fast_info", None)
        if raw is None:
            return fast_info
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "items"):
            return dict(raw.items())
        if hasattr(raw, "__dict__"):
            return {
                key: value
                for key, value in raw.__dict__.items()
                if not key.startswith("_")
            }
        return fast_info

    @staticmethod
    def _extract_series(
        stock: Any,
        attr: str,
        *,
        value_field: str = "value",
        max_items: int = 5,
    ) -> list[dict[str, Any]]:
        """Convert a pandas Series attribute into a list of dicts."""
        series = getattr(stock, attr, None)
        if series is None:
            return []
        try:
            if series.empty:
                return []
            recent = series.tail(max_items)
            return [
                {
                    "date": idx.to_pydatetime(),
                    value_field: float(value),
                }
                for idx, value in recent.items()
            ]
        except Exception:
            return []

    @staticmethod
    def _safe_get_history_metadata(stock: Any) -> dict[str, Any]:
        """Safely access get_history_metadata(), handling API quirks."""
        try:
            metadata = stock.get_history_metadata()
            return metadata or {}
        except Exception:
            return {}

    @staticmethod
    def _extract_dataframe_row(stock: Any, attr: str) -> dict[str, Any] | None:
        """Serialize the latest row of a yfinance DataFrame attribute."""
        df = getattr(stock, attr, None)
        if df is None:
            return None
        try:
            if df.empty:
                return None
            row = df.iloc[0].to_dict()
            result: dict[str, Any] = {}
            for key, value in row.items():
                result[key] = TickerAnalysisController._normalize_value(value)
            try:
                period = df.index[0]
                result["period"] = str(period)
            except Exception:
                pass
            return result
        except Exception:
            return None

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Convert numpy / pandas scalars to builtin types."""
        if isinstance(value, (np.generic,)):
            return float(value)
        return value

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
            
            prices = hist["Close"].to_numpy()
            opens = hist["Open"].to_numpy() if "Open" in hist.columns else None
            highs = hist["High"].to_numpy() if "High" in hist.columns else None
            lows = hist["Low"].to_numpy() if "Low" in hist.columns else None
            volumes = hist["Volume"].to_numpy() if "Volume" in hist.columns else None
            dates = [idx.to_pydatetime() for idx in hist.index]
            
            # Take last 'days' data points consistently across arrays
            if len(prices) > days:
                slice_idx = -days
                prices = prices[slice_idx:]
                if opens is not None:
                    opens = opens[slice_idx:]
                if highs is not None:
                    highs = highs[slice_idx:]
                if lows is not None:
                    lows = lows[slice_idx:]
                if volumes is not None:
                    volumes = volumes[slice_idx:]
                dates = dates[slice_idx:]
            
            actual_start = dates[0]
            actual_end = dates[-1]
            
            self._log(
                f"✓ Fetched {len(prices)} data points from "
                f"{actual_start.date()} to {actual_end.date()}"
            )
            self._log(f"  Current price: ${prices[-1]:.2f}")
            
            fast_info = self._extract_fast_info(stock)
            dividends = self._extract_series(
                stock, "dividends", value_field="amount", max_items=5
            )
            splits = self._extract_series(
                stock, "splits", value_field="ratio", max_items=5
            )
            history_metadata = self._safe_get_history_metadata(stock)
            recommendations = self._extract_dataframe_row(stock, "recommendations_summary")
            price_targets = self._extract_dataframe_row(stock, "analyst_price_targets")
            
            if history_metadata:
                warning = history_metadata.get("warning")
                if warning:
                    self._log(f"⚠ Yahoo Finance metadata warning: {warning}")
            
            result = FetchResult(
                prices=prices,
                opens=opens,
                highs=highs,
                lows=lows,
                start_date=actual_start,
                end_date=actual_end,
                volumes=volumes,
                dates=dates,
                fast_info=fast_info,
                history_metadata=history_metadata,
                dividends=dividends,
                splits=splits,
                recommendations=recommendations,
                price_targets=price_targets,
            )
            
            # Update state
            self._state.prices = prices
            self._state.open_prices = opens
            self._state.high_prices = highs
            self._state.low_prices = lows
            self._state.volumes = volumes
            self._state.price_dates = dates
            self._state.start_date = actual_start
            self._state.end_date = actual_end
            self._state.dividends = dividends
            self._state.splits = splits
            self._state.fast_info = fast_info
            self._state.history_metadata = history_metadata
            self._state.recommendations = recommendations
            self._state.price_targets = price_targets
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
    ) -> tuple[OptionPricingResult, OptionPricingResult] | None:
        """
        Price call and put options using Monte Carlo.
        
        Uses config.option_maturity for time to expiry and config.strike_pct
        to calculate strike price as a percentage of spot.
        
        Args:
            config: Simulation configuration
            params: Market parameters
            
        Returns:
            Tuple of (call_result, put_result) if successful, None otherwise
        """
        try:
            # Calculate strike from percentage of spot price
            strike = params.spot_price * (config.strike_pct / 100.0)
            option_maturity = config.option_maturity
            
            # Determine moneyness description
            if config.strike_pct < 98:
                moneyness = "ITM Call"
            elif config.strike_pct > 102:
                moneyness = "OTM Call"
            else:
                moneyness = "ATM"
            
            self._log(f"Pricing {moneyness} options (K=${strike:.2f}, T={option_maturity:.2f}y)...")
            
            self._option_sim = BlackScholesSimulation(
                name=f"{config.ticker} Option Pricing"
            )
            self._option_sim.set_seed(config.seed)
            
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
            
            days_to_expiry = int(option_maturity * 252)
            self._log(f"\n{moneyness} Options (K=${strike:.2f}, {days_to_expiry}d to expiry):")
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
    ) -> tuple[GreeksResult, GreeksResult] | None:
        """
        Calculate option Greeks using finite differences.
        
        Uses config.option_maturity for time to expiry and config.strike_pct
        to calculate strike price as a percentage of spot.
        
        Args:
            config: Simulation configuration
            params: Market parameters
            
        Returns:
            Tuple of (call_greeks, put_greeks) if successful, None otherwise
        """
        if not config.compute_greeks:
            return None

        try:
            # Calculate strike from percentage of spot price
            strike = params.spot_price * (config.strike_pct / 100.0)
            option_maturity = config.option_maturity
            
            self._log("Calculating Greeks...")
            
            if self._option_sim is None:
                self._option_sim = BlackScholesSimulation(
                    name=f"{config.ticker} Greeks"
                )
            self._option_sim.set_seed(config.seed)
            
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
        
        d1 = (
            np.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        if option_type == "call":
            price = (
                spot * norm.cdf(d1)
                - strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
            )
        else:
            price = (
                strike * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(-d2)
                - spot * norm.cdf(-d1)
            )
        
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
        d1 = (
            np.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * sqrt_t)
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

