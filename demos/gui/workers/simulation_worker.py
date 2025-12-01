"""
Background workers for non-blocking operations.

This module provides QThread-based workers for data fetching and
Monte Carlo simulations, ensuring the UI remains responsive during
long-running operations.

Note: Chart generation has been moved to interactive widgets that
render directly from simulation data, eliminating the need for
file-based chart generation in the worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

if TYPE_CHECKING:
    from ..controllers.analysis_controller import TickerAnalysisController
    from ..models.state import SimulationConfig, TickerAnalysisState


class DataFetchWorker(QThread):
    """
    Worker thread for fetching market data from Yahoo Finance.
    
    This worker runs the data fetch operation in a separate thread
    to prevent UI freezing during network operations.
    
    Signals:
        started_fetch: Emitted when fetch begins
        progress: Emitted with status message during fetch
        finished_fetch: Emitted when fetch completes successfully
        error: Emitted when an error occurs
    """
    
    started_fetch = Signal()
    progress = Signal(str)
    finished_fetch = Signal(object)  # FetchResult
    error = Signal(str)

    def __init__(
        self,
        controller: "TickerAnalysisController",
        ticker: str,
        days: int = 252,
        parent=None,
    ):
        """
        Initialize the data fetch worker.
        
        Args:
            controller: The analysis controller to use
            ticker: Stock ticker symbol to fetch
            days: Number of historical days
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._controller = controller
        self._ticker = ticker
        self._days = days

    def run(self) -> None:
        """Execute the data fetch operation."""
        self.started_fetch.emit()
        self.progress.emit(f"Connecting to Yahoo Finance for {self._ticker}...")
        
        try:
            result = self._controller.fetch_ticker_data(self._ticker, self._days)
            
            if result is not None:
                self.progress.emit("Estimating market parameters...")
                self._controller.estimate_parameters(result.prices)
                self.finished_fetch.emit(result)
            else:
                self.error.emit(f"Failed to fetch data for {self._ticker}")
                
        except Exception as e:
            self.error.emit(str(e))


class SimulationWorker(QThread):
    """
    Worker thread for running Monte Carlo simulations.
    
    This worker handles path simulation, option pricing, and Greeks
    calculation in a background thread with progress reporting.
    
    Charts are rendered interactively by UI widgets from the simulation
    data, so this worker only computes numerical results.
    
    Signals:
        started: Emitted when simulation begins
        phase_started: Emitted with phase name when a new phase begins
        progress: Emitted with (phase, current, total) during simulation
        finished: Emitted with results when simulation completes
        error: Emitted when an error occurs
    """
    
    started = Signal()
    phase_started = Signal(str)
    progress = Signal(str, int, int)  # phase, current, total
    finished = Signal(object)  # SimulationResult
    error = Signal(str)

    def __init__(
        self,
        controller: "TickerAnalysisController",
        config: "SimulationConfig",
        state: "TickerAnalysisState",
        parent=None,
    ):
        """
        Initialize the simulation worker.
        
        Args:
            controller: The analysis controller to use
            config: Simulation configuration
            state: Current application state with market data
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._controller = controller
        self._config = config
        self._state = state
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the simulation."""
        self._is_cancelled = True

    def run(self) -> None:
        """Execute the simulation pipeline (path simulation, pricing, Greeks)."""
        self.started.emit()
        
        try:
            if not self._state.has_market_data():
                self.error.emit("No market data available. Fetch data first.")
                return
                
            params = self._state.parameters
            if params is None:
                self.error.emit("Market parameters not estimated.")
                return

            # Phase 1: Path simulation
            self.phase_started.emit("Path Simulation")
            self.progress.emit("Path Simulation", 0, 3)
            
            paths = self._controller.run_path_simulation(self._config, params)
            
            if self._is_cancelled:
                return
            
            if paths is None:
                self.error.emit("Path simulation failed.")
                return
            
            self.progress.emit("Path Simulation", 1, 3)

            # Phase 2: Option pricing
            self.phase_started.emit("Option Pricing")
            pricing_result = self._controller.price_options(
                self._config, params, option_maturity=0.25
            )
            
            if self._is_cancelled:
                return
            
            if pricing_result is None:
                self.error.emit("Option pricing failed.")
                return
            
            call_result, put_result = pricing_result
            self.progress.emit("Option Pricing", 2, 3)

            # Phase 3: Greeks calculation (optional)
            call_greeks = None
            put_greeks = None
            
            if self._config.compute_greeks:
                self.phase_started.emit("Greeks Calculation")
                greeks_result = self._controller.calculate_greeks(
                    self._config, params, option_maturity=0.25
                )
                
                if self._is_cancelled:
                    return
                
                if greeks_result is not None:
                    call_greeks, put_greeks = greeks_result
            
            self.progress.emit("Complete", 3, 3)

            # Build final result (no chart_paths - charts rendered interactively)
            from ..controllers.analysis_controller import (
                SimulationResult as ControllerSimResult,
            )
            
            result = ControllerSimResult(
                paths=paths,
                call_pricing=call_result,
                put_pricing=put_result,
                call_greeks=call_greeks,
                put_greeks=put_greeks,
                chart_paths={},  # Empty - charts rendered by widgets
            )
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Simulation error: {str(e)}")


class ChartExportWorker(QThread):
    """
    Worker for exporting charts to PNG files.
    
    Used when the user requests to save charts or generate reports.
    Charts are exported from the interactive widgets' figures.
    
    Signals:
        export_complete: Emitted with output path when export finishes
        error: Emitted when an error occurs
    """
    
    export_complete = Signal(object)  # Path
    error = Signal(str)

    def __init__(
        self,
        export_func,
        output_path: Path,
        dpi: int = 150,
        parent=None,
    ):
        """
        Initialize the chart export worker.
        
        Args:
            export_func: Callable that exports the chart (widget.export_to_png)
            output_path: Path to save the PNG file
            dpi: Resolution for export
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self._export_func = export_func
        self._output_path = output_path
        self._dpi = dpi

    def run(self) -> None:
        """Execute the chart export."""
        try:
            self._export_func(self._output_path, dpi=self._dpi)
            self.export_complete.emit(self._output_path)
        except Exception as e:
            self.error.emit(f"Chart export error: {e}")
