#!/usr/bin/env python3
"""
Quant Black-Scholes Simulator - Main Entry Point

A Bloomberg-lite PySide6 application for Monte Carlo simulations
on stock tickers using Black-Scholes option pricing.

Usage:
    python quant_black_scholes.py

Features:
    - Fetch live ticker data from Yahoo Finance
    - Run Monte Carlo path simulations
    - Price European options with Greeks
    - Interactive what-if analysis
    - 3D option price surfaces
    - Export reports with charts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QWidget,
)

from demos.gui.controllers.analysis_controller import TickerAnalysisController
from demos.gui.models.state import SimulationConfig, TickerAnalysisState
from demos.gui.widgets.log_console import LogConsoleWidget
from demos.gui.widgets.market_data_tab import MarketDataTab
from demos.gui.widgets.monte_carlo_tab import MonteCarloTab
from demos.gui.widgets.option_calculator import OptionCalculatorDialog
from demos.gui.widgets.options_greeks_tab import OptionsGreeksTab
from demos.gui.widgets.sidebar import SidebarWidget
from demos.gui.widgets.surfaces_tab import SurfacesTab
from demos.gui.workers.simulation_worker import DataFetchWorker, SimulationWorker

if TYPE_CHECKING:
    from demos.gui.controllers.analysis_controller import FetchResult, SimulationResult


class QuantBlackScholesWindow(QMainWindow):
    """
    Main application window for the Quant Black-Scholes Simulator.
    
    Assembles all UI components and coordinates between the controller,
    workers, and display widgets. Follows the Mediator pattern to
    manage communication between components.
    """

    WINDOW_TITLE = "Quant Black-Scholes Simulator"
    WINDOW_MIN_WIDTH = 1400
    WINDOW_MIN_HEIGHT = 800

    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Initialize state and controller
        self._state = TickerAnalysisState()
        self._controller = TickerAnalysisController(self._state, self)
        
        # Workers
        self._fetch_worker: DataFetchWorker | None = None
        self._sim_worker: SimulationWorker | None = None
        
        # Calculator dialog (lazy init)
        self._calculator_dialog: OptionCalculatorDialog | None = None
        
        # Set up UI
        self._setup_window()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()
        self._load_stylesheet()
        self._connect_signals()
        
        # Initial state
        self._update_ui_running_state(False)

    def _setup_window(self) -> None:
        """Configure main window properties."""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(self.WINDOW_MIN_WIDTH, self.WINDOW_MIN_HEIGHT)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.WINDOW_MIN_WIDTH) // 2
        y = (screen.height() - self.WINDOW_MIN_HEIGHT) // 2
        self.move(x, y)

    def _setup_toolbar(self) -> None:
        """Create and populate the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)
        
        # Run action
        self._run_action = QAction("▶ Run Analysis", self)
        self._run_action.setToolTip("Fetch data and run full simulation")
        self._run_action.triggered.connect(self._on_run_clicked)
        toolbar.addAction(self._run_action)
        
        # Stop action
        self._stop_action = QAction("⏹ Stop", self)
        self._stop_action.setToolTip("Stop current simulation")
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._on_stop_clicked)
        toolbar.addAction(self._stop_action)
        
        toolbar.addSeparator()
        
        # Calculator action
        self._calc_action = QAction("🧮 Calculator", self)
        self._calc_action.setToolTip("Open Black-Scholes option calculator")
        self._calc_action.triggered.connect(self._on_calculator_clicked)
        toolbar.addAction(self._calc_action)
        
        toolbar.addSeparator()
        
        # Export action
        self._export_action = QAction("📄 Export Report", self)
        self._export_action.setToolTip("Export analysis report as HTML")
        self._export_action.triggered.connect(self._on_export_clicked)
        toolbar.addAction(self._export_action)
        
        # Open folder action
        self._folder_action = QAction("📁 Open Output", self)
        self._folder_action.setToolTip("Open output folder in file explorer")
        self._folder_action.triggered.connect(self._on_open_folder_clicked)
        toolbar.addAction(self._folder_action)

    def _setup_central_widget(self) -> None:
        """Set up the main layout with sidebar, tabs, and log panel."""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Sidebar
        self._sidebar = SidebarWidget()
        layout.addWidget(self._sidebar)
        
        # Main splitter for center + right panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Tab widget (center)
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        
        # Create tabs
        self._market_tab = MarketDataTab()
        self._mc_tab = MonteCarloTab()
        self._options_tab = OptionsGreeksTab()
        self._surfaces_tab = SurfacesTab()
        
        self._tabs.addTab(self._market_tab, "📊 Market Data")
        self._tabs.addTab(self._mc_tab, "🎲 Monte Carlo")
        self._tabs.addTab(self._options_tab, "📈 Options & Greeks")
        self._tabs.addTab(self._surfaces_tab, "🌐 3D Surfaces")
        
        splitter.addWidget(self._tabs)
        
        # Log console (right)
        self._log_console = LogConsoleWidget()
        splitter.addWidget(self._log_console)
        
        # Set splitter sizes
        splitter.setSizes([900, 320])
        
        layout.addWidget(splitter, 1)

    def _setup_status_bar(self) -> None:
        """Set up the status bar."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        
        self._status_bar.showMessage("Ready")

    def _load_stylesheet(self) -> None:
        """Load the dark theme stylesheet."""
        qss_path = Path(__file__).parent / "assets" / "dark_theme.qss"
        
        if qss_path.exists():
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            # Fallback minimal dark style
            self.setStyleSheet("""
                QWidget { background-color: #1a1a2e; color: #e0e0e0; }
            """)

    def _connect_signals(self) -> None:
        """Connect all signals between components."""
        # Sidebar signals
        self._sidebar.run_requested.connect(self._on_run_clicked)
        self._sidebar.fetch_requested.connect(self._on_fetch_only_clicked)
        self._sidebar.config_changed.connect(self._on_config_changed)
        
        # Controller signals
        self._controller.log_message.connect(self._log_console.on_log_message)
        self._controller.error_occurred.connect(self._on_error)
        
        # Options tab sensitivity
        self._options_tab.sensitivity_requested.connect(self._on_sensitivity_requested)
        
        # Interactive chart exports
        self._mc_tab.chart_export_requested.connect(self._on_chart_export_requested)
        self._surfaces_tab.surface_export_requested.connect(self._on_surface_export_requested)

    def _update_ui_running_state(self, running: bool) -> None:
        """Update UI state for running/stopped status."""
        self._state.is_running = running
        
        self._run_action.setEnabled(not running)
        self._stop_action.setEnabled(running)
        self._sidebar.set_running(running)
        self._log_console.set_running(running)
        
        if running:
            self._status_bar.showMessage("Running simulation...")
        else:
            self._status_bar.showMessage("Ready")

    @Slot()
    def _on_run_clicked(self) -> None:
        """Handle Run Analysis action."""
        config = self._sidebar.get_config()
        self._state.config = config
        
        if self._sidebar.is_auto_fetch_enabled() or not self._state.has_market_data():
            # Fetch data first, then run simulation
            self._start_fetch(config.ticker, config.historical_days, run_after=True)
        else:
            # Run simulation with existing data
            self._start_simulation(config)

    @Slot()
    def _on_fetch_only_clicked(self) -> None:
        """Handle Fetch Data Only action."""
        ticker = self._sidebar.get_ticker()
        days = self._sidebar.get_historical_days()
        self._start_fetch(ticker, days, run_after=False)

    @Slot()
    def _on_stop_clicked(self) -> None:
        """Handle Stop action."""
        if self._sim_worker and self._sim_worker.isRunning():
            self._sim_worker.cancel()
            self._log_console.log_warning("Simulation cancelled by user")
        
        self._update_ui_running_state(False)

    @Slot()
    def _on_calculator_clicked(self) -> None:
        """Open the Option Calculator dialog."""
        if self._calculator_dialog is None:
            self._calculator_dialog = OptionCalculatorDialog(self)
        
        # Pre-fill with current parameters if available
        if self._state.parameters:
            self._calculator_dialog.set_parameters(
                spot=self._state.parameters.spot_price,
                vol=self._state.parameters.volatility,
                rate=self._state.config.risk_free_rate,
            )
        
        self._calculator_dialog.show()
        self._calculator_dialog.raise_()

    @Slot()
    def _on_export_clicked(self) -> None:
        """Handle Export Report action."""
        if not self._state.has_simulation_results():
            QMessageBox.warning(
                self,
                "No Results",
                "Run a simulation first to generate a report.",
            )
            return
        
        # Generate HTML report
        self._export_html_report()

    @Slot()
    def _on_open_folder_clicked(self) -> None:
        """Open the output folder in file explorer."""
        import os
        import subprocess
        
        output_dir = self._state.output_directory
        if output_dir is None:
            output_dir = self._controller.get_output_directory(
                self._state.config.ticker
            )
        
        # Cross-platform folder open
        if sys.platform == "win32":
            os.startfile(str(output_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(output_dir)])
        else:
            subprocess.run(["xdg-open", str(output_dir)])

    @Slot(object)
    def _on_config_changed(self, config: SimulationConfig) -> None:
        """Handle configuration changes from sidebar."""
        self._state.config = config

    @Slot(str)
    def _on_error(self, message: str) -> None:
        """Handle error from controller or workers."""
        self._log_console.log_error(message)
        self._status_bar.showMessage(f"Error: {message}")

    @Slot(float, float)
    def _on_sensitivity_requested(self, s0: float, sigma: float) -> None:
        """Handle what-if analysis parameter change."""
        if not self._state.has_market_data():
            return
        
        strike = self._state.parameters.spot_price  # ATM strike
        T = 0.25  # 3-month option
        r = self._state.config.risk_free_rate
        
        # Calculate adjusted prices using analytical formulas
        call_price = TickerAnalysisController.calculate_bs_price(
            spot=s0, strike=strike, time_to_maturity=T,
            risk_free_rate=r, volatility=sigma, option_type="call",
        )
        
        put_price = TickerAnalysisController.calculate_bs_price(
            spot=s0, strike=strike, time_to_maturity=T,
            risk_free_rate=r, volatility=sigma, option_type="put",
        )
        
        self._options_tab.update_adjusted_prices(call_price, put_price)
        
        # Calculate adjusted Greeks using analytical formulas
        call_greeks = self._calculate_analytical_greeks(
            s0, strike, T, r, sigma, option_type="call"
        )
        put_greeks = self._calculate_analytical_greeks(
            s0, strike, T, r, sigma, option_type="put"
        )
        
        self._options_tab.update_adjusted_greeks(call_greeks, put_greeks)

    def _calculate_analytical_greeks(
        self,
        spot: float,
        strike: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> dict[str, float]:
        """
        Calculate analytical Black-Scholes Greeks.
        
        Args:
            spot: Current stock price
            strike: Strike price
            T: Time to maturity in years
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            
        Returns:
            Dictionary with delta, gamma, vega, theta, rho
        """
        from math import exp, log, sqrt

        from scipy.stats import norm
        
        # Handle edge cases
        if T <= 0 or sigma <= 0:
            return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
        
        sqrt_T = sqrt(T)
        d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Standard normal PDF and CDF
        n_d1 = norm.pdf(d1)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        
        # Gamma (same for call and put)
        gamma = n_d1 / (spot * sigma * sqrt_T)
        
        # Vega (same for call and put, per 1% vol change)
        vega = spot * n_d1 * sqrt_T * 0.01
        
        if option_type == "call":
            delta = N_d1
            theta = (
                -spot * n_d1 * sigma / (2 * sqrt_T) - r * strike * exp(-r * T) * N_d2
            ) / 365  # Per day
            rho = strike * T * exp(-r * T) * N_d2 * 0.01  # Per 1%
        else:
            delta = N_d1 - 1
            theta = (
                -spot * n_d1 * sigma / (2 * sqrt_T)
                + r * strike * exp(-r * T) * norm.cdf(-d2)
            ) / 365  # Per day
            rho = -strike * T * exp(-r * T) * norm.cdf(-d2) * 0.01  # Per 1%
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    @Slot(str, object)
    def _on_chart_export_requested(self, chart_name: str, export_func) -> None:
        """Handle interactive chart export request."""
        ticker = self._state.config.ticker.upper()
        default_name = f"{ticker}_{chart_name}.png"
        
        dest, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {chart_name.replace('_', ' ').title()} Chart",
            default_name,
            "PNG Images (*.png)",
        )
        
        if dest:
            try:
                export_func(Path(dest))
                self._log_console.log_success(f"Chart saved to {dest}")
            except Exception as e:
                self._on_error(f"Failed to export chart: {e}")

    @Slot(str, object)
    def _on_surface_export_requested(self, surface_type: str, export_func) -> None:
        """Handle 3D surface export request."""
        ticker = self._state.config.ticker.upper()
        default_name = f"{ticker}_{surface_type}_surface_3d.png"
        
        dest, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {surface_type.title()} Option Surface",
            default_name,
            "PNG Images (*.png)",
        )
        
        if dest:
            try:
                export_func(Path(dest))
                self._log_console.log_success(f"Surface saved to {dest}")
            except Exception as e:
                self._on_error(f"Failed to export surface: {e}")

    def _start_fetch(
        self,
        ticker: str,
        days: int,
        run_after: bool = False,
    ) -> None:
        """
        Start data fetch in background worker.
        
        Args:
            ticker: Ticker symbol
            days: Historical days
            run_after: Whether to run simulation after fetch
        """
        self._update_ui_running_state(True)
        self._log_console.set_phase("Fetching Data")
        
        # Reset state for new ticker
        if self._state.config.ticker != ticker:
            self._state.reset()
        
        self._fetch_worker = DataFetchWorker(
            self._controller, ticker, days, self
        )
        
        self._fetch_worker.progress.connect(self._log_console.on_log_message)
        self._fetch_worker.error.connect(self._on_fetch_error)
        
        if run_after:
            self._fetch_worker.finished_fetch.connect(self._on_fetch_complete_run)
        else:
            self._fetch_worker.finished_fetch.connect(self._on_fetch_complete)
        
        self._fetch_worker.start()

    @Slot(object)
    def _on_fetch_complete(self, result: "FetchResult") -> None:
        """Handle fetch completion without running simulation."""
        self._update_ui_running_state(False)
        self._log_console.set_complete(True)
        
        # Update UI
        self._market_tab.update_from_state(self._state)
        self._tabs.setCurrentWidget(self._market_tab)
        
        self._status_bar.showMessage(
            f"Fetched {len(result.prices)} data points for {self._state.config.ticker}"
        )

    @Slot(object)
    def _on_fetch_complete_run(self, result: "FetchResult") -> None:
        """Handle fetch completion and start simulation."""
        self._market_tab.update_from_state(self._state)
        self._start_simulation(self._state.config)

    @Slot(str)
    def _on_fetch_error(self, message: str) -> None:
        """Handle fetch error."""
        self._update_ui_running_state(False)
        self._log_console.set_complete(False)
        self._on_error(message)

    def _start_simulation(self, config: SimulationConfig) -> None:
        """
        Start Monte Carlo simulation in background worker.
        
        Args:
            config: Simulation configuration
        """
        self._update_ui_running_state(True)
        
        self._sim_worker = SimulationWorker(
            self._controller, config, self._state, self
        )
        
        self._sim_worker.phase_started.connect(self._log_console.set_phase)
        self._sim_worker.progress.connect(self._log_console.on_simulation_progress)
        self._sim_worker.finished.connect(self._on_simulation_complete)
        self._sim_worker.error.connect(self._on_simulation_error)
        
        self._sim_worker.start()

    @Slot(object)
    def _on_simulation_complete(self, result: "SimulationResult") -> None:
        """Handle simulation completion."""
        self._update_ui_running_state(False)
        self._log_console.set_complete(True)
        
        # Update all tabs
        self._market_tab.update_from_state(self._state)
        self._mc_tab.update_from_state(self._state)
        self._options_tab.update_from_state(self._state)
        self._surfaces_tab.update_from_state(self._state)
        
        # Switch to Monte Carlo tab
        self._tabs.setCurrentWidget(self._mc_tab)
        
        self._log_console.log_success("Analysis complete!")
        self._status_bar.showMessage(
            f"Analysis complete for {self._state.config.ticker}"
        )

    @Slot(str)
    def _on_simulation_error(self, message: str) -> None:
        """Handle simulation error."""
        self._update_ui_running_state(False)
        self._log_console.set_complete(False)
        self._on_error(message)

    def _export_html_report(self) -> None:
        """Generate and save an HTML report."""
        from datetime import datetime
        
        ticker = self._state.config.ticker.upper()
        output_dir = self._controller.get_output_directory(ticker)
        report_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Generate HTML content
        html = self._generate_report_html()
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        self._log_console.log_success(f"Report saved to {report_path}")
        self._status_bar.showMessage(f"Report exported to {report_path}")
        
        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")

    def _generate_report_html(self) -> str:
        """Generate HTML report content with charts from interactive widgets."""
        import base64
        import io
        from datetime import datetime
        
        ticker = self._state.config.ticker.upper()
        params = self._state.parameters
        config = self._state.config
        
        # Export charts to memory buffer and encode as base64
        def export_chart_to_base64(chart_widget, dpi: int = 120) -> str:
            try:
                buffer = io.BytesIO()
                chart_widget._figure.savefig(
                    buffer, format='png', dpi=dpi,
                    bbox_inches='tight', facecolor='#1a1a2e'
                )
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            except Exception:
                return ""
        
        # Build chart sections from interactive widgets
        chart_sections = []
        
        # Monte Carlo charts
        if self._state.has_simulation_results():
            charts = [
                ("Historical vs Simulated", self._mc_tab._hist_chart),
                ("Return Distributions", self._mc_tab._returns_chart),
                ("Forecast Distribution", self._mc_tab._forecast_chart),
            ]
            
            for name, chart in charts:
                img_data = export_chart_to_base64(chart)
                if img_data:
                    chart_sections.append(f'''
                <div class="chart">
                    <h3>{name}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{name}">
                </div>
                    ''')
            
            # 3D surfaces if enabled
            if config.generate_3d_plots:
                surfaces = [
                    ("Call Option Surface", self._surfaces_tab._call_surface),
                    ("Put Option Surface", self._surfaces_tab._put_surface),
                ]
                
                for name, surface in surfaces:
                    img_data = export_chart_to_base64(surface, dpi=100)
                    if img_data:
                        chart_sections.append(f'''
                <div class="chart">
                    <h3>{name}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{name}">
                </div>
                        ''')
        
        # Extract values for cleaner HTML template
        call_price = self._state.call_result.price if self._state.call_result else 0
        call_se = self._state.call_result.std_error if self._state.call_result else 0
        put_price = self._state.put_result.price if self._state.put_result else 0
        put_se = self._state.put_result.std_error if self._state.put_result else 0
        drift_class = 'positive' if params.drift >= 0 else 'negative'
        drift_pct = params.drift * 100
        
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{ticker} Black-Scholes Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #1a1a2e;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #ffffff;
        }}
        .header {{
            border-bottom: 2px solid #4a6fa5;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: #1f1f35;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}
        .metric {{
            background-color: #252542;
            border-radius: 6px;
            padding: 16px;
        }}
        .metric-label {{
            color: #888;
            font-size: 12px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4a6fa5;
        }}
        .chart {{
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #3d3d5c;
        }}
        th {{
            color: #888;
            font-weight: normal;
        }}
        .positive {{ color: #00d26a; }}
        .negative {{ color: #f23645; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #3d3d5c;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{ticker} Black-Scholes Analysis</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Market Parameters</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${params.spot_price:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Annual Drift</div>
                    <div class="metric-value {drift_class}">{drift_pct:+.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Annual Volatility</div>
                    <div class="metric-value">{params.volatility*100:.2f}%</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Option Pricing (3M ATM)</h2>
            <table>
                <tr>
                    <th>Option Type</th>
                    <th>Price</th>
                    <th>Std Error</th>
                </tr>
                <tr>
                    <td>Call</td>
                    <td class="positive">${call_price:.2f}</td>
                    <td>±${call_se:.4f}</td>
                </tr>
                <tr>
                    <td>Put</td>
                    <td class="negative">${put_price:.2f}</td>
                    <td>±${put_se:.4f}</td>
                </tr>
            </table>
        </div>
        
        {''.join(chart_sections)}
        
        <div class="section">
            <h2>Configuration</h2>
            <table>
                <tr><td>Historical Days</td><td>{config.historical_days}</td></tr>
                <tr><td>Forecast Horizon</td><td>{config.forecast_horizon} years</td></tr>
                <tr><td>Simulations</td><td>{config.n_simulations:,}</td></tr>
                <tr><td>Risk-Free Rate</td><td>{config.risk_free_rate*100:.2f}%</td></tr>
                <tr><td>Random Seed</td><td>{config.seed}</td></tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by Quant Black-Scholes Simulator | McFramework</p>
        </div>
    </div>
</body>
</html>
'''
        return html

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Stop any running workers
        if self._sim_worker and self._sim_worker.isRunning():
            self._sim_worker.cancel()
            self._sim_worker.wait()
        
        if self._fetch_worker and self._fetch_worker.isRunning():
            self._fetch_worker.wait()
        
        event.accept()


def main():
    """Main entry point for the application."""
    # Set up multiprocessing for macOS
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Quant Black-Scholes Simulator")
    app.setOrganizationName("McFramework")
    
    # Create and show main window
    window = QuantBlackScholesWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

