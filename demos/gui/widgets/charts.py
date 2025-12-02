"""
Interactive chart widgets using matplotlib FigureCanvas.

This module provides reusable, interactive chart components that embed
matplotlib figures directly in the Qt UI with zoom, pan, and hover support.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# =============================================================================
# Theme Constants
# =============================================================================

DARK_THEME = {
    "background": "#1a1a32",
    "foreground": "#e8e8f0",
    "grid": "#3a3a55",
    "grid_alpha": 0.25,
    "spine": "#3a3a55",
    "tick": "#9a9aaa",
    "historical": "#5aa0e5",
    "simulated": "#6a6a7a",
    "simulated_alpha": 0.12,
    "mean": "#ff7a7a",
    "confidence": "#ff7a7a",
    "confidence_alpha": 0.12,
    "call": "#2adf7a",
    "put": "#ff5a6a",
    "neutral": "#8a8a9a",
}


# =============================================================================
# Base Interactive Chart
# =============================================================================


class InteractiveChart(QWidget):
    """
    Base class for interactive matplotlib charts.
    
    Provides a FigureCanvas with NavigationToolbar for zoom, pan, and save.
    All charts inherit from this and implement update_data() for their
    specific visualization.
    
    Features:
    - Crosshair cursor for precise value reading
    - Hover tooltips with data info
    - Reset view button in toolbar
    
    Signals:
        hover_data: Emitted with data info when mouse hovers over data point
    """
    
    hover_data = Signal(str)

    def __init__(
        self,
        figsize: tuple[float, float] = (8, 5),
        dpi: int = 100,
        parent: QWidget | None = None,
    ):
        """
        Initialize the interactive chart.
        
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        self._figure = Figure(figsize=figsize, dpi=dpi)
        self._figure.patch.set_facecolor(DARK_THEME["background"])
        
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setParent(self)
        
        # Create axes
        self._axes: Axes = self._figure.add_subplot(111)
        self._configure_axes()
        
        # Crosshair lines
        self._crosshair_h = None
        self._crosshair_v = None
        self._crosshair_enabled = True
        
        # Tooltip annotation
        self._tooltip = None
        
        # Navigation toolbar
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._style_toolbar()
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        
        # Connect events
        self._canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self._canvas.mpl_connect('axes_leave_event', self._on_axes_leave)

    def _setup_crosshair(self) -> None:
        """Set up crosshair lines on the axes."""
        if self._crosshair_h is not None:
            return
        
        self._crosshair_h = self._axes.axhline(
            y=0, color=DARK_THEME["grid"], linestyle='--',
            linewidth=0.8, alpha=0.6, visible=False
        )
        self._crosshair_v = self._axes.axvline(
            x=0, color=DARK_THEME["grid"], linestyle='--',
            linewidth=0.8, alpha=0.6, visible=False
        )

    def _update_crosshair(self, x: float, y: float) -> None:
        """Update crosshair position."""
        if not self._crosshair_enabled:
            return
        
        self._setup_crosshair()
        
        if self._crosshair_h is not None:
            self._crosshair_h.set_ydata([y, y])
            self._crosshair_h.set_visible(True)
        
        if self._crosshair_v is not None:
            self._crosshair_v.set_xdata([x, x])
            self._crosshair_v.set_visible(True)

    def _hide_crosshair(self) -> None:
        """Hide the crosshair lines."""
        if self._crosshair_h is not None:
            self._crosshair_h.set_visible(False)
        if self._crosshair_v is not None:
            self._crosshair_v.set_visible(False)

    def _on_axes_leave(self, event) -> None:
        """Handle mouse leaving the axes."""
        self._hide_crosshair()
        if self._tooltip is not None:
            self._tooltip.set_visible(False)
        self._canvas.draw_idle()

    def _configure_axes(self) -> None:
        """Apply dark theme styling to axes."""
        ax = self._axes
        ax.set_facecolor(DARK_THEME["background"])
        ax.tick_params(colors=DARK_THEME["tick"], labelsize=9)
        
        for spine in ax.spines.values():
            spine.set_color(DARK_THEME["spine"])
        
        ax.grid(True, alpha=DARK_THEME["grid_alpha"], color=DARK_THEME["grid"])
        ax.title.set_color(DARK_THEME["foreground"])
        ax.xaxis.label.set_color(DARK_THEME["foreground"])
        ax.yaxis.label.set_color(DARK_THEME["foreground"])

    def _style_toolbar(self) -> None:
        """Apply dark styling to the navigation toolbar."""
        self._toolbar.setStyleSheet("""
            QToolBar {
                background-color: #252542;
                border: none;
                spacing: 4px;
                padding: 2px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px;
                color: #cccccc;
            }
            QToolButton:hover {
                background-color: #3d3d5c;
                border-radius: 4px;
            }
            QToolButton:pressed {
                background-color: #4a6fa5;
            }
        """)

    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement for hover tooltips. Override in subclasses."""
        pass

    def clear(self) -> None:
        """Clear the chart."""
        self._axes.clear()
        self._configure_axes()
        self._canvas.draw()

    def refresh(self) -> None:
        """Refresh the canvas."""
        # self._figure.tight_layout() # commented out to prevent layout issues
        self._canvas.draw()

    def export_to_png(self, path: Path, dpi: int = 150) -> None:
        """
        Export the chart to a PNG file.
        
        Args:
            path: Output file path
            dpi: Resolution for export
        """
        self._figure.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_THEME["background"])


# =============================================================================
# Historical vs Simulated Paths Chart
# =============================================================================


class HistoricalVsSimulatedChart(InteractiveChart):
    """
    Chart showing historical prices alongside simulated future paths.
    
    Features:
    - Historical price line (solid blue)
    - Multiple simulated paths (transparent gray)
    - Mean forecast line (dashed red)
    - 90% confidence band (shaded)
    - Hover tooltip showing price at cursor
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the chart."""
        super().__init__(figsize=(10, 5), parent=parent)
        self._historical_data: np.ndarray | None = None
        self._simulated_data: np.ndarray | None = None
        self._annotation = None

    def update_data(
        self,
        historical: np.ndarray,
        simulated: np.ndarray,
        ticker: str,
        forecast_horizon: float,
        max_paths: int = 50,
    ) -> None:
        """
        Update the chart with new data.
        
        Args:
            historical: Array of historical prices
            simulated: Array of simulated paths (n_paths, n_steps)
            ticker: Ticker symbol for title
            forecast_horizon: Forecast horizon in years
            max_paths: Maximum number of paths to display
        """
        self._historical_data = historical
        self._simulated_data = simulated
        
        self._axes.clear()
        self._configure_axes()
        
        n_hist = len(historical)
        n_paths, n_steps = simulated.shape
        
        # Time arrays
        hist_time = np.arange(n_hist)
        future_time = np.arange(n_hist, n_hist + n_steps)
        
        # Plot historical data
        self._axes.plot(
            hist_time, historical,
            color=DARK_THEME["historical"],
            linewidth=2,
            label='Historical',
            zorder=10,
        )
        
        # Plot simulated paths (limit for performance)
        paths_to_plot = min(max_paths, n_paths)
        for i in range(paths_to_plot):
            self._axes.plot(
                future_time, simulated[i],
                color=DARK_THEME["simulated"],
                alpha=DARK_THEME["simulated_alpha"],
                linewidth=0.5,
            )
        
        # Mean forecast
        mean_forecast = np.mean(simulated, axis=0)
        self._axes.plot(
            future_time, mean_forecast,
            color=DARK_THEME["mean"],
            linewidth=2.5,
            linestyle='--',
            label='Mean Forecast',
            zorder=11,
        )
        
        # Confidence bands
        lower = np.percentile(simulated, 5, axis=0)
        upper = np.percentile(simulated, 95, axis=0)
        self._axes.fill_between(
            future_time, lower, upper,
            color=DARK_THEME["confidence"],
            alpha=DARK_THEME["confidence_alpha"],
            label='90% Confidence',
        )
        
        # Today marker
        self._axes.axvline(
            x=n_hist - 1,
            color=DARK_THEME["foreground"],
            linestyle=':',
            linewidth=1.5,
            alpha=0.5,
        )
        self._axes.text(
            n_hist - 1, self._axes.get_ylim()[1] * 0.98,
            'Today',
            ha='right', va='top',
            color=DARK_THEME["foreground"],
            fontsize=9,
        )
        
        # Labels
        self._axes.set_xlabel('Trading Days')
        self._axes.set_ylabel('Price ($)')
        self._axes.set_title(
            f'{ticker} - Historical Data & {forecast_horizon:.1f}-Year Monte Carlo Forecast',
            color=DARK_THEME["foreground"],
            fontsize=11,
        )
        self._axes.legend(loc='upper left', framealpha=0.8)
        
        self.refresh()

    def _on_mouse_move(self, event) -> None:
        """Show price tooltip and crosshair on hover."""
        if event.inaxes != self._axes:
            self._hide_crosshair()
            if self._annotation:
                self._annotation.set_visible(False)
            self._canvas.draw_idle()
            return
        
        if self._historical_data is None:
            return
        
        # Update crosshair
        if event.xdata is not None and event.ydata is not None:
            self._update_crosshair(event.xdata, event.ydata)
        
        x = int(round(event.xdata)) if event.xdata else 0
        n_hist = len(self._historical_data)
        
        if 0 <= x < n_hist:
            price = self._historical_data[x]
            label = f'Day {x}: ${price:.2f}'
            tooltip_text = f'Historical\nDay {x}\n${price:.2f}'
        elif self._simulated_data is not None:
            sim_idx = x - n_hist
            if 0 <= sim_idx < self._simulated_data.shape[1]:
                mean_price = np.mean(self._simulated_data[:, sim_idx])
                p5 = np.percentile(self._simulated_data[:, sim_idx], 5)
                p95 = np.percentile(self._simulated_data[:, sim_idx], 95)
                label = f'Day {x}: ${mean_price:.2f} (mean)'
                tooltip_text = (
                    f'Forecast Day {sim_idx+1}\n'
                    f'Mean: ${mean_price:.2f}\n5%: ${p5:.2f}\n95%: ${p95:.2f}'
                )
            else:
                self._canvas.draw_idle()
                return
        else:
            self._canvas.draw_idle()
            return
        
        # Update or create annotation
        if self._annotation is None:
            self._annotation = self._axes.annotate(
                tooltip_text,
                xy=(event.xdata, event.ydata),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                color=DARK_THEME["foreground"],
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=DARK_THEME["background"],
                    edgecolor=DARK_THEME["grid"],
                    alpha=0.9
                ),
            )
        else:
            self._annotation.set_text(tooltip_text)
            self._annotation.xy = (event.xdata, event.ydata)
            self._annotation.set_visible(True)
        
        self._canvas.draw_idle()
        self.hover_data.emit(label)


# =============================================================================
# Return Distribution Chart
# =============================================================================


class ReturnDistributionChart(InteractiveChart):
    """
    Side-by-side histograms comparing historical and simulated returns.
    
    Features:
    - Historical returns histogram (left)
    - Simulated returns histogram (right)
    - Mean lines on each
    - Hover to show bin details
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the chart."""
        super().__init__(figsize=(10, 4), parent=parent)
        
        # Create two subplots
        self._figure.clear()
        self._axes_hist, self._axes_sim = self._figure.subplots(1, 2)
        self._configure_subplot(self._axes_hist)
        self._configure_subplot(self._axes_sim)

    def _configure_subplot(self, ax: "Axes") -> None:
        """Apply dark theme to a subplot."""
        ax.set_facecolor(DARK_THEME["background"])
        ax.tick_params(colors=DARK_THEME["tick"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(DARK_THEME["spine"])
        ax.grid(True, alpha=DARK_THEME["grid_alpha"], color=DARK_THEME["grid"])

    def clear(self) -> None:
        """Clear both subplots."""
        self._axes_hist.clear()
        self._axes_sim.clear()
        self._configure_subplot(self._axes_hist)
        self._configure_subplot(self._axes_sim)
        self._canvas.draw()

    def update_data(
        self,
        historical_prices: np.ndarray,
        simulated_paths: np.ndarray,
        ticker: str,
    ) -> None:
        """
        Update the chart with return distributions.
        
        Args:
            historical_prices: Historical price array
            simulated_paths: Simulated paths array (n_paths, n_steps)
            ticker: Ticker symbol for titles
        """
        # Calculate returns
        hist_returns = np.diff(np.log(historical_prices))
        
        sim_returns = []
        for path in simulated_paths:
            sim_returns.extend(np.diff(np.log(path)))
        sim_returns = np.array(sim_returns)
        
        # Clear and reconfigure
        self._axes_hist.clear()
        self._axes_sim.clear()
        self._configure_subplot(self._axes_hist)
        self._configure_subplot(self._axes_sim)
        
        # Historical histogram
        self._axes_hist.hist(
            hist_returns, bins=50,
            alpha=0.7,
            color=DARK_THEME["historical"],
            edgecolor='#333',
        )
        hist_mean = np.mean(hist_returns)
        self._axes_hist.axvline(
            hist_mean, color=DARK_THEME["mean"],
            linestyle='--', linewidth=2,
            label=f'Mean: {hist_mean:.6f}',
        )
        self._axes_hist.set_xlabel('Log Returns', color=DARK_THEME["foreground"])
        self._axes_hist.set_ylabel('Frequency', color=DARK_THEME["foreground"])
        self._axes_hist.set_title(
            f'{ticker} - Historical Returns',
            color=DARK_THEME["foreground"], fontsize=10,
        )
        self._axes_hist.legend(fontsize=8)
        
        # Simulated histogram
        self._axes_sim.hist(
            sim_returns, bins=50,
            alpha=0.7,
            color=DARK_THEME["neutral"],
            edgecolor='#333',
        )
        sim_mean = np.mean(sim_returns)
        self._axes_sim.axvline(
            sim_mean, color=DARK_THEME["mean"],
            linestyle='--', linewidth=2,
            label=f'Mean: {sim_mean:.6f}',
        )
        self._axes_sim.set_xlabel('Log Returns', color=DARK_THEME["foreground"])
        self._axes_sim.set_ylabel('Frequency', color=DARK_THEME["foreground"])
        self._axes_sim.set_title(
            f'{ticker} - Simulated Returns',
            color=DARK_THEME["foreground"], fontsize=10,
        )
        self._axes_sim.legend(fontsize=8)
        
        self.refresh()


# =============================================================================
# Forecast Distribution Chart
# =============================================================================


class ForecastDistributionChart(InteractiveChart):
    """
    Histogram of forecasted final prices with key statistics.
    
    Features:
    - Price distribution histogram
    - Current price, mean, median markers
    - 5th and 95th percentile annotations
    - Hover to show probability info
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the chart."""
        super().__init__(figsize=(8, 4), parent=parent)
        self._final_prices: np.ndarray | None = None

    def update_data(
        self,
        simulated_paths: np.ndarray,
        current_price: float,
        ticker: str,
        forecast_horizon: float,
    ) -> None:
        """
        Update the chart with forecast distribution.
        
        Args:
            simulated_paths: Simulated paths (n_paths, n_steps)
            current_price: Current stock price
            ticker: Ticker symbol
            forecast_horizon: Horizon in years
        """
        final_prices = simulated_paths[:, -1]
        self._final_prices = final_prices
        
        self._axes.clear()
        self._configure_axes()
        
        # Statistics
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        p5 = np.percentile(final_prices, 5)
        p95 = np.percentile(final_prices, 95)
        
        # Histogram
        self._axes.hist(
            final_prices, bins=40,
            alpha=0.7,
            color='steelblue',
            edgecolor='#333',
        )
        
        # Marker lines
        self._axes.axvline(
            current_price, color=DARK_THEME["historical"],
            linestyle='--', linewidth=2,
            label=f'Current: ${current_price:.2f}',
        )
        self._axes.axvline(
            mean_price, color=DARK_THEME["mean"],
            linestyle='--', linewidth=2,
            label=f'Mean: ${mean_price:.2f}',
        )
        self._axes.axvline(
            median_price, color='orange',
            linestyle='--', linewidth=2,
            label=f'Median: ${median_price:.2f}',
        )
        
        # Labels
        self._axes.set_xlabel('Price ($)', color=DARK_THEME["foreground"])
        self._axes.set_ylabel('Frequency', color=DARK_THEME["foreground"])
        self._axes.set_title(
            f'{ticker} - Price Distribution in {forecast_horizon:.1f} Years\n'
            f'(5th: ${p5:.2f}, 95th: ${p95:.2f})',
            color=DARK_THEME["foreground"], fontsize=10,
        )
        self._axes.legend(loc='upper right', fontsize=8)
        
        self.refresh()


# =============================================================================
# 3D Option Surface Chart
# =============================================================================


class OptionSurfaceChart(InteractiveChart):
    """
    3D surface plot of option prices as function of stock price and time.
    
    Features:
    - Rotatable 3D view
    - Simulated paths overlaid
    - Mean path highlighted
    - Mouse drag to rotate
    """

    def __init__(self, option_type: str = "call", parent: QWidget | None = None):
        """
        Initialize the 3D surface chart.
        
        Args:
            option_type: "call" or "put"
            parent: Optional parent widget
        """
        self._option_type = option_type
        super().__init__(figsize=(8, 6), parent=parent)
        
        # Replace 2D axes with 3D
        self._figure.clear()
        self._axes = self._figure.add_subplot(111, projection='3d')
        self._configure_3d_axes()

    def _configure_3d_axes(self) -> None:
        """Configure 3D axes styling."""
        ax = self._axes
        ax.set_facecolor(DARK_THEME["background"])
        ax.tick_params(colors=DARK_THEME["tick"], labelsize=8)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(DARK_THEME["grid"])
        ax.yaxis.pane.set_edgecolor(DARK_THEME["grid"])
        ax.zaxis.pane.set_edgecolor(DARK_THEME["grid"])

    def update_data(
        self,
        paths: np.ndarray,
        spot_price: float,
        volatility: float,
        risk_free_rate: float,
        forecast_horizon: float,
        ticker: str,
        max_paths: int = 50,
    ) -> None:
        """
        Update the 3D surface with option prices.
        
        Args:
            paths: Simulated stock paths (n_paths, n_steps)
            spot_price: Current stock price (used as strike)
            volatility: Volatility sigma
            risk_free_rate: Risk-free rate r
            forecast_horizon: Time horizon T
            ticker: Ticker symbol
            max_paths: Max paths to overlay
        """
        self._figure.clear()
        self._axes = self._figure.add_subplot(111, projection='3d')
        self._configure_3d_axes()
        
        n_paths, n_steps = paths.shape
        t = np.linspace(0, forecast_horizon, n_steps)
        K = spot_price  # ATM
        
        # Create surface grid
        S_min, S_max = np.min(paths), np.max(paths)
        S_vals = np.linspace(S_min, S_max, 80)
        T_vals = np.linspace(0, forecast_horizon, 80)
        T_grid, S_grid = np.meshgrid(T_vals, S_vals)
        tau_grid = forecast_horizon - T_grid
        
        # Black-Scholes pricing
        C_grid = self._bs_price(S_grid, K, risk_free_rate, volatility, tau_grid)
        
        # Colormap based on option type
        cmap = 'viridis' if self._option_type == 'call' else 'plasma'
        path_color = DARK_THEME["mean"] if self._option_type == 'call' else DARK_THEME["put"]
        
        # Plot surface
        self._axes.plot_surface(
            T_grid, S_grid, C_grid,
            cmap=cmap, alpha=0.4,
            rstride=4, cstride=4, linewidth=0,
        )
        
        # Overlay paths
        paths_to_plot = min(max_paths, n_paths)
        for i in range(paths_to_plot):
            tau = forecast_horizon - t
            path_prices = self._bs_price(paths[i], K, risk_free_rate, volatility, tau)
            self._axes.plot(
                t, paths[i], path_prices,
                color='white', alpha=0.5,
            )
        
        # Mean path
        mean_path = np.mean(paths, axis=0)
        tau_mean = forecast_horizon - t
        mean_prices = self._bs_price(mean_path, K, risk_free_rate, volatility, tau_mean)
        self._axes.plot(
            t, mean_path, mean_prices,
            color=path_color, linewidth=2.5, label='Mean Path',
        )
        
        # Labels
        label_color = DARK_THEME["foreground"]
        self._axes.set_xlabel('Time (years)', fontsize=9, color=label_color)
        self._axes.set_ylabel('Stock Price ($)', fontsize=9, color=label_color)
        self._axes.set_zlabel(
            f'{self._option_type.title()} Price ($)', fontsize=9, color=label_color
        )
        self._axes.set_title(
            f'{ticker} - {self._option_type.title()} Option Price Surface (K=${K:.2f})',
            color=DARK_THEME["foreground"], fontsize=10,
        )
        
        # Adjust view angle
        self._axes.view_init(elev=25, azim=-60)
        self._axes.set_box_aspect([1.5, 1, 0.8])
        
        self.refresh()

    def _bs_price(
        self,
        S: np.ndarray,
        K: float,
        r: float,
        sigma: float,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Calculate Black-Scholes price (vectorized)."""
        from scipy.stats import norm
        
        S = np.asarray(S)
        tau = np.asarray(tau)
        
        # Handle tau <= 0 (at expiry)
        intrinsic = np.maximum(S - K, 0) if self._option_type == 'call' else np.maximum(K - S, 0)
        
        result = np.zeros_like(S, dtype=float)
        mask = tau > 0
        
        if np.any(mask):
            tau_pos = tau[mask] if tau.ndim > 0 else tau
            S_pos = S[mask] if S.ndim > 0 else S
            
            d1 = (np.log(S_pos / K) + (r + 0.5 * sigma**2) * tau_pos) / (sigma * np.sqrt(tau_pos))
            d2 = d1 - sigma * np.sqrt(tau_pos)
            
            if self._option_type == 'call':
                prices = S_pos * norm.cdf(d1) - K * np.exp(-r * tau_pos) * norm.cdf(d2)
            else:
                prices = K * np.exp(-r * tau_pos) * norm.cdf(-d2) - S_pos * norm.cdf(-d1)
            
            if S.ndim > 0:
                result[mask] = prices
            else:
                result = prices
        
        if S.ndim > 0:
            result[~mask] = intrinsic[~mask] if intrinsic.ndim > 0 else intrinsic
        
        return result


# =============================================================================
# Market Data Charts
# =============================================================================


class SparklineChart(InteractiveChart):
    """Lightweight sparkline chart built atop the InteractiveChart base."""

    def __init__(self, parent: QWidget | None = None, n_points: int = 60):
        super().__init__(figsize=(6, 2), parent=parent)
        self._n_points = n_points
        self._crosshair_enabled = False
        self._configure_sparkline_axes()

    def _configure_sparkline_axes(self) -> None:
        ax = self._axes
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888888', labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        ax.grid(True, alpha=0.2, color='#444444')

    def clear(self) -> None:
        super().clear()
        self._configure_sparkline_axes()

    def update_data(
        self,
        prices: np.ndarray | None,
        ticker: str = "",
        dates: list[datetime] | None = None,
        events: list[tuple[int, str]] | None = None,
    ) -> None:
        self._axes.clear()
        self._configure_sparkline_axes()

        if prices is None or len(prices) == 0:
            self._axes.text(
                0.5,
                0.5,
                "No data",
                ha='center',
                va='center',
                color='#888888',
                fontsize=12,
                transform=self._axes.transAxes,
            )
            self.refresh()
            return

        display_prices = prices[-self._n_points :] if len(prices) > self._n_points else prices
        if dates is not None and len(dates) == len(prices):
            display_dates = dates[-len(display_prices):]
        else:
            display_dates = None
        x = np.arange(len(display_prices))

        is_up = display_prices[-1] >= display_prices[0]
        color = '#00d26a' if is_up else '#f23645'

        self._axes.plot(x, display_prices, color=color, linewidth=1.5)
        self._axes.fill_between(x, display_prices, alpha=0.2, color=color)

        current_price = display_prices[-1]
        self._axes.annotate(
            f'${current_price:.2f}',
            xy=(len(display_prices) - 1, current_price),
            xytext=(5, 0),
            textcoords='offset points',
            color=color,
            fontsize=10,
            fontweight='bold',
            va='center',
        )

        if ticker:
            self._axes.set_title(
                f'{ticker} - Last {len(display_prices)} Days',
                color='#cccccc',
                fontsize=10,
                loc='left',
            )

        if events:
            for idx, label in events:
                if 0 <= idx < len(display_prices):
                    price_point = display_prices[idx]
                    self._axes.scatter(
                        idx,
                        price_point,
                        color='#ffd166',
                        s=30,
                        zorder=5,
                        marker='o',
                    )
                    self._axes.annotate(
                        label,
                        xy=(idx, price_point),
                        xytext=(0, 12),
                        textcoords='offset points',
                        color='#ffd166',
                        fontsize=8,
                        ha='center',
                    )

        self._axes.set_ylabel('Price ($)', color='#888888', fontsize=9)
        if display_dates:
            tick_indices = np.linspace(0, len(display_prices) - 1, num=3, dtype=int)
            tick_labels = [display_dates[idx].strftime("%b %d") for idx in tick_indices]
            self._axes.set_xticks(tick_indices)
            self._axes.set_xticklabels(tick_labels, color='#888888', fontsize=8)
            self._axes.set_xlabel('Date', color='#888888', fontsize=9)
        else:
            self._axes.set_xlabel('Trading Days', color='#888888', fontsize=9)

        self.refresh()


class CandlestickChart(InteractiveChart):
    """Interactive candlestick chart with aligned volume subplot.

    Features:
    - Candlestick OHLC price bars
    - Volume bars aligned below
    - Crosshair spanning both subplots
    - Hover tooltip with OHLC + volume info
    """

    def __init__(self, parent: QWidget | None = None, max_points: int = 120):
        super().__init__(figsize=(8, 4), parent=parent)
        self._max_points = max_points
        # Enable crosshair for this chart
        self._crosshair_enabled = True
        self._figure.clear()
        grid = self._figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        self._price_ax = self._figure.add_subplot(grid[0])
        self._volume_ax = self._figure.add_subplot(grid[1], sharex=self._price_ax)
        self._configure_candlestick_axes()

        # Crosshair lines for price and volume axes
        self._price_crosshair_h = None
        self._price_crosshair_v = None
        self._volume_crosshair_v = None

        # Tooltip annotation on price axes
        self._candle_annotation = None

        # Data cache for hover lookups
        self._x_values: np.ndarray | None = None
        self._opens: np.ndarray | None = None
        self._highs: np.ndarray | None = None
        self._lows: np.ndarray | None = None
        self._closes: np.ndarray | None = None
        self._volumes: np.ndarray | None = None
        self._dates: list[datetime] | None = None

        # Cached axis limits to prevent rescaling on hover
        self._price_xlim: tuple[float, float] | None = None
        self._price_ylim: tuple[float, float] | None = None
        self._volume_ylim: tuple[float, float] | None = None

        # Track if user has zoomed/panned (don't restore limits in that case)
        self._user_has_zoomed = False
        self._canvas.mpl_connect('button_release_event', self._on_button_release)

        # Connect to toolbar home button to reset zoom flag
        self._toolbar.actions()[0].triggered.connect(self._on_home_clicked)

    def _configure_axes(self) -> None:
        """Override base hook to avoid configuring unused _axes."""
        # Candlestick chart manages its own axes pair.
        pass

    def _configure_candlestick_axes(self) -> None:
        background = '#101020'
        grid_color = '#2a2a40'

        for ax in (self._price_ax, self._volume_ax):
            ax.set_facecolor(background)
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.tick_params(colors='#bbbbbb', labelsize=8)
            ax.grid(True, color=grid_color, alpha=0.3, linestyle='--', linewidth=0.5)

        self._volume_ax.ticklabel_format(style='plain', axis='y')
        self._volume_ax.set_ylabel('Volume', color='#bbbbbb', fontsize=8)
        self._price_ax.set_ylabel('Price ($)', color='#bbbbbb', fontsize=9)

    # -------------------------------------------------------------------------
    # Crosshair helpers (overrides base class for dual-axis support)
    # -------------------------------------------------------------------------

    def _setup_candlestick_crosshair(self) -> None:
        """Create crosshair lines on both price and volume axes."""
        if self._price_crosshair_h is not None:
            return

        # Get current axis limits to position lines within valid range
        xlim = self._price_ax.get_xlim()
        ylim = self._price_ax.get_ylim()
        x_mid = (xlim[0] + xlim[1]) / 2
        y_mid = (ylim[0] + ylim[1]) / 2

        line_style = dict(
            color=DARK_THEME["grid"],
            linestyle='--',
            linewidth=0.8,
            alpha=0.6,
            visible=False,
        )
        # Create lines at midpoint (within current limits) to avoid autoscale issues
        self._price_crosshair_h = self._price_ax.axhline(y=y_mid, **line_style)
        self._price_crosshair_v = self._price_ax.axvline(x=x_mid, **line_style)
        self._volume_crosshair_v = self._volume_ax.axvline(x=x_mid, **line_style)

        # Prevent these artists from affecting autoscale
        self._price_crosshair_h.set_clip_on(True)
        self._price_crosshair_v.set_clip_on(True)
        self._volume_crosshair_v.set_clip_on(True)

    def _update_candlestick_crosshair(self, x: float, y: float) -> None:
        """Update crosshair position across both subplots."""
        if not self._crosshair_enabled:
            return

        self._setup_candlestick_crosshair()

        if self._price_crosshair_h is not None:
            self._price_crosshair_h.set_ydata([y, y])
            self._price_crosshair_h.set_visible(True)

        if self._price_crosshair_v is not None:
            self._price_crosshair_v.set_xdata([x, x])
            self._price_crosshair_v.set_visible(True)

        if self._volume_crosshair_v is not None:
            self._volume_crosshair_v.set_xdata([x, x])
            self._volume_crosshair_v.set_visible(True)

    def _hide_candlestick_crosshair(self) -> None:
        """Hide crosshair lines on both subplots."""
        for line in (
            self._price_crosshair_h,
            self._price_crosshair_v,
            self._volume_crosshair_v,
        ):
            if line is not None:
                line.set_visible(False)

    def _on_axes_leave(self, event) -> None:
        """Handle mouse leaving the axes (override)."""
        self._hide_candlestick_crosshair()
        if self._candle_annotation is not None:
            self._candle_annotation.set_visible(False)
        self._restore_axis_limits()
        self._canvas.draw_idle()

    def _on_button_release(self, event) -> None:
        """Detect when user finishes a zoom/pan operation."""
        # Check if toolbar is in zoom or pan mode
        mode = self._toolbar.mode if hasattr(self._toolbar, 'mode') else ''
        if mode in ('zoom rect', 'pan/zoom'):
            self._user_has_zoomed = True

    def _on_home_clicked(self) -> None:
        """Reset zoom flag when user clicks Home button."""
        self._user_has_zoomed = False

    def _restore_axis_limits(self) -> None:
        """Restore cached axis limits to prevent rescaling (only if user hasn't zoomed)."""
        if self._user_has_zoomed:
            return
        if self._price_xlim is not None:
            self._price_ax.set_xlim(self._price_xlim)
        if self._price_ylim is not None:
            self._price_ax.set_ylim(self._price_ylim)
        if self._volume_ylim is not None:
            self._volume_ax.set_ylim(self._volume_ylim)

    def _on_mouse_move(self, event) -> None:
        """Handle hover: show crosshair and tooltip with OHLC + volume."""
        if event.inaxes not in (self._price_ax, self._volume_ax):
            self._hide_candlestick_crosshair()
            if self._candle_annotation is not None:
                self._candle_annotation.set_visible(False)
            self._restore_axis_limits()
            self._canvas.draw_idle()
            return

        if self._x_values is None or len(self._x_values) == 0:
            return

        x = event.xdata
        if x is None:
            return

        # Find nearest candle index
        idx = int(np.argmin(np.abs(self._x_values - x)))
        if idx < 0 or idx >= len(self._x_values):
            return

        # Snap crosshair to candle center
        snap_x = self._x_values[idx]
        snap_y = self._closes[idx] if event.inaxes == self._price_ax else 0

        self._update_candlestick_crosshair(snap_x, snap_y)

        # Build tooltip text
        date_str = self._dates[idx].strftime("%Y-%m-%d") if self._dates else ""
        o = self._opens[idx]
        h = self._highs[idx]
        lo = self._lows[idx]
        c = self._closes[idx]
        vol = self._volumes[idx] if self._volumes is not None else 0
        vol_str = f"{vol:,.0f}" if vol else "N/A"

        tooltip_text = (
            f"{date_str}\n"
            f"O: ${o:.2f}  H: ${h:.2f}\n"
            f"L: ${lo:.2f}  C: ${c:.2f}\n"
            f"Vol: {vol_str}"
        )

        # Position annotation near cursor on price axes
        if self._candle_annotation is None:
            self._candle_annotation = self._price_ax.annotate(
                tooltip_text,
                xy=(snap_x, c),
                xytext=(12, 12),
                textcoords='offset points',
                fontsize=9,
                color=DARK_THEME["foreground"],
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='#1a1a32',
                    edgecolor=DARK_THEME["grid"],
                    alpha=0.95,
                ),
                zorder=100,
            )
        else:
            self._candle_annotation.set_text(tooltip_text)
            self._candle_annotation.xy = (snap_x, c)
            self._candle_annotation.set_visible(True)

        # Restore axis limits before redrawing to prevent zoom-out
        self._restore_axis_limits()
        self._canvas.draw_idle()
        self.hover_data.emit(f"{date_str} C=${c:.2f}")

    def clear(self) -> None:
        self._price_ax.clear()
        self._volume_ax.clear()
        self._configure_candlestick_axes()
        # Reset crosshair lines (will be recreated on next hover)
        self._price_crosshair_h = None
        self._price_crosshair_v = None
        self._volume_crosshair_v = None
        self._candle_annotation = None
        # Clear cached data
        self._x_values = None
        self._opens = None
        self._highs = None
        self._lows = None
        self._closes = None
        self._volumes = None
        self._dates = None
        self.refresh()

    def update_data(
        self,
        opens: np.ndarray | None,
        highs: np.ndarray | None,
        lows: np.ndarray | None,
        closes: np.ndarray | None,
        volumes: np.ndarray | None,
        dates: list[datetime] | None,
    ) -> None:
        self._price_ax.clear()
        self._volume_ax.clear()
        self._configure_candlestick_axes()
        # Reset crosshair and zoom state for fresh data
        self._price_crosshair_h = None
        self._price_crosshair_v = None
        self._volume_crosshair_v = None
        self._candle_annotation = None
        self._user_has_zoomed = False

        if (
            opens is None
            or highs is None
            or lows is None
            or closes is None
            or dates is None
            or len(closes) == 0
        ):
            self._price_ax.text(
                0.5,
                0.5,
                "No OHLC data available",
                transform=self._price_ax.transAxes,
                ha='center',
                va='center',
                color='#888888',
            )
            self.refresh()
            return

        total_points = min(
            len(opens),
            len(highs),
            len(lows),
            len(closes),
            len(dates),
        )

        if total_points == 0:
            self.refresh()
            return

        slice_len = min(self._max_points, total_points)
        slice_obj = slice(total_points - slice_len, total_points)

        opens = np.asarray(opens)[slice_obj]
        highs = np.asarray(highs)[slice_obj]
        lows = np.asarray(lows)[slice_obj]
        closes = np.asarray(closes)[slice_obj]
        volumes_slice = np.asarray(volumes)[slice_obj] if volumes is not None else None
        # dates may be a list, so slice explicitly
        date_slice = dates[slice_obj.start:slice_obj.stop]

        x_values = mdates.date2num(date_slice)
        width = 0.6

        # Cache data for hover lookups
        self._x_values = x_values
        self._opens = opens
        self._highs = highs
        self._lows = lows
        self._closes = closes
        self._volumes = volumes_slice
        self._dates = date_slice

        for idx, x_val in enumerate(x_values):
            open_price = float(opens[idx])
            high_price = float(highs[idx])
            low_price = float(lows[idx])
            close_price = float(closes[idx])

            color = '#00d26a' if close_price >= open_price else '#f23645'
            body_bottom = min(open_price, close_price)
            body_height = max(abs(close_price - open_price), 0.01)

            self._price_ax.vlines(
                x_val,
                low_price,
                high_price,
                color=color,
                linewidth=1.0,
            )
            candle = Rectangle(
                (x_val - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.9,
            )
            self._price_ax.add_patch(candle)

        self._price_ax.set_title(
            f"Last {slice_len} Sessions",
            color='#dddddd',
            fontsize=10,
            loc='left',
        )
        self._price_ax.xaxis.set_visible(False)

        if volumes_slice is not None:
            colors = [
                '#00d26a' if closes[i] >= opens[i] else '#f23645'
                for i in range(len(x_values))
            ]
            self._volume_ax.bar(
                x_values,
                volumes_slice,
                width=width,
                color=colors,
                alpha=0.6,
            )

        formatter = mdates.DateFormatter("%b %d")
        self._volume_ax.xaxis.set_major_formatter(formatter)
        self._figure.autofmt_xdate()

        self.refresh()

        # Cache axis limits after plotting to prevent rescaling on hover
        self._price_xlim = self._price_ax.get_xlim()
        self._price_ylim = self._price_ax.get_ylim()
        self._volume_ylim = self._volume_ax.get_ylim()

