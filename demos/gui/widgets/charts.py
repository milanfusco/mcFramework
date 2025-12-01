"""
Interactive chart widgets using matplotlib FigureCanvas.

This module provides reusable, interactive chart components that embed
matplotlib figures directly in the Qt UI with zoom, pan, and hover support.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# =============================================================================
# Theme Constants
# =============================================================================

DARK_THEME = {
    "background": "#1a1a2e",
    "foreground": "#e0e0e0",
    "grid": "#444444",
    "grid_alpha": 0.3,
    "spine": "#444444",
    "tick": "#888888",
    "historical": "#4a9eff",
    "simulated": "#666666",
    "simulated_alpha": 0.15,
    "mean": "#ff6b6b",
    "confidence": "#ff6b6b",
    "confidence_alpha": 0.15,
    "call": "#00d26a",
    "put": "#f23645",
    "neutral": "#888888",
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
        
        # Navigation toolbar
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._style_toolbar()
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)
        
        # Connect hover events
        self._canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

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
        self._figure.tight_layout()
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
        """Show price tooltip on hover."""
        if event.inaxes != self._axes:
            if self._annotation:
                self._annotation.set_visible(False)
                self._canvas.draw_idle()
            return
        
        if self._historical_data is None:
            return
        
        x = int(round(event.xdata)) if event.xdata else 0
        n_hist = len(self._historical_data)
        
        if 0 <= x < n_hist:
            price = self._historical_data[x]
            label = f'Day {x}: ${price:.2f}'
        elif self._simulated_data is not None:
            sim_idx = x - n_hist
            if 0 <= sim_idx < self._simulated_data.shape[1]:
                mean_price = np.mean(self._simulated_data[:, sim_idx])
                label = f'Day {x}: ${mean_price:.2f} (mean)'
            else:
                return
        else:
            return
        
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

