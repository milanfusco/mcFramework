"""
Monte Carlo simulation results tab with interactive charts.

This module provides the Monte Carlo tab displaying simulation charts
including historical vs simulated paths, return distributions, and
forecast distributions using embedded matplotlib FigureCanvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .charts import (
    ForecastDistributionChart,
    HistoricalVsSimulatedChart,
    ReturnDistributionChart,
)
from .empty_state import MonteCarloEmptyState

if TYPE_CHECKING:
    from ..models.state import TickerAnalysisState


class ForecastSummary(QGroupBox):
    """
    Summary widget showing forecast statistics.
    
    Displays key statistics from the Monte Carlo simulation including
    mean forecast, confidence intervals, and probability estimates.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the forecast summary."""
        super().__init__("Forecast Summary", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the summary UI."""
        layout = QGridLayout(self)
        layout.setSpacing(8)
        
        # Labels for statistics
        self._stats_labels: dict[str, QLabel] = {}
        
        stats = [
            ("current", "Current Price:"),
            ("mean", "Mean Forecast:"),
            ("median", "Median Forecast:"),
            ("p5", "5th Percentile:"),
            ("p95", "95th Percentile:"),
            ("prob_up", "P(Price > Current):"),
        ]
        
        for i, (key, label_text) in enumerate(stats):
            label = QLabel(label_text)
            value = QLabel("—")
            value.setObjectName("summaryValue")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            
            layout.addWidget(label, i, 0)
            layout.addWidget(value, i, 1)
            
            self._stats_labels[key] = value

    def update_stats(
        self,
        current: float,
        mean: float,
        median: float,
        p5: float,
        p95: float,
        prob_up: float,
    ) -> None:
        """
        Update the displayed statistics.
        
        Args:
            current: Current stock price
            mean: Mean forecasted price
            median: Median forecasted price
            p5: 5th percentile
            p95: 95th percentile
            prob_up: Probability of price increase
        """
        self._stats_labels["current"].setText(f"${current:.2f}")
        self._stats_labels["mean"].setText(f"${mean:.2f}")
        self._stats_labels["median"].setText(f"${median:.2f}")
        self._stats_labels["p5"].setText(f"${p5:.2f}")
        self._stats_labels["p95"].setText(f"${p95:.2f}")
        self._stats_labels["prob_up"].setText(f"{prob_up:.1f}%")
        
        # Color the probability
        color = '#00d26a' if prob_up >= 50 else '#f23645'
        self._stats_labels["prob_up"].setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )

    def clear(self) -> None:
        """Clear all statistics."""
        for label in self._stats_labels.values():
            label.setText("—")
            label.setStyleSheet("")


class ChartExportPanel(QFrame):
    """
    Panel with export buttons for all charts.
    
    Provides quick access to save any chart as PNG.
    
    Signals:
        export_requested: Emitted with chart name when export is clicked
    """
    
    export_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the export panel."""
        super().__init__(parent)
        self.setObjectName("chartExportPanel")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the export panel UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        label = QLabel("Export:")
        label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(label)
        
        charts = [
            ("historical", "Historical vs Simulated"),
            ("returns", "Return Distribution"),
            ("forecast", "Forecast Distribution"),
        ]
        
        for chart_id, name in charts:
            btn = QPushButton(name)
            btn.setObjectName("exportBtn")
            btn.setToolTip(f"Export {name} chart as PNG")
            btn.clicked.connect(lambda checked, cid=chart_id: self.export_requested.emit(cid))
            layout.addWidget(btn)
        
        layout.addStretch()


class MonteCarloTab(QWidget):
    """
    Monte Carlo results tab with interactive simulation charts.
    
    This tab displays the main simulation visualizations using embedded
    matplotlib figures that support zoom, pan, and hover interactions:
    - Historical vs Simulated paths
    - Return distributions (historical vs simulated)
    - Forecast price distribution
    
    Shows an empty state when no simulation has been run.
    
    Signals:
        chart_export_requested: Emitted with (chart_name, export_func)
    """
    
    chart_export_requested = Signal(str, object)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the Monte Carlo tab."""
        super().__init__(parent)
        self._has_data = False
        self._content_widget: QWidget | None = None
        self._scroll_content: QWidget | None = None
        self._middle_splitter: QSplitter | None = None
        self._current_content_width: int | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the tab UI with interactive charts."""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for empty/content states
        self._stack = QStackedWidget()
        
        # Empty state (index 0)
        self._empty_state = MonteCarloEmptyState()
        self._stack.addWidget(self._empty_state)
        
        # Content widget (index 1)
        content = QWidget()
        self._content_widget = content
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(8)
        content_layout.setContentsMargins(8, 8, 8, 8)
        
        # Export panel at top
        self._export_panel = ChartExportPanel()
        self._export_panel.export_requested.connect(self._on_export_requested)
        content_layout.addWidget(self._export_panel)
        
        # Scroll area for charts
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        self._scroll_content = scroll_content
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Historical vs Simulated chart (interactive)
        hist_group = QGroupBox("Historical vs Simulated Paths")
        hist_layout = QVBoxLayout(hist_group)
        hist_layout.setContentsMargins(4, 12, 4, 4)
        self._hist_chart = HistoricalVsSimulatedChart()
        self._hist_chart.setMinimumHeight(350)
        hist_layout.addWidget(self._hist_chart)
        scroll_layout.addWidget(hist_group)
        
        # Returns and Summary side by side
        middle_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._middle_splitter = middle_splitter
        
        # Return distributions chart (interactive)
        returns_group = QGroupBox("Return Distributions")
        returns_layout = QVBoxLayout(returns_group)
        returns_layout.setContentsMargins(4, 12, 4, 4)
        self._returns_chart = ReturnDistributionChart()
        self._returns_chart.setMinimumHeight(280)
        returns_layout.addWidget(self._returns_chart)
        middle_splitter.addWidget(returns_group)
        
        # Forecast summary
        self._forecast_summary = ForecastSummary()
        self._forecast_summary.setMaximumWidth(280)
        middle_splitter.addWidget(self._forecast_summary)
        
        middle_splitter.setSizes([700, 280])
        scroll_layout.addWidget(middle_splitter)
        
        # Forecast distribution chart (interactive)
        forecast_group = QGroupBox("Forecast Price Distribution")
        forecast_layout = QVBoxLayout(forecast_group)
        forecast_layout.setContentsMargins(4, 12, 4, 4)
        self._forecast_chart = ForecastDistributionChart()
        self._forecast_chart.setMinimumHeight(300)
        forecast_layout.addWidget(self._forecast_chart)
        scroll_layout.addWidget(forecast_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        content_layout.addWidget(scroll, 1)
        
        self._stack.addWidget(content)
        layout.addWidget(self._stack)
        
        # Start with empty state
        self._stack.setCurrentIndex(0)

    def set_content_width(self, width: int) -> None:
        """Allow parent window to broadcast available width."""
        self._current_content_width = width if width > 0 else None
        self._apply_content_width()

    def _apply_content_width(self) -> None:
        """Clamp scroll content width based on available space."""
        width = self._current_content_width
        if self._content_widget:
            if width is None:
                self._content_widget.setMaximumWidth(16777215)
            else:
                self._content_widget.setMaximumWidth(width)
        if self._scroll_content and width is not None:
            self._scroll_content.setMaximumWidth(width)
        if self._middle_splitter and width is not None:
            right = self._forecast_summary.maximumWidth() or 280
            left = max(400, width - right - 48)
            self._middle_splitter.setSizes([left, right])

    def _on_export_requested(self, chart_name: str) -> None:
        """Handle chart export request."""
        chart_map = {
            "historical": self._hist_chart,
            "returns": self._returns_chart,
            "forecast": self._forecast_chart,
        }
        
        chart = chart_map.get(chart_name)
        if chart:
            self.chart_export_requested.emit(chart_name, chart.export_to_png)

    def update_from_state(self, state: "TickerAnalysisState") -> None:
        """
        Update all charts from the application state.
        
        Args:
            state: Current application state
        """
        # Check if we have the required data
        if not state.has_market_data() or not state.has_simulation_results():
            self._stack.setCurrentIndex(0)
            self._has_data = False
            return
        
        # Show content
        self._stack.setCurrentIndex(1)
        self._has_data = True
        
        prices = state.prices
        paths = state.simulated_paths
        ticker = state.config.ticker.upper()
        horizon = state.config.forecast_horizon
        
        # Update Historical vs Simulated chart
        self._hist_chart.update_data(
            historical=prices,
            simulated=paths,
            ticker=ticker,
            forecast_horizon=horizon,
        )
        
        # Update Return Distribution chart
        self._returns_chart.update_data(
            historical_prices=prices,
            simulated_paths=paths,
            ticker=ticker,
        )
        
        # Update Forecast Distribution chart
        current_price = state.get_current_price()
        self._forecast_chart.update_data(
            simulated_paths=paths,
            current_price=current_price,
            ticker=ticker,
            forecast_horizon=horizon,
        )
        
        # Update summary statistics
        final_prices = paths[:, -1]
        self._forecast_summary.update_stats(
            current=current_price,
            mean=float(np.mean(final_prices)),
            median=float(np.median(final_prices)),
            p5=float(np.percentile(final_prices, 5)),
            p95=float(np.percentile(final_prices, 95)),
            prob_up=float(100 * np.mean(final_prices > current_price)),
        )

    def clear(self) -> None:
        """Clear all charts and show empty state."""
        self._hist_chart.clear()
        self._returns_chart.clear()
        self._forecast_chart.clear()
        self._forecast_summary.clear()
        self._has_data = False
        self._stack.setCurrentIndex(0)

    def set_run_callback(self, callback) -> None:
        """Set callback for the empty state action button."""
        self._empty_state.set_action_callback(callback)

    def export_chart(self, chart_name: str, path: Path) -> bool:
        """
        Export a specific chart to PNG.
        
        Args:
            chart_name: Chart identifier ("historical", "returns", "forecast")
            path: Output file path
            
        Returns:
            True if exported successfully
        """
        chart_map = {
            "historical": self._hist_chart,
            "returns": self._returns_chart,
            "forecast": self._forecast_chart,
        }
        
        chart = chart_map.get(chart_name)
        if chart:
            try:
                chart.export_to_png(path)
                return True
            except Exception:
                return False
        return False
