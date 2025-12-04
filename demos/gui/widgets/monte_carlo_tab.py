"""
Monte Carlo simulation results tab with interactive charts.

This module provides the Monte Carlo tab displaying simulation charts
including historical vs simulated paths, return distributions, and
forecast distributions using embedded matplotlib FigureCanvas.

Integrates with StatsEngine to display comprehensive statistical analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .charts import (
    ForecastDistributionChart,
    HistoricalVsSimulatedChart,
    ReturnDistributionChart,
)
from .empty_state import MonteCarloEmptyState
from .stats_panel import StatsPanel

if TYPE_CHECKING:
    from ..models.state import TickerAnalysisState


class SummaryStatCard(QFrame):
    """Styled card for individual summary statistics."""
    
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("summaryStatCard")
        self.setStyleSheet("""
            QFrame#summaryStatCard {
                background-color: #252542;
                border-radius: 8px;
                border: 1px solid #3a3a55;
            }
            QFrame#summaryStatCard:hover {
                border: 1px solid #4a6fa5;
                background-color: #2a2a4a;
            }
        """)
        self._setup_ui(title)
    
    def _setup_ui(self, title: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(14, 12, 14, 12)
        
        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #8a8a9a; font-size: 11px; font-weight: 500;"
        )
        layout.addWidget(self._title)
        
        self._value = QLabel("—")
        self._value.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e0e0e0;"
        )
        layout.addWidget(self._value)
    
    def set_value(self, value: str, color: str | None = None) -> None:
        self._value.setText(value)
        if color:
            self._value.setStyleSheet(
                f"font-size: 20px; font-weight: bold; color: {color};"
            )
        else:
            self._value.setStyleSheet(
                "font-size: 20px; font-weight: bold; color: #e0e0e0;"
            )
    
    def clear(self) -> None:
        self._value.setText("—")
        self._value.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e0e0e0;"
        )


class ProbabilityIndicator(QFrame):
    """Visual probability indicator with bar and percentage."""
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("probIndicator")
        self._prob = 50.0
        self._bar_container: QFrame | None = None
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QFrame#probIndicator {
                background-color: #1f2f3f;
                border-radius: 10px;
                border: 1px solid #2a4a6a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 14, 16, 14)
        
        # Title
        title = QLabel("Probability of Price Increase")
        title.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Percentage display
        self._pct_label = QLabel("50.0%")
        self._pct_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #e0e0e0;"
        )
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._pct_label)
        
        # Visual bar container
        self._bar_container = QFrame()
        self._bar_container.setFixedHeight(16)
        self._bar_container.setStyleSheet(
            "background-color: #1a1a2a; border-radius: 8px;"
        )
        bar_layout = QHBoxLayout(self._bar_container)
        bar_layout.setContentsMargins(2, 2, 2, 2)
        bar_layout.setSpacing(0)
        
        self._bar_fill = QFrame()
        self._bar_fill.setStyleSheet(
            "background-color: #5aa0e5; border-radius: 6px;"
        )
        self._bar_fill.setFixedHeight(12)
        bar_layout.addWidget(self._bar_fill)
        bar_layout.addStretch()
        
        layout.addWidget(self._bar_container)
        
        # Interpretation label
        self._interpretation = QLabel("Neutral outlook")
        self._interpretation.setStyleSheet(
            "color: #7a8a9a; font-size: 12px; font-weight: 500;"
        )
        self._interpretation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._interpretation)
    
    def set_probability(self, prob: float) -> None:
        """Set the probability value (0-100)."""
        self._prob = prob
        self._pct_label.setText(f"{prob:.1f}%")
        
        # Update bar width
        if self._bar_container:
            bar_width = self._bar_container.width() - 4  # Account for margins
            self._bar_fill.setFixedWidth(max(0, int(bar_width * prob / 100)))
        
        # Color based on probability
        if prob >= 65:
            color = "#2adf7a"
            interpretation = "Bullish outlook"
        elif prob >= 50:
            color = "#5aa0e5"
            interpretation = "Slightly bullish"
        elif prob >= 35:
            color = "#f0b90b"
            interpretation = "Slightly bearish"
        else:
            color = "#ff5a6a"
            interpretation = "Bearish outlook"
        
        self._pct_label.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {color};"
        )
        self._bar_fill.setStyleSheet(
            f"background-color: {color}; border-radius: 6px;"
        )
        self._interpretation.setText(interpretation)
        self._interpretation.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: 500;"
        )
    
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Update bar width on resize
        if self._bar_container:
            bar_width = self._bar_container.width() - 4
            self._bar_fill.setFixedWidth(max(0, int(bar_width * self._prob / 100)))
    
    def clear(self) -> None:
        self._prob = 50.0
        self._pct_label.setText("—")
        self._pct_label.setStyleSheet(
            "font-size: 36px; font-weight: bold; color: #e0e0e0;"
        )
        self._bar_fill.setFixedWidth(0)
        self._interpretation.setText("")
        self._interpretation.setStyleSheet(
            "color: #7a8a9a; font-size: 12px; font-weight: 500;"
        )


class ForecastSummary(QWidget):
    """
    Summary widget showing forecast statistics.
    
    Displays key statistics from the Monte Carlo simulation including
    mean forecast, confidence intervals, and probability estimates
    with styled cards matching the Full Stats panel.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the forecast summary."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the summary UI with styled cards."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 20, 16, 16)
        
        # Current Price Section
        current_label = QLabel("Current Price")
        current_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(current_label)
        
        self._current_card = SummaryStatCard("Spot Price")
        self._current_card.setMinimumHeight(70)
        layout.addWidget(self._current_card)
        
        # Spacer
        layout.addSpacing(8)
        
        # Forecast Section
        forecast_label = QLabel("Forecast Statistics")
        forecast_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(forecast_label)
        
        # Mean and Median in a row
        forecast_row = QHBoxLayout()
        forecast_row.setSpacing(12)
        
        self._mean_card = SummaryStatCard("Mean")
        self._mean_card.setMinimumHeight(70)
        forecast_row.addWidget(self._mean_card)
        
        self._median_card = SummaryStatCard("Median")
        self._median_card.setMinimumHeight(70)
        forecast_row.addWidget(self._median_card)
        
        layout.addLayout(forecast_row)
        
        # Spacer
        layout.addSpacing(8)
        
        # Percentiles Section
        pct_label = QLabel("90% Range (P5 – P95)")
        pct_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(pct_label)
        
        pct_row = QHBoxLayout()
        pct_row.setSpacing(12)
        
        self._p5_card = SummaryStatCard("5th Percentile")
        self._p5_card.setMinimumHeight(70)
        pct_row.addWidget(self._p5_card)
        
        self._p95_card = SummaryStatCard("95th Percentile")
        self._p95_card.setMinimumHeight(70)
        pct_row.addWidget(self._p95_card)
        
        layout.addLayout(pct_row)
        
        # Spacer
        layout.addSpacing(12)
        
        # Probability Indicator (give it more room)
        self._prob_indicator = ProbabilityIndicator()
        self._prob_indicator.setMinimumHeight(120)
        layout.addWidget(self._prob_indicator)
        
        layout.addStretch()

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
        self._current_card.set_value(f"${current:,.2f}", color="#5aa0e5")
        
        # Color mean/median based on comparison to current
        mean_color = "#2adf7a" if mean > current else "#ff5a6a" if mean < current else "#e0e0e0"
        median_color = "#2adf7a" if median > current else "#ff5a6a" if median < current else "#e0e0e0"
        
        self._mean_card.set_value(f"${mean:,.2f}", color=mean_color)
        self._median_card.set_value(f"${median:,.2f}", color=median_color)
        
        # Percentiles with colors
        self._p5_card.set_value(f"${p5:,.2f}", color="#f0b90b")
        self._p95_card.set_value(f"${p95:,.2f}", color="#2adf7a")
        
        # Update probability indicator
        self._prob_indicator.set_probability(prob_up)

    def clear(self) -> None:
        """Clear all statistics."""
        self._current_card.clear()
        self._mean_card.clear()
        self._median_card.clear()
        self._p5_card.clear()
        self._p95_card.clear()
        self._prob_indicator.clear()


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
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Export panel at top (compact)
        self._export_panel = ChartExportPanel()
        self._export_panel.export_requested.connect(self._on_export_requested)
        content_layout.addWidget(self._export_panel)
        
        # Main splitter: Charts on left, Stats on right
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter = main_splitter
        
        # ===== LEFT SIDE: All Charts =====
        charts_scroll = QScrollArea()
        charts_scroll.setWidgetResizable(True)
        charts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        charts_scroll.setStyleSheet("QScrollArea { border: none; }")
        
        charts_container = QWidget()
        self._scroll_content = charts_container
        charts_layout = QVBoxLayout(charts_container)
        charts_layout.setSpacing(16)
        charts_layout.setContentsMargins(12, 12, 8, 12)
        
        # Historical vs Simulated chart
        hist_group = QGroupBox("Historical vs Simulated Paths")
        hist_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                padding-top: 12px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        hist_layout = QVBoxLayout(hist_group)
        hist_layout.setContentsMargins(8, 20, 8, 8)
        self._hist_chart = HistoricalVsSimulatedChart()
        self._hist_chart.setMinimumHeight(420)
        hist_layout.addWidget(self._hist_chart)
        charts_layout.addWidget(hist_group)
        
        # Return distributions chart
        returns_group = QGroupBox("Return Distributions")
        returns_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                padding-top: 12px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        returns_layout = QVBoxLayout(returns_group)
        returns_layout.setContentsMargins(8, 20, 8, 8)
        self._returns_chart = ReturnDistributionChart()
        self._returns_chart.setMinimumHeight(380)
        returns_layout.addWidget(self._returns_chart)
        charts_layout.addWidget(returns_group)
        
        # Forecast distribution chart
        forecast_group = QGroupBox("Forecast Price Distribution")
        forecast_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                padding-top: 12px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        forecast_layout = QVBoxLayout(forecast_group)
        forecast_layout.setContentsMargins(8, 20, 8, 8)
        self._forecast_chart = ForecastDistributionChart()
        self._forecast_chart.setMinimumHeight(400)
        forecast_layout.addWidget(self._forecast_chart)
        charts_layout.addWidget(forecast_group)
        
        charts_layout.addStretch()
        charts_scroll.setWidget(charts_container)
        main_splitter.addWidget(charts_scroll)
        
        # ===== RIGHT SIDE: Statistics Panel =====
        stats_container = QWidget()
        stats_container.setMinimumWidth(360)
        stats_container.setMaximumWidth(480)
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(8, 12, 12, 12)
        stats_layout.setSpacing(0)
        
        # Stats tabs with better styling
        self._stats_tabs = QTabWidget()
        self._stats_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._stats_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3a3a55;
                border-radius: 8px;
                background-color: #1a1a2e;
            }
            QTabBar::tab {
                background-color: #252542;
                color: #8a8a9a;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #1a1a2e;
                color: #e0e0e0;
                border: 1px solid #3a3a55;
                border-bottom: none;
            }
            QTabBar::tab:hover:!selected {
                background-color: #2a2a4a;
                color: #c0c0d0;
            }
        """)
        
        # Summary tab with scroll
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        summary_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._forecast_summary = ForecastSummary()
        summary_scroll.setWidget(self._forecast_summary)
        self._stats_tabs.addTab(summary_scroll, "📊 Summary")
        
        # Full Statistics tab with scroll
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._stats_panel = StatsPanel("Detailed Statistics")
        stats_scroll.setWidget(self._stats_panel)
        self._stats_tabs.addTab(stats_scroll, "📈 Full Stats")
        
        stats_layout.addWidget(self._stats_tabs)
        main_splitter.addWidget(stats_container)
        
        # Set initial splitter sizes (70% charts, 30% stats)
        main_splitter.setSizes([700, 400])
        main_splitter.setStretchFactor(0, 2)  # Charts stretch more
        main_splitter.setStretchFactor(1, 1)  # Stats stretch less
        
        # Keep reference for width adjustments
        self._middle_splitter = main_splitter
        
        content_layout.addWidget(main_splitter, 1)
        
        self._stack.addWidget(content)
        layout.addWidget(self._stack)
        
        # Start with empty state
        self._stack.setCurrentIndex(0)

    def set_content_width(self, width: int) -> None:
        """Allow parent window to broadcast available width."""
        self._current_content_width = width if width > 0 else None
        self._apply_content_width()

    def _apply_content_width(self) -> None:
        """Adjust splitter proportions based on available space."""
        width = self._current_content_width
        if self._content_widget:
            if width is None:
                self._content_widget.setMaximumWidth(16777215)
            else:
                self._content_widget.setMaximumWidth(width)
        
        if self._middle_splitter and width is not None:
            # Responsive layout: adjust stats panel width based on available space
            if width > 1400:
                # Wide screen: give stats more room
                stats_width = min(480, int(width * 0.30))
            elif width > 1000:
                # Medium screen: balanced
                stats_width = min(420, int(width * 0.35))
            else:
                # Narrow screen: stats panel takes more proportion
                stats_width = min(380, int(width * 0.40))
            
            charts_width = max(500, width - stats_width - 24)
            self._middle_splitter.setSizes([charts_width, stats_width])

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
        
        # Update summary statistics (quick view)
        final_prices = paths[:, -1]
        self._forecast_summary.update_stats(
            current=current_price,
            mean=float(np.mean(final_prices)),
            median=float(np.median(final_prices)),
            p5=float(np.percentile(final_prices, 5)),
            p95=float(np.percentile(final_prices, 95)),
            prob_up=float(100 * np.mean(final_prices > current_price)),
        )
        
        # Update full statistics panel if stats are enabled
        self._update_stats_panel(state, final_prices)

    def _update_stats_panel(
        self,
        state: "TickerAnalysisState",
        final_prices: np.ndarray,
    ) -> None:
        """
        Update the full statistics panel using StatsEngine.
        
        Args:
            state: Application state with stats configuration
            final_prices: Array of simulated final prices
        """
        stats_cfg = state.config.stats
        
        if not stats_cfg.compute_stats:
            self._stats_panel.clear()
            return
        
        try:
            from mcframework.stats_engine import StatsContext, build_default_engine
            
            # Build StatsContext from config
            ctx = StatsContext(
                n=len(final_prices),
                confidence=stats_cfg.confidence,
                ci_method=stats_cfg.ci_method,
                bootstrap=stats_cfg.bootstrap_method,
                n_bootstrap=stats_cfg.n_bootstrap,
                nan_policy=stats_cfg.nan_policy,
                percentiles=stats_cfg.percentiles,
                target=state.get_current_price(),
            )
            
            # Build engine with appropriate flags
            engine = build_default_engine(
                include_dist_free=stats_cfg.enable_chebyshev_ci,
                include_target_bounds=True,
            )
            
            # Select which metrics to compute based on config
            select = [
                "mean", "std", "skew", "kurtosis", "percentiles", "ci_mean",
            ]
            if stats_cfg.enable_bootstrap_ci:
                select.append("ci_mean_bootstrap")
            if stats_cfg.enable_chebyshev_ci:
                select.append("ci_mean_chebyshev")
            
            # Compute statistics
            result = engine.compute(final_prices, ctx, select=select)
            
            # Store in state for other components
            state.forecast_stats = result
            
            # Update the panel
            self._stats_panel.update_from_result(result, stats_cfg.confidence)
            
        except ImportError:
            # StatsEngine not available, clear the panel
            self._stats_panel.clear()
        except Exception as e:
            # Log error but don't crash
            print(f"Error computing statistics: {e}")
            self._stats_panel.clear()

    def clear(self) -> None:
        """Clear all charts and show empty state."""
        self._hist_chart.clear()
        self._returns_chart.clear()
        self._forecast_chart.clear()
        self._forecast_summary.clear()
        self._stats_panel.clear()
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
