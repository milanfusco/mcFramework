"""
Market Data tab widget.

This module provides the Market Data tab displaying current price,
estimated parameters, and a sparkline of recent closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Matplotlib imports for embedded charts
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .empty_state import MarketDataEmptyState

if TYPE_CHECKING:
    from ..models.state import MarketParameters, TickerAnalysisState


class SparklineCanvas(FigureCanvas):
    """
    Matplotlib canvas for displaying a sparkline chart.
    
    This lightweight chart shows price history in a compact form
    suitable for the Market Data tab.
    """

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the sparkline canvas.
        
        Args:
            parent: Optional parent widget
        """
        self._figure = Figure(figsize=(6, 2), dpi=100)
        self._figure.patch.set_facecolor('#1a1a2e')
        
        super().__init__(self._figure)
        self.setParent(parent)
        
        self._axes = self._figure.add_subplot(111)
        self._configure_axes()

    def _configure_axes(self) -> None:
        """Configure axes styling for dark theme."""
        self._axes.set_facecolor('#1a1a2e')
        self._axes.tick_params(colors='#888888', labelsize=8)
        self._axes.spines['top'].set_visible(False)
        self._axes.spines['right'].set_visible(False)
        self._axes.spines['bottom'].set_color('#444444')
        self._axes.spines['left'].set_color('#444444')
        self._axes.grid(True, alpha=0.2, color='#444444')

    def update_data(
        self,
        prices: np.ndarray,
        ticker: str = "",
        n_points: int = 60,
    ) -> None:
        """
        Update the sparkline with new price data.
        
        Args:
            prices: Array of historical prices
            ticker: Ticker symbol for title
            n_points: Number of recent points to display
        """
        self._axes.clear()
        self._configure_axes()
        
        if prices is None or len(prices) == 0:
            self._axes.text(
                0.5, 0.5, "No data",
                ha='center', va='center',
                color='#888888', fontsize=12,
                transform=self._axes.transAxes
            )
            self.draw()
            return
        
        # Take last n_points
        display_prices = prices[-n_points:] if len(prices) > n_points else prices
        x = np.arange(len(display_prices))
        
        # Determine color based on trend
        is_up = display_prices[-1] >= display_prices[0]
        color = '#00d26a' if is_up else '#f23645'
        
        # Plot line and fill
        self._axes.plot(x, display_prices, color=color, linewidth=1.5)
        self._axes.fill_between(x, display_prices, alpha=0.2, color=color)
        
        # Add current price annotation
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
        
        # Title
        if ticker:
            self._axes.set_title(
                f'{ticker} - Last {len(display_prices)} Days',
                color='#cccccc',
                fontsize=10,
                loc='left',
            )
        
        self._axes.set_ylabel('Price ($)', color='#888888', fontsize=9)
        self._axes.set_xlabel('Trading Days', color='#888888', fontsize=9)
        
        self._figure.tight_layout()
        self.draw()

    def clear(self) -> None:
        """Clear the sparkline."""
        self._axes.clear()
        self._configure_axes()
        self.draw()


class MetricCard(QFrame):
    """
    Card widget for displaying a single metric.
    
    This component provides consistent styling for key metrics
    displayed in the Market Data tab.
    """

    def __init__(
        self,
        title: str,
        value: str = "—",
        subtitle: str = "",
        parent: QWidget | None = None,
    ):
        """
        Initialize the metric card.
        
        Args:
            title: Metric title
            value: Initial value text
            subtitle: Optional subtitle
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(160)
        self.setMinimumHeight(90)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(16, 14, 16, 14)
        
        # Title
        self._title_label = QLabel(title)
        self._title_label.setObjectName("metricTitle")
        layout.addWidget(self._title_label)
        
        # Value
        self._value_label = QLabel(value)
        self._value_label.setObjectName("metricValue")
        layout.addWidget(self._value_label)
        
        # Subtitle (optional)
        if subtitle:
            self._subtitle_label = QLabel(subtitle)
            self._subtitle_label.setObjectName("metricSubtitle")
            layout.addWidget(self._subtitle_label)
        else:
            self._subtitle_label = None
        
        layout.addStretch()

    def set_value(self, value: str, subtitle: str = "", animate: bool = True) -> None:
        """
        Update the displayed value with optional animation.
        
        Args:
            value: New value text
            subtitle: Optional new subtitle
            animate: Whether to animate the change
        """
        old_value = self._value_label.text()
        self._value_label.setText(value)
        
        if self._subtitle_label and subtitle:
            self._subtitle_label.setText(subtitle)
        
        # Flash animation on value change
        if animate and old_value != value and value != "—":
            self._flash_value()

    def _flash_value(self) -> None:
        """Flash the value label to indicate change."""
        original_style = self._value_label.styleSheet()
        
        # Brief highlight
        self._value_label.setStyleSheet(
            original_style + "background-color: rgba(90, 159, 213, 0.3); "
            "border-radius: 4px; padding: 2px 4px;"
        )
        
        # Reset after delay
        QTimer.singleShot(300, lambda: self._value_label.setStyleSheet(original_style))

    def set_color(self, color: str) -> None:
        """Set the value text color."""
        self._value_label.setStyleSheet(f"color: {color};")


class ParametersTable(QTableWidget):
    """
    Table widget for displaying estimated market parameters.
    
    This table shows the full set of parameters estimated from
    historical data in a structured format.
    """

    PARAMETERS = [
        ("Spot Price (S₀)", "spot_price", "${:.2f}"),
        ("Drift (μ)", "drift", "{:.4f}"),
        ("Drift (Annual %)", "drift", "{:.2f}%", lambda x: x * 100),
        ("Volatility (σ)", "volatility", "{:.4f}"),
        ("Volatility (Annual %)", "volatility", "{:.2f}%", lambda x: x * 100),
        ("Daily Return Mean", "daily_return_mean", "{:.6f}"),
        ("Daily Return Std", "daily_return_std", "{:.6f}"),
    ]

    def __init__(self, parent: QWidget | None = None):
        """Initialize the parameters table."""
        super().__init__(parent)
        
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.setRowCount(len(self.PARAMETERS))
        
        # Style
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        
        # Populate row labels
        for i, (name, _, _, *_) in enumerate(self.PARAMETERS):
            item = QTableWidgetItem(name)
            self.setItem(i, 0, item)
            self.setItem(i, 1, QTableWidgetItem("—"))

    def update_parameters(self, params: "MarketParameters") -> None:
        """
        Update the table with new parameters.
        
        Args:
            params: Market parameters to display
        """
        for i, param_def in enumerate(self.PARAMETERS):
            name, attr, fmt, *transform = param_def
            
            value = getattr(params, attr, 0.0)
            if transform:
                value = transform[0](value)
            
            formatted = fmt.format(value)
            item = QTableWidgetItem(formatted)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(i, 1, item)

    def clear(self) -> None:
        """Clear all parameter values."""
        for i in range(len(self.PARAMETERS)):
            self.setItem(i, 1, QTableWidgetItem("—"))


class MarketDataTab(QWidget):
    """
    Market Data tab displaying price information and sparkline.
    
    This tab provides a quick overview of the current market data:
    - Key metrics in card format
    - Sparkline of recent price history
    - Full parameters table
    
    Shows an empty state when no data is loaded.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the Market Data tab."""
        super().__init__(parent)
        self._has_data = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the tab UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget to toggle between empty state and content
        self._stack = QStackedWidget()
        
        # Empty state (index 0)
        self._empty_state = MarketDataEmptyState()
        self._stack.addWidget(self._empty_state)
        
        # Content widget (index 1)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(16, 16, 16, 16)
        
        # Metrics cards row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        
        self._price_card = MetricCard("Current Price", "—", "Last close")
        self._drift_card = MetricCard("Annual Drift", "—", "Estimated μ")
        self._vol_card = MetricCard("Annual Volatility", "—", "Estimated σ")
        self._data_points_card = MetricCard("Data Points", "—", "Trading days")
        
        metrics_layout.addWidget(self._price_card)
        metrics_layout.addWidget(self._drift_card)
        metrics_layout.addWidget(self._vol_card)
        metrics_layout.addWidget(self._data_points_card)
        
        content_layout.addLayout(metrics_layout)
        
        # Splitter for sparkline and table
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Sparkline group
        sparkline_group = QGroupBox("Price History")
        sparkline_layout = QVBoxLayout(sparkline_group)
        self._sparkline = SparklineCanvas()
        self._sparkline.setMinimumHeight(200)
        sparkline_layout.addWidget(self._sparkline)
        splitter.addWidget(sparkline_group)
        
        # Parameters table group
        params_group = QGroupBox("Estimated Parameters")
        params_layout = QVBoxLayout(params_group)
        self._params_table = ParametersTable()
        params_layout.addWidget(self._params_table)
        splitter.addWidget(params_group)
        
        content_layout.addWidget(splitter, 1)
        
        self._stack.addWidget(content)
        layout.addWidget(self._stack)
        
        # Start with empty state
        self._stack.setCurrentIndex(0)

    def update_from_state(self, state: "TickerAnalysisState") -> None:
        """
        Update all components from the application state.
        
        Args:
            state: Current application state
        """
        # Check if we have data
        has_data = state.prices is not None and len(state.prices) > 0
        
        if has_data:
            # Show content
            self._stack.setCurrentIndex(1)
            self._has_data = True
            
            # Update sparkline
            self._sparkline.update_data(state.prices, state.config.ticker)
            self._data_points_card.set_value(str(len(state.prices)))
            
            # Update metric cards
            if state.parameters is not None:
                params = state.parameters
                
                # Price
                self._price_card.set_value(f"${params.spot_price:.2f}")
                
                # Drift with color
                drift_pct = params.drift * 100
                self._drift_card.set_value(f"{drift_pct:+.2f}%")
                drift_color = '#00d26a' if params.drift >= 0 else '#f23645'
                self._drift_card.set_color(drift_color)
                
                # Volatility
                vol_pct = params.volatility * 100
                self._vol_card.set_value(f"{vol_pct:.2f}%")
                
                # Update table
                self._params_table.update_parameters(params)
        else:
            # Show empty state
            self._stack.setCurrentIndex(0)
            self._has_data = False

    def clear(self) -> None:
        """Clear all displayed data and show empty state."""
        self._sparkline.clear()
        self._price_card.set_value("—")
        self._drift_card.set_value("—")
        self._drift_card.set_color("#ffffff")  # Reset color
        self._vol_card.set_value("—")
        self._data_points_card.set_value("—")
        self._params_table.clear()
        self._has_data = False
        self._stack.setCurrentIndex(0)

    def set_fetch_callback(self, callback) -> None:
        """Set callback for the empty state action button."""
        self._empty_state.set_action_callback(callback)

