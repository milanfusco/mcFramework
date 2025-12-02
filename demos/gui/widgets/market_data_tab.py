"""
Market Data tab widget.

This module provides the Market Data tab displaying current price,
estimated parameters, and a sparkline of recent closes.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .charts import CandlestickChart, SparklineChart
from .empty_state import MarketDataEmptyState

if TYPE_CHECKING:
    from ..models.state import MarketParameters, TickerAnalysisState


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
        self.setMinimumWidth(110)
        self.setMinimumHeight(85)
        # Allow cards to shrink/expand proportionally
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(8, 8, 8, 8)

        # Title - allow word wrap for long titles
        self._title_label = QLabel(title)
        self._title_label.setObjectName("metricTitle")
        self._title_label.setWordWrap(True)
        self._title_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._title_label)

        # Value - allow word wrap for large numbers
        self._value_label = QLabel(value)
        self._value_label.setObjectName("metricValue")
        self._value_label.setWordWrap(True)
        layout.addWidget(self._value_label)

        # Subtitle (optional)
        if subtitle:
            self._subtitle_label = QLabel(subtitle)
            self._subtitle_label.setObjectName("metricSubtitle")
            self._subtitle_label.setStyleSheet("font-size: 9px;")
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
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

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
        self._sparkline_points = 60
        self._content_widget: QWidget | None = None
        self._current_content_width: int | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the tab UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        self._empty_state = MarketDataEmptyState()
        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(self._create_scrollable_content())
        layout.addWidget(self._stack)
        self._stack.setCurrentIndex(0)

    def _create_scrollable_content(self) -> QWidget:
        """Build the scrollable container that hosts all market widgets."""
        container = QWidget()  # level 1
        container_layout = QVBoxLayout(container)  # level 2
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()  # level 3
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        content = QWidget()  # level 4
        self._content_widget = content
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(32, 16, 32, 16)

        self._build_metric_cards(content_layout)
        self._build_chart_section(content_layout)
        self._build_splitter_section(content_layout)
        self._build_sentiment_section(content_layout)

        content_layout.addStretch()

        scroll_area.setWidget(content)
        container_layout.addWidget(scroll_area, 1)
        return container

    def set_content_width(self, width: int) -> None:
        """Adjust maximum width to match available center panel size."""
        self._current_content_width = width if width > 0 else None
        self._apply_content_width()

    def _apply_content_width(self) -> None:
        """Apply stored width constraints to the scroll content."""
        if self._content_widget is None:
            return
        width = self._current_content_width
        if width is None:
            self._content_widget.setMaximumWidth(16777215)
            self._content_widget.setMinimumWidth(0)
            return
        target = max(720, width - 48)
        self._content_widget.setMaximumWidth(target)
        self._content_widget.setMinimumWidth(min(target, 1100))

    def _build_metric_cards(self, parent_layout: QVBoxLayout) -> None:
        """Create the horizontal row of metric cards."""
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)

        card_defs = [
            ("_price_card", MetricCard("Price", "—", "Last close")),
            ("_drift_card", MetricCard("Drift (μ)", "—", "Annual")),
            ("_vol_card", MetricCard("Volatility (σ)", "—", "Annual")),
            ("_data_points_card", MetricCard("Data Points", "—", "Trading days")),
            ("_dividend_card", MetricCard("Div Yield", "—", "TTM")),
            ("_volume_card", MetricCard("Avg Vol", "—", "10-day")),
        ]

        for attr_name, card in card_defs:
            setattr(self, attr_name, card)
            # Equal stretch factor so cards share width evenly
            metrics_layout.addWidget(card, 1)

        parent_layout.addLayout(metrics_layout)

    def _build_chart_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the candlestick chart section."""
        chart_group = QGroupBox("Candlestick & Volume")
        chart_layout = QVBoxLayout(chart_group)
        self._candlestick = CandlestickChart(max_points=120)
        self._candlestick.setMinimumHeight(380)
        chart_layout.addWidget(self._candlestick)
        parent_layout.addWidget(chart_group)

    def _build_splitter_section(self, parent_layout: QVBoxLayout) -> None:
        """Create splitter containing sparkline and parameters table."""
        splitter = QSplitter(Qt.Orientation.Vertical)

        sparkline_group = QGroupBox("Price History")
        sparkline_layout = QVBoxLayout(sparkline_group)
        self._sparkline = SparklineChart(n_points=self._sparkline_points)
        self._sparkline.setMinimumHeight(240)
        sparkline_layout.addWidget(self._sparkline)
        splitter.addWidget(sparkline_group)

        params_group = QGroupBox("Estimated Parameters")
        params_layout = QVBoxLayout(params_group)
        self._params_table = ParametersTable()
        self._params_table.setMinimumHeight(265)
        params_layout.addWidget(self._params_table)
        splitter.addWidget(params_group)

        parent_layout.addWidget(splitter, 1)

    def _build_sentiment_section(self, parent_layout: QVBoxLayout) -> None:
        """Create analyst sentiment widgets."""
        sentiment_group = QGroupBox("Analyst Sentiment")
        sentiment_layout = QVBoxLayout(sentiment_group)

        self._recommendations_period = QLabel("Period: —")
        self._recommendations_period.setObjectName("recommendationsPeriod")

        self._recs_table = QTableWidget(0, 2)
        self._recs_table.setMinimumHeight(205)
        self._recs_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._recs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._recs_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._recs_table.verticalHeader().setVisible(False)
        self._recs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self._price_target_label = QLabel("No analyst price targets available.")
        self._price_target_label.setObjectName("priceTargetLabel")

        sentiment_layout.addWidget(self._recommendations_period)
        sentiment_layout.addWidget(self._recs_table)
        sentiment_layout.addWidget(self._price_target_label)
        parent_layout.addWidget(sentiment_group)

    def update_from_state(self, state: "TickerAnalysisState") -> None:
        """
        Update all components from the application state.

        Args:
            state: Current application state
        """
        # Check if we have data
        has_data = state.prices is not None and len(state.prices) > 0

        if has_data:
            self._show_content()
            self._update_charts(state)
            self._data_points_card.set_value(str(len(state.prices)))
            self._update_market_metrics(state)
            self._update_metric_cards(state.parameters)
            self._update_recommendations_table(state.recommendations)
            self._update_price_targets_label(state.price_targets)
        else:
            self._show_empty_state()

    def clear(self) -> None:
        """Clear all displayed data and show empty state."""
        self._sparkline.clear()
        self._candlestick.clear()
        self._reset_metric_cards()
        self._recs_table.setRowCount(0)
        self._recommendations_period.setText("Period: —")
        self._price_target_label.setText("No analyst price targets available.")
        self._params_table.clear()
        self._has_data = False
        self._stack.setCurrentIndex(0)

    def set_fetch_callback(self, callback) -> None:
        """Set callback for the empty state action button."""
        self._empty_state.set_action_callback(callback)

    def _show_content(self) -> None:
        """Display populated content view."""
        if self._stack.currentIndex() != 1:
            self._stack.setCurrentIndex(1)
        self._has_data = True

    def _show_empty_state(self) -> None:
        """Display the empty-state view."""
        if self._stack.currentIndex() != 0:
            self._stack.setCurrentIndex(0)
        self._has_data = False

    def _update_charts(self, state: "TickerAnalysisState") -> None:
        """Refresh sparkline and candlestick charts."""
        events = self._build_event_markers(state)
        self._sparkline.update_data(
            state.prices,
            state.config.ticker,
            dates=state.price_dates,
            events=events,
        )
        self._candlestick.update_data(
            opens=state.open_prices,
            highs=state.high_prices,
            lows=state.low_prices,
            closes=state.prices,
            volumes=state.volumes,
            dates=state.price_dates,
        )

    def _update_metric_cards(self, params: "MarketParameters" | None) -> None:
        """Refresh the core metric cards and parameters table."""
        if params is None:
            return

        self._price_card.set_value(f"${params.spot_price:.2f}")

        drift_pct = params.drift * 100
        self._drift_card.set_value(f"{drift_pct:+.2f}%")
        drift_color = "#00d26a" if params.drift >= 0 else "#f23645"
        self._drift_card.set_color(drift_color)

        vol_pct = params.volatility * 100
        self._vol_card.set_value(f"{vol_pct:.2f}%")

        self._params_table.update_parameters(params)

    def _reset_metric_cards(self) -> None:
        """Return metric cards to default state."""
        self._price_card.set_value("—")
        self._drift_card.set_value("—")
        self._drift_card.set_color("#ffffff")
        self._vol_card.set_value("—")
        self._data_points_card.set_value("—")
        self._dividend_card.set_value("—", subtitle="Trailing 12M", animate=False)
        self._volume_card.set_value("—", animate=False)

    def _build_event_markers(self, state: "TickerAnalysisState") -> list[tuple[int, str]]:
        """Convert dividend/split events into sparkline annotations."""
        if state.price_dates is None:
            return []
        window = min(self._sparkline_points, len(state.price_dates))
        recent_dates = state.price_dates[-window:]
        date_to_index = {recent_dates[idx].date(): idx for idx in range(len(recent_dates))}
        markers: list[tuple[int, str]] = []
        for event in state.dividends or []:
            event_date = event.get("date")
            if not isinstance(event_date, datetime):
                continue
            idx = date_to_index.get(event_date.date())
            if idx is not None:
                markers.append((idx, "Div"))
        for event in state.splits or []:
            event_date = event.get("date")
            if not isinstance(event_date, datetime):
                continue
            idx = date_to_index.get(event_date.date())
            if idx is not None:
                markers.append((idx, "Split"))
        return markers

    def _update_market_metrics(self, state: "TickerAnalysisState") -> None:
        """Update dividend and volume cards from state extras."""
        fast_info = state.fast_info or {}
        dividend_yield = fast_info.get("dividendYield")
        last_dividend = (state.dividends or [])[-1] if state.dividends else None
        subtitle = (
            f"Last: {last_dividend['date'].strftime('%b %d, %Y')}"
            if last_dividend and isinstance(last_dividend.get("date"), datetime)
            else "Trailing 12M"
        )
        if dividend_yield is not None:
            self._dividend_card.set_value(
                f"{dividend_yield * 100:.2f}%",
                subtitle=subtitle,
                animate=False,
            )
        elif last_dividend and last_dividend.get("amount") is not None:
            amount = last_dividend["amount"]
            self._dividend_card.set_value(
                f"${amount:.2f}",
                subtitle=subtitle,
                animate=False,
            )
        else:
            self._dividend_card.set_value("—", subtitle="No dividends", animate=False)

        avg_volume = (
            fast_info.get("tenDayAverageVolume")
            or fast_info.get("tenDayAverageVolume3Month")
            or fast_info.get("threeMonthAverageVolume")
        )
        if avg_volume:
            self._volume_card.set_value(self._format_compact_number(avg_volume), animate=False)
        elif state.volumes is not None and len(state.volumes) > 0:
            recent_vol = float(np.nanmean(state.volumes[-10:]))
            self._volume_card.set_value(self._format_compact_number(recent_vol), animate=False)
        else:
            self._volume_card.set_value("—", animate=False)

    def _update_recommendations_table(self, data: dict[str, Any] | None) -> None:
        """Populate the recommendations table with yfinance summary data."""
        self._recs_table.setRowCount(0)
        if not data:
            self._recommendations_period.setText("Period: —")
            return
        self._recommendations_period.setText(f"Period: {data.get('period', '—')}")
        for key, value in data.items():
            if key == "period":
                continue
            row = self._recs_table.rowCount()
            self._recs_table.insertRow(row)
            metric_item = QTableWidgetItem(self._format_key(key))
            metric_item.setFlags(metric_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            value_item = QTableWidgetItem(self._format_number(value))
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._recs_table.setItem(row, 0, metric_item)
            self._recs_table.setItem(row, 1, value_item)

    def _update_price_targets_label(self, targets: dict[str, Any] | None) -> None:
        """Update analyst price target summary label."""
        if not targets:
            self._price_target_label.setText("No analyst price targets available.")
            return
        parts: list[str] = []
        if targets.get("targetMean") is not None:
            parts.append(f"Mean ${targets['targetMean']:,.2f}")
        if targets.get("targetMedian") is not None:
            parts.append(f"Median ${targets['targetMedian']:,.2f}")
        range_parts: list[str] = []
        if targets.get("targetHigh") is not None:
            range_parts.append(f"High ${targets['targetHigh']:,.2f}")
        if targets.get("targetLow") is not None:
            range_parts.append(f"Low ${targets['targetLow']:,.2f}")
        text = " | ".join(parts)
        if range_parts:
            range_text = " / ".join(range_parts)
            text = f"{text} ({range_text})" if text else range_text
        self._price_target_label.setText(text or "Analyst targets unavailable.")

    @staticmethod
    def _format_number(value: Any) -> str:
        """Format numeric table values with commas."""
        if value is None:
            return "—"
        if isinstance(value, (int, float, np.number, np.generic)):
            return f"{float(value):,.2f}"
        return str(value)

    @staticmethod
    def _format_compact_number(value: float | int) -> str:
        """Format large numbers compactly (e.g., 44.9M, 1.2B)."""
        if value is None:
            return "—"
        value = float(value)
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:,.0f}"

    @staticmethod
    def _format_key(key: str) -> str:
        """Convert camelCase/underscored keys into readable labels."""
        if not key:
            return ""
        lowered = key.lower()
        for prefix in ("rating", "target"):
            if lowered.startswith(prefix):
                key = key[len(prefix) :]
                break
        return key.replace("_", " ").strip().title()
