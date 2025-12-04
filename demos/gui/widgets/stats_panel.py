"""
Statistical analysis display widgets.

This module provides widgets for displaying comprehensive statistics
computed by the StatsEngine, including descriptive statistics,
percentiles, and multiple confidence interval methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from mcframework.stats_engine import ComputeResult


# =============================================================================
# Styling Constants
# =============================================================================

CARD_STYLE = """
    QFrame#statCard {
        background-color: #252542;
        border-radius: 6px;
        border: 1px solid #3a3a55;
    }
    QFrame#statCard:hover {
        border: 1px solid #4a6fa5;
    }
"""

CI_CARD_STYLE = """
    QFrame#ciCard {
        background-color: #1f2f3f;
        border-radius: 6px;
        border: 1px solid #2a4a6a;
    }
"""

DISABLED_STYLE = """
    QFrame {
        background-color: #1a1a2a;
        border: 1px dashed #3a3a55;
    }
    QLabel {
        color: #5a5a6a;
    }
"""


# =============================================================================
# Individual Statistic Card
# =============================================================================

class StatCard(QFrame):
    """
    Individual statistic display card.
    
    Displays a single statistic with a title, value, and optional subtitle.
    Designed to be compact and fit in narrow panels.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ):
        """
        Initialize the stat card.
        
        Args:
            title: Title label for the statistic
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setStyleSheet(CARD_STYLE)
        self._setup_ui(title)

    def _setup_ui(self, title: str) -> None:
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(10, 8, 10, 8)

        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #8a8a9a; font-size: 10px; font-weight: 500;"
        )
        layout.addWidget(self._title)

        self._value = QLabel("—")
        self._value.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #e0e0e0;"
        )
        self._value.setWordWrap(True)
        layout.addWidget(self._value)

        self._subtitle = QLabel("")
        self._subtitle.setStyleSheet("color: #6a6a7a; font-size: 9px;")
        self._subtitle.setVisible(False)
        layout.addWidget(self._subtitle)

    def set_value(
        self,
        value: str,
        subtitle: str = "",
        color: str | None = None,
    ) -> None:
        """
        Set the displayed value.
        
        Args:
            value: Value text to display
            subtitle: Optional subtitle text
            color: Optional color override for the value
        """
        self._value.setText(value)
        
        if subtitle:
            self._subtitle.setText(subtitle)
            self._subtitle.setVisible(True)
        else:
            self._subtitle.setVisible(False)
        
        if color:
            self._value.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {color};"
            )
        else:
            self._value.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #e0e0e0;"
            )

    def set_disabled(self, disabled: bool = True) -> None:
        """Set the card to disabled/enabled state."""
        if disabled:
            self._value.setText("—")
            self._subtitle.setText("disabled")
            self._subtitle.setVisible(True)
            self.setStyleSheet(DISABLED_STYLE)
        else:
            self._subtitle.setVisible(False)
            self.setStyleSheet(CARD_STYLE)

    def clear(self) -> None:
        """Clear the displayed value."""
        self._value.setText("—")
        self._subtitle.setVisible(False)
        self._value.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #e0e0e0;"
        )


# =============================================================================
# Confidence Interval Card
# =============================================================================

class CICard(QFrame):
    """
    Confidence interval display card.
    
    Shows a CI with method name, bounds, and optional width indicator.
    Designed for vertical stacking in narrow panels.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
    ):
        """
        Initialize the CI card.
        
        Args:
            title: CI method name (e.g., "Parametric", "Bootstrap")
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("ciCard")
        self.setStyleSheet(CI_CARD_STYLE)
        self._setup_ui(title)

    def _setup_ui(self, title: str) -> None:
        """Set up the card UI."""
        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 10, 14, 10)

        # Left: Title and method badge
        left_col = QVBoxLayout()
        left_col.setSpacing(2)
        
        self._title = QLabel(title)
        self._title.setStyleSheet(
            "color: #5aa0e5; font-size: 11px; font-weight: 600;"
        )
        left_col.addWidget(self._title)
        
        self._method_badge = QLabel("")
        self._method_badge.setStyleSheet(
            "color: #7a8a9a; font-size: 9px;"
        )
        self._method_badge.setVisible(False)
        left_col.addWidget(self._method_badge)
        
        layout.addLayout(left_col)
        layout.addStretch()

        # Right: Bounds display in compact format
        bounds_widget = QWidget()
        bounds_layout = QHBoxLayout(bounds_widget)
        bounds_layout.setSpacing(6)
        bounds_layout.setContentsMargins(0, 0, 0, 0)

        self._low_value = QLabel("—")
        self._low_value.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #f0b90b;"
        )
        bounds_layout.addWidget(self._low_value)

        dash = QLabel("–")
        dash.setStyleSheet("color: #6a6a7a; font-size: 13px;")
        bounds_layout.addWidget(dash)

        self._high_value = QLabel("—")
        self._high_value.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #2adf7a;"
        )
        bounds_layout.addWidget(self._high_value)

        layout.addWidget(bounds_widget)

        # Width indicator (hidden, shown via tooltip or subtitle)
        self._width_label = QLabel("")
        self._width_label.setVisible(False)
        
        # Store for tooltip
        self._low_label = QLabel("")  # Kept for compatibility
        self._high_label = QLabel("")  # Kept for compatibility

    def set_ci(
        self,
        low: float,
        high: float,
        method: str = "",
        format_currency: bool = True,
    ) -> None:
        """
        Set the confidence interval values.
        
        Args:
            low: Lower bound
            high: Upper bound
            method: Method name for badge (e.g., "z", "bca")
            format_currency: Format values as currency
        """
        if format_currency:
            self._low_value.setText(f"${low:,.2f}")
            self._high_value.setText(f"${high:,.2f}")
            width = high - low
            self.setToolTip(f"Width: ${width:,.2f}")
        else:
            self._low_value.setText(f"{low:.4f}")
            self._high_value.setText(f"{high:.4f}")
            width = high - low
            self.setToolTip(f"Width: {width:.4f}")
        
        if method:
            self._method_badge.setText(method)
            self._method_badge.setVisible(True)
        else:
            self._method_badge.setVisible(False)
        
        self.setStyleSheet(CI_CARD_STYLE)

    def set_disabled(self, disabled: bool = True, reason: str = "disabled") -> None:
        """Set the card to disabled state."""
        if disabled:
            self._low_value.setText("—")
            self._high_value.setText("—")
            self._method_badge.setText(reason)
            self._method_badge.setVisible(True)
            self._method_badge.setStyleSheet("color: #5a5a6a; font-size: 9px;")
            self.setStyleSheet(DISABLED_STYLE)
            self.setToolTip("")
        else:
            self._method_badge.setStyleSheet("color: #7a8a9a; font-size: 9px;")
            self.setStyleSheet(CI_CARD_STYLE)

    def clear(self) -> None:
        """Clear the displayed values."""
        self._low_value.setText("—")
        self._high_value.setText("—")
        self._method_badge.setVisible(False)
        self.setToolTip("")


# =============================================================================
# Percentile Bar
# =============================================================================

class PercentileBar(QFrame):
    """
    Visual percentile display with labeled markers.
    
    Shows P5, P25, P50, P75, P95 in a compact grid format.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the percentile bar."""
        super().__init__(parent)
        self.setObjectName("percentileBar")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the percentile bar UI."""
        layout = QGridLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 8, 4, 8)

        self._pct_labels: dict[int, QLabel] = {}
        self._pct_values: dict[int, QLabel] = {}

        percentiles = [5, 25, 50, 75, 95]
        colors = ["#ff5a6a", "#f0b90b", "#5aa0e5", "#f0b90b", "#2adf7a"]

        # Create a 2-row grid: labels on top, values below
        for col, (pct, color) in enumerate(zip(percentiles, colors)):
            label = QLabel(f"P{pct}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: 600;")
            layout.addWidget(label, 0, col)
            self._pct_labels[pct] = label

            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setStyleSheet(
                "color: #e0e0e0; font-size: 11px; font-weight: 500;"
            )
            layout.addWidget(value, 1, col)
            self._pct_values[pct] = value

    def set_percentiles(
        self,
        percentiles: dict[int, float],
        format_currency: bool = True,
    ) -> None:
        """
        Set percentile values.
        
        Args:
            percentiles: Dict mapping percentile (5, 25, 50, 75, 95) to value
            format_currency: Format as currency
        """
        for pct, value in percentiles.items():
            if pct in self._pct_values:
                if format_currency:
                    self._pct_values[pct].setText(f"${value:,.2f}")
                else:
                    self._pct_values[pct].setText(f"{value:.4f}")

    def clear(self) -> None:
        """Clear all percentile values."""
        for value_label in self._pct_values.values():
            value_label.setText("—")


# =============================================================================
# Main Statistics Panel
# =============================================================================

class StatsPanel(QGroupBox):
    """
    Comprehensive statistics display panel.
    
    Displays all statistics computed by the StatsEngine including:
    - Descriptive statistics (mean, std, skew, kurtosis)
    - Percentiles (P5, P25, P50, P75, P95)
    - Confidence intervals (parametric, bootstrap, Chebyshev)
    
    Signals:
        stat_clicked: Emitted when a statistic card is clicked
    """

    stat_clicked = Signal(str)  # metric name

    def __init__(
        self,
        title: str = "Statistical Analysis",
        parent: QWidget | None = None,
    ):
        """
        Initialize the statistics panel.
        
        Args:
            title: Panel title
            parent: Optional parent widget
        """
        super().__init__(title, parent)
        self._format_currency = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 20, 16, 16)

        # Descriptive Statistics Section
        desc_label = QLabel("Descriptive Statistics")
        desc_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(desc_label)

        # 2x2 grid for descriptive stats
        desc_grid = QGridLayout()
        desc_grid.setSpacing(10)

        self._cards: dict[str, StatCard] = {}

        desc_stats = [
            ("mean", "Mean"),
            ("std", "Std Dev"),
            ("skew", "Skewness"),
            ("kurtosis", "Kurtosis"),
        ]

        for i, (key, title) in enumerate(desc_stats):
            card = StatCard(title)
            self._cards[key] = card
            desc_grid.addWidget(card, i // 2, i % 2)

        layout.addLayout(desc_grid)

        # Spacer
        layout.addSpacing(8)

        # Percentiles Section
        pct_label = QLabel("Distribution Percentiles")
        pct_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(pct_label)

        self._percentile_bar = PercentileBar()
        layout.addWidget(self._percentile_bar)

        # Spacer
        layout.addSpacing(8)

        # Confidence Intervals Section
        ci_label = QLabel("Confidence Intervals")
        ci_label.setStyleSheet(
            "color: #9a9aaa; font-size: 11px; font-weight: 600; "
            "text-transform: uppercase; letter-spacing: 1px;"
        )
        layout.addWidget(ci_label)

        # Confidence level indicator (moved up)
        self._confidence_label = QLabel("Confidence Level: —")
        self._confidence_label.setStyleSheet(
            "color: #7a8a9a; font-size: 11px;"
        )
        layout.addWidget(self._confidence_label)

        self._ci_cards: dict[str, CICard] = {}

        # Stack CI cards vertically for better fit in narrow panels
        ci_layout = QVBoxLayout()
        ci_layout.setSpacing(8)

        ci_types = [
            ("parametric", "Parametric (z/t)"),
            ("bootstrap", "Bootstrap"),
            ("chebyshev", "Chebyshev"),
        ]

        for key, title in ci_types:
            card = CICard(title)
            self._ci_cards[key] = card
            ci_layout.addWidget(card)

        layout.addLayout(ci_layout)

        layout.addStretch()

    def set_format_currency(self, currency: bool) -> None:
        """Set whether to format values as currency."""
        self._format_currency = currency

    def update_from_result(
        self,
        result: "ComputeResult",
        confidence: float = 0.95,
    ) -> None:
        """
        Update the display from a StatsEngine ComputeResult.
        
        Args:
            result: ComputeResult from StatsEngine.compute()
            confidence: Confidence level used
        """
        metrics = result.metrics
        fmt = self._format_currency

        # Update descriptive statistics
        if "mean" in metrics:
            val = metrics["mean"]
            if fmt:
                self._cards["mean"].set_value(f"${val:,.2f}")
            else:
                self._cards["mean"].set_value(f"{val:.4f}")

        if "std" in metrics:
            val = metrics["std"]
            if fmt:
                self._cards["std"].set_value(f"${val:,.2f}")
            else:
                self._cards["std"].set_value(f"{val:.4f}")

        if "skew" in metrics:
            val = metrics["skew"]
            # Color based on skewness direction
            color = "#2adf7a" if val > 0.5 else "#ff5a6a" if val < -0.5 else "#e0e0e0"
            self._cards["skew"].set_value(f"{val:.3f}", color=color)

        if "kurtosis" in metrics:
            val = metrics["kurtosis"]
            # Color based on tail heaviness
            color = "#f0b90b" if abs(val) > 1 else "#e0e0e0"
            self._cards["kurtosis"].set_value(f"{val:.3f}", color=color)

        # Update percentiles
        if "percentiles" in metrics:
            self._percentile_bar.set_percentiles(
                metrics["percentiles"],
                format_currency=fmt,
            )

        # Update confidence intervals
        if "ci_mean" in metrics:
            ci = metrics["ci_mean"]
            self._ci_cards["parametric"].set_ci(
                low=ci.get("low", 0),
                high=ci.get("high", 0),
                method=ci.get("method", ""),
                format_currency=fmt,
            )
        else:
            self._ci_cards["parametric"].set_disabled(True, "not computed")

        if "ci_mean_bootstrap" in metrics:
            ci = metrics["ci_mean_bootstrap"]
            self._ci_cards["bootstrap"].set_ci(
                low=ci.get("low", 0),
                high=ci.get("high", 0),
                method=ci.get("method", "").replace("bootstrap-", ""),
                format_currency=fmt,
            )
        else:
            self._ci_cards["bootstrap"].set_disabled(True, "disabled")

        if "ci_mean_chebyshev" in metrics:
            ci = metrics["ci_mean_chebyshev"]
            self._ci_cards["chebyshev"].set_ci(
                low=ci.get("low", 0),
                high=ci.get("high", 0),
                method="dist-free",
                format_currency=fmt,
            )
        else:
            self._ci_cards["chebyshev"].set_disabled(True, "disabled")

        # Update confidence label
        self._confidence_label.setText(f"Confidence Level: {confidence:.0%}")

    def update_from_array(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        enable_bootstrap: bool = True,
        enable_chebyshev: bool = True,
    ) -> None:
        """
        Compute and display statistics from a numpy array.
        
        This is a convenience method that creates a StatsEngine,
        computes statistics, and updates the display.
        
        Args:
            data: Data array to analyze
            confidence: Confidence level
            enable_bootstrap: Whether to compute bootstrap CI
            enable_chebyshev: Whether to compute Chebyshev CI
        """
        from mcframework.stats_engine import StatsContext, build_default_engine

        engine = build_default_engine(
            include_dist_free=enable_chebyshev,
            include_target_bounds=False,
        )

        ctx = StatsContext(
            n=len(data),
            confidence=confidence,
            percentiles=(5, 25, 50, 75, 95),
            n_bootstrap=5000 if enable_bootstrap else 100,
        )

        # Select metrics to compute
        select = ["mean", "std", "skew", "kurtosis", "percentiles", "ci_mean"]
        if enable_bootstrap:
            select.append("ci_mean_bootstrap")
        if enable_chebyshev:
            select.append("ci_mean_chebyshev")

        result = engine.compute(data, ctx, select=select)
        self.update_from_result(result, confidence)

    def clear(self) -> None:
        """Clear all displayed statistics."""
        for card in self._cards.values():
            card.clear()

        self._percentile_bar.clear()

        for ci_card in self._ci_cards.values():
            ci_card.clear()

        self._confidence_label.setText("Confidence Level: —")


# =============================================================================
# Compact Stats Row
# =============================================================================

class CompactStatsRow(QFrame):
    """
    Compact single-row statistics display.
    
    Shows key statistics in a horizontal layout, suitable for
    embedding in other widgets or as a summary bar.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the compact stats row."""
        super().__init__(parent)
        self.setObjectName("compactStatsRow")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the compact row UI."""
        self.setStyleSheet("""
            QFrame#compactStatsRow {
                background-color: #1f1f35;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(12, 6, 12, 6)

        self._labels: dict[str, tuple[QLabel, QLabel]] = {}

        stats = [
            ("mean", "μ"),
            ("std", "σ"),
            ("skew", "Skew"),
            ("ci", "95% CI"),
        ]

        for key, symbol in stats:
            name_label = QLabel(f"{symbol}:")
            name_label.setStyleSheet("color: #8a8a9a; font-size: 10px;")

            value_label = QLabel("—")
            value_label.setStyleSheet(
                "color: #e0e0e0; font-size: 11px; font-weight: 600;"
            )

            layout.addWidget(name_label)
            layout.addWidget(value_label)
            self._labels[key] = (name_label, value_label)

            if key != "ci":  # Add separator except for last
                sep = QLabel("|")
                sep.setStyleSheet("color: #3a3a55; font-size: 10px;")
                layout.addWidget(sep)

        layout.addStretch()

    def update_stats(
        self,
        mean: float,
        std: float,
        skew: float,
        ci_low: float,
        ci_high: float,
        format_currency: bool = True,
    ) -> None:
        """
        Update the displayed statistics.
        
        Args:
            mean: Mean value
            std: Standard deviation
            skew: Skewness
            ci_low: CI lower bound
            ci_high: CI upper bound
            format_currency: Format as currency
        """
        if format_currency:
            self._labels["mean"][1].setText(f"${mean:,.2f}")
            self._labels["std"][1].setText(f"${std:,.2f}")
            self._labels["ci"][1].setText(f"[${ci_low:,.2f}, ${ci_high:,.2f}]")
        else:
            self._labels["mean"][1].setText(f"{mean:.4f}")
            self._labels["std"][1].setText(f"{std:.4f}")
            self._labels["ci"][1].setText(f"[{ci_low:.4f}, {ci_high:.4f}]")

        self._labels["skew"][1].setText(f"{skew:.2f}")

    def clear(self) -> None:
        """Clear all values."""
        for _, value_label in self._labels.values():
            value_label.setText("—")

