"""
Options & Greeks tab widget.

This module provides the Options & Greeks tab displaying option pricing
results, Greeks sensitivities, and interactive parameter sliders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .empty_state import OptionsEmptyState

if TYPE_CHECKING:
    from ..models.state import GreeksResult, OptionPricingResult, TickerAnalysisState


class OptionPriceCard(QFrame):
    """
    Card widget displaying option pricing result.
    
    Shows the option type (Call/Put), price, and confidence interval
    in a styled card format.
    """

    def __init__(
        self,
        option_type: str,
        parent: QWidget | None = None,
    ):
        """
        Initialize the option price card.
        
        Args:
            option_type: "Call" or "Put"
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._option_type = option_type
        
        self.setObjectName("optionCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumWidth(220)
        self.setMinimumHeight(140)
        
        # Color based on type
        self._color = '#2adf7a' if option_type == "Call" else '#ff5a6a'
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 16, 20, 16)
        
        # Type header
        header = QLabel(f"{self._option_type} Option")
        header.setObjectName("optionTypeHeader")
        header.setStyleSheet(f"""
            color: {self._color}; 
            font-weight: 600; 
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(header)
        
        # Price
        self._price_label = QLabel("—")
        self._price_label.setObjectName("optionPrice")
        self._price_label.setStyleSheet("""
            font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
            font-size: 28px; 
            font-weight: 600;
            color: #ffffff;
        """)
        layout.addWidget(self._price_label)
        
        # Standard error
        self._se_label = QLabel("SE: —")
        self._se_label.setObjectName("optionSE")
        self._se_label.setStyleSheet("color: #8a8a9a; font-size: 11px;")
        layout.addWidget(self._se_label)
        
        # Confidence interval
        self._ci_label = QLabel("95% CI: [—, —]")
        self._ci_label.setObjectName("optionCI")
        self._ci_label.setStyleSheet("color: #8a8a9a; font-size: 11px;")
        layout.addWidget(self._ci_label)
        
        layout.addStretch()

    def update_result(self, result: "OptionPricingResult") -> None:
        """
        Update the card with pricing result.
        
        Args:
            result: Option pricing result
        """
        self._price_label.setText(f"${result.price:.2f}")
        self._se_label.setText(f"SE: ±${result.std_error:.4f}")
        
        ci_low, ci_high = result.confidence_interval
        self._ci_label.setText(f"95% CI: [${ci_low:.2f}, ${ci_high:.2f}]")

    def clear(self) -> None:
        """Clear the card."""
        self._price_label.setText("—")
        self._se_label.setText("SE: —")
        self._ci_label.setText("95% CI: [—, —]")


class GreeksTable(QTableWidget):
    """
    Table widget displaying option Greeks.
    
    Shows Delta, Gamma, Vega, Theta, and Rho for both
    call and put options in a structured table format.
    """

    GREEKS = [
        ("Delta (Δ)", "delta", "Sensitivity to underlying price", "{:+.4f}"),
        ("Gamma (Γ)", "gamma", "Rate of change of Delta", "{:.6f}"),
        ("Vega (ν)", "vega", "Sensitivity to volatility (per 1%)", "{:.4f}"),
        ("Theta (Θ)", "theta", "Time decay (per day)", "{:.4f}"),
        ("Rho (ρ)", "rho", "Sensitivity to interest rate (per 1%)", "{:.4f}"),
    ]

    def __init__(self, parent: QWidget | None = None):
        """Initialize the Greeks table."""
        super().__init__(parent)
        
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Greek", "Call", "Put", "Description"])
        self.setRowCount(len(self.GREEKS))
        
        # Style
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        
        # Disable scrollbars - show all rows at once
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Set fixed height for 5 rows (header ~30px + 5 rows ~30px each + padding)
        self.setFixedHeight(200)
        
        # Set column widths
        self.setColumnWidth(0, 100)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 100)
        
        # Populate row labels
        for i, (name, _, desc, _) in enumerate(self.GREEKS):
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(desc)
            self.setItem(i, 0, name_item)
            self.setItem(i, 1, QTableWidgetItem("—"))
            self.setItem(i, 2, QTableWidgetItem("—"))
            
            desc_item = QTableWidgetItem(desc)
            desc_item.setForeground(Qt.GlobalColor.gray)
            self.setItem(i, 3, desc_item) 

    def update_greeks(
        self,
        call_greeks: "GreeksResult | None",
        put_greeks: "GreeksResult | None",
    ) -> None:
        """
        Update the table with Greeks values.
        
        Args:
            call_greeks: Call option Greeks
            put_greeks: Put option Greeks
        """
        for i, (_, attr, _, fmt) in enumerate(self.GREEKS):
            # Call value
            if call_greeks:
                call_val = getattr(call_greeks, attr, 0.0)
                call_item = QTableWidgetItem(fmt.format(call_val))
                call_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            else:
                call_item = QTableWidgetItem("—")
            self.setItem(i, 1, call_item)
            
            # Put value
            if put_greeks:
                put_val = getattr(put_greeks, attr, 0.0)
                put_item = QTableWidgetItem(fmt.format(put_val))
                put_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            else:
                put_item = QTableWidgetItem("—")
            self.setItem(i, 2, put_item)

    def clear(self) -> None:
        """Clear all Greeks values."""
        for i in range(len(self.GREEKS)):
            self.setItem(i, 1, QTableWidgetItem("—"))
            self.setItem(i, 2, QTableWidgetItem("—"))


class ParameterSlider(QWidget):
    """
    Labeled slider for adjusting simulation parameters.
    
    Provides a slider with min/max labels and current value display,
    useful for sensitivity analysis.
    
    Signals:
        value_changed: Emitted with new value when slider changes
    """
    
    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        initial: float,
        step: float = 0.01,
        format_str: str = "{:.2f}",
        suffix: str = "",
        parent: QWidget | None = None,
    ):
        """
        Initialize the parameter slider.
        
        Args:
            label: Slider label
            min_val: Minimum value
            max_val: Maximum value
            initial: Initial value
            step: Step size
            format_str: Format string for value display
            suffix: Suffix for value display
            parent: Optional parent widget
        """
        super().__init__(parent)
        
        self._min_val = min_val
        self._max_val = max_val
        self._step = step
        self._format_str = format_str
        self._suffix = suffix
        
        self._setup_ui(label, initial)

    def _setup_ui(self, label: str, initial: float) -> None:
        """Set up the slider UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header row with label and value
        header = QHBoxLayout()
        
        self._label = QLabel(label)
        header.addWidget(self._label)
        
        header.addStretch()
        
        self._value_label = QLabel()
        self._value_label.setObjectName("sliderValue")
        header.addWidget(self._value_label)
        
        layout.addLayout(header)
        
        # Slider
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(int((self._max_val - self._min_val) / self._step))
        self._slider.setValue(int((initial - self._min_val) / self._step))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)
        
        # Min/max labels
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel(f"{self._format_str.format(self._min_val)}{self._suffix}"))
        range_layout.addStretch()
        range_layout.addWidget(QLabel(f"{self._format_str.format(self._max_val)}{self._suffix}"))
        layout.addLayout(range_layout)
        
        # Update initial value display
        self._update_value_display()

    def _on_slider_changed(self, position: int) -> None:
        """Handle slider value change."""
        self._update_value_display()
        self.value_changed.emit(self.get_value())

    def _update_value_display(self) -> None:
        """Update the value label."""
        value = self.get_value()
        self._value_label.setText(f"{self._format_str.format(value)}{self._suffix}")

    def get_value(self) -> float:
        """Get the current slider value."""
        return self._min_val + self._slider.value() * self._step

    def set_value(self, value: float) -> None:
        """Set the slider value."""
        position = int((value - self._min_val) / self._step)
        self._slider.setValue(position)

    def set_range(self, min_val: float, max_val: float) -> None:
        """Update the slider range."""
        current = self.get_value()
        self._min_val = min_val
        self._max_val = max_val
        self._slider.setMaximum(int((max_val - min_val) / self._step))
        
        # Clamp current value to new range
        clamped = max(min_val, min(max_val, current))
        self.set_value(clamped)


class SensitivityPanel(QGroupBox):
    """
    Panel for what-if analysis with parameter sliders.
    
    Allows users to adjust S0 and sigma to see how option
    prices and Greeks change in real-time.
    
    Signals:
        parameters_changed: Emitted with (s0, sigma) when sliders change
    """
    
    parameters_changed = Signal(float, float)  # s0, sigma

    def __init__(self, parent: QWidget | None = None):
        """Initialize the sensitivity panel."""
        super().__init__("What-If Analysis", parent)
        self._base_s0 = 100.0
        self._base_sigma = 0.2
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Spot price slider
        self._s0_slider = ParameterSlider(
            label="Spot Price (S₀)",
            min_val=50.0,
            max_val=200.0,
            initial=100.0,
            step=1.0,
            format_str="${:.0f}",
        )
        self._s0_slider.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self._s0_slider)
        
        # Volatility slider
        self._sigma_slider = ParameterSlider(
            label="Volatility (σ)",
            min_val=0.05,
            max_val=1.0,
            initial=0.2,
            step=0.01,
            format_str="{:.0%}",
        )
        self._sigma_slider.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self._sigma_slider)
        
        # Prices display
        prices_frame = QFrame()
        prices_frame.setObjectName("sensitivityResults")
        prices_layout = QGridLayout(prices_frame)
        prices_layout.setSpacing(8)
        
        # Header
        prices_layout.addWidget(
            self._make_header_label("Adjusted Prices"), 0, 0, 1, 2
        )
        
        prices_layout.addWidget(QLabel("Call:"), 1, 0)
        self._adj_call_label = QLabel("—")
        self._adj_call_label.setStyleSheet("color: #00d26a; font-weight: bold;")
        prices_layout.addWidget(self._adj_call_label, 1, 1)
        
        prices_layout.addWidget(QLabel("Put:"), 2, 0)
        self._adj_put_label = QLabel("—")
        self._adj_put_label.setStyleSheet("color: #f23645; font-weight: bold;")
        prices_layout.addWidget(self._adj_put_label, 2, 1)
        
        layout.addWidget(prices_frame)
        
        # Greeks display
        greeks_frame = QFrame()
        greeks_frame.setObjectName("sensitivityGreeks")
        greeks_layout = QGridLayout(greeks_frame)
        greeks_layout.setSpacing(4)
        
        # Greeks header
        greeks_layout.addWidget(
            self._make_header_label("Adjusted Greeks"), 0, 0, 1, 3
        )
        
        # Column headers
        call_header = QLabel("Call")
        call_header.setStyleSheet("color: #00d26a; font-size: 10px;")
        put_header = QLabel("Put")
        put_header.setStyleSheet("color: #f23645; font-size: 10px;")
        greeks_layout.addWidget(call_header, 1, 1)
        greeks_layout.addWidget(put_header, 1, 2)
        
        # Greek rows
        self._greek_labels: dict[str, tuple[QLabel, QLabel]] = {}
        greeks = ["Delta", "Gamma", "Vega", "Theta", "Rho"]
        
        for i, greek in enumerate(greeks):
            row = i + 2
            name_label = QLabel(f"{greek}:")
            name_label.setStyleSheet("color: #888; font-size: 11px;")
            greeks_layout.addWidget(name_label, row, 0)
            
            call_val = QLabel("—")
            call_val.setStyleSheet("font-size: 11px;")
            call_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            greeks_layout.addWidget(call_val, row, 1)
            
            put_val = QLabel("—")
            put_val.setStyleSheet("font-size: 11px;")
            put_val.setAlignment(Qt.AlignmentFlag.AlignRight)
            greeks_layout.addWidget(put_val, row, 2)
            
            self._greek_labels[greek.lower()] = (call_val, put_val)
        
        layout.addWidget(greeks_frame)
        layout.addStretch()

    def _make_header_label(self, text: str) -> QLabel:
        """Create a styled header label."""
        label = QLabel(text)
        label.setStyleSheet(
            "color: #aaa; font-size: 11px; font-weight: bold; "
            "border-bottom: 1px solid #444; padding-bottom: 4px;"
        )
        return label

    def _on_parameter_changed(self) -> None:
        """Handle parameter slider change."""
        s0 = self._s0_slider.get_value()
        sigma = self._sigma_slider.get_value()
        self.parameters_changed.emit(s0, sigma)

    def set_base_parameters(self, s0: float, sigma: float) -> None:
        """
        Set the base parameters and update slider ranges.
        
        Args:
            s0: Base spot price
            sigma: Base volatility
        """
        self._base_s0 = s0
        self._base_sigma = sigma
        
        # Update S0 slider range to ±50% of base
        s0_min = max(1.0, s0 * 0.5)
        s0_max = s0 * 1.5
        self._s0_slider.set_range(s0_min, s0_max)
        self._s0_slider.set_value(s0)
        
        # Update sigma slider range
        sigma_min = max(0.01, sigma * 0.5)
        sigma_max = min(2.0, sigma * 2.0)
        self._sigma_slider.set_range(sigma_min, sigma_max)
        self._sigma_slider.set_value(sigma)

    def update_adjusted_prices(self, call_price: float, put_price: float) -> None:
        """
        Update the adjusted price display.
        
        Args:
            call_price: Adjusted call price
            put_price: Adjusted put price
        """
        self._adj_call_label.setText(f"${call_price:.2f}")
        self._adj_put_label.setText(f"${put_price:.2f}")

    def update_adjusted_greeks(
        self,
        call_greeks: dict[str, float],
        put_greeks: dict[str, float],
    ) -> None:
        """
        Update the adjusted Greeks display.
        
        Args:
            call_greeks: Dict with delta, gamma, vega, theta, rho for call
            put_greeks: Dict with delta, gamma, vega, theta, rho for put
        """
        formats = {
            "delta": "{:+.4f}",
            "gamma": "{:.6f}",
            "vega": "{:.4f}",
            "theta": "{:.4f}",
            "rho": "{:.4f}",
        }
        
        for greek, (call_label, put_label) in self._greek_labels.items():
            fmt = formats.get(greek, "{:.4f}")
            
            call_val = call_greeks.get(greek, 0.0)
            call_label.setText(fmt.format(call_val))
            
            put_val = put_greeks.get(greek, 0.0)
            put_label.setText(fmt.format(put_val))

    def clear(self) -> None:
        """Clear the adjusted prices and Greeks."""
        self._adj_call_label.setText("—")
        self._adj_put_label.setText("—")
        
        for call_label, put_label in self._greek_labels.values():
            call_label.setText("—")
            put_label.setText("—")


class OptionsGreeksTab(QWidget):
    """
    Options & Greeks tab with pricing results and sensitivity analysis.
    
    This tab displays:
    - Call and Put option pricing cards
    - Greeks table for both option types
    - Interactive what-if analysis sliders
    
    Shows an empty state when no option pricing has been run.
    
    Signals:
        sensitivity_requested: Emitted with (s0, sigma) for recalculation
    """
    
    sensitivity_requested = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the Options & Greeks tab."""
        super().__init__(parent)
        self._has_data = False
        self._content_widget: QWidget | None = None
        self._current_content_width: int | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the tab UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget for empty/content states
        self._stack = QStackedWidget()
        
        # Empty state (index 0)
        self._empty_state = OptionsEmptyState()
        self._stack.addWidget(self._empty_state)
        
        # Content widget (index 1)
        content = QWidget()
        self._content_widget = content
        content_layout = QHBoxLayout(content)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(16, 16, 16, 16)
        
        # Left side: Cards and Greeks table stacked
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Option pricing cards
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)
        
        self._call_card = OptionPriceCard("Call")
        self._put_card = OptionPriceCard("Put")
        
        cards_layout.addWidget(self._call_card)
        cards_layout.addWidget(self._put_card)
        cards_layout.addStretch()
        
        left_layout.addLayout(cards_layout)
        
        # Greeks table (compact)
        greeks_group = QGroupBox("Option Greeks")
        greeks_layout = QVBoxLayout(greeks_group)
        greeks_layout.setContentsMargins(8, 16, 8, 8)
        self._greeks_table = GreeksTable()
        greeks_layout.addWidget(self._greeks_table)
        left_layout.addWidget(greeks_group)
        
        # Add stretch to push content to top
        left_layout.addStretch()
        
        content_layout.addWidget(left_panel, 2)
        
        # Right side: Sensitivity panel
        self._sensitivity_panel = SensitivityPanel()
        content_layout.addWidget(self._sensitivity_panel, 1)
        
        self._stack.addWidget(content)
        layout.addWidget(self._stack)
        
        # Start with empty state
        self._stack.setCurrentIndex(0)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._sensitivity_panel.parameters_changed.connect(
            self._on_sensitivity_changed
        )

    def _on_sensitivity_changed(self, s0: float, sigma: float) -> None:
        """Handle sensitivity parameter change."""
        self.sensitivity_requested.emit(s0, sigma)

    def update_from_state(self, state: "TickerAnalysisState") -> None:
        """
        Update all components from the application state.
        
        Args:
            state: Current application state
        """
        # Check if we have option pricing data
        has_data = state.call_result is not None or state.put_result is not None
        
        if has_data:
            # Show content
            self._stack.setCurrentIndex(1)
            self._has_data = True
            
            # Update pricing cards
            if state.call_result:
                self._call_card.update_result(state.call_result)
            else:
                self._call_card.clear()
            
            if state.put_result:
                self._put_card.update_result(state.put_result)
            else:
                self._put_card.clear()
            
            # Update Greeks table
            self._greeks_table.update_greeks(state.call_greeks, state.put_greeks)
            
            # Update sensitivity panel base parameters
            if state.parameters:
                self._sensitivity_panel.set_base_parameters(
                    state.parameters.spot_price,
                    state.parameters.volatility,
                )
        else:
            # Show empty state
            self._stack.setCurrentIndex(0)
            self._has_data = False

    def update_adjusted_prices(self, call_price: float, put_price: float) -> None:
        """
        Update the sensitivity panel with adjusted prices.
        
        Args:
            call_price: Adjusted call option price
            put_price: Adjusted put option price
        """
        self._sensitivity_panel.update_adjusted_prices(call_price, put_price)

    def update_adjusted_greeks(
        self,
        call_greeks: dict[str, float],
        put_greeks: dict[str, float],
    ) -> None:
        """
        Update the sensitivity panel with adjusted Greeks.
        
        Args:
            call_greeks: Adjusted call option Greeks
            put_greeks: Adjusted put option Greeks
        """
        self._sensitivity_panel.update_adjusted_greeks(call_greeks, put_greeks)

    def clear(self) -> None:
        """Clear all displayed data and show empty state."""
        self._call_card.clear()
        self._put_card.clear()
        self._greeks_table.clear()
        self._sensitivity_panel.clear()
        self._has_data = False
        self._stack.setCurrentIndex(0)

    def set_run_callback(self, callback) -> None:
        """Set callback for the empty state action button."""
        self._empty_state.set_action_callback(callback)

    def set_content_width(self, width: int) -> None:
        """Allow parent window to cap content width."""
        self._current_content_width = width if width > 0 else None
        if self._content_widget is None:
            return
        if self._current_content_width is None:
            self._content_widget.setMaximumWidth(16777215)
        else:
            self._content_widget.setMaximumWidth(self._current_content_width)

