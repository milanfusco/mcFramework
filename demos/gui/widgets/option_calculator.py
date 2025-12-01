"""
Option Calculator dialog.

This module provides a standalone Black-Scholes option calculator
dialog for quick option pricing calculations.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class OptionCalculatorDialog(QDialog):
    """
    Standalone Black-Scholes option calculator dialog.
    
    Provides a simple interface for calculating option prices
    and Greeks using analytical Black-Scholes formulas.
    """

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the calculator dialog.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Black-Scholes Option Calculator")
        self.setMinimumWidth(500)
        self.setModal(False)  # Non-modal for continuous use
        
        self._setup_ui()
        self._connect_signals()
        
        # Initial calculation
        self._calculate()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Input parameters
        input_group = QGroupBox("Parameters")
        input_layout = QFormLayout(input_group)
        input_layout.setSpacing(8)
        
        # Spot price
        self._spot_spin = QDoubleSpinBox()
        self._spot_spin.setRange(0.01, 100000)
        self._spot_spin.setValue(100.0)
        self._spot_spin.setPrefix("$")
        self._spot_spin.setDecimals(2)
        input_layout.addRow("Spot Price (S):", self._spot_spin)
        
        # Strike price
        self._strike_spin = QDoubleSpinBox()
        self._strike_spin.setRange(0.01, 100000)
        self._strike_spin.setValue(100.0)
        self._strike_spin.setPrefix("$")
        self._strike_spin.setDecimals(2)
        input_layout.addRow("Strike Price (K):", self._strike_spin)
        
        # Time to maturity
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(0.001, 30)
        self._time_spin.setValue(0.25)
        self._time_spin.setSuffix(" years")
        self._time_spin.setDecimals(3)
        self._time_spin.setSingleStep(0.01)
        input_layout.addRow("Time to Maturity (T):", self._time_spin)
        
        # Risk-free rate
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(0, 1)
        self._rate_spin.setValue(0.05)
        self._rate_spin.setDecimals(4)
        self._rate_spin.setSingleStep(0.001)
        input_layout.addRow("Risk-Free Rate (r):", self._rate_spin)
        
        # Volatility
        self._vol_spin = QDoubleSpinBox()
        self._vol_spin.setRange(0.001, 5)
        self._vol_spin.setValue(0.20)
        self._vol_spin.setDecimals(4)
        self._vol_spin.setSingleStep(0.01)
        input_layout.addRow("Volatility (σ):", self._vol_spin)
        
        # Option type
        type_layout = QHBoxLayout()
        self._call_radio = QRadioButton("Call")
        self._put_radio = QRadioButton("Put")
        self._call_radio.setChecked(True)
        type_layout.addWidget(self._call_radio)
        type_layout.addWidget(self._put_radio)
        type_layout.addStretch()
        input_layout.addRow("Option Type:", type_layout)
        
        layout.addWidget(input_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QGridLayout(results_group)
        results_layout.setSpacing(12)
        
        # Price result (large display)
        results_layout.addWidget(QLabel("Option Price:"), 0, 0)
        self._price_label = QLabel("—")
        self._price_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #00d26a;"
        )
        results_layout.addWidget(self._price_label, 0, 1)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        results_layout.addWidget(sep, 1, 0, 1, 2)
        
        # Greeks
        results_layout.addWidget(QLabel("Delta (Δ):"), 2, 0)
        self._delta_label = QLabel("—")
        results_layout.addWidget(self._delta_label, 2, 1)
        
        results_layout.addWidget(QLabel("Gamma (Γ):"), 3, 0)
        self._gamma_label = QLabel("—")
        results_layout.addWidget(self._gamma_label, 3, 1)
        
        results_layout.addWidget(QLabel("Vega (ν):"), 4, 0)
        self._vega_label = QLabel("—")
        results_layout.addWidget(self._vega_label, 4, 1)
        
        results_layout.addWidget(QLabel("Theta (Θ):"), 5, 0)
        self._theta_label = QLabel("—")
        results_layout.addWidget(self._theta_label, 5, 1)
        
        results_layout.addWidget(QLabel("Rho (ρ):"), 6, 0)
        self._rho_label = QLabel("—")
        results_layout.addWidget(self._rho_label, 6, 1)
        
        layout.addWidget(results_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._calculate_btn = QPushButton("Calculate")
        self._calculate_btn.setDefault(True)
        button_layout.addWidget(self._calculate_btn)
        
        self._reset_btn = QPushButton("Reset")
        button_layout.addWidget(self._reset_btn)
        
        button_layout.addStretch()
        
        self._close_btn = QPushButton("Close")
        button_layout.addWidget(self._close_btn)
        
        layout.addLayout(button_layout)

    def _connect_signals(self) -> None:
        """Connect UI signals."""
        # Auto-calculate on value change
        self._spot_spin.valueChanged.connect(self._calculate)
        self._strike_spin.valueChanged.connect(self._calculate)
        self._time_spin.valueChanged.connect(self._calculate)
        self._rate_spin.valueChanged.connect(self._calculate)
        self._vol_spin.valueChanged.connect(self._calculate)
        self._call_radio.toggled.connect(self._calculate)
        
        # Buttons
        self._calculate_btn.clicked.connect(self._calculate)
        self._reset_btn.clicked.connect(self._reset)
        self._close_btn.clicked.connect(self.close)

    def _calculate(self) -> None:
        """Calculate option price and Greeks."""
        from ..controllers.analysis_controller import TickerAnalysisController
        
        spot = self._spot_spin.value()
        strike = self._strike_spin.value()
        time = self._time_spin.value()
        rate = self._rate_spin.value()
        vol = self._vol_spin.value()
        option_type = "call" if self._call_radio.isChecked() else "put"
        
        # Calculate price
        price = TickerAnalysisController.calculate_bs_price(
            spot, strike, time, rate, vol, option_type
        )
        
        # Calculate Greeks
        greeks = TickerAnalysisController.calculate_bs_greeks(
            spot, strike, time, rate, vol, option_type
        )
        
        # Update display
        color = '#00d26a' if option_type == "call" else '#f23645'
        self._price_label.setText(f"${price:.4f}")
        self._price_label.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {color};"
        )
        
        self._delta_label.setText(f"{greeks.delta:+.4f}")
        self._gamma_label.setText(f"{greeks.gamma:.6f}")
        self._vega_label.setText(f"{greeks.vega:.4f}")
        self._theta_label.setText(f"{greeks.theta:.4f}")
        self._rho_label.setText(f"{greeks.rho:.4f}")

    def _reset(self) -> None:
        """Reset all inputs to defaults."""
        self._spot_spin.setValue(100.0)
        self._strike_spin.setValue(100.0)
        self._time_spin.setValue(0.25)
        self._rate_spin.setValue(0.05)
        self._vol_spin.setValue(0.20)
        self._call_radio.setChecked(True)

    def set_parameters(
        self,
        spot: float | None = None,
        strike: float | None = None,
        time: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
    ) -> None:
        """
        Set calculator parameters from external source.
        
        Args:
            spot: Spot price
            strike: Strike price
            time: Time to maturity
            rate: Risk-free rate
            vol: Volatility
        """
        if spot is not None:
            self._spot_spin.setValue(spot)
        if strike is not None:
            self._strike_spin.setValue(strike)
        if time is not None:
            self._time_spin.setValue(time)
        if rate is not None:
            self._rate_spin.setValue(rate)
        if vol is not None:
            self._vol_spin.setValue(vol)

