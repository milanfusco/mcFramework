"""
Sidebar widget with simulation controls.

This module provides the sidebar panel containing all input controls
for configuring and running simulations, following a clean component-based
design with grouped sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..models.state import SimulationConfig


@dataclass
class ScenarioPreset:
    """Predefined scenario configuration."""
    name: str
    ticker: str
    historical_days: int
    forecast_horizon: float
    n_simulations: int
    volatility_hint: str  # Descriptive hint about expected volatility


# Built-in scenario presets
SCENARIO_PRESETS: list[ScenarioPreset] = [
    ScenarioPreset(
        name="Default (AAPL)",
        ticker="AAPL",
        historical_days=252,
        forecast_horizon=1.0,
        n_simulations=1000,
        volatility_hint="Moderate volatility, large cap",
    ),
    ScenarioPreset(
        name="High Vol Tech (TSLA)",
        ticker="TSLA",
        historical_days=252,
        forecast_horizon=1.0,
        n_simulations=2000,
        volatility_hint="High volatility growth stock",
    ),
    ScenarioPreset(
        name="Index (SPY)",
        ticker="SPY",
        historical_days=504,
        forecast_horizon=1.0,
        n_simulations=1000,
        volatility_hint="Low volatility index ETF",
    ),
    ScenarioPreset(
        name="Crypto-adjacent (COIN)",
        ticker="COIN",
        historical_days=252,
        forecast_horizon=0.5,
        n_simulations=3000,
        volatility_hint="Very high volatility",
    ),
    ScenarioPreset(
        name="Stable Dividend (JNJ)",
        ticker="JNJ",
        historical_days=504,
        forecast_horizon=2.0,
        n_simulations=1000,
        volatility_hint="Low volatility dividend stock",
    ),
]


class ControlGroup(QGroupBox):
    """
    Styled group box for organizing related controls.
    
    This reusable component provides consistent styling for
    grouped input controls in the sidebar.
    """

    def __init__(self, title: str, parent: QWidget | None = None):
        """
        Initialize the control group.
        
        Args:
            title: Group title
            parent: Optional parent widget
        """
        super().__init__(title, parent)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(12, 16, 12, 12)

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the group."""
        self._layout.addWidget(widget)

    def add_row(self, label: str, widget: QWidget) -> QHBoxLayout:
        """
        Add a labeled row to the group.
        
        Args:
            label: Label text
            widget: Widget to add
            
        Returns:
            The created row layout
        """
        row = QHBoxLayout()
        row.setSpacing(8)
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(100)
        row.addWidget(label_widget)
        row.addWidget(widget, 1)
        
        self._layout.addLayout(row)
        return row


class SidebarWidget(QWidget):
    """
    Main sidebar widget containing all simulation controls.
    
    This widget organizes controls into logical groups:
    - Ticker & Data: ticker symbol and historical days
    - Simulation: horizon, simulations, paths, seed
    - Options: Greeks, 3D plots, auto-fetch toggles
    - Presets: Quick scenario selection
    
    Signals:
        config_changed: Emitted when any configuration value changes
        run_requested: Emitted when Run button is clicked
        fetch_requested: Emitted when Fetch Only is clicked
        preset_selected: Emitted with preset index when a preset is chosen
    """
    
    config_changed = Signal(object)  # SimulationConfig
    run_requested = Signal()
    fetch_requested = Signal()
    preset_selected = Signal(int)

    # Sidebar width constant
    SIDEBAR_WIDTH = 280

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the sidebar widget.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setFixedWidth(self.SIDEBAR_WIDTH)
        self.setObjectName("sidebar")
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the sidebar UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Ticker & Data Group
        self._ticker_group = self._create_ticker_group()
        layout.addWidget(self._ticker_group)
        
        # Simulation Parameters Group
        self._sim_group = self._create_simulation_group()
        layout.addWidget(self._sim_group)
        
        # Options Group
        self._options_group = self._create_options_group()
        layout.addWidget(self._options_group)
        
        # Presets Group
        self._presets_group = self._create_presets_group()
        layout.addWidget(self._presets_group)
        
        # Action Buttons
        self._action_buttons = self._create_action_buttons()
        layout.addWidget(self._action_buttons)
        
        # Spacer
        layout.addStretch()

    def _create_ticker_group(self) -> ControlGroup:
        """Create the ticker and data controls group."""
        group = ControlGroup("Ticker & Data")
        
        # Ticker input
        self._ticker_input = QLineEdit()
        self._ticker_input.setPlaceholderText("e.g., AAPL")
        self._ticker_input.setText("AAPL")
        self._ticker_input.setMaxLength(10)
        group.add_row("Ticker:", self._ticker_input)
        
        # Historical days
        self._days_spin = QSpinBox()
        self._days_spin.setRange(30, 2520)  # ~10 years max
        self._days_spin.setValue(252)
        self._days_spin.setSuffix(" days")
        self._days_spin.setToolTip("Number of historical trading days to fetch")
        group.add_row("History:", self._days_spin)
        
        return group

    def _create_simulation_group(self) -> ControlGroup:
        """Create the simulation parameters group."""
        group = ControlGroup("Simulation")
        
        # Forecast horizon
        self._horizon_spin = QDoubleSpinBox()
        self._horizon_spin.setRange(0.1, 10.0)
        self._horizon_spin.setValue(1.0)
        self._horizon_spin.setSingleStep(0.25)
        self._horizon_spin.setSuffix(" years")
        self._horizon_spin.setDecimals(2)
        group.add_row("Horizon:", self._horizon_spin)
        
        # Number of simulations
        self._sims_spin = QSpinBox()
        self._sims_spin.setRange(100, 100000)
        self._sims_spin.setValue(1000)
        self._sims_spin.setSingleStep(500)
        self._sims_spin.setToolTip("Number of Monte Carlo simulations")
        group.add_row("Simulations:", self._sims_spin)
        
        # Paths for visualization
        self._paths_spin = QSpinBox()
        self._paths_spin.setRange(10, 1000)
        self._paths_spin.setValue(100)
        self._paths_spin.setToolTip("Number of paths to visualize in charts")
        group.add_row("Viz Paths:", self._paths_spin)
        
        # Random seed
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip("Random seed for reproducibility")
        group.add_row("Seed:", self._seed_spin)
        
        # Risk-free rate
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(0.0, 0.20)
        self._rate_spin.setValue(0.05)
        self._rate_spin.setSingleStep(0.005)
        self._rate_spin.setDecimals(3)
        self._rate_spin.setSuffix(" (r)")
        self._rate_spin.setToolTip("Risk-free interest rate")
        group.add_row("Rate:", self._rate_spin)
        
        return group

    def _create_options_group(self) -> ControlGroup:
        """Create the options toggles group."""
        group = ControlGroup("Options")
        
        # Compute Greeks checkbox
        self._greeks_check = QCheckBox("Calculate Greeks")
        self._greeks_check.setChecked(True)
        self._greeks_check.setToolTip("Compute option sensitivities (Delta, Gamma, etc.)")
        group.add_widget(self._greeks_check)
        
        # 3D plots checkbox
        self._3d_check = QCheckBox("Generate 3D Surfaces")
        self._3d_check.setChecked(True)
        self._3d_check.setToolTip("Generate 3D option price surfaces")
        group.add_widget(self._3d_check)
        
        # Auto-fetch checkbox
        self._auto_fetch_check = QCheckBox("Auto-fetch on Run")
        self._auto_fetch_check.setChecked(True)
        self._auto_fetch_check.setToolTip("Automatically fetch fresh data when running")
        group.add_widget(self._auto_fetch_check)
        
        return group

    def _create_presets_group(self) -> ControlGroup:
        """Create the scenario presets group."""
        group = ControlGroup("Scenarios")
        
        # Preset dropdown
        self._preset_combo = QComboBox()
        for preset in SCENARIO_PRESETS:
            self._preset_combo.addItem(preset.name)
        group.add_widget(self._preset_combo)
        
        # Preset hint label
        self._preset_hint = QLabel()
        self._preset_hint.setWordWrap(True)
        self._preset_hint.setStyleSheet("color: #888; font-size: 11px;")
        self._update_preset_hint(0)
        group.add_widget(self._preset_hint)
        
        # Apply preset button
        self._apply_preset_btn = QPushButton("Apply Preset")
        self._apply_preset_btn.setToolTip("Apply the selected scenario preset")
        group.add_widget(self._apply_preset_btn)
        
        return group

    def _create_action_buttons(self) -> QFrame:
        """Create the main action buttons."""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 8, 0, 0)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)
        
        # Fetch Only button
        self._fetch_btn = QPushButton("Fetch Data Only")
        self._fetch_btn.setToolTip("Fetch market data without running simulation")
        layout.addWidget(self._fetch_btn)
        
        # Run button (primary action)
        self._run_btn = QPushButton("▶ Run Analysis")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setToolTip("Fetch data and run full simulation")
        layout.addWidget(self._run_btn)
        
        return frame

    def _connect_signals(self) -> None:
        """Connect internal signals to handlers."""
        # Value change signals
        self._ticker_input.textChanged.connect(self._on_config_changed)
        self._days_spin.valueChanged.connect(self._on_config_changed)
        self._horizon_spin.valueChanged.connect(self._on_config_changed)
        self._sims_spin.valueChanged.connect(self._on_config_changed)
        self._paths_spin.valueChanged.connect(self._on_config_changed)
        self._seed_spin.valueChanged.connect(self._on_config_changed)
        self._rate_spin.valueChanged.connect(self._on_config_changed)
        self._greeks_check.stateChanged.connect(self._on_config_changed)
        self._3d_check.stateChanged.connect(self._on_config_changed)
        
        # Preset signals
        self._preset_combo.currentIndexChanged.connect(self._update_preset_hint)
        self._apply_preset_btn.clicked.connect(self._on_apply_preset)
        
        # Action signals
        self._fetch_btn.clicked.connect(self.fetch_requested.emit)
        self._run_btn.clicked.connect(self.run_requested.emit)

    def _on_config_changed(self) -> None:
        """Handle configuration value changes."""
        config = self.get_config()
        self.config_changed.emit(config)

    def _update_preset_hint(self, index: int) -> None:
        """Update the preset hint text."""
        if 0 <= index < len(SCENARIO_PRESETS):
            preset = SCENARIO_PRESETS[index]
            self._preset_hint.setText(preset.volatility_hint)

    def _on_apply_preset(self) -> None:
        """Apply the selected preset."""
        index = self._preset_combo.currentIndex()
        if 0 <= index < len(SCENARIO_PRESETS):
            preset = SCENARIO_PRESETS[index]
            self.apply_preset(preset)
            self.preset_selected.emit(index)

    def apply_preset(self, preset: ScenarioPreset) -> None:
        """
        Apply a scenario preset to the controls.
        
        Args:
            preset: The preset to apply
        """
        # Block signals to prevent multiple config_changed emissions
        self._ticker_input.blockSignals(True)
        self._days_spin.blockSignals(True)
        self._horizon_spin.blockSignals(True)
        self._sims_spin.blockSignals(True)
        
        self._ticker_input.setText(preset.ticker)
        self._days_spin.setValue(preset.historical_days)
        self._horizon_spin.setValue(preset.forecast_horizon)
        self._sims_spin.setValue(preset.n_simulations)
        
        self._ticker_input.blockSignals(False)
        self._days_spin.blockSignals(False)
        self._horizon_spin.blockSignals(False)
        self._sims_spin.blockSignals(False)
        
        # Emit single config changed signal
        self._on_config_changed()

    def get_config(self) -> "SimulationConfig":
        """
        Get the current configuration from control values.
        
        Returns:
            SimulationConfig with current values
        """
        from ..models.state import SimulationConfig
        
        return SimulationConfig(
            ticker=self._ticker_input.text().strip().upper() or "AAPL",
            historical_days=self._days_spin.value(),
            forecast_horizon=self._horizon_spin.value(),
            n_simulations=self._sims_spin.value(),
            n_paths_viz=self._paths_spin.value(),
            seed=self._seed_spin.value(),
            compute_greeks=self._greeks_check.isChecked(),
            generate_3d_plots=self._3d_check.isChecked(),
            risk_free_rate=self._rate_spin.value(),
        )

    def set_config(self, config: "SimulationConfig") -> None:
        """
        Set control values from a configuration.
        
        Args:
            config: Configuration to apply
        """
        self._ticker_input.setText(config.ticker)
        self._days_spin.setValue(config.historical_days)
        self._horizon_spin.setValue(config.forecast_horizon)
        self._sims_spin.setValue(config.n_simulations)
        self._paths_spin.setValue(config.n_paths_viz)
        self._seed_spin.setValue(config.seed)
        self._greeks_check.setChecked(config.compute_greeks)
        self._3d_check.setChecked(config.generate_3d_plots)
        self._rate_spin.setValue(config.risk_free_rate)

    def is_auto_fetch_enabled(self) -> bool:
        """Check if auto-fetch is enabled."""
        return self._auto_fetch_check.isChecked()

    def set_running(self, running: bool) -> None:
        """
        Update UI state for running/stopped status.
        
        Args:
            running: Whether a simulation is running
        """
        self._run_btn.setEnabled(not running)
        self._fetch_btn.setEnabled(not running)
        self._apply_preset_btn.setEnabled(not running)
        
        if running:
            self._run_btn.setText("⏳ Running...")
        else:
            self._run_btn.setText("▶ Run Analysis")

    def get_ticker(self) -> str:
        """Get the current ticker symbol."""
        return self._ticker_input.text().strip().upper() or "AAPL"

    def get_historical_days(self) -> int:
        """Get the historical days value."""
        return self._days_spin.value()

