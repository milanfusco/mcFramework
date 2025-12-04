"""
Sidebar widget with simulation controls.

This module provides the sidebar panel containing all input controls
for configuring and running simulations, following a clean component-based
design with grouped sections.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from PySide6.QtCore import (
    QPropertyAnimation,
    QSequentialAnimationGroup,
    QSettings,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..models.state import SimulationConfig, StatsConfig


# =============================================================================
# Constants
# =============================================================================

# Layout constants
SIDEBAR_WIDTH = 280
GROUP_SPACING = 12
CONTENT_MARGINS = (12, 12, 12, 12)
GROUP_CONTENT_MARGINS = (10, 14, 10, 10)
GROUP_CONTENT_SPACING = 6
LABEL_MIN_WIDTH = 100

# Ticker input constraints
TICKER_MAX_LENGTH = 10  # Allows ^GSPC, BRK.A, etc.
MAX_RECENT_TICKERS = 10
DEFAULT_RECENT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "^SPX"]

# Historical days constraints
MIN_HISTORICAL_DAYS = 30
MAX_HISTORICAL_DAYS = 2520  # ~10 years
DEFAULT_HISTORICAL_DAYS = 252

# Simulation constraints
MIN_SIMULATIONS = 100
MAX_SIMULATIONS = 100_000_000
DEFAULT_SIMULATIONS = 1000
SIMULATIONS_STEP = 500

MIN_VIZ_PATHS = 10
MAX_VIZ_PATHS = 1000
DEFAULT_VIZ_PATHS = 100

MIN_SEED = 0
MAX_SEED = 999_999
DEFAULT_SEED = 42

# Horizon constraints
MIN_HORIZON = 0.1
MAX_HORIZON = 10.0
DEFAULT_HORIZON = 1.0
HORIZON_STEP = 0.25

# Rate constraints
MIN_RATE = 0.0
MAX_RATE = 0.20
DEFAULT_RATE = 0.05
RATE_STEP = 0.005
RATE_DECIMALS = 3

# Option maturity constraints
MIN_MATURITY = 0.01  # ~3.6 days
MAX_MATURITY = 2.0   # 2 years
DEFAULT_MATURITY = 0.25  # 3 months
MATURITY_STEP = 0.01
MATURITY_DECIMALS = 2

# Strike percentage constraints (as % of spot)
MIN_STRIKE_PCT = 50.0
MAX_STRIKE_PCT = 150.0
DEFAULT_STRIKE_PCT = 100.0  # ATM
STRIKE_PCT_STEP = 1.0

# Strike price constraints (absolute $)
MIN_STRIKE_PRICE = 1.0
MAX_STRIKE_PRICE = 10000.0
DEFAULT_STRIKE_PRICE = 100.0
STRIKE_PRICE_STEP = 1.0
STRIKE_PRICE_DECIMALS = 2

# Statistics configuration constraints
MIN_CONFIDENCE = 0.80
MAX_CONFIDENCE = 0.99
DEFAULT_CONFIDENCE = 0.95
CONFIDENCE_STEP = 0.01
CONFIDENCE_DECIMALS = 2

MIN_BOOTSTRAP = 100
MAX_BOOTSTRAP = 100_000
DEFAULT_BOOTSTRAP = 10_000
BOOTSTRAP_STEP = 1000

CI_METHODS = ["auto", "z", "t"]  # Parametric methods for ci_mean
BOOTSTRAP_METHODS = ["percentile", "bca"]
NAN_POLICIES = ["omit", "propagate"]

# Animation constants
PULSE_FADE_DURATION_MS = 400
PULSE_OPACITY_MIN = 0.6
PULSE_OPACITY_MAX = 1.0

# Settings keys
SETTINGS_ORG = "McFramework"
SETTINGS_APP = "QuantBlackScholes"
SETTINGS_RECENT_TICKERS_KEY = "sidebar/recent_tickers"


# =============================================================================
# Utility Functions
# =============================================================================

@contextmanager
def block_signals(*widgets: QWidget) -> Generator[None, None, None]:
    """
    Context manager to temporarily block signals from widgets.
    
    Args:
        *widgets: Widgets to block signals on
        
    Yields:
        None
    """
    for w in widgets:
        w.blockSignals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.blockSignals(False)


# =============================================================================
# Data Classes
# =============================================================================

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


# =============================================================================
# Group Widget Classes
# =============================================================================

class GroupMixin:
    """
    Mixin providing common functionality for group widgets.
    
    This eliminates duplicate add_row() implementations across group classes.
    Subclasses must define self._layout as a QVBoxLayout.
    """

    _layout: QVBoxLayout  # Type hint for subclasses

    def add_row(
        self, label: str, widget: QWidget, label_width: int = LABEL_MIN_WIDTH
    ) -> QHBoxLayout:
        """
        Add a labeled row to the group.
        
        Args:
            label: Label text
            widget: Widget to add
            label_width: Minimum width for the label
            
        Returns:
            The created row layout
        """
        row = QHBoxLayout()
        row.setSpacing(8)
        
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(label_width)
        row.addWidget(label_widget)
        row.addWidget(widget, 1)
        
        self._layout.addLayout(row)
        return row

    def add_widget(self, widget: QWidget) -> None:
        """Add a widget to the group."""
        self._layout.addWidget(widget)


class CollapsibleGroup(GroupMixin, QGroupBox):
    """
    Collapsible group box for organizing controls.
    
    Click on the title to expand/collapse the group content.
    """

    def __init__(
        self, title: str, collapsed: bool = False, parent: QWidget | None = None
    ):
        """
        Initialize the collapsible group.

        Args:
            title: Group title
            collapsed: Initial collapsed state
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._title = title
        self._collapsed = collapsed
        
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setSpacing(GROUP_CONTENT_SPACING)
        self._layout.setContentsMargins(10, 6, 10, 10)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header button
        self._header = QPushButton(self._get_title_text())
        self._header.setObjectName("collapsibleHeader")
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.clicked.connect(self._toggle)
        self._header.setStyleSheet("""
            QPushButton#collapsibleHeader {
                background-color: transparent;
                border: none;
                color: #9a9aaa;
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
                text-align: left;
                padding: 8px 12px;
            }
            QPushButton#collapsibleHeader:hover {
                color: #c0c0d0;
            }
        """)
        layout.addWidget(self._header)
        
        layout.addWidget(self._content)
        self._content.setVisible(not collapsed)

    def _get_title_text(self) -> str:
        """Get title with collapse indicator."""
        arrow = "▼" if not self._collapsed else "▶"
        return f"{arrow}  {self._title}"

    def _toggle(self) -> None:
        """Toggle collapsed state."""
        self._collapsed = not self._collapsed
        self._header.setText(self._get_title_text())
        self._content.setVisible(not self._collapsed)


class ControlGroup(GroupMixin, QGroupBox):
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
        self._layout.setSpacing(GROUP_CONTENT_SPACING)
        self._layout.setContentsMargins(*GROUP_CONTENT_MARGINS)


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
        reset_requested: Emitted when Reset button is clicked
    """
    
    config_changed = Signal(object)  # SimulationConfig
    run_requested = Signal()
    fetch_requested = Signal()
    preset_selected = Signal(int)
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the sidebar widget.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setObjectName("sidebar")
        
        # Initialize animation state
        self._pulse_group: QSequentialAnimationGroup | None = None
        self._opacity_effect: QGraphicsOpacityEffect | None = None
        
        # Ticker validation state
        self._ticker_valid = True
        
        self._setup_ui()
        self._connect_signals()

    # -------------------------------------------------------------------------
    # Settings Persistence
    # -------------------------------------------------------------------------

    def _load_recent_tickers(self) -> list[str]:
        """
        Load recent tickers from persistent settings.
        
        Returns:
            List of recently used ticker symbols
        """
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        tickers = settings.value(SETTINGS_RECENT_TICKERS_KEY, DEFAULT_RECENT_TICKERS)
        if isinstance(tickers, list):
            return tickers[:MAX_RECENT_TICKERS]
        return list(DEFAULT_RECENT_TICKERS)

    def _save_recent_tickers(self) -> None:
        """Save recent tickers to persistent settings."""
        settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        settings.setValue(SETTINGS_RECENT_TICKERS_KEY, self._recent_tickers)

    def _setup_ui(self) -> None:
        """Set up the sidebar UI components."""
        # Main layout for the sidebar
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
        """)
        
        # Container widget for scrollable content
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(GROUP_SPACING)
        layout.setContentsMargins(*CONTENT_MARGINS)
        
        # Ticker & Data Group
        self._ticker_group = self._create_ticker_group()
        layout.addWidget(self._ticker_group)
        
        # Simulation Parameters Group
        self._sim_group = self._create_simulation_group()
        layout.addWidget(self._sim_group)
        
        # Option Pricing Group
        self._option_pricing_group = self._create_option_pricing_group()
        layout.addWidget(self._option_pricing_group)
        
        # Statistics Configuration Group
        self._stats_group = self._create_stats_group()
        layout.addWidget(self._stats_group)
        
        # Advanced Options Group
        self._options_group = self._create_options_group()
        layout.addWidget(self._options_group)
        
        # Presets Group
        self._presets_group = self._create_presets_group()
        layout.addWidget(self._presets_group)
        
        # Action Buttons
        self._action_buttons = self._create_action_buttons()
        layout.addWidget(self._action_buttons)
        
        # Small spacer at bottom
        layout.addSpacing(12)
        
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def _create_ticker_group(self) -> ControlGroup:
        """Create the ticker and data controls group."""
        group = ControlGroup("Ticker & Data")
        
        # Load recent tickers from persistent storage
        self._recent_tickers: list[str] = self._load_recent_tickers()
        
        # Ticker input
        self._ticker_input = QLineEdit()
        self._ticker_input.setPlaceholderText("e.g., AAPL, ^SPX")
        self._ticker_input.setText("AAPL")
        self._ticker_input.setMaxLength(TICKER_MAX_LENGTH)
        self._ticker_input.setToolTip(
            "Enter ticker symbol (AAPL, MSFT) or index (^SPX, ^DJI, ^VIX)"
        )
        self._ticker_input.setAccessibleName("Ticker Symbol")
        self._ticker_input.setAccessibleDescription(
            "Enter a stock ticker symbol like AAPL or MSFT"
        )
        group.add_row("Ticker:", self._ticker_input)
        
        # Recent tickers dropdown
        self._recent_combo = QComboBox()
        self._recent_combo.addItem("Recent Tickers...")
        for ticker in self._recent_tickers:
            self._recent_combo.addItem(ticker)
        self._recent_combo.setToolTip("Select from recently used tickers")
        self._recent_combo.currentIndexChanged.connect(self._on_recent_selected)
        group.add_widget(self._recent_combo)
        
        # Historical days
        self._days_spin = QSpinBox()
        self._days_spin.setRange(MIN_HISTORICAL_DAYS, MAX_HISTORICAL_DAYS)
        self._days_spin.setValue(DEFAULT_HISTORICAL_DAYS)
        self._days_spin.setSuffix(" days")
        self._days_spin.setToolTip("Number of historical trading days to fetch")
        group.add_row("History:", self._days_spin)
        
        return group

    def _on_recent_selected(self, index: int) -> None:
        """Handle selection from recent tickers dropdown."""
        if index > 0:  # Skip the placeholder item
            ticker = self._recent_combo.itemText(index)
            self._ticker_input.setText(ticker)
            self._recent_combo.setCurrentIndex(0)  # Reset to placeholder

    def add_to_recent(self, ticker: str) -> None:
        """
        Add a ticker to the recent history and persist to settings.
        
        Args:
            ticker: Ticker symbol to add
        """
        ticker = ticker.upper().strip()
        if not ticker:
            return
        
        # Remove if already exists (will re-add at top)
        if ticker in self._recent_tickers:
            self._recent_tickers.remove(ticker)
        
        # Add to front
        self._recent_tickers.insert(0, ticker)
        
        # Trim to max
        self._recent_tickers = self._recent_tickers[:MAX_RECENT_TICKERS]
        
        # Update combo box
        self._recent_combo.clear()
        self._recent_combo.addItem("Recent Tickers...")
        for t in self._recent_tickers:
            self._recent_combo.addItem(t)
        
        # Persist to settings
        self._save_recent_tickers()

    def _create_simulation_group(self) -> ControlGroup:
        """Create the simulation parameters group."""
        group = ControlGroup("Simulation")
        
        # Forecast horizon
        self._horizon_spin = QDoubleSpinBox()
        self._horizon_spin.setRange(MIN_HORIZON, MAX_HORIZON)
        self._horizon_spin.setValue(DEFAULT_HORIZON)
        self._horizon_spin.setSingleStep(HORIZON_STEP)
        self._horizon_spin.setSuffix(" years")
        self._horizon_spin.setDecimals(2)
        group.add_row("Horizon:", self._horizon_spin)
        
        # Number of simulations
        self._sims_spin = QSpinBox()
        self._sims_spin.setRange(MIN_SIMULATIONS, MAX_SIMULATIONS)
        self._sims_spin.setValue(DEFAULT_SIMULATIONS)
        self._sims_spin.setSingleStep(SIMULATIONS_STEP)
        self._sims_spin.setToolTip("Number of Monte Carlo simulations")
        group.add_row("Simulations:", self._sims_spin)
        
        # Paths for visualization
        self._paths_spin = QSpinBox()
        self._paths_spin.setRange(MIN_VIZ_PATHS, MAX_VIZ_PATHS)
        self._paths_spin.setValue(DEFAULT_VIZ_PATHS)
        self._paths_spin.setToolTip("Number of paths to visualize in charts")
        group.add_row("Viz Paths:", self._paths_spin)
        
        # Random seed
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(MIN_SEED, MAX_SEED)
        self._seed_spin.setValue(DEFAULT_SEED)
        self._seed_spin.setToolTip("Random seed for reproducibility")
        group.add_row("Seed:", self._seed_spin)
        
        # Risk-free rate
        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(MIN_RATE, MAX_RATE)
        self._rate_spin.setValue(DEFAULT_RATE)
        self._rate_spin.setSingleStep(RATE_STEP)
        self._rate_spin.setDecimals(RATE_DECIMALS)
        self._rate_spin.setSuffix(" (r)")
        self._rate_spin.setToolTip("Risk-free interest rate")
        group.add_row("Rate:", self._rate_spin)
        
        return group

    def _create_option_pricing_group(self) -> ControlGroup:
        """Create the option pricing parameters group."""
        group = ControlGroup("Option Contract")
        
        # Initialize spot price early (before it's used in label updates)
        self._current_spot_price: float = 100.0
        
        # Option maturity
        self._maturity_spin = QDoubleSpinBox()
        self._maturity_spin.setRange(MIN_MATURITY, MAX_MATURITY)
        self._maturity_spin.setValue(DEFAULT_MATURITY)
        self._maturity_spin.setSingleStep(MATURITY_STEP)
        self._maturity_spin.setSuffix(" years")
        self._maturity_spin.setDecimals(MATURITY_DECIMALS)
        self._maturity_spin.setToolTip(
            "Time to option expiration (0.25 = 3 months, 1.0 = 1 year)"
        )
        group.add_row("Maturity:", self._maturity_spin)
        
        # Days equivalent label
        self._maturity_days_label = QLabel()
        self._maturity_days_label.setStyleSheet("color: #888; font-size: 10px;")
        self._update_maturity_days_label()
        group.add_widget(self._maturity_days_label)
        
        # Strike mode toggle
        self._strike_mode_combo = QComboBox()
        self._strike_mode_combo.addItems(["% of Spot", "Manual $"])
        self._strike_mode_combo.setToolTip("Choose strike input mode")
        self._strike_mode_combo.currentIndexChanged.connect(self._on_strike_mode_changed)
        group.add_row("Strike Mode:", self._strike_mode_combo)
        
        # Strike percentage spinner (visible when mode = 0)
        self._strike_pct_spin = QDoubleSpinBox()
        self._strike_pct_spin.setRange(MIN_STRIKE_PCT, MAX_STRIKE_PCT)
        self._strike_pct_spin.setValue(DEFAULT_STRIKE_PCT)
        self._strike_pct_spin.setSingleStep(STRIKE_PCT_STEP)
        self._strike_pct_spin.setSuffix("%")
        self._strike_pct_spin.setDecimals(0)
        self._strike_pct_spin.setToolTip(
            "Strike as % of spot (100% = ATM, <100% = ITM call, >100% = OTM call)"
        )
        self._strike_pct_label = QLabel("Strike %:")
        self._strike_pct_label.setMinimumWidth(LABEL_MIN_WIDTH)
        pct_row = QHBoxLayout()
        pct_row.setSpacing(8)
        pct_row.addWidget(self._strike_pct_label)
        pct_row.addWidget(self._strike_pct_spin, 1)
        group._layout.addLayout(pct_row)
        
        # Strike price spinner (hidden initially, visible when mode = 1)
        self._strike_price_spin = QDoubleSpinBox()
        self._strike_price_spin.setRange(MIN_STRIKE_PRICE, MAX_STRIKE_PRICE)
        self._strike_price_spin.setValue(DEFAULT_STRIKE_PRICE)
        self._strike_price_spin.setSingleStep(STRIKE_PRICE_STEP)
        self._strike_price_spin.setPrefix("$")
        self._strike_price_spin.setDecimals(STRIKE_PRICE_DECIMALS)
        self._strike_price_spin.setToolTip("Enter strike price in dollars")
        self._strike_price_label = QLabel("Strike $:")
        self._strike_price_label.setMinimumWidth(LABEL_MIN_WIDTH)
        price_row = QHBoxLayout()
        price_row.setSpacing(8)
        price_row.addWidget(self._strike_price_label)
        price_row.addWidget(self._strike_price_spin, 1)
        group._layout.addLayout(price_row)
        # Hide initially
        self._strike_price_spin.setVisible(False)
        self._strike_price_label.setVisible(False)
        
        # Calculated strike value label (shows $ value when in % mode)
        self._strike_value_label = QLabel()
        self._strike_value_label.setStyleSheet("color: #5a9fd5; font-size: 10px;")
        self._update_strike_value_label()
        group.add_widget(self._strike_value_label)
        
        # Moneyness hint label
        self._strike_hint_label = QLabel()
        self._strike_hint_label.setStyleSheet("color: #888; font-size: 10px;")
        self._update_strike_hint_label()
        group.add_widget(self._strike_hint_label)
        
        return group
    
    def _on_strike_mode_changed(self, index: int) -> None:
        """Handle strike mode toggle between % and manual $."""
        is_pct_mode = index == 0
        
        # Toggle visibility of the spinners and their labels
        self._strike_pct_spin.setVisible(is_pct_mode)
        self._strike_pct_label.setVisible(is_pct_mode)
        self._strike_price_spin.setVisible(not is_pct_mode)
        self._strike_price_label.setVisible(not is_pct_mode)
        
        # Sync values between modes
        if is_pct_mode:
            # Switching to % mode - convert $ to %
            if self._current_spot_price > 0:
                pct = (self._strike_price_spin.value() / self._current_spot_price) * 100
                self._strike_pct_spin.setValue(pct)
        else:
            # Switching to $ mode - convert % to $
            strike_price = self._current_spot_price * (self._strike_pct_spin.value() / 100)
            self._strike_price_spin.setValue(strike_price)
        
        self._update_strike_value_label()
        self._update_strike_hint_label()
        self._on_config_changed()

    def _update_maturity_days_label(self) -> None:
        """Update the maturity days equivalent label."""
        years = self._maturity_spin.value()
        days = int(years * 252)  # Trading days
        calendar_days = int(years * 365)
        self._maturity_days_label.setText(f"≈ {days} trading days ({calendar_days} calendar)")

    def _update_strike_value_label(self) -> None:
        """Update the calculated strike value display."""
        pct = self.get_strike_pct()
        strike_price = self._current_spot_price * (pct / 100)
        
        if self._strike_mode_combo.currentIndex() == 0:
            # In % mode - show calculated $ value
            self._strike_value_label.setText(f"= ${strike_price:,.2f} strike")
            self._strike_value_label.setVisible(True)
        else:
            # In $ mode - show equivalent %
            self._strike_value_label.setText(f"= {pct:.1f}% of spot")
            self._strike_value_label.setVisible(True)

    def _update_strike_hint_label(self) -> None:
        """Update the strike moneyness hint label."""
        pct = self.get_strike_pct()
        if pct < 98:
            hint = "ITM Call / OTM Put"
            color = "#2adf7a"
        elif pct > 102:
            hint = "OTM Call / ITM Put"
            color = "#ff5a6a"
        else:
            hint = "At-the-Money (ATM)"
            color = "#f0b90b"
        self._strike_hint_label.setText(hint)
        self._strike_hint_label.setStyleSheet(f"color: {color}; font-size: 10px;")
    
    def get_strike_pct(self) -> float:
        """Get the strike percentage regardless of input mode."""
        if self._strike_mode_combo.currentIndex() == 0:
            # % mode
            return self._strike_pct_spin.value()
        else:
            # $ mode - convert to %
            if self._current_spot_price > 0:
                return (self._strike_price_spin.value() / self._current_spot_price) * 100
            return 100.0
    
    def set_spot_price(self, spot: float) -> None:
        """
        Update the current spot price for strike calculations.
        
        Called when market data is fetched to enable accurate
        $ to % conversions.
        
        Args:
            spot: Current spot price
        """
        if spot <= 0:
            return
        
        old_spot = self._current_spot_price
        self._current_spot_price = spot
        
        # Update $ spinner range based on spot
        self._strike_price_spin.setRange(
            spot * (MIN_STRIKE_PCT / 100),
            spot * (MAX_STRIKE_PCT / 100),
        )
        
        # If in $ mode, adjust value proportionally
        if self._strike_mode_combo.currentIndex() == 1 and old_spot > 0:
            ratio = self._strike_price_spin.value() / old_spot
            self._strike_price_spin.setValue(spot * ratio)
        elif self._strike_mode_combo.currentIndex() == 0:
            # Update the displayed $ value
            self._update_strike_value_label()

    def _create_stats_group(self) -> CollapsibleGroup:
        """Create the statistics configuration group (collapsible)."""
        group = CollapsibleGroup("Statistics (StatsEngine)", collapsed=True)
        
        # Compute extended stats checkbox
        self._compute_stats_check = QCheckBox("Enable Extended Stats")
        self._compute_stats_check.setChecked(True)
        self._compute_stats_check.setToolTip(
            "Compute comprehensive statistics using StatsEngine "
            "(skewness, kurtosis, multiple CI methods, etc.)"
        )
        group.add_widget(self._compute_stats_check)
        
        # Confidence level
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(MIN_CONFIDENCE, MAX_CONFIDENCE)
        self._confidence_spin.setValue(DEFAULT_CONFIDENCE)
        self._confidence_spin.setSingleStep(CONFIDENCE_STEP)
        self._confidence_spin.setDecimals(CONFIDENCE_DECIMALS)
        self._confidence_spin.setToolTip(
            "Confidence level for confidence intervals (e.g., 0.95 = 95%)"
        )
        group.add_row("Confidence:", self._confidence_spin)
        
        # --- Parametric CI Section ---
        parametric_header = QLabel("─── Parametric CI ───")
        parametric_header.setStyleSheet(
            "color: #6a7a8a; font-size: 10px; margin-top: 8px;"
        )
        parametric_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group.add_widget(parametric_header)
        
        # CI Method selection (parametric: auto, z, t)
        self._ci_method_combo = QComboBox()
        self._ci_method_combo.addItems(CI_METHODS)
        self._ci_method_combo.setCurrentText("auto")
        self._ci_method_combo.setToolTip(
            "Parametric CI method for ci_mean:\n"
            "• auto: Use t for n<30, z otherwise\n"
            "• z: Normal distribution (large samples)\n"
            "• t: Student's t (small samples)"
        )
        group.add_row("Method:", self._ci_method_combo)
        
        # --- Bootstrap CI Section ---
        bootstrap_header = QLabel("─── Bootstrap CI ───")
        bootstrap_header.setStyleSheet(
            "color: #6a7a8a; font-size: 10px; margin-top: 8px;"
        )
        bootstrap_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group.add_widget(bootstrap_header)
        
        # Enable Bootstrap CI checkbox
        self._enable_bootstrap_check = QCheckBox("Enable Bootstrap CI")
        self._enable_bootstrap_check.setChecked(True)
        self._enable_bootstrap_check.setToolTip(
            "Compute bootstrap confidence interval (ci_mean_bootstrap)\n"
            "using resampling methods"
        )
        self._enable_bootstrap_check.stateChanged.connect(self._on_bootstrap_toggled)
        group.add_widget(self._enable_bootstrap_check)
        
        # Bootstrap method dropdown
        self._bootstrap_method_combo = QComboBox()
        self._bootstrap_method_combo.addItems(BOOTSTRAP_METHODS)
        self._bootstrap_method_combo.setCurrentText("percentile")
        self._bootstrap_method_combo.setToolTip(
            "Bootstrap flavor:\n"
            "• percentile: Simple percentile method\n"
            "• bca: Bias-corrected and accelerated"
        )
        group.add_row("  Flavor:", self._bootstrap_method_combo)
        
        # Number of bootstrap samples
        self._n_bootstrap_spin = QSpinBox()
        self._n_bootstrap_spin.setRange(MIN_BOOTSTRAP, MAX_BOOTSTRAP)
        self._n_bootstrap_spin.setValue(DEFAULT_BOOTSTRAP)
        self._n_bootstrap_spin.setSingleStep(BOOTSTRAP_STEP)
        self._n_bootstrap_spin.setToolTip(
            "Number of bootstrap resamples"
        )
        group.add_row("  Resamples:", self._n_bootstrap_spin)
        
        # --- Chebyshev CI Section ---
        chebyshev_header = QLabel("─── Chebyshev CI ───")
        chebyshev_header.setStyleSheet(
            "color: #6a7a8a; font-size: 10px; margin-top: 8px;"
        )
        chebyshev_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group.add_widget(chebyshev_header)
        
        # Enable Chebyshev CI checkbox
        self._enable_chebyshev_check = QCheckBox("Enable Chebyshev CI")
        self._enable_chebyshev_check.setChecked(True)
        self._enable_chebyshev_check.setToolTip(
            "Compute distribution-free CI (ci_mean_chebyshev)\n"
            "using Chebyshev's inequality.\n"
            "Wider but makes no distributional assumptions."
        )
        group.add_widget(self._enable_chebyshev_check)
        
        # --- General Options ---
        general_header = QLabel("─── General ───")
        general_header.setStyleSheet(
            "color: #6a7a8a; font-size: 10px; margin-top: 8px;"
        )
        general_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group.add_widget(general_header)
        
        # NaN policy
        self._nan_policy_combo = QComboBox()
        self._nan_policy_combo.addItems(NAN_POLICIES)
        self._nan_policy_combo.setCurrentText("omit")
        self._nan_policy_combo.setToolTip(
            "How to handle NaN/infinite values:\n"
            "• omit: Drop non-finite values\n"
            "• propagate: Include in calculations"
        )
        group.add_row("NaN Policy:", self._nan_policy_combo)
        
        return group
    
    def _on_bootstrap_toggled(self, state: int) -> None:
        """Handle Bootstrap CI enable/disable toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self._bootstrap_method_combo.setEnabled(enabled)
        self._n_bootstrap_spin.setEnabled(enabled)
        self._on_config_changed()

    def _create_options_group(self) -> CollapsibleGroup:
        """Create the options toggles group (collapsible)."""
        group = CollapsibleGroup("Advanced Options", collapsed=True)
        
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

    def _create_presets_group(self) -> CollapsibleGroup:
        """Create the scenario presets group (collapsible)."""
        group = CollapsibleGroup("Scenarios", collapsed=True)
        
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
        self._fetch_btn.setAccessibleName("Fetch Data")
        self._fetch_btn.setAccessibleDescription(
            "Fetch historical market data without running Monte Carlo simulation"
        )
        layout.addWidget(self._fetch_btn)
        
        # Run button (primary action)
        self._run_btn = QPushButton("▶ Run Analysis")
        self._run_btn.setObjectName("runButton")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setToolTip("Fetch data and run full simulation")
        self._run_btn.setDefault(True)  # Enter key activates this button
        self._run_btn.setAccessibleName("Run Analysis")
        self._run_btn.setAccessibleDescription(
            "Fetch market data and run full Monte Carlo simulation with options pricing"
        )
        layout.addWidget(self._run_btn)
        
        # Reset button
        self._reset_btn = QPushButton("↺ Reset Workspace")
        self._reset_btn.setObjectName("resetButton")
        self._reset_btn.setToolTip("Clear all data and reset to initial state")
        self._reset_btn.setAccessibleName("Reset Workspace")
        self._reset_btn.setAccessibleDescription(
            "Clear all charts, data, and results, returning to initial empty state"
        )
        layout.addWidget(self._reset_btn)
        
        return frame

    def _connect_signals(self) -> None:
        """Connect internal signals to handlers."""
        # Ticker validation and quick-entry
        self._ticker_input.textChanged.connect(self._validate_ticker)
        self._ticker_input.textChanged.connect(self._on_config_changed)
        self._ticker_input.returnPressed.connect(self._on_ticker_enter_pressed)
        
        # Value change signals
        self._days_spin.valueChanged.connect(self._on_config_changed)
        self._horizon_spin.valueChanged.connect(self._on_config_changed)
        self._sims_spin.valueChanged.connect(self._on_config_changed)
        self._paths_spin.valueChanged.connect(self._on_config_changed)
        self._seed_spin.valueChanged.connect(self._on_config_changed)
        self._rate_spin.valueChanged.connect(self._on_config_changed)
        self._greeks_check.stateChanged.connect(self._on_config_changed)
        self._3d_check.stateChanged.connect(self._on_config_changed)
        
        # Option pricing signals
        self._maturity_spin.valueChanged.connect(self._on_config_changed)
        self._maturity_spin.valueChanged.connect(self._update_maturity_days_label)
        self._strike_pct_spin.valueChanged.connect(self._on_config_changed)
        self._strike_pct_spin.valueChanged.connect(self._update_strike_value_label)
        self._strike_pct_spin.valueChanged.connect(self._update_strike_hint_label)
        self._strike_price_spin.valueChanged.connect(self._on_config_changed)
        self._strike_price_spin.valueChanged.connect(self._update_strike_value_label)
        self._strike_price_spin.valueChanged.connect(self._update_strike_hint_label)
        
        # Statistics configuration signals
        self._compute_stats_check.stateChanged.connect(self._on_config_changed)
        self._confidence_spin.valueChanged.connect(self._on_config_changed)
        self._ci_method_combo.currentTextChanged.connect(self._on_config_changed)
        self._enable_bootstrap_check.stateChanged.connect(self._on_config_changed)
        self._bootstrap_method_combo.currentTextChanged.connect(self._on_config_changed)
        self._n_bootstrap_spin.valueChanged.connect(self._on_config_changed)
        self._enable_chebyshev_check.stateChanged.connect(self._on_config_changed)
        self._nan_policy_combo.currentTextChanged.connect(self._on_config_changed)
        
        # Preset signals
        self._preset_combo.currentIndexChanged.connect(self._update_preset_hint)
        self._apply_preset_btn.clicked.connect(self._on_apply_preset)
        
        # Action signals
        self._fetch_btn.clicked.connect(self.fetch_requested.emit)
        self._run_btn.clicked.connect(self.run_requested.emit)
        self._reset_btn.clicked.connect(self.reset_requested.emit)

    # -------------------------------------------------------------------------
    # Ticker Validation
    # -------------------------------------------------------------------------

    def _validate_ticker(self, text: str) -> None:
        """
        Validate ticker input and provide visual feedback.
        
        Supports:
        - Standard tickers: AAPL, MSFT, GOOGL
        - Index symbols: ^SPX, ^GSPC, ^DJI, ^VIX
        - Share classes: BRK.A, BRK.B
        - Preferred/warrants: BAC-PL, SPCE.WS
        
        Args:
            text: Current ticker input text
        """
        import re
        
        text = text.strip().upper()
        
        # Pattern allows:
        # - Optional ^ prefix for indexes
        # - Alphanumeric base (1-5 chars)
        # - Optional suffix with . or - followed by 1-2 alphanumeric chars
        pattern = r"^\^?[A-Z]{1,5}([.\-][A-Z0-9]{1,2})?$"
        is_valid = bool(text) and bool(re.match(pattern, text))
        
        self._ticker_valid = is_valid
        
        if is_valid:
            self._ticker_input.setStyleSheet("")
        else:
            self._ticker_input.setStyleSheet(
                "QLineEdit { border: 1px solid #ff5a6a; }"
            )
        
        # Update run button state (only if not currently running)
        if self._run_btn.isEnabled() or not is_valid:
            self._run_btn.setEnabled(is_valid)
            self._fetch_btn.setEnabled(is_valid)

    def _on_ticker_enter_pressed(self) -> None:
        """Handle Enter key in ticker input - immediately run analysis."""
        if not self._ticker_valid:
            return
        
        ticker = self._ticker_input.text().strip().upper()
        if ticker:
            self.add_to_recent(ticker)
            self.run_requested.emit()

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
        # Use context manager to block signals during batch update
        with block_signals(
            self._ticker_input,
            self._days_spin,
            self._horizon_spin,
            self._sims_spin,
        ):
            pass
        self._ticker_input.setText(preset.ticker)
        self._days_spin.setValue(preset.historical_days)
        self._horizon_spin.setValue(preset.forecast_horizon)
        self._sims_spin.setValue(preset.n_simulations)
        
        # Emit single config changed signal
        self._on_config_changed()

    def get_config(self) -> "SimulationConfig":
        """
        Get the current configuration from control values.
        
        Returns:
            SimulationConfig with current values
        """
        from ..models.state import SimulationConfig, StatsConfig
        
        stats_config = StatsConfig(
            confidence=self._confidence_spin.value(),
            ci_method=self._ci_method_combo.currentText(),
            enable_bootstrap_ci=self._enable_bootstrap_check.isChecked(),
            bootstrap_method=self._bootstrap_method_combo.currentText(),
            n_bootstrap=self._n_bootstrap_spin.value(),
            enable_chebyshev_ci=self._enable_chebyshev_check.isChecked(),
            nan_policy=self._nan_policy_combo.currentText(),
            compute_stats=self._compute_stats_check.isChecked(),
        )
        
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
            option_maturity=self._maturity_spin.value(),
            strike_pct=self.get_strike_pct(),
            stats=stats_config,
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
        self._maturity_spin.setValue(config.option_maturity)
        self._strike_pct_spin.setValue(config.strike_pct)
        # Also update $ value if we have spot price
        if self._current_spot_price > 0:
            self._strike_price_spin.setValue(
                self._current_spot_price * (config.strike_pct / 100)
            )
        
        # Set stats configuration
        self._compute_stats_check.setChecked(config.stats.compute_stats)
        self._confidence_spin.setValue(config.stats.confidence)
        self._ci_method_combo.setCurrentText(config.stats.ci_method)
        self._enable_bootstrap_check.setChecked(config.stats.enable_bootstrap_ci)
        self._bootstrap_method_combo.setCurrentText(config.stats.bootstrap_method)
        self._n_bootstrap_spin.setValue(config.stats.n_bootstrap)
        self._enable_chebyshev_check.setChecked(config.stats.enable_chebyshev_ci)
        self._nan_policy_combo.setCurrentText(config.stats.nan_policy)
        # Update bootstrap controls enabled state
        self._on_bootstrap_toggled(
            Qt.CheckState.Checked.value if config.stats.enable_bootstrap_ci 
            else Qt.CheckState.Unchecked.value
        )

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
        self._reset_btn.setEnabled(not running)
        
        if running:
            self._run_btn.setText("⏳ Running...")
            self._start_pulse_animation()
        else:
            self._run_btn.setText("▶ Run Analysis")
            self._stop_pulse_animation()

    def _start_pulse_animation(self) -> None:
        """Start a real pulse animation with opacity fade on the run button."""
        if self._pulse_group is not None:
            return
        
        # Apply running style
        self._run_btn.setStyleSheet("""
            QPushButton#runButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3abf6c, stop:1 #2a9f5c);
                border: 2px solid #4acf7c;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
        """)
        
        # Create opacity effect
        self._opacity_effect = QGraphicsOpacityEffect(self._run_btn)
        self._run_btn.setGraphicsEffect(self._opacity_effect)
        
        # Create fade out animation
        fade_out = QPropertyAnimation(self._opacity_effect, b"opacity")
        fade_out.setDuration(PULSE_FADE_DURATION_MS)
        fade_out.setStartValue(PULSE_OPACITY_MAX)
        fade_out.setEndValue(PULSE_OPACITY_MIN)
        
        # Create fade in animation
        fade_in = QPropertyAnimation(self._opacity_effect, b"opacity")
        fade_in.setDuration(PULSE_FADE_DURATION_MS)
        fade_in.setStartValue(PULSE_OPACITY_MIN)
        fade_in.setEndValue(PULSE_OPACITY_MAX)
        
        # Combine into sequential group with infinite loop
        self._pulse_group = QSequentialAnimationGroup()
        self._pulse_group.addAnimation(fade_out)
        self._pulse_group.addAnimation(fade_in)
        self._pulse_group.setLoopCount(-1)  # Infinite loop
        self._pulse_group.start()

    def _stop_pulse_animation(self) -> None:
        """Stop the pulse animation and clean up effects."""
        if self._pulse_group is not None:
            self._pulse_group.stop()
            self._pulse_group = None
        
        # Remove graphics effect
        self._run_btn.setGraphicsEffect(None)
        self._opacity_effect = None
        
        # Reset to default stylesheet
        self._run_btn.setStyleSheet("")

    def get_ticker(self) -> str:
        """Get the current ticker symbol."""
        return self._ticker_input.text().strip().upper() or "AAPL"

    def get_historical_days(self) -> int:
        """Get the historical days value."""
        return self._days_spin.value()

    def get_stats_config(self) -> "StatsConfig":
        """
        Get the current statistics configuration.
        
        Returns:
            StatsConfig with current values
        """
        from ..models.state import StatsConfig
        
        return StatsConfig(
            confidence=self._confidence_spin.value(),
            ci_method=self._ci_method_combo.currentText(),
            enable_bootstrap_ci=self._enable_bootstrap_check.isChecked(),
            bootstrap_method=self._bootstrap_method_combo.currentText(),
            n_bootstrap=self._n_bootstrap_spin.value(),
            enable_chebyshev_ci=self._enable_chebyshev_check.isChecked(),
            nan_policy=self._nan_policy_combo.currentText(),
            compute_stats=self._compute_stats_check.isChecked(),
        )

    def is_stats_enabled(self) -> bool:
        """Check if extended statistics computation is enabled."""
        return self._compute_stats_check.isChecked()

