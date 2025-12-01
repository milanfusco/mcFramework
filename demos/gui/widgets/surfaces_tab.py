"""
3D Surfaces tab with interactive option price surfaces.

This module provides the 3D Surfaces tab displaying option price
surfaces for call and put options using embedded matplotlib 3D plots
with rotation and zoom capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .charts import OptionSurfaceChart

if TYPE_CHECKING:
    from ..models.state import TickerAnalysisState


class SurfaceExportPanel(QFrame):
    """
    Panel with export buttons for 3D surface charts.
    
    Signals:
        export_requested: Emitted with surface type when export is clicked
    """
    
    export_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the export panel."""
        super().__init__(parent)
        self.setObjectName("surfaceExportPanel")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the export panel UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        label = QLabel("Export:")
        label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(label)
        
        surfaces = [
            ("call", "Call Surface"),
            ("put", "Put Surface"),
        ]
        
        for surface_id, name in surfaces:
            btn = QPushButton(name)
            btn.setObjectName("exportBtn")
            btn.setToolTip(f"Export {name} as PNG")
            btn.clicked.connect(
                lambda checked, sid=surface_id: self.export_requested.emit(sid)
            )
            layout.addWidget(btn)
        
        layout.addStretch()


class SurfaceInfoPanel(QFrame):
    """
    Information panel explaining the 3D surface plots.
    
    Provides context about what the surfaces represent and
    how to interact with them.
    """

    def __init__(self, parent: QWidget | None = None):
        """Initialize the info panel."""
        super().__init__(parent)
        self.setObjectName("surfaceInfoPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        title = QLabel("Understanding 3D Option Surfaces")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #ccc;")
        layout.addWidget(title)
        
        # Explanation text
        explanation = QLabel(
            "The 3D surfaces show how option prices evolve:<br><br>"
            "• <b>X-axis (Time)</b>: Time from now until maturity<br>"
            "• <b>Y-axis (Stock Price)</b>: Underlying asset price<br>"
            "• <b>Z-axis (Option Price)</b>: Theoretical option value<br><br>"
            "The colored paths represent Monte Carlo simulated "
            "trajectories. The highlighted line shows the mean path."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #aaa; font-size: 11px;")
        explanation.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(explanation)
        
        # Interaction hints
        hints_title = QLabel("Interactions")
        hints_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #ccc; margin-top: 8px;"
        )
        layout.addWidget(hints_title)

        hints = QLabel(
            "• <b>Drag</b>: Rotate the 3D view<br>"
            "• <b>Scroll</b>: Zoom in/out<br>"
            "• <b>Toolbar</b>: Pan, reset, save<br>"
            "• <b>Home button</b>: Reset view"
        )
        hints.setTextFormat(Qt.TextFormat.RichText)
        hints.setWordWrap(True)
        hints.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(hints)
        
        # Key observations
        key_title = QLabel("Key Observations")
        key_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #ccc; margin-top: 8px;"
        )
        layout.addWidget(key_title)
        
        observations = QLabel(
            "• <span style='color: #00d26a;'>Call surfaces</span> "
            "increase as stock price rises<br>"
            "• <span style='color: #f23645;'>Put surfaces</span> "
            "increase as stock price falls<br>"
            "• Both surfaces decrease as time approaches maturity<br>"
            "• Surface curvature reflects gamma (convexity)<br>"
        )
        observations.setWordWrap(True)
        observations.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(observations)
        
        layout.addStretch()


class SurfacesTab(QWidget):
    """
    3D Surfaces tab with interactive option price surfaces.
    
    Shows call and put option price surfaces as 3D plots with
    simulated price paths overlaid. Supports mouse rotation,
    zoom, and PNG export.
    
    Signals:
        surface_export_requested: Emitted with (surface_type, export_func)
    """
    
    surface_export_requested = Signal(str, object)

    def __init__(self, parent: QWidget | None = None):
        """Initialize the Surfaces tab."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the tab UI with interactive 3D charts."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Export panel
        self._export_panel = SurfaceExportPanel()
        self._export_panel.export_requested.connect(self._on_export_requested)
        layout.addWidget(self._export_panel)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Scroll area for surfaces
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(12)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Call surface (interactive 3D)
        call_group = QGroupBox("Call Option Price Surface")
        call_layout = QVBoxLayout(call_group)
        call_layout.setContentsMargins(4, 12, 4, 4)
        self._call_surface = OptionSurfaceChart(option_type="call")
        self._call_surface.setMinimumHeight(450)
        call_layout.addWidget(self._call_surface)
        scroll_layout.addWidget(call_group)
        
        # Put surface (interactive 3D)
        put_group = QGroupBox("Put Option Price Surface")
        put_layout = QVBoxLayout(put_group)
        put_layout.setContentsMargins(4, 12, 4, 4)
        self._put_surface = OptionSurfaceChart(option_type="put")
        self._put_surface.setMinimumHeight(450)
        put_layout.addWidget(self._put_surface)
        scroll_layout.addWidget(put_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_splitter.addWidget(scroll)
        
        # Info panel on the right
        self._info_panel = SurfaceInfoPanel()
        self._info_panel.setMaximumWidth(260)
        main_splitter.addWidget(self._info_panel)
        
        main_splitter.setSizes([700, 260])
        layout.addWidget(main_splitter, 1)

    def _on_export_requested(self, surface_type: str) -> None:
        """Handle surface export request."""
        surface_map = {
            "call": self._call_surface,
            "put": self._put_surface,
        }
        
        surface = surface_map.get(surface_type)
        if surface:
            self.surface_export_requested.emit(surface_type, surface.export_to_png)

    def update_from_state(self, state: "TickerAnalysisState") -> None:
        """
        Update surfaces from the application state.
        
        Args:
            state: Current application state
        """
        # Check if we have required data and 3D plots are enabled
        if not state.has_simulation_results():
            return
        
        if not state.config.generate_3d_plots:
            return
        
        if state.parameters is None:
            return
        
        paths = state.simulated_paths
        params = state.parameters
        config = state.config
        ticker = config.ticker.upper()
        
        # Update Call surface
        self._call_surface.update_data(
            paths=paths,
            spot_price=params.spot_price,
            volatility=params.volatility,
            risk_free_rate=config.risk_free_rate,
            forecast_horizon=config.forecast_horizon,
            ticker=ticker,
        )
        
        # Update Put surface
        self._put_surface.update_data(
            paths=paths,
            spot_price=params.spot_price,
            volatility=params.volatility,
            risk_free_rate=config.risk_free_rate,
            forecast_horizon=config.forecast_horizon,
            ticker=ticker,
        )

    def clear(self) -> None:
        """Clear all surfaces."""
        self._call_surface.clear()
        self._put_surface.clear()

    def export_surface(self, surface_type: str, path: Path) -> bool:
        """
        Export a specific surface to PNG.
        
        Args:
            surface_type: "call" or "put"
            path: Output file path
            
        Returns:
            True if exported successfully
        """
        surface_map = {
            "call": self._call_surface,
            "put": self._put_surface,
        }
        
        surface = surface_map.get(surface_type)
        if surface:
            try:
                surface.export_to_png(path)
                return True
            except Exception:
                return False
        return False
