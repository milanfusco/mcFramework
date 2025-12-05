"""
Empty state widget for displaying placeholder content.

This module provides a reusable empty state component that shows
instructional content when tabs have no data to display.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class EmptyStateWidget(QFrame):
    """
    Placeholder widget for empty tab states.
    
    Displays an icon, title, description, and optional action button
    when a tab has no data to show.
    """

    def __init__(
        self,
        icon: str = "📊",
        title: str = "No Data",
        description: str = "Data will appear here once available.",
        action_text: str | None = None,
        parent: QWidget | None = None,
    ):
        """
        Initialize the empty state widget.
        
        Args:
            icon: Emoji or icon character to display
            title: Main title text
            description: Descriptive text explaining the empty state
            action_text: Optional text for action button
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("emptyState")
        
        self._action_callback = None
        self._setup_ui(icon, title, description, action_text)

    def _setup_ui(
        self,
        icon: str,
        title: str,
        description: str,
        action_text: str | None,
    ) -> None:
        """Set up the empty state UI."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 60, 40, 60)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setObjectName("emptyStateIcon")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("""
            font-size: 64px;
            color: #4a6fa5;
            padding: 20px;
        """)
        layout.addWidget(icon_label)
        
        # Title
        title_label = QLabel(title)
        title_label.setObjectName("emptyStateTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #e0e0e0;
            margin-top: 8px;
        """)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setObjectName("emptyStateDescription")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(400)
        desc_label.setStyleSheet("""
            font-size: 13px;
            color: #888888;
            line-height: 1.5;
        """)
        layout.addWidget(desc_label)
        
        # Action button (optional)
        if action_text:
            self._action_btn = QPushButton(action_text)
            self._action_btn.setObjectName("emptyStateAction")
            self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._action_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a6fa5;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 12px 24px;
                    font-size: 13px;
                    font-weight: bold;
                    margin-top: 16px;
                }
                QPushButton:hover {
                    background-color: #5a8fd5;
                }
                QPushButton:pressed {
                    background-color: #3d5a80;
                }
            """)
            self._action_btn.clicked.connect(self._on_action_clicked)
            layout.addWidget(self._action_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        else:
            self._action_btn = None
        
        # Set frame styling
        self.setStyleSheet("""
            QFrame#emptyState {
                background-color: transparent;
                border: 2px dashed #3d3d5c;
                border-radius: 12px;
                margin: 20px;
            }
        """)

    def set_action_callback(self, callback) -> None:
        """
        Set the callback for the action button.
        
        Args:
            callback: Callable to invoke when action button is clicked
        """
        self._action_callback = callback

    def _on_action_clicked(self) -> None:
        """Handle action button click."""
        if self._action_callback:
            self._action_callback()


class MarketDataEmptyState(EmptyStateWidget):
    """Empty state for Market Data tab."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            icon="📈",
            title="No Market Data",
            description=(
                "Enter a stock ticker symbol in the sidebar and click "
                "'Fetch Data Only' or 'Run Analysis' to load historical price data."
            ),
            action_text="Fetch Data",
            parent=parent,
        )


class MonteCarloEmptyState(EmptyStateWidget):
    """Empty state for Monte Carlo tab."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            icon="🎲",
            title="No Simulation Results",
            description=(
                "Run a Monte Carlo simulation to see price path projections, "
                "return distributions, and forecast statistics."
            ),
            action_text="Run Analysis",
            parent=parent,
        )


class OptionsEmptyState(EmptyStateWidget):
    """Empty state for Options & Greeks tab."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            icon="📊",
            title="No Option Pricing Data",
            description=(
                "Option prices and Greeks will appear here after running "
                "a simulation. Use the what-if sliders to explore sensitivity."
            ),
            action_text="Run Analysis",
            parent=parent,
        )


class SurfacesEmptyState(EmptyStateWidget):
    """Empty state for 3D Surfaces tab."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            icon="🌐",
            title="No 3D Surfaces",
            description=(
                "Interactive 3D option price surfaces will appear here. "
                "Make sure 'Generate 3D Surfaces' is enabled in the sidebar options."
            ),
            action_text="Run Analysis",
            parent=parent,
        )

