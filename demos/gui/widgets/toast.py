"""
Toast notification widget.

This module provides a non-intrusive toast notification system
for displaying success, error, and info messages.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QPropertyAnimation,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class ToastType(Enum):
    """Toast notification types."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# Styling for each toast type
TOAST_STYLES = {
    ToastType.SUCCESS: {
        "icon": "✓",
        "bg": "#1a4d2e",
        "border": "#2a8f5c",
        "text": "#8fe8a0",
    },
    ToastType.ERROR: {
        "icon": "✗",
        "bg": "#4d1a1a",
        "border": "#a03030",
        "text": "#f8a0a0",
    },
    ToastType.WARNING: {
        "icon": "⚠",
        "bg": "#4d3a1a",
        "border": "#a08030",
        "text": "#f8d8a0",
    },
    ToastType.INFO: {
        "icon": "ℹ",
        "bg": "#1a2a4d",
        "border": "#3070a0",
        "text": "#a0c8f8",
    },
}


class ToastWidget(QFrame):
    """
    Individual toast notification widget.
    
    A small, animated notification that appears briefly and fades out.
    
    Signals:
        closed: Emitted when the toast is closed
    """
    
    closed = Signal()

    def __init__(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration: int = 3000,
        parent: QWidget | None = None,
    ):
        """
        Initialize a toast notification.
        
        Args:
            message: Message to display
            toast_type: Type of notification (affects styling)
            duration: How long to show in milliseconds (0 = indefinite)
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("toast")
        # Ensure toast widgets capture mouse events for close button
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        
        self._duration = duration
        self._toast_type = toast_type
        self._setup_ui(message)
        self._setup_animation()
        
        if duration > 0:
            QTimer.singleShot(duration, self._fade_out)

    def _setup_ui(self, message: str) -> None:
        """Set up the toast UI."""
        style = TOAST_STYLES[self._toast_type]
        
        self.setStyleSheet(f"""
            QFrame#toast {{
                background-color: {style['bg']};
                border: 1px solid {style['border']};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        
        # Icon
        icon = QLabel(style["icon"])
        icon.setStyleSheet(f"""
            font-size: 12px;
            color: {style['text']};
        """)
        layout.addWidget(icon)
        
        # Message
        msg_label = QLabel(message)
        msg_label.setStyleSheet(f"""
            color: {style['text']};
            font-size: 11px;
            font-weight: 500;
        """)
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(200)
        layout.addWidget(msg_label, 1)
        
        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(14, 14)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {style['text']};
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: #ffffff;
            }}
        """)
        close_btn.clicked.connect(self._fade_out)
        layout.addWidget(close_btn)
        
        self.setMinimumWidth(160)
        self.setMaximumWidth(260)
        self.adjustSize()

    def _setup_animation(self) -> None:
        """Set up fade animation."""
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        
        # Fade in
        self._fade_in_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_in_anim.setDuration(200)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.start()

    def _fade_out(self) -> None:
        """Fade out and close the toast."""
        self._fade_out_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_out_anim.setDuration(300)
        self._fade_out_anim.setStartValue(1.0)
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.finished.connect(self._on_fade_complete)
        self._fade_out_anim.start()

    def _on_fade_complete(self) -> None:
        """Handle fade complete."""
        self.closed.emit()
        self.deleteLater()


class ToastManager(QWidget):
    """
    Manager for displaying toast notifications.
    
    Can be embedded in a toolbar or used as an overlay.
    Shows the most recent toast inline.
    """

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the toast manager.
        
        Args:
            parent: Parent widget (toolbar or main window)
        """
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self._toasts: list[ToastWidget] = []
        
        # Horizontal layout for inline toolbar display
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

    def show_toast(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        duration: int = 3000,
    ) -> ToastWidget:
        """
        Show a toast notification.
        
        Args:
            message: Message to display
            toast_type: Type of notification
            duration: How long to show (ms, 0 = indefinite)
            
        Returns:
            The created toast widget
        """
        # Remove old toasts if we have too many (keep max 1 for toolbar)
        while len(self._toasts) >= 1:
            old_toast = self._toasts[0]
            old_toast.deleteLater()
            self._toasts.remove(old_toast)
        
        toast = ToastWidget(message, toast_type, duration, self)
        toast.closed.connect(lambda: self._remove_toast(toast))
        
        self._toasts.append(toast)
        self.layout().addWidget(toast)
        
        return toast

    def show_success(self, message: str, duration: int = 3000) -> ToastWidget:
        """Show a success toast."""
        return self.show_toast(message, ToastType.SUCCESS, duration)

    def show_error(self, message: str, duration: int = 5000) -> ToastWidget:
        """Show an error toast."""
        return self.show_toast(message, ToastType.ERROR, duration)

    def show_warning(self, message: str, duration: int = 4000) -> ToastWidget:
        """Show a warning toast."""
        return self.show_toast(message, ToastType.WARNING, duration)

    def show_info(self, message: str, duration: int = 3000) -> ToastWidget:
        """Show an info toast."""
        return self.show_toast(message, ToastType.INFO, duration)

    def _remove_toast(self, toast: ToastWidget) -> None:
        """Remove a toast from the list."""
        if toast in self._toasts:
            self._toasts.remove(toast)

