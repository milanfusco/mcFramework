"""
Log console widget.

This module provides a log console panel for displaying simulation
progress and messages with a progress bar.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LogLevel(Enum):
    """Log message severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


# Color mapping for log levels
LOG_COLORS: dict[LogLevel, str] = {
    LogLevel.INFO: "#cccccc",
    LogLevel.SUCCESS: "#00d26a",
    LogLevel.WARNING: "#ffa500",
    LogLevel.ERROR: "#f23645",
    LogLevel.DEBUG: "#888888",
}


class LogConsoleWidget(QWidget):
    """
    Console widget for displaying log messages and progress.
    
    Provides a terminal-style log view with color-coded messages,
    a progress bar for tracking simulation progress, and phase
    indicators for the current operation.
    """
    
    # Console width constant
    CONSOLE_WIDTH = 320

    def __init__(self, parent: QWidget | None = None):
        """
        Initialize the log console.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self.setObjectName("logConsole")
        self.setFixedWidth(self.CONSOLE_WIDTH)
        
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the console UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("Console")
        title.setObjectName("consoleTitle")
        title.setStyleSheet("font-weight: bold; color: #ccc;")
        header.addWidget(title)
        
        header.addStretch()
        
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("consoleClearBtn")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)
        
        layout.addLayout(header)
        
        # Log text area
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setObjectName("consoleLogText")
        self._log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d1a;
                color: #cccccc;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._log_text, 1)
        
        # Progress section
        progress_frame = QFrame()
        progress_frame.setObjectName("progressFrame")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setSpacing(4)
        progress_layout.setContentsMargins(0, 8, 0, 0)
        
        # Phase indicator
        self._phase_label = QLabel("Ready")
        self._phase_label.setObjectName("phaseLabel")
        self._phase_label.setStyleSheet("color: #888; font-size: 11px;")
        progress_layout.addWidget(self._phase_label)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("simulationProgress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m")
        progress_layout.addWidget(self._progress_bar)
        
        # Status label
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setStyleSheet("color: #666; font-size: 10px;")
        progress_layout.addWidget(self._status_label)
        
        layout.addWidget(progress_frame)

    def _format_timestamp(self) -> str:
        """Get formatted timestamp for log messages."""
        return datetime.now().strftime("%H:%M:%S")

    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """
        Add a log message to the console.
        
        Args:
            message: Message text
            level: Log level for color coding
        """
        timestamp = self._format_timestamp()
        color = LOG_COLORS.get(level, LOG_COLORS[LogLevel.INFO])
        
        # Format the message with HTML
        formatted = f'<span style="color: #666;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        
        self._log_text.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, LogLevel.INFO)

    def log_success(self, message: str) -> None:
        """Log a success message."""
        self.log(message, LogLevel.SUCCESS)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, LogLevel.WARNING)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, LogLevel.ERROR)

    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.log(message, LogLevel.DEBUG)

    @Slot(str)
    def on_log_message(self, message: str) -> None:
        """
        Handle log message from controller signal.
        
        Auto-detects message type from content.
        
        Args:
            message: Log message
        """
        # Detect level from message content
        if message.startswith("✓") or "success" in message.lower():
            level = LogLevel.SUCCESS
        elif message.startswith("✗") or "error" in message.lower() or "failed" in message.lower():
            level = LogLevel.ERROR
        elif "warning" in message.lower():
            level = LogLevel.WARNING
        else:
            level = LogLevel.INFO
        
        self.log(message, level)

    @Slot(str)
    def set_phase(self, phase: str) -> None:
        """
        Set the current operation phase.
        
        Args:
            phase: Phase name to display
        """
        self._phase_label.setText(f"⏳ {phase}...")
        self._phase_label.setStyleSheet("color: #4da6ff; font-size: 11px;")

    @Slot(int, int)
    def update_progress(self, current: int, total: int) -> None:
        """
        Update the progress bar.
        
        Args:
            current: Current progress value
            total: Total value (maximum)
        """
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        
        if total > 0:
            pct = (current / total) * 100
            self._status_label.setText(f"{pct:.1f}% complete")

    @Slot(str, int, int)
    def on_simulation_progress(self, phase: str, current: int, total: int) -> None:
        """
        Handle simulation progress update.
        
        Args:
            phase: Current phase name
            current: Current progress
            total: Total steps
        """
        self.set_phase(phase)
        self.update_progress(current, total)

    def set_running(self, running: bool) -> None:
        """
        Update UI state for running/stopped status.
        
        Args:
            running: Whether simulation is running
        """
        if running:
            self._phase_label.setText("⏳ Starting...")
            self._phase_label.setStyleSheet("color: #4da6ff; font-size: 11px;")
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        else:
            self._phase_label.setText("Ready")
            self._phase_label.setStyleSheet("color: #888; font-size: 11px;")

    def set_complete(self, success: bool = True) -> None:
        """
        Set the console to completion state.
        
        Args:
            success: Whether the operation completed successfully
        """
        if success:
            self._phase_label.setText("✓ Complete")
            self._phase_label.setStyleSheet("color: #00d26a; font-size: 11px;")
            self._progress_bar.setValue(self._progress_bar.maximum())
            self._status_label.setText("100% complete")
        else:
            self._phase_label.setText("✗ Failed")
            self._phase_label.setStyleSheet("color: #f23645; font-size: 11px;")

    def clear(self) -> None:
        """Clear all log messages."""
        self._log_text.clear()
        self._progress_bar.setValue(0)
        self._phase_label.setText("Ready")
        self._phase_label.setStyleSheet("color: #888; font-size: 11px;")
        self._status_label.setText("")

    def get_log_text(self) -> str:
        """
        Get all log text as plain string.
        
        Returns:
            All logged messages as plain text
        """
        return self._log_text.toPlainText()

