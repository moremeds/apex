"""
Structured JSON logger with log categories.

Provides structured logging with:
- JSON output format
- Log categories (SYSTEM, RISK, TRADING, DATA, ALERT)
- Standard schema for log entries
- Configurable timezone support (via logging_setup.set_log_timezone)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict

from .logging_setup import get_current_timestamp


class LogCategory(Enum):
    """Log entry categories."""

    SYSTEM = "SYSTEM"  # System events (startup, shutdown, errors)
    RISK = "RISK"  # Risk calculation and limit breaches
    TRADING = "TRADING"  # Position updates and reconciliation
    DATA = "DATA"  # Market data quality and staleness
    ALERT = "ALERT"  # Critical alerts requiring attention


class StructuredLogger:
    """
    Structured JSON logger.

    Outputs logs in JSON format with standard schema:
    {
        "timestamp": "2024-03-15T10:30:45.123Z",
        "level": "INFO",
        "category": "RISK",
        "message": "Portfolio delta breach detected",
        "data": {...}
    }
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize structured logger.

        Args:
            logger: Python logger instance.
        """
        self.logger = logger

    def log(
        self,
        level: str,
        category: LogCategory,
        message: str,
        data: Dict[str, Any] | None = None,
    ) -> None:
        """
        Log a structured message.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            category: Log category enum.
            message: Log message.
            data: Optional additional data dict.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": get_current_timestamp(),
            "level": level.upper(),
            "category": category.value,
            "message": message,
        }

        if data:
            log_entry["data"] = data

        json_str = json.dumps(log_entry)

        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(json_str)

    def info(self, category: LogCategory, message: str, data: Dict[str, Any] | None = None) -> None:
        """Log info message."""
        self.log("INFO", category, message, data)

    def warning(
        self, category: LogCategory, message: str, data: Dict[str, Any] | None = None
    ) -> None:
        """Log warning message."""
        self.log("WARNING", category, message, data)

    def error(
        self, category: LogCategory, message: str, data: Dict[str, Any] | None = None
    ) -> None:
        """Log error message."""
        self.log("ERROR", category, message, data)

    def critical(
        self, category: LogCategory, message: str, data: Dict[str, Any] | None = None
    ) -> None:
        """Log critical message."""
        self.log("CRITICAL", category, message, data)

    def debug(
        self, category: LogCategory, message: str, data: Dict[str, Any] | None = None
    ) -> None:
        """Log debug message."""
        self.log("DEBUG", category, message, data)
