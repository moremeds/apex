"""
Internal utilities for systematic backtesting.

This module provides:
- Logging: Structured JSON logging with context propagation
"""

from .structured import (
    ContextLogger,
    LogContext,
    StructuredFormatter,
    get_logger,
    setup_logging,
)

__all__ = [
    "LogContext",
    "ContextLogger",
    "StructuredFormatter",
    "setup_logging",
    "get_logger",
]
