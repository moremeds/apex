"""
Structured JSON logging for the backtest framework.

Provides machine-parseable logs with context propagation for
experiments, trials, and runs.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Tuple


@dataclass
class LogContext:
    """Context for structured log entries."""

    experiment_id: Optional[str] = None
    trial_id: Optional[str] = None
    run_id: Optional[str] = None
    symbol: Optional[str] = None
    phase: Optional[str] = None


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Output format:
    {
        "timestamp": "2025-12-31T10:30:00.000Z",
        "level": "INFO",
        "logger": "backtest.runner",
        "message": "Starting experiment",
        "context": {
            "experiment_id": "exp_abc123",
            "trial_id": null
        },
        "extra": {}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if present
        if hasattr(record, "context") and record.context:
            entry["context"] = asdict(record.context)
        else:
            entry["context"] = {}

        # Add any extra fields
        extra: Dict[str, Any] = {}
        skip_keys = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "context",
            "message",
            "taskName",
        }

        for key, value in record.__dict__.items():
            if key not in skip_keys:
                try:
                    json.dumps(value)  # Check if serializable
                    extra[key] = value
                except (TypeError, ValueError):
                    extra[key] = str(value)

        if extra:
            entry["extra"] = extra

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry)


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context.

    Usage:
        logger = ContextLogger(
            logging.getLogger(__name__),
            LogContext(experiment_id="exp_123")
        )
        logger.info("Starting trial", trial_id="trial_456")
    """

    def __init__(self, logger: logging.Logger, context: LogContext):
        super().__init__(logger, {})
        self.context = context

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        # Merge context
        extra = kwargs.get("extra", {})
        extra["context"] = self.context

        # Allow overriding context fields via kwargs
        for field in ["experiment_id", "trial_id", "run_id", "symbol", "phase"]:
            if field in kwargs:
                setattr(extra["context"], field, kwargs.pop(field))

        kwargs["extra"] = extra
        return msg, kwargs

    def with_context(self, **kwargs: Any) -> "ContextLogger":
        """Create new logger with updated context."""
        new_context = LogContext(
            experiment_id=kwargs.get("experiment_id", self.context.experiment_id),
            trial_id=kwargs.get("trial_id", self.context.trial_id),
            run_id=kwargs.get("run_id", self.context.run_id),
            symbol=kwargs.get("symbol", self.context.symbol),
            phase=kwargs.get("phase", self.context.phase),
        )
        return ContextLogger(self.logger, new_context)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
) -> None:
    """
    Configure logging for the backtest framework.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        json_format: If True, use JSON format; else use human-readable

    Example:
        setup_logging(level="DEBUG", log_file=Path("backtest.log"), json_format=True)
    """
    root_logger = logging.getLogger("backtest")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if json_format:
        formatter: logging.Formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log initial message
    root_logger.info(f"Logging initialized at {level} level")


def get_logger(name: str, context: Optional[LogContext] = None) -> ContextLogger:
    """
    Get a logger with optional context.

    Args:
        name: Logger name (usually __name__)
        context: Optional initial context

    Returns:
        ContextLogger instance

    Example:
        logger = get_logger(__name__, LogContext(experiment_id="exp_123"))
        logger.info("Processing trial", trial_id="trial_456")
    """
    return ContextLogger(logging.getLogger(name), context or LogContext())
