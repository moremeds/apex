"""
Trading signal logger with JSONL output and daily rotation.

Logs trading signals to date-stamped JSONL files for analysis and replay.
Uses async file I/O via queue to avoid blocking the event bus.

Usage:
    logger = SignalLogger(log_dir="logs/signals", env="dev")
    logger.attach(event_bus)  # Subscribe to TRADING_SIGNAL events

    # Or manually log signals
    logger.log_signal(signal)

    # Cleanup
    logger.stop()
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Optional

from ..events.event_types import EventType

# Module logger for internal messages
_logger = logging.getLogger(__name__)


class SignalLogger:
    """
    JSONL logger for trading signals with daily file rotation.

    Features:
    - Daily rotation with date-based filenames (trading_signals_{env}_{date}.jsonl)
    - Async file I/O via QueueHandler + QueueListener (non-blocking)
    - Automatic directory creation
    - Graceful handling of serialization errors
    """

    def __init__(
        self,
        log_dir: str = "logs/signals",
        env: str = "dev",
        retention_days: int = 30,
    ) -> None:
        """
        Initialize the signal logger.

        Args:
            log_dir: Directory for log files
            env: Environment name (included in filename)
            retention_days: Number of days to retain log files
        """
        self.log_dir = Path(log_dir)
        self.env = env
        self.retention_days = retention_days

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current date for file naming
        self._current_date = datetime.now().strftime("%Y-%m-%d")

        # Async logging queue
        self._log_queue: Queue = Queue(maxsize=5000)
        self._queue_listener: Optional[logging.handlers.QueueListener] = None
        self._file_handler: Optional[logging.Handler] = None

        # Dedicated logger for signal output
        self._signal_logger = self._setup_logger()

        # Event bus reference for cleanup
        self._subscribed_bus: Any = None

        log_file = self._get_log_filename()
        _logger.info(f"SignalLogger initialized: {log_file}")

    def _get_log_filename(self) -> Path:
        """Get the current log filename."""
        return self.log_dir / f"trading_signals_{self.env}_{self._current_date}.jsonl"

    def _setup_logger(self) -> logging.Logger:
        """Set up the dedicated async file logger."""
        # Create a unique logger name to avoid conflicts
        logger_name = f"apex.trading_signals.{self.env}.{id(self)}"
        signal_logger = logging.getLogger(logger_name)
        signal_logger.setLevel(logging.INFO)
        signal_logger.handlers.clear()
        signal_logger.propagate = False

        # File handler with daily rotation
        log_file = self._get_log_filename()
        self._file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=self.retention_days,
            encoding="utf-8",
        )
        self._file_handler.namer = self._log_namer
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Queue handler for async I/O
        queue_handler = logging.handlers.QueueHandler(self._log_queue)
        signal_logger.addHandler(queue_handler)

        # Queue listener processes the queue and writes to file
        self._queue_listener = logging.handlers.QueueListener(
            self._log_queue,
            self._file_handler,
            respect_handler_level=True,
        )
        self._queue_listener.start()

        return signal_logger

    def stop(self) -> None:
        """Stop the logger and flush pending writes."""
        if self._queue_listener:
            self._queue_listener.stop()
            self._queue_listener = None
        if self._file_handler:
            self._file_handler.close()
            self._file_handler = None
        _logger.debug("SignalLogger stopped")

    def _recreate_logger(self) -> None:
        """Recreate the logger with a new date-based filename."""
        # Stop existing handlers
        if self._queue_listener:
            self._queue_listener.stop()
        if self._file_handler:
            self._file_handler.close()

        # Clear existing handlers from the logger
        self._signal_logger.handlers.clear()

        # Recreate with new filename
        self._signal_logger = self._setup_logger()
        _logger.info(f"Logger recreated for date: {self._current_date}")

    def attach(self, event_bus: Any) -> None:
        """
        Subscribe to TRADING_SIGNAL events on the event bus.

        Args:
            event_bus: PriorityEventBus instance
        """
        self._subscribed_bus = event_bus
        event_bus.subscribe(EventType.TRADING_SIGNAL, self._handle_event)
        _logger.debug("SignalLogger attached to event bus")

    def detach(self) -> None:
        """Unsubscribe from the event bus."""
        if not self._subscribed_bus:
            return
        try:
            self._subscribed_bus.unsubscribe(EventType.TRADING_SIGNAL, self._handle_event)
        except Exception as e:
            _logger.warning(f"Failed to detach SignalLogger: {e}")
        finally:
            self._subscribed_bus = None

    def _handle_event(self, payload: Any) -> None:
        """Event bus callback for trading signals."""
        self.log_signal(payload)

    def log_signal(self, signal: Any) -> None:
        """
        Log a trading signal to the JSONL file.

        Args:
            signal: TradingSignal, TradingSignalEvent, or dict
        """
        if signal is None:
            return

        # Check if date has changed (midnight rollover)
        self._check_date_rollover()

        record = self._serialize(signal)
        if not record:
            return

        try:
            json_line = json.dumps(record, default=str, ensure_ascii=False)
            self._signal_logger.info(json_line)
        except Exception as e:
            _logger.error(f"Failed to serialize signal: {e}")

    def _check_date_rollover(self) -> None:
        """Check if date changed and recreate logger if needed."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self._current_date:
            _logger.info(f"Date rolled over: {self._current_date} -> {current_date}")
            self._current_date = current_date
            # Recreate the logger with new date filename
            self._recreate_logger()

    def _serialize(self, signal: Any) -> Optional[dict]:
        """
        Serialize a signal to a JSON-serializable dict.

        Args:
            signal: Signal object or dict

        Returns:
            Serialized dict or None on failure
        """
        try:
            if hasattr(signal, "to_dict"):
                data = signal.to_dict()
            elif isinstance(signal, dict):
                data = dict(signal)
            else:
                # Fallback: extract common attributes
                data = {
                    "signal_id": getattr(signal, "signal_id", None),
                    "symbol": getattr(signal, "symbol", None),
                    "direction": str(getattr(signal, "direction", "")),
                    "strength": getattr(signal, "strength", None),
                    "timestamp": str(getattr(signal, "timestamp", "")),
                    "message": getattr(signal, "message", None) or getattr(signal, "reason", None),
                }

            # Add logging metadata
            data["logged_at"] = datetime.now().isoformat()
            data["logger_env"] = self.env

            return data

        except Exception as e:
            _logger.error(f"Signal serialization failed: {e}")
            return None

    def _log_namer(self, default_name: str) -> str:
        """
        Custom namer for rotated log files.

        Renames rotated files to include the date in JSONL format.
        """
        import re

        # Extract date from the rotated filename
        match = re.search(r"\.(\d{4}-\d{2}-\d{2})$", default_name)
        if match:
            date_str = match.group(1)
            return str(self.log_dir / f"trading_signals_{self.env}_{date_str}.jsonl")
        return default_name
