"""
Logging setup with file rotation, categories, and trace ID support.

Provides:
- 5 log categories: system, adapter, risk, data, perf
- Automatic module → category routing
- Cycle ID correlation in all logs
- File logging with size-based or time-based rotation
- Console output (when dashboard is disabled or verbose mode)
- JSON or standard text formatting
- Configurable timezone for log timestamps

Categories:
- system: Startup, shutdown, config, errors, TUI
- adapter: Broker connections (IB, Futu), reconnects, API calls
- risk: Risk calculations, breaches, signals
- data: Market data, positions, reconciliation, stores
- perf: Timing, latency, performance diagnostics
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import os
import re
import json
from queue import Queue
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from zoneinfo import ZoneInfo
from config.models import LoggingConfig

# Import trace context for cycle ID
from .trace_context import get_cycle_id

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Global run number for this session (determined at startup)
_session_run_number: Optional[int] = None

# Global timezone setting for log timestamps (None = local time)
_log_timezone: Optional[ZoneInfo] = None

# Global verbose flag (set via --verbose CLI flag)
_verbose_mode: bool = False

# Global console output flag (set when --no-dashboard)
_console_enabled: bool = False

# Global log level override (set via --log-level CLI flag)
_log_level_override: Optional[str] = None

# Configured category loggers
_category_loggers: Dict[str, logging.Logger] = {}

# Queue listeners for async file logging (one per category)
_queue_listeners: List[logging.handlers.QueueListener] = []

# =============================================================================
# LOG CATEGORIES AND ROUTING
# =============================================================================

# Available log categories
CATEGORIES = ["system", "adapter", "risk", "data", "perf"]

# Category file suffixes
CATEGORY_SUFFIXES = {
    "system": "sys",
    "adapter": "adp",
    "risk": "rsk",
    "data": "dat",
    "perf": "prf",
}

# Module path → category routing
# More specific paths should come first
MODULE_ROUTING: List[tuple[str, str]] = [
    # Adapters
    ("src.infrastructure.adapters.ib", "adapter"),
    ("src.infrastructure.adapters.futu", "adapter"),
    ("src.infrastructure.adapters.yahoo", "adapter"),
    ("src.infrastructure.adapters.broker_manager", "adapter"),
    ("src.infrastructure.adapters.market_data_manager", "adapter"),
    ("src.infrastructure.adapters.market_data_fetcher", "adapter"),
    ("src.infrastructure.adapters.file_loader", "data"),

    # Stores
    ("src.infrastructure.stores", "data"),

    # Risk domain
    ("src.domain.services.risk", "risk"),

    # Data domain (reconciler, mdqc, etc.)
    ("src.domain.services.pos_reconciler", "data"),
    ("src.domain.services.mdqc", "data"),
    ("src.domain.services.market_alert_detector", "data"),
    ("src.domain.services.strategy_detector", "data"),
    ("src.domain.services.correlation_analyzer", "data"),
    ("src.domain.services.event_risk_detector", "risk"),
    ("src.domain.services.position_risk_analyzer", "risk"),
    ("src.domain.services.strategy_risk_analyzer", "risk"),

    # Observability
    ("src.infrastructure.observability", "perf"),

    # Monitoring
    ("src.infrastructure.monitoring", "system"),

    # Application layer
    ("src.application", "system"),

    # TUI
    ("src.tui", "system"),

    # Models (rarely log, but route to data)
    ("src.models", "data"),

    # Default fallback
    ("src", "system"),
]


def get_category_for_module(module_name: str) -> str:
    """
    Determine the log category for a given module name.

    Args:
        module_name: Full module path (e.g., "src.infrastructure.adapters.ib.adapter").

    Returns:
        Category name (system, adapter, risk, data, or perf).
    """
    for prefix, category in MODULE_ROUTING:
        if module_name.startswith(prefix):
            return category
    return "system"  # Default fallback


# =============================================================================
# TIMEZONE SUPPORT
# =============================================================================

def set_log_timezone(tz: Optional[str] = None) -> None:
    """
    Set the timezone for log timestamps.

    Args:
        tz: Timezone name (e.g., "America/New_York", "US/Eastern", "UTC").
            If None, uses local system time.
    """
    global _log_timezone
    if tz is None:
        _log_timezone = None
    else:
        _log_timezone = ZoneInfo(tz)


def get_log_timezone() -> Optional[ZoneInfo]:
    """Get the current log timezone setting."""
    return _log_timezone


def get_current_timestamp() -> str:
    """
    Get the current timestamp formatted for logging.

    Returns:
        ISO format timestamp with timezone info.
    """
    if _log_timezone is not None:
        now = datetime.now(_log_timezone)
        return now.isoformat()
    else:
        return datetime.now().isoformat()


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

def set_verbose_mode(enabled: bool) -> None:
    """Enable or disable verbose mode (DEBUG level logging)."""
    global _verbose_mode
    _verbose_mode = enabled


def is_verbose_mode() -> bool:
    """Check if verbose mode is enabled."""
    return _verbose_mode


def set_console_enabled(enabled: bool) -> None:
    """Enable or disable console output."""
    global _console_enabled
    _console_enabled = enabled


def is_console_enabled() -> bool:
    """Check if console output is enabled."""
    return _console_enabled


def set_log_level_override(level: Optional[str]) -> None:
    """Set a global log level override."""
    global _log_level_override
    _log_level_override = level.upper() if level else None


def get_effective_log_level() -> str:
    """Get the effective log level (considering verbose mode and overrides)."""
    if _verbose_mode:
        return "DEBUG"
    if _log_level_override:
        return _log_level_override
    return "INFO"


# =============================================================================
# JSON FORMATTER WITH CYCLE ID
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with cycle ID support.

    Formats log records as single-line JSON with:
    - Timestamp (with timezone)
    - Level
    - Category (derived from logger name)
    - Cycle ID (for correlation)
    - Message
    - Extra data
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # If the message is already JSON (from StructuredLogger), parse and enhance
        msg = record.getMessage()

        if msg.startswith('{') and msg.endswith('}'):
            try:
                log_entry = json.loads(msg)
                # Normalize field names from StructuredLogger format
                if "timestamp" in log_entry:
                    log_entry["ts"] = log_entry.pop("timestamp")
                if "category" in log_entry:
                    log_entry["cat"] = log_entry.pop("category").lower()
                if "message" in log_entry:
                    log_entry["msg"] = log_entry.pop("message")
                # Add cycle ID if not present
                if "cycle" not in log_entry:
                    log_entry["cycle"] = get_cycle_id()
                return json.dumps(log_entry)
            except json.JSONDecodeError:
                pass

        # Build structured log entry
        log_entry = {
            "ts": get_current_timestamp(),
            "level": record.levelname,
            "cat": self._get_category(record.name),
            "cycle": get_cycle_id(),
            "msg": msg,
        }

        # Add extra data if present
        if hasattr(record, "data") and record.data:
            log_entry["data"] = record.data

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

    def _get_category(self, logger_name: str) -> str:
        """Extract category from logger name."""
        if logger_name.startswith("apex."):
            parts = logger_name.split(".")
            if len(parts) >= 2 and parts[1] in CATEGORIES:
                return parts[1]
        return "system"


class ConsoleFormatter(logging.Formatter):
    """
    Console formatter with cycle ID and color support.

    Format: [LEVEL] [cycle] message
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        cycle_id = get_cycle_id()
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.RESET
            return f"{color}[{level:7}]{reset} [{cycle_id}] {record.getMessage()}"
        else:
            return f"[{level:7}] [{cycle_id}] {record.getMessage()}"


# =============================================================================
# LOGGER FACTORY
# =============================================================================

def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for the given module, automatically routed to the correct category.

    This is the primary function modules should use to get their logger.
    It automatically routes the module to the appropriate category logger.

    Args:
        module_name: Module name (typically __name__).

    Returns:
        Logger instance that routes to the appropriate category.

    Example:
        from src.utils.logging_setup import get_logger
        logger = get_logger(__name__)
        logger.info("Processing...")
    """
    category = get_category_for_module(module_name)
    category_logger_name = f"apex.{category}"

    # Return the category logger (will be configured by setup_category_logging)
    logger = logging.getLogger(category_logger_name)

    # If category loggers haven't been set up yet, ensure basic config
    if not logger.handlers and category_logger_name not in _category_loggers:
        # Temporary setup - will be replaced by setup_category_logging
        logger.setLevel(logging.DEBUG)

    return logger


# =============================================================================
# RUN NUMBER MANAGEMENT
# =============================================================================

def _get_next_run_number(log_dir: str, env: str, date_str: str) -> int:
    """Find the next available run number for today's date."""
    # Look in the date-specific subdirectory
    log_path = Path(log_dir) / date_str
    if not log_path.exists():
        return 1

    # Pattern: live_risk_{env}_{category}_{date}_{N}.log
    pattern = re.compile(
        rf'^live_risk_{re.escape(env)}_(?:sys|adp|rsk|dat|prf|mkt)_{re.escape(date_str)}_(\d+)\.log$'
    )

    max_num = 0
    for filename in os.listdir(log_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num + 1


def _get_session_run_number(log_dir: str, env: str) -> int:
    """Get or initialize the session run number."""
    global _session_run_number

    if _session_run_number is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
        _session_run_number = _get_next_run_number(log_dir, env, date_str)

    return _session_run_number


def reset_session_run_number() -> None:
    """Reset the session run number (for testing)."""
    global _session_run_number
    _session_run_number = None


# =============================================================================
# CATEGORY LOGGING SETUP
# =============================================================================

def setup_category_logging(
    env: str,
    log_dir: str = "./logs",
    level: str = "INFO",
    console: bool = False,
    verbose: bool = False,
) -> Dict[str, logging.Logger]:
    """
    Set up separate log files for each category.

    Creates log files in date-specific subdirectory:
    - logs/{date}/live_risk_{env}_sys_{date}_{run}.log - System events
    - logs/{date}/live_risk_{env}_adp_{date}_{run}.log - Adapter events
    - logs/{date}/live_risk_{env}_rsk_{date}_{run}.log - Risk events
    - logs/{date}/live_risk_{env}_dat_{date}_{run}.log - Data events
    - logs/{date}/live_risk_{env}_prf_{date}_{run}.log - Performance events

    Args:
        env: Environment name (dev/prod/demo).
        log_dir: Base directory for log files.
        level: Default logging level.
        console: Enable console output.
        verbose: Enable verbose (DEBUG) mode.

    Returns:
        Dict mapping category name to logger.
    """
    global _category_loggers, _queue_listeners

    # Clean up existing handlers and listeners before reconfiguration
    # to avoid file handle leaks (HYG-006)
    for listener in _queue_listeners:
        listener.stop()
    _queue_listeners.clear()

    for category in CATEGORIES:
        logger_name = f"apex.{category}"
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:  # Copy list to avoid mutation during iteration
            handler.close()
            logger.removeHandler(handler)

    # Set global flags
    set_verbose_mode(verbose)
    set_console_enabled(console)
    if not verbose:
        set_log_level_override(level)

    # Create date-specific log directory (logs/{date}/)
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_path = Path(log_dir) / date_str
    log_path.mkdir(parents=True, exist_ok=True)

    # Get run number for this session
    run_number = _get_session_run_number(log_dir, env)

    effective_level = get_effective_log_level()

    for category in CATEGORIES:
        suffix = CATEGORY_SUFFIXES[category]
        filename = f"live_risk_{env}_{suffix}_{date_str}_{run_number}.log"

        logger_name = f"apex.{category}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, effective_level, logging.INFO))
        logger.propagate = False  # Handlers already cleaned up at start of function

        # JSON formatter for files
        json_formatter = JSONFormatter()

        # File handler (actual disk writer)
        file_path = str(log_path / filename)
        file_handler = logging.FileHandler(
            filename=file_path,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(getattr(logging, effective_level, logging.INFO))

        # Use QueueHandler for async file logging (non-blocking writes)
        log_queue: Queue = Queue(-1)  # Unbounded queue
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)

        # Start QueueListener to process queue -> file handler
        listener = logging.handlers.QueueListener(
            log_queue, file_handler, respect_handler_level=True
        )
        listener.start()
        _queue_listeners.append(listener)

        # Console handler (only when dashboard is disabled)
        if console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_formatter = ConsoleFormatter(use_colors=True)
            console_handler.setFormatter(console_formatter)
            console_level = logging.DEBUG if verbose else logging.WARNING
            console_handler.setLevel(console_level)
            logger.addHandler(console_handler)

        _category_loggers[category] = logger

    return _category_loggers


def get_category_loggers() -> Dict[str, logging.Logger]:
    """Get all configured category loggers."""
    return _category_loggers


def flush_all_loggers() -> None:
    """Flush all file handlers to ensure logs are written to disk."""
    for category in CATEGORIES:
        logger_name = f"apex.{category}"
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.flush()

    # Also flush legacy loggers for backward compatibility
    for logger_name in ["apex.system", "apex.market"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.flush()


def shutdown_logging() -> None:
    """Shutdown all queue listeners (call during application shutdown)."""
    global _queue_listeners
    for listener in _queue_listeners:
        listener.stop()
    _queue_listeners.clear()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Keep old function for backward compatibility (now routes to new system)
def setup_logging(config: LoggingConfig, logger_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with file rotation and retention.

    DEPRECATED: Use setup_category_logging() instead.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.handlers.clear()

    if config.json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(filename=config.file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if logger_name:
        logger.propagate = False

    return logger


# Legacy custom_namer kept for compatibility
def custom_namer(default_name: str, env: str = "dev") -> str:
    """Custom namer for rotated log files (legacy compatibility)."""
    dir_name = os.path.dirname(default_name)
    base_name = os.path.basename(default_name)

    match = re.match(r'^(.+?)\.log', base_name)
    if not match:
        return default_name

    log_base = match.group(1)

    time_match = re.search(r'\.log\.(\d{4}-\d{2}-\d{2})$', base_name)
    if time_match:
        date_str = time_match.group(1)
        number = _find_next_number(dir_name, log_base, date_str)
        new_name = f"{log_base}_{date_str}_{number}.log"
        return os.path.join(dir_name, new_name) if dir_name else new_name

    size_match = re.search(r'\.log\.(\d+)$', base_name)
    if size_match:
        date_str = datetime.now().strftime('%Y-%m-%d')
        rotation_num = int(size_match.group(1))
        new_name = f"{log_base}_{date_str}_{rotation_num}.log"
        return os.path.join(dir_name, new_name) if dir_name else new_name

    return default_name


def _find_next_number(dir_path: str, log_base: str, date_str: str) -> int:
    """Find the next available number for a given date."""
    if not dir_path or not os.path.exists(dir_path):
        return 1

    pattern = re.compile(rf'^{re.escape(log_base)}_{re.escape(date_str)}_(\d+)\.log$')

    max_num = 0
    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num + 1
