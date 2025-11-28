"""
Logging setup with file rotation and retention policy.

Provides:
- File logging with size-based or time-based rotation
- Configurable retention policy (number of backup files)
- JSON or standard text formatting
- Console and file output handlers
- Custom naming: live_risk_{env}_{category}_{date}_{run_number}.log
- New log file created on each program run with incremental number
- Configurable timezone for log timestamps (defaults to local time)
"""

from __future__ import annotations
import logging
import logging.handlers
import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from config.models import LoggingConfig

# Global run number for this session (determined at startup)
_session_run_number: Optional[int] = None

# Global timezone setting for log timestamps (None = local time)
_log_timezone: Optional[ZoneInfo] = None


def set_log_timezone(tz: Optional[str] = None) -> None:
    """
    Set the timezone for log timestamps.

    Args:
        tz: Timezone name (e.g., "America/New_York", "US/Eastern", "UTC").
            If None, uses local system time.

    Example:
        >>> set_log_timezone("America/New_York")  # Eastern Time
        >>> set_log_timezone("UTC")               # UTC time
        >>> set_log_timezone(None)                # Local system time
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
        - If timezone is set: "2025-11-28T09:30:45.123456-05:00"
        - If no timezone (local): "2025-11-28T09:30:45.123456" (no suffix)
    """
    if _log_timezone is not None:
        now = datetime.now(_log_timezone)
        return now.isoformat()
    else:
        # Local time without timezone suffix
        return datetime.now().isoformat()


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Formats log records as single-line JSON for easy parsing.
    Uses the configured timezone for timestamps.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        # If the message is already JSON (from StructuredLogger), pass through
        if record.msg.startswith('{') and record.msg.endswith('}'):
            return record.msg

        # Otherwise, create basic JSON structure
        import json

        log_entry = {
            "timestamp": get_current_timestamp(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def custom_namer(default_name: str, env: str = "dev") -> str:
    """
    Custom namer for rotated log files.

    Converts:
      - live_risk_dev_sys.log.2025-11-24 -> live_risk_dev_sys_2025-11-24_1.log (time-based)
      - live_risk_prod_mkt.log.1 -> live_risk_prod_mkt_{today}_1.log (size-based)

    Args:
        default_name: Default rotated filename from handler.
        env: Environment name (dev/prod).

    Returns:
        Custom formatted filename.
    """
    # Get directory and base filename
    dir_name = os.path.dirname(default_name)
    base_name = os.path.basename(default_name)

    # Extract base log name (without .log extension)
    # e.g., "live_risk_dev_sys.log.2025-11-24" -> "live_risk_dev_sys"
    match = re.match(r'^(.+?)\.log', base_name)
    if not match:
        return default_name

    log_base = match.group(1)

    # Check if it's time-based rotation (has date suffix)
    # e.g., live_risk_dev_sys.log.2025-11-24
    time_match = re.search(r'\.log\.(\d{4}-\d{2}-\d{2})$', base_name)
    if time_match:
        date_str = time_match.group(1)
        # Find next available number for this date
        number = _find_next_number(dir_name, log_base, date_str)
        new_name = f"{log_base}_{date_str}_{number}.log"
        return os.path.join(dir_name, new_name) if dir_name else new_name

    # Check if it's size-based rotation (has numeric suffix)
    # e.g., live_risk_dev_sys.log.1
    size_match = re.search(r'\.log\.(\d+)$', base_name)
    if size_match:
        date_str = datetime.now().strftime('%Y-%m-%d')
        rotation_num = int(size_match.group(1))
        new_name = f"{log_base}_{date_str}_{rotation_num}.log"
        return os.path.join(dir_name, new_name) if dir_name else new_name

    return default_name


def _find_next_number(dir_path: str, log_base: str, date_str: str) -> int:
    """
    Find the next available number for a given date.

    Args:
        dir_path: Directory containing log files.
        log_base: Base log name (e.g., "live_risk_dev_sys").
        date_str: Date string (e.g., "2025-11-24").

    Returns:
        Next available number (starting from 1).
    """
    if not dir_path or not os.path.exists(dir_path):
        return 1

    # Pattern: live_risk_dev_sys_2025-11-24_N.log
    pattern = re.compile(rf'^{re.escape(log_base)}_{re.escape(date_str)}_(\d+)\.log$')

    max_num = 0
    for filename in os.listdir(dir_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num + 1


def setup_logging(config: LoggingConfig, logger_name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with file rotation and retention.

    Args:
        config: Logging configuration.
        logger_name: Name of the logger to configure. If None, configures root logger.

    Returns:
        Configured logger instance.

    Example:
        >>> from config.config_manager import ConfigManager
        >>> cfg = ConfigManager().load()
        >>> logger = setup_logging(cfg.logging)
        >>> logger.info("Application started")
    """
    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    if config.json:
        file_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

    # Console handler (always add for visibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if config.file:
        # Create log directory if it doesn't exist
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if config.rotation == "size":
            # Size-based rotation
            file_handler = logging.handlers.RotatingFileHandler(
                filename=config.file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            # Apply custom naming
            file_handler.namer = custom_namer
        elif config.rotation == "time":
            # Time-based rotation
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=config.file,
                when=config.when,
                interval=config.interval,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            # Apply custom naming
            file_handler.namer = custom_namer
        else:
            # No rotation, just basic file handler
            file_handler = logging.FileHandler(
                filename=config.file,
                encoding='utf-8'
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger if configuring a named logger
    if logger_name:
        logger.propagate = False

    return logger


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name (usually __name__).
        config: Optional logging config. If provided, sets up the logger.

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    logger = logging.getLogger(name)

    if config and not logger.handlers:
        # Configure this logger if config provided and not already configured
        setup_logging(config, name)

    return logger


def _get_next_run_number(log_dir: str, env: str, date_str: str) -> int:
    """
    Find the next available run number for today's date.

    Scans existing log files to find the highest run number used today,
    then returns the next number.

    Args:
        log_dir: Directory containing log files.
        env: Environment name (dev/prod).
        date_str: Date string (e.g., "2025-11-28").

    Returns:
        Next available run number (starting from 1).
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return 1

    # Pattern: live_risk_{env}_{category}_{date}_{N}.log
    # We check for any category (sys or mkt) to find the max run number
    pattern = re.compile(
        rf'^live_risk_{re.escape(env)}_(?:sys|mkt)_{re.escape(date_str)}_(\d+)\.log$'
    )

    max_num = 0
    for filename in os.listdir(log_path):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    return max_num + 1


def _get_session_run_number(log_dir: str, env: str) -> int:
    """
    Get or initialize the session run number.

    This ensures all loggers in the same session use the same run number.

    Args:
        log_dir: Directory containing log files.
        env: Environment name (dev/prod).

    Returns:
        Run number for this session.
    """
    global _session_run_number

    if _session_run_number is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
        _session_run_number = _get_next_run_number(log_dir, env, date_str)

    return _session_run_number


def setup_category_logging(env: str, log_dir: str = "./logs", level: str = "INFO") -> Dict[str, logging.Logger]:
    """
    Set up separate log files for different log categories.

    Creates separate log files with incremental run numbers:
    - live_risk_{env}_sys_{date}_{run_number}.log - System events, errors, connections
    - live_risk_{env}_mkt_{date}_{run_number}.log - Market data, positions, trading

    Each program run creates new log files with an incremented run number.
    For example, if run 3 times on 2025-11-28:
    - live_risk_dev_sys_2025-11-28_1.log (first run)
    - live_risk_dev_sys_2025-11-28_2.log (second run)
    - live_risk_dev_sys_2025-11-28_3.log (third run)

    Note: Logs are written to files only (no console output) to avoid
    interfering with the terminal dashboard UI.

    Args:
        env: Environment name (dev/prod).
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Dict mapping category to logger.

    Example:
        >>> loggers = setup_category_logging("dev")
        >>> loggers["system"].info("System started")
        >>> loggers["market"].info("Fetched market data")
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get run number for this session
    date_str = datetime.now().strftime('%Y-%m-%d')
    run_number = _get_session_run_number(str(log_path), env)

    loggers = {}
    categories = {
        "system": f"live_risk_{env}_sys_{date_str}_{run_number}.log",
        "market": f"live_risk_{env}_mkt_{date_str}_{run_number}.log",
    }

    for category, filename in categories.items():
        logger_name = f"apex.{category}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.handlers.clear()
        logger.propagate = False

        # JSON formatter
        formatter = JSONFormatter()

        # File handler - use simple FileHandler since we create new file each run
        file_path = str(log_path / filename)
        file_handler = logging.FileHandler(
            filename=file_path,
            mode='a',  # Append mode (file is new anyway)
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Flush immediately after adding handler to ensure file is created
        file_handler.flush()

        loggers[category] = logger

    return loggers


def flush_all_loggers() -> None:
    """
    Flush all file handlers to ensure logs are written to disk.

    Call this before program exit or when you need to ensure
    all pending log messages are written.
    """
    for logger_name in ["apex.system", "apex.market"]:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            handler.flush()


def reset_session_run_number() -> None:
    """
    Reset the session run number.

    Useful for testing to force a new run number on next setup_category_logging call.
    """
    global _session_run_number
    _session_run_number = None
