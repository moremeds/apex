"""Utility modules."""

from .logging_setup import (
    flush_all_loggers,
    get_current_timestamp,
    get_log_timezone,
    get_logger,
    is_console_enabled,
    is_verbose_mode,
    reset_session_run_number,
    set_console_enabled,
    set_log_timezone,
    set_verbose_mode,
    setup_category_logging,
    setup_logging,
)
from .perf_logger import (
    log_timing,
    log_timing_async,
    timed,
)
from .result import (
    Err,
    Ok,
    Result,
    collect_results,
    try_result,
)
from .structured_logger import StructuredLogger
from .trace_context import (
    generate_cycle_id,
    get_cycle_id,
    new_cycle,
    set_cycle_id,
)

__all__ = [
    # Logging setup
    "StructuredLogger",
    "setup_category_logging",
    "setup_logging",
    "flush_all_loggers",
    "reset_session_run_number",
    "set_log_timezone",
    "get_log_timezone",
    "get_current_timestamp",
    "get_logger",
    "set_verbose_mode",
    "set_console_enabled",
    "is_verbose_mode",
    "is_console_enabled",
    # Trace context
    "get_cycle_id",
    "set_cycle_id",
    "new_cycle",
    "generate_cycle_id",
    # Performance logging
    "log_timing",
    "log_timing_async",
    "timed",
    # Result type
    "Result",
    "Ok",
    "Err",
    "try_result",
    "collect_results",
]
