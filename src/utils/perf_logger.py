"""
Performance logging utilities.

Provides timing context managers and decorators for automatic
performance logging. All timing logs include the current cycle ID
for correlation.

Usage:
    # Context manager
    with log_timing("snapshot_build"):
        snapshot = engine.build_snapshot(...)

    # Async context manager
    async with log_timing_async("market_data_fetch"):
        data = await fetch_market_data(...)

    # Decorator
    @timed("position_reconcile")
    def reconcile_positions(...):
        ...
"""

from __future__ import annotations

import time
import logging
import functools
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Callable, Any, Generator, AsyncGenerator

from .trace_context import get_cycle_id

# Performance logger - uses 'apex.perf' category
_perf_logger: Optional[logging.Logger] = None


def get_perf_logger() -> logging.Logger:
    """Get or create the performance logger."""
    global _perf_logger
    if _perf_logger is None:
        _perf_logger = logging.getLogger("apex.perf")
    return _perf_logger


def set_perf_logger(logger: logging.Logger) -> None:
    """Set the performance logger (for testing or custom configuration)."""
    global _perf_logger
    _perf_logger = logger


@contextmanager
def log_timing(
    operation: str,
    warn_threshold_ms: float = 500.0,
    error_threshold_ms: float = 2000.0,
    extra: Optional[dict] = None,
) -> Generator[dict, None, None]:
    """
    Context manager to log operation timing.

    Logs timing to the perf category with cycle ID. Automatically
    escalates log level based on duration thresholds.

    Args:
        operation: Name of the operation being timed.
        warn_threshold_ms: Duration above which to log as WARNING.
        error_threshold_ms: Duration above which to log as ERROR.
        extra: Additional data to include in the log.

    Yields:
        Dict that can be updated with additional context during execution.

    Example:
        with log_timing("snapshot_build", warn_threshold_ms=200) as ctx:
            ctx["positions"] = len(positions)
            snapshot = engine.build_snapshot(...)
    """
    logger = get_perf_logger()
    context = extra.copy() if extra else {}
    start_time = time.perf_counter()

    try:
        yield context
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        cycle_id = get_cycle_id()

        log_data = {
            "cycle": cycle_id,
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            **context,
        }

        # Determine log level based on duration
        if duration_ms >= error_threshold_ms:
            logger.error(f"[{cycle_id}] SLOW {operation}: {duration_ms:.1f}ms", extra={"data": log_data})
        elif duration_ms >= warn_threshold_ms:
            logger.warning(f"[{cycle_id}] {operation}: {duration_ms:.1f}ms (slow)", extra={"data": log_data})
        else:
            logger.debug(f"[{cycle_id}] {operation}: {duration_ms:.1f}ms", extra={"data": log_data})


@asynccontextmanager
async def log_timing_async(
    operation: str,
    warn_threshold_ms: float = 500.0,
    error_threshold_ms: float = 2000.0,
    extra: Optional[dict] = None,
) -> AsyncGenerator[dict, None]:
    """
    Async context manager to log operation timing.

    Same as log_timing but for async operations.

    Args:
        operation: Name of the operation being timed.
        warn_threshold_ms: Duration above which to log as WARNING.
        error_threshold_ms: Duration above which to log as ERROR.
        extra: Additional data to include in the log.

    Yields:
        Dict that can be updated with additional context during execution.

    Example:
        async with log_timing_async("market_data_fetch") as ctx:
            data = await fetch_market_data(symbols)
            ctx["symbols"] = len(symbols)
    """
    logger = get_perf_logger()
    context = extra.copy() if extra else {}
    start_time = time.perf_counter()

    try:
        yield context
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        cycle_id = get_cycle_id()

        log_data = {
            "cycle": cycle_id,
            "operation": operation,
            "duration_ms": round(duration_ms, 2),
            **context,
        }

        # Determine log level based on duration
        if duration_ms >= error_threshold_ms:
            logger.error(f"[{cycle_id}] SLOW {operation}: {duration_ms:.1f}ms", extra={"data": log_data})
        elif duration_ms >= warn_threshold_ms:
            logger.warning(f"[{cycle_id}] {operation}: {duration_ms:.1f}ms (slow)", extra={"data": log_data})
        else:
            logger.debug(f"[{cycle_id}] {operation}: {duration_ms:.1f}ms", extra={"data": log_data})


def timed(
    operation: Optional[str] = None,
    warn_threshold_ms: float = 500.0,
    error_threshold_ms: float = 2000.0,
) -> Callable:
    """
    Decorator to log function timing.

    Args:
        operation: Name of the operation. Defaults to function name.
        warn_threshold_ms: Duration above which to log as WARNING.
        error_threshold_ms: Duration above which to log as ERROR.

    Returns:
        Decorated function.

    Example:
        @timed("position_reconcile")
        def reconcile_positions(ib_positions, manual_positions):
            ...

        @timed(warn_threshold_ms=100)
        def calculate_greeks(position):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_timing(op_name, warn_threshold_ms, error_threshold_ms):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with log_timing_async(op_name, warn_threshold_ms, error_threshold_ms):
                return await func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if a function is an async coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# Convenience timing functions for common operations
def log_snapshot_timing() -> Generator[dict, None, None]:
    """Pre-configured timing for snapshot builds."""
    return log_timing("snapshot_build", warn_threshold_ms=250, error_threshold_ms=1000)


def log_market_data_timing() -> AsyncGenerator[dict, None]:
    """Pre-configured timing for market data fetches."""
    return log_timing_async("market_data_fetch", warn_threshold_ms=1000, error_threshold_ms=5000)


def log_position_fetch_timing() -> AsyncGenerator[dict, None]:
    """Pre-configured timing for position fetches."""
    return log_timing_async("position_fetch", warn_threshold_ms=500, error_threshold_ms=2000)


def log_reconcile_timing() -> Generator[dict, None, None]:
    """Pre-configured timing for position reconciliation."""
    return log_timing("position_reconcile", warn_threshold_ms=100, error_threshold_ms=500)
