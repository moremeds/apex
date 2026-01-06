"""Tests for parallel runner retry policy."""

from __future__ import annotations

import pytest

from src.backtest.execution.parallel import (
    ParallelConfig,
    is_transient_error,
    TRANSIENT_ERROR_TYPES,
    DETERMINISTIC_ERROR_TYPES,
)


class TestIsTransientError:
    """Tests for error classification."""

    def test_timeout_is_transient(self) -> None:
        """Timeout errors should be retried."""
        assert is_transient_error(TimeoutError("operation timed out"))

    def test_connection_error_is_transient(self) -> None:
        """Connection errors should be retried."""
        assert is_transient_error(ConnectionError("connection reset"))

    def test_os_error_is_transient(self) -> None:
        """OS resource errors should be retried."""
        assert is_transient_error(OSError("Too many open files"))
        assert is_transient_error(OSError("Resource temporarily unavailable"))

    def test_value_error_not_transient(self) -> None:
        """ValueError should NOT be retried - it's a logic error."""
        assert not is_transient_error(ValueError("invalid parameter"))

    def test_type_error_not_transient(self) -> None:
        """TypeError should NOT be retried - it's a programming error."""
        assert not is_transient_error(TypeError("expected str, got int"))

    def test_key_error_not_transient(self) -> None:
        """KeyError should NOT be retried - deterministic failure."""
        assert not is_transient_error(KeyError("missing_key"))

    def test_assertion_error_not_transient(self) -> None:
        """AssertionError should NOT be retried."""
        assert not is_transient_error(AssertionError("test failed"))

    def test_zero_division_not_transient(self) -> None:
        """ZeroDivisionError should NOT be retried."""
        assert not is_transient_error(ZeroDivisionError("division by zero"))

    def test_message_based_transient_detection(self) -> None:
        """Errors with transient-like messages should be retried."""
        # Generic Exception with transient message
        assert is_transient_error(Exception("connection reset by peer"))
        assert is_transient_error(Exception("Resource temporarily unavailable"))
        assert is_transient_error(Exception("request timed out"))

    def test_generic_exception_not_transient(self) -> None:
        """Generic exceptions without transient patterns are not retried."""
        assert not is_transient_error(Exception("some random error"))
        assert not is_transient_error(RuntimeError("unexpected state"))


class TestParallelConfigRetryDelay:
    """Tests for exponential backoff calculation."""

    def test_first_retry_uses_base_delay(self) -> None:
        """First retry should use base delay."""
        config = ParallelConfig(
            retry_base_delay=1.0,
            retry_max_delay=30.0,
            retry_jitter=0.0,  # No jitter for deterministic test
        )
        delay = config.calculate_retry_delay(1)
        assert delay == 1.0

    def test_exponential_backoff(self) -> None:
        """Delays should increase exponentially."""
        config = ParallelConfig(
            retry_base_delay=1.0,
            retry_max_delay=100.0,  # High max to not cap
            retry_jitter=0.0,
        )
        assert config.calculate_retry_delay(1) == 1.0  # 1 * 2^0
        assert config.calculate_retry_delay(2) == 2.0  # 1 * 2^1
        assert config.calculate_retry_delay(3) == 4.0  # 1 * 2^2
        assert config.calculate_retry_delay(4) == 8.0  # 1 * 2^3

    def test_max_delay_cap(self) -> None:
        """Delay should be capped at max_delay."""
        config = ParallelConfig(
            retry_base_delay=1.0,
            retry_max_delay=5.0,
            retry_jitter=0.0,
        )
        # 5th retry would be 16s without cap
        delay = config.calculate_retry_delay(5)
        assert delay == 5.0  # Capped at max

    def test_jitter_adds_randomness(self) -> None:
        """Jitter should add some randomness to delays."""
        config = ParallelConfig(
            retry_base_delay=10.0,
            retry_max_delay=100.0,
            retry_jitter=0.1,  # ±10%
        )
        # Run multiple times, should vary
        delays = [config.calculate_retry_delay(1) for _ in range(100)]

        # All delays should be within jitter range (9 to 11 for base 10 with 10% jitter)
        assert all(9.0 <= d <= 11.0 for d in delays)

        # With jitter, not all values should be exactly the same
        assert len(set(delays)) > 1

    def test_delay_never_negative(self) -> None:
        """Delay should never be negative."""
        config = ParallelConfig(
            retry_base_delay=0.1,
            retry_max_delay=100.0,
            retry_jitter=0.5,  # High jitter could theoretically go negative
        )
        for _ in range(100):
            delay = config.calculate_retry_delay(1)
            assert delay >= 0


class TestExecutionProgressWithRetries:
    """Tests for progress tracking with retries."""

    def test_retried_runs_tracked(self) -> None:
        """Retried runs should be tracked separately."""
        from src.backtest.execution.parallel import ExecutionProgress

        progress = ExecutionProgress(total_runs=10)
        progress.completed_runs = 8
        progress.retried_runs = 2  # 2 of the 8 required retries
        progress.failed_runs = 2

        stats = progress.to_dict()
        assert stats["completed"] == 8
        assert stats["retried"] == 2
        assert stats["failed"] == 2
