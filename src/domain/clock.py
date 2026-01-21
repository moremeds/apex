"""
Clock abstraction for live vs simulated time.

This module provides clock implementations that allow strategies to work
identically in both live trading and backtesting modes. Strategies use
the clock interface to get current time, schedule timers, and sleep without
knowing whether time is real or simulated.

Usage:
    # Live trading
    clock = SystemClock()
    now = clock.now()

    # Backtesting
    clock = SimulatedClock(start_time=datetime(2024, 1, 1, 9, 30))
    now = clock.now()  # Returns start_time
    clock.advance_to(datetime(2024, 1, 1, 9, 31))
    now = clock.now()  # Returns advanced time
"""

import asyncio
import heapq
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import backtrader as bt

logger = logging.getLogger(__name__)


class Clock(ABC):
    """
    Abstract clock interface.

    Provides time-related operations that work in both live and simulated modes.
    All time-dependent code should use Clock instead of direct time functions.
    """

    @abstractmethod
    def now(self) -> datetime:
        """
        Get current time.

        Returns:
            Current datetime (local time in live, simulated time in backtest).
        """
        ...

    @abstractmethod
    def timestamp(self) -> float:
        """
        Get current timestamp (seconds since epoch).

        Returns:
            Unix timestamp as float.
        """
        ...

    @abstractmethod
    async def sleep(self, seconds: float) -> None:
        """
        Sleep for specified duration.

        In live mode, this performs real sleep.
        In simulated mode, this advances time immediately.

        Args:
            seconds: Duration to sleep in seconds.
        """
        ...

    @abstractmethod
    def set_timer(self, delay: float, callback: Callable[[], None]) -> str:
        """
        Set a timer to fire after delay seconds.

        Args:
            delay: Delay in seconds before callback fires.
            callback: Function to call when timer fires.

        Returns:
            Timer ID for cancellation.
        """
        ...

    @abstractmethod
    def cancel_timer(self, timer_id: str) -> bool:
        """
        Cancel a timer.

        Args:
            timer_id: ID returned by set_timer.

        Returns:
            True if timer was found and cancelled.
        """
        ...

    def elapsed_since(self, reference: datetime) -> float:
        """
        Calculate elapsed time since a reference datetime.

        Args:
            reference: Reference datetime.

        Returns:
            Elapsed time in seconds.
        """
        return (self.now() - reference).total_seconds()

    def is_after(self, target: datetime) -> bool:
        """Check if current time is after target."""
        return self.now() > target

    def is_before(self, target: datetime) -> bool:
        """Check if current time is before target."""
        return self.now() < target


class SystemClock(Clock):
    """
    Real system clock for live trading.

    Uses actual system time and asyncio for scheduling.
    """

    def __init__(self) -> None:
        self._timers: Dict[str, asyncio.TimerHandle] = {}
        self._next_timer_id = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def now(self) -> datetime:
        """Get current system time."""
        return datetime.now()

    def timestamp(self) -> float:
        """Get current Unix timestamp."""
        return time.time()

    async def sleep(self, seconds: float) -> None:
        """Sleep using asyncio."""
        await asyncio.sleep(seconds)

    def set_timer(self, delay: float, callback: Callable[[], None]) -> str:
        """Schedule a timer using asyncio."""
        timer_id = f"sys-timer-{self._next_timer_id}"
        self._next_timer_id += 1

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        handle = loop.call_later(delay, self._execute_timer, timer_id, callback)
        self._timers[timer_id] = handle

        logger.debug(f"Timer {timer_id} scheduled for {delay}s from now")
        return timer_id

    def _execute_timer(self, timer_id: str, callback: Callable[[], None]) -> None:
        """Execute timer callback and cleanup."""
        self._timers.pop(timer_id, None)
        try:
            callback()
        except Exception as e:
            logger.exception(f"Timer {timer_id} callback error: {e}")

    def cancel_timer(self, timer_id: str) -> bool:
        """Cancel a scheduled timer."""
        handle = self._timers.pop(timer_id, None)
        if handle:
            handle.cancel()
            logger.debug(f"Timer {timer_id} cancelled")
            return True
        return False


class SimulatedClock(Clock):
    """
    Simulated clock for backtesting.

    Time advances only when explicitly advanced via advance_to() or advance_by().
    Timers fire when their scheduled time is reached during advancement.

    Usage:
        clock = SimulatedClock(datetime(2024, 1, 1, 9, 30))

        # Set a timer for 5 minutes from now
        clock.set_timer(300, lambda: print("Timer fired!"))

        # Advance time (timer will fire during this)
        clock.advance_by(timedelta(minutes=10))
    """

    def __init__(self, start_time: datetime):
        """
        Initialize simulated clock.

        Args:
            start_time: Initial simulated time.
        """
        self._current_time = start_time
        self._timers: List[Tuple[datetime, str, Callable[[], None]]] = []  # min-heap
        self._next_timer_id = 0
        self._cancelled_timers: set = set()

    def now(self) -> datetime:
        """Get current simulated time."""
        return self._current_time

    def timestamp(self) -> float:
        """Get current simulated timestamp."""
        return self._current_time.timestamp()

    async def sleep(self, seconds: float) -> None:
        """
        In simulation, sleep advances time immediately.

        This allows strategies using sleep to work in backtest mode.
        """
        self.advance_by(timedelta(seconds=seconds))

    def set_timer(self, delay: float, callback: Callable[[], None]) -> str:
        """
        Schedule timer for future simulated time.

        Timer will fire when advance_to/advance_by reaches its scheduled time.
        """
        fire_time = self._current_time + timedelta(seconds=delay)
        timer_id = f"sim-timer-{self._next_timer_id}"
        self._next_timer_id += 1

        # Use heapq for efficient timer management
        heapq.heappush(self._timers, (fire_time, timer_id, callback))

        logger.debug(f"Simulated timer {timer_id} scheduled for {fire_time}")
        return timer_id

    def cancel_timer(self, timer_id: str) -> bool:
        """
        Cancel a scheduled timer.

        Due to heap structure, we mark as cancelled and skip during execution.
        """
        if timer_id in self._cancelled_timers:
            return False
        self._cancelled_timers.add(timer_id)
        logger.debug(f"Simulated timer {timer_id} cancelled")
        return True

    def advance_to(self, new_time: datetime) -> int:
        """
        Advance clock to new time, firing any due timers.

        This is called by the backtest engine when processing historical events.
        Timers scheduled between current time and new_time will fire in order.

        Args:
            new_time: Target datetime to advance to.

        Returns:
            Number of timers that fired.

        Raises:
            ValueError: If new_time is before current time.
        """
        if new_time < self._current_time:
            raise ValueError(f"Cannot advance backwards: {self._current_time} -> {new_time}")

        timers_fired = 0

        # Fire all timers scheduled before or at new_time
        while self._timers and self._timers[0][0] <= new_time:
            fire_time, timer_id, callback = heapq.heappop(self._timers)

            # Skip cancelled timers
            if timer_id in self._cancelled_timers:
                self._cancelled_timers.discard(timer_id)
                continue

            # Advance to timer fire time
            self._current_time = fire_time

            # Execute callback
            try:
                callback()
                timers_fired += 1
                logger.debug(f"Timer {timer_id} fired at {fire_time}")
            except Exception as e:
                logger.exception(f"Timer {timer_id} callback error: {e}")

        # Advance to final target time
        self._current_time = new_time

        return timers_fired

    def advance_by(self, delta: timedelta) -> int:
        """
        Advance clock by a time delta.

        Args:
            delta: Time delta to advance by.

        Returns:
            Number of timers that fired.
        """
        return self.advance_to(self._current_time + delta)

    def reset(self, new_time: datetime) -> None:
        """
        Reset clock to a new time and clear all timers.

        Useful for running multiple backtests with the same clock instance.

        Args:
            new_time: New starting time.
        """
        self._current_time = new_time
        self._timers.clear()
        self._cancelled_timers.clear()
        logger.debug(f"Simulated clock reset to {new_time}")

    @property
    def pending_timers(self) -> int:
        """Get count of pending (non-cancelled) timers."""
        return sum(1 for _, tid, _ in self._timers if tid not in self._cancelled_timers)


class BacktraderClock(Clock):
    """
    Clock adapter for Backtrader integration.

    Reads time from Backtrader's data feed datetime.
    Used by ApexStrategyWrapper to provide Clock interface to Apex strategies.

    Note: This clock doesn't support set_timer directly - use BacktraderScheduler
    instead for time-based actions in Backtrader.
    """

    def __init__(self, bt_strategy: "bt.Strategy"):
        """
        Initialize with Backtrader strategy reference.

        Args:
            bt_strategy: Backtrader strategy instance.
        """
        self._bt_strategy = bt_strategy

    def now(self) -> datetime:
        """Get current time from Backtrader data."""
        try:
            result: datetime = self._bt_strategy.data.datetime.datetime(0)
            return result
        except (IndexError, AttributeError):
            # No data yet - return epoch
            return datetime(1970, 1, 1)

    def timestamp(self) -> float:
        """Get current timestamp from Backtrader data."""
        return self.now().timestamp()

    async def sleep(self, seconds: float) -> None:
        """
        No-op in Backtrader - time advances via data.

        Strategies should not rely on sleep in Backtrader mode.
        Use scheduler for time-based actions instead.
        """
        logger.warning(
            "Clock.sleep() called in Backtrader mode - "
            "time advances via data, not sleep. Use scheduler instead."
        )

    def set_timer(self, delay: float, callback: Callable[[], None]) -> str:
        """
        Timers not supported directly in Backtrader.

        Use BacktraderScheduler.schedule_daily() or similar instead.
        """
        logger.warning(
            "Clock.set_timer() not supported in Backtrader mode. "
            "Use BacktraderScheduler instead."
        )
        return ""

    def cancel_timer(self, timer_id: str) -> bool:
        """Timers not supported in Backtrader."""
        return False


# Type alias for clock factory
ClockFactory = Callable[[], Clock]


def create_system_clock() -> SystemClock:
    """Factory function for SystemClock."""
    return SystemClock()


def create_simulated_clock(start_time: datetime) -> SimulatedClock:
    """Factory function for SimulatedClock."""
    return SimulatedClock(start_time)
