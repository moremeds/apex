"""
Scheduler façade for live/backtest parity.

This module provides a unified scheduling interface that works identically
in live trading and backtesting. Strategies use the Scheduler to schedule
time-based actions without knowing the execution context.

Inspired by:
- Zipline's schedule_function()
- LEAN's Schedule.On()
- RQAlpha's run_daily/run_weekly

Why this matters:
Without a scheduler abstraction, strategies using time-based actions
(rebalance at market close, daily reset, weekly lookback) will behave
differently in live vs backtest. This breaks the core parity principle.

Usage:
    # In strategy.on_start():
    self.context.scheduler.schedule_daily(
        action_id="rebalance",
        callback=self.rebalance,
        time_of_day=time(15, 55),  # 3:55 PM
    )

    # For end-of-day actions:
    self.context.scheduler.schedule_before_close(
        action_id="eod_report",
        callback=self.generate_report,
        minutes_before=5,
    )
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from ..clock import Clock, SimulatedClock

logger = logging.getLogger(__name__)


class ScheduleFrequency(Enum):
    """Frequency of scheduled actions."""

    ONCE = "once"  # Run once at specified time
    DAILY = "daily"  # Run daily at specified time
    WEEKLY = "weekly"  # Run weekly on specified day/time
    MONTHLY = "monthly"  # Run monthly on specified day/time
    ON_BAR = "on_bar"  # Run after each bar close
    ON_SESSION_OPEN = "on_session_open"  # Run at market open
    ON_SESSION_CLOSE = "on_session_close"  # Run at/before market close


@dataclass
class ScheduledAction:
    """A scheduled action registration."""

    action_id: str
    callback: Callable[[], None]
    frequency: ScheduleFrequency
    time_of_day: Optional[time] = None  # For DAILY, WEEKLY, etc.
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    day_of_month: Optional[int] = None  # 1-31
    minutes_before_close: Optional[int] = None  # For relative scheduling
    enabled: bool = True

    # Tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


class Scheduler(ABC):
    """
    Abstract scheduler interface.

    Strategies use this to schedule time-based actions without
    knowing whether they're in live or backtest mode.

    All scheduler implementations must provide these methods.
    """

    @abstractmethod
    def schedule_once(
        self,
        action_id: str,
        callback: Callable[[], None],
        at_time: datetime,
    ) -> None:
        """
        Schedule a one-time action at a specific datetime.

        Args:
            action_id: Unique identifier for this action.
            callback: Function to call when action fires.
            at_time: Datetime when action should fire.
        """
        ...

    @abstractmethod
    def schedule_daily(
        self,
        action_id: str,
        callback: Callable[[], None],
        time_of_day: time,
    ) -> None:
        """
        Schedule an action to run daily at a specific time.

        Args:
            action_id: Unique identifier for this action.
            callback: Function to call when action fires.
            time_of_day: Time of day to run (e.g., time(15, 55)).
        """
        ...

    @abstractmethod
    def schedule_weekly(
        self,
        action_id: str,
        callback: Callable[[], None],
        day_of_week: int,
        time_of_day: time,
    ) -> None:
        """
        Schedule an action to run weekly.

        Args:
            action_id: Unique identifier for this action.
            callback: Function to call when action fires.
            day_of_week: Day of week (0=Monday, 6=Sunday).
            time_of_day: Time of day to run.
        """
        ...

    @abstractmethod
    def schedule_on_bar_close(
        self,
        action_id: str,
        callback: Callable[[], None],
    ) -> None:
        """
        Schedule an action to run after each bar closes.

        Args:
            action_id: Unique identifier for this action.
            callback: Function to call after each bar.
        """
        ...

    @abstractmethod
    def schedule_before_close(
        self,
        action_id: str,
        callback: Callable[[], None],
        minutes_before: int = 5,
    ) -> None:
        """
        Schedule an action to run N minutes before market close.

        Args:
            action_id: Unique identifier for this action.
            callback: Function to call before close.
            minutes_before: Minutes before close to trigger.
        """
        ...

    @abstractmethod
    def cancel(self, action_id: str) -> bool:
        """
        Cancel a scheduled action.

        Args:
            action_id: ID of action to cancel.

        Returns:
            True if action was found and cancelled.
        """
        ...

    @abstractmethod
    def get_scheduled_actions(self) -> List[ScheduledAction]:
        """
        Get all registered scheduled actions.

        Returns:
            List of all scheduled actions.
        """
        ...

    def disable(self, action_id: str) -> bool:
        """
        Disable an action without removing it.

        Args:
            action_id: ID of action to disable.

        Returns:
            True if action was found and disabled.
        """
        for action in self.get_scheduled_actions():
            if action.action_id == action_id:
                action.enabled = False
                return True
        return False

    def enable(self, action_id: str) -> bool:
        """
        Re-enable a disabled action.

        Args:
            action_id: ID of action to enable.

        Returns:
            True if action was found and enabled.
        """
        for action in self.get_scheduled_actions():
            if action.action_id == action_id:
                action.enabled = True
                return True
        return False


class LiveScheduler(Scheduler):
    """
    Scheduler implementation for live trading.

    Uses asyncio for time-based scheduling. Actions run when their
    scheduled real-world time is reached.
    """

    def __init__(
        self,
        clock: "Clock",
        market_close_time: time = time(16, 0),  # 4:00 PM default
    ):
        """
        Initialize live scheduler.

        Args:
            clock: Clock instance for time operations.
            market_close_time: Market close time for before_close scheduling.
        """
        self._clock = clock
        self._market_close = market_close_time
        self._actions: Dict[str, ScheduledAction] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = True

    def schedule_once(
        self,
        action_id: str,
        callback: Callable[[], None],
        at_time: datetime,
    ) -> None:
        """Schedule a one-time action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ONCE,
        )
        self._actions[action_id] = action

        # Calculate delay
        delay = (at_time - self._clock.now()).total_seconds()
        if delay > 0:
            task = asyncio.create_task(self._run_once_async(action_id, delay, callback))
            self._tasks[action_id] = task
            logger.debug(f"Scheduled once: {action_id} at {at_time}")

    async def _run_once_async(self, action_id: str, delay: float, callback: Callable) -> None:
        """Run a one-time action after delay."""
        await asyncio.sleep(delay)
        action = self._actions.get(action_id)
        if action and action.enabled and self._running:
            try:
                callback()
                action.last_triggered = self._clock.now()
                action.trigger_count += 1
            except Exception as e:
                logger.error(f"Scheduled action {action_id} error: {e}")

    def schedule_daily(
        self,
        action_id: str,
        callback: Callable[[], None],
        time_of_day: time,
    ) -> None:
        """Schedule a daily action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action

        task = asyncio.create_task(self._run_daily_async(action_id, time_of_day, callback))
        self._tasks[action_id] = task
        logger.debug(f"Scheduled daily: {action_id} at {time_of_day}")

    async def _run_daily_async(self, action_id: str, time_of_day: time, callback: Callable) -> None:
        """Run a daily action."""
        while self._running:
            action = self._actions.get(action_id)
            if not action or not action.enabled:
                break

            # Calculate next trigger time
            now = self._clock.now()
            target = now.replace(
                hour=time_of_day.hour,
                minute=time_of_day.minute,
                second=0,
                microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)

            # Wait until trigger time
            delay = (target - now).total_seconds()
            await asyncio.sleep(delay)

            # Execute if still enabled
            action = self._actions.get(action_id)
            if action and action.enabled and self._running:
                try:
                    callback()
                    action.last_triggered = self._clock.now()
                    action.trigger_count += 1
                    logger.debug(f"Daily action triggered: {action_id}")
                except Exception as e:
                    logger.error(f"Daily action {action_id} error: {e}")

    def schedule_weekly(
        self,
        action_id: str,
        callback: Callable[[], None],
        day_of_week: int,
        time_of_day: time,
    ) -> None:
        """Schedule a weekly action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.WEEKLY,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action

        task = asyncio.create_task(
            self._run_weekly_async(action_id, day_of_week, time_of_day, callback)
        )
        self._tasks[action_id] = task
        logger.debug(f"Scheduled weekly: {action_id} on day {day_of_week}")

    async def _run_weekly_async(
        self,
        action_id: str,
        day_of_week: int,
        time_of_day: time,
        callback: Callable,
    ) -> None:
        """Run a weekly action."""
        while self._running:
            action = self._actions.get(action_id)
            if not action or not action.enabled:
                break

            # Calculate next trigger time
            now = self._clock.now()
            days_ahead = day_of_week - now.weekday()
            if days_ahead < 0 or (days_ahead == 0 and now.time() >= time_of_day):
                days_ahead += 7

            target = now.replace(
                hour=time_of_day.hour,
                minute=time_of_day.minute,
                second=0,
                microsecond=0,
            ) + timedelta(days=days_ahead)

            delay = (target - now).total_seconds()
            await asyncio.sleep(delay)

            action = self._actions.get(action_id)
            if action and action.enabled and self._running:
                try:
                    callback()
                    action.last_triggered = self._clock.now()
                    action.trigger_count += 1
                except Exception as e:
                    logger.error(f"Weekly action {action_id} error: {e}")

    def schedule_on_bar_close(
        self,
        action_id: str,
        callback: Callable[[], None],
    ) -> None:
        """
        Schedule an action for each bar close.

        In live mode, this is triggered externally when bars complete.
        """
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_BAR,
        )
        self._actions[action_id] = action
        logger.debug(f"Scheduled on_bar_close: {action_id}")

    def schedule_before_close(
        self,
        action_id: str,
        callback: Callable[[], None],
        minutes_before: int = 5,
    ) -> None:
        """Schedule an action before market close."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_SESSION_CLOSE,
            minutes_before_close=minutes_before,
        )
        self._actions[action_id] = action

        # Calculate trigger time
        trigger_time = time(
            hour=self._market_close.hour,
            minute=self._market_close.minute - minutes_before,
        )

        # Schedule as daily at the trigger time
        task = asyncio.create_task(self._run_daily_async(action_id, trigger_time, callback))
        self._tasks[action_id] = task
        logger.debug(f"Scheduled before_close: {action_id} at {minutes_before}min before close")

    def cancel(self, action_id: str) -> bool:
        """Cancel a scheduled action."""
        action = self._actions.pop(action_id, None)
        task = self._tasks.pop(action_id, None)

        if task:
            task.cancel()

        if action:
            logger.debug(f"Cancelled action: {action_id}")
            return True
        return False

    def get_scheduled_actions(self) -> List[ScheduledAction]:
        """Get all scheduled actions."""
        return list(self._actions.values())

    def trigger_bar_close_actions(self) -> None:
        """
        Trigger all ON_BAR actions.

        Called externally when a bar completes.
        """
        for action in self._actions.values():
            if action.enabled and action.frequency == ScheduleFrequency.ON_BAR:
                try:
                    action.callback()
                    action.last_triggered = self._clock.now()
                    action.trigger_count += 1
                except Exception as e:
                    logger.error(f"Bar close action {action.action_id} error: {e}")

    def stop(self) -> None:
        """Stop scheduler and cancel all tasks."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        logger.debug("LiveScheduler stopped")


class SimulatedScheduler(Scheduler):
    """
    Scheduler implementation for backtests.

    Scheduled actions are triggered by clock advancement, not real time.
    The backtest engine calls advance_to() to process scheduled events.
    """

    def __init__(self, clock: "SimulatedClock"):
        """
        Initialize simulated scheduler.

        Args:
            clock: SimulatedClock instance.
        """
        self._clock = clock
        self._actions: Dict[str, ScheduledAction] = {}
        # Min-heap of (trigger_time, action_id) for one-time events
        self._pending: List[tuple] = []
        # Market close time for before_close scheduling
        self._market_close = time(16, 0)

    def schedule_once(
        self,
        action_id: str,
        callback: Callable[[], None],
        at_time: datetime,
    ) -> None:
        """Schedule a one-time action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ONCE,
        )
        self._actions[action_id] = action
        heapq.heappush(self._pending, (at_time, action_id))
        logger.debug(f"Simulated schedule_once: {action_id} at {at_time}")

    def schedule_daily(
        self,
        action_id: str,
        callback: Callable[[], None],
        time_of_day: time,
    ) -> None:
        """Schedule a daily action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action
        logger.debug(f"Simulated schedule_daily: {action_id} at {time_of_day}")

    def schedule_weekly(
        self,
        action_id: str,
        callback: Callable[[], None],
        day_of_week: int,
        time_of_day: time,
    ) -> None:
        """Schedule a weekly action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.WEEKLY,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action
        logger.debug(f"Simulated schedule_weekly: {action_id}")

    def schedule_on_bar_close(
        self,
        action_id: str,
        callback: Callable[[], None],
    ) -> None:
        """Schedule an action for each bar close."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_BAR,
        )
        self._actions[action_id] = action
        logger.debug(f"Simulated schedule_on_bar_close: {action_id}")

    def schedule_before_close(
        self,
        action_id: str,
        callback: Callable[[], None],
        minutes_before: int = 5,
    ) -> None:
        """Schedule an action before market close."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_SESSION_CLOSE,
            minutes_before_close=minutes_before,
        )
        self._actions[action_id] = action
        logger.debug(f"Simulated schedule_before_close: {action_id}")

    def cancel(self, action_id: str) -> bool:
        """Cancel a scheduled action."""
        if action_id in self._actions:
            del self._actions[action_id]
            logger.debug(f"Cancelled simulated action: {action_id}")
            return True
        return False

    def get_scheduled_actions(self) -> List[ScheduledAction]:
        """Get all scheduled actions."""
        return list(self._actions.values())

    def advance_to(self, target_time: datetime) -> int:
        """
        Process scheduled actions as time advances.

        Called by backtest engine to trigger scheduled events.

        Args:
            target_time: Target datetime to advance to.

        Returns:
            Number of actions triggered.
        """
        triggered = 0
        current = self._clock.now()

        # Process daily/weekly/before_close actions
        for action in self._actions.values():
            if not action.enabled:
                continue

            if action.frequency == ScheduleFrequency.DAILY:
                triggered += self._check_daily(action, current, target_time)
            elif action.frequency == ScheduleFrequency.WEEKLY:
                triggered += self._check_weekly(action, current, target_time)
            elif action.frequency == ScheduleFrequency.ON_SESSION_CLOSE:
                triggered += self._check_before_close(action, current, target_time)

        # Process one-time events
        while self._pending and self._pending[0][0] <= target_time:
            trigger_time, action_id = heapq.heappop(self._pending)
            maybe_action = self._actions.get(action_id)
            if maybe_action is not None and maybe_action.enabled:
                action = maybe_action
                try:
                    action.callback()
                    action.last_triggered = trigger_time
                    action.trigger_count += 1
                    triggered += 1
                except Exception as e:
                    logger.error(f"Simulated action {action_id} error: {e}")

        return triggered

    def _check_daily(
        self,
        action: ScheduledAction,
        current: datetime,
        target: datetime,
    ) -> int:
        """Check if daily action should trigger."""
        if action.time_of_day is None:
            return 0

        triggered = 0
        check_time = current

        while check_time.date() <= target.date():
            trigger_dt = check_time.replace(
                hour=action.time_of_day.hour,
                minute=action.time_of_day.minute,
                second=0,
                microsecond=0,
            )

            # Check if trigger time is in the advancement window
            if current < trigger_dt <= target:
                try:
                    action.callback()
                    action.last_triggered = trigger_dt
                    action.trigger_count += 1
                    triggered += 1
                except Exception as e:
                    logger.error(f"Daily action {action.action_id} error: {e}")

            check_time += timedelta(days=1)

        return triggered

    def _check_weekly(
        self,
        action: ScheduledAction,
        current: datetime,
        target: datetime,
    ) -> int:
        """Check if weekly action should trigger."""
        if action.time_of_day is None or action.day_of_week is None:
            return 0

        triggered = 0
        check_time = current

        while check_time <= target:
            if check_time.weekday() == action.day_of_week:
                trigger_dt = check_time.replace(
                    hour=action.time_of_day.hour,
                    minute=action.time_of_day.minute,
                    second=0,
                    microsecond=0,
                )

                if current < trigger_dt <= target:
                    try:
                        action.callback()
                        action.last_triggered = trigger_dt
                        action.trigger_count += 1
                        triggered += 1
                    except Exception as e:
                        logger.error(f"Weekly action {action.action_id} error: {e}")

            check_time += timedelta(days=1)

        return triggered

    def _check_before_close(
        self,
        action: ScheduledAction,
        current: datetime,
        target: datetime,
    ) -> int:
        """Check if before_close action should trigger."""
        if action.minutes_before_close is None:
            return 0

        triggered = 0
        check_time = current

        while check_time.date() <= target.date():
            trigger_dt = check_time.replace(
                hour=self._market_close.hour,
                minute=self._market_close.minute - action.minutes_before_close,
                second=0,
                microsecond=0,
            )

            if current < trigger_dt <= target:
                try:
                    action.callback()
                    action.last_triggered = trigger_dt
                    action.trigger_count += 1
                    triggered += 1
                except Exception as e:
                    logger.error(f"Before close action {action.action_id} error: {e}")

            check_time += timedelta(days=1)

        return triggered

    def trigger_bar_close_actions(self) -> None:
        """Trigger all ON_BAR actions."""
        for action in self._actions.values():
            if action.enabled and action.frequency == ScheduleFrequency.ON_BAR:
                try:
                    action.callback()
                    action.last_triggered = self._clock.now()
                    action.trigger_count += 1
                except Exception as e:
                    logger.error(f"Bar close action {action.action_id} error: {e}")

    def reset(self) -> None:
        """Reset scheduler state."""
        self._actions.clear()
        self._pending.clear()


class NullScheduler(Scheduler):
    """
    Null scheduler that does nothing.

    Used when scheduler functionality is not needed.
    """

    def schedule_once(self, action_id: str, callback: Callable, at_time: datetime) -> None:
        pass

    def schedule_daily(self, action_id: str, callback: Callable, time_of_day: time) -> None:
        pass

    def schedule_weekly(
        self, action_id: str, callback: Callable, day_of_week: int, time_of_day: time
    ) -> None:
        pass

    def schedule_on_bar_close(self, action_id: str, callback: Callable) -> None:
        pass

    def schedule_before_close(
        self, action_id: str, callback: Callable, minutes_before: int = 5
    ) -> None:
        pass

    def cancel(self, action_id: str) -> bool:
        return False

    def get_scheduled_actions(self) -> List[ScheduledAction]:
        return []
