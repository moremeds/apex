"""
IndicatorEngine - Computes indicators on bar close and publishes updates.

Subscribes to BAR_CLOSE events, calculates all registered indicators
using a ThreadPoolExecutor, and publishes INDICATOR_UPDATE events for
downstream rule evaluation.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from src.domain.events.domain_events import BarCloseEvent, IndicatorUpdateEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger

from .indicators.base import Indicator
from .indicators.registry import get_indicator_registry

logger = get_logger(__name__)


BarKey = Tuple[str, str]  # (symbol, timeframe)
StateKey = Tuple[str, str, str]  # (symbol, timeframe, indicator)


class EventBusProtocol(Protocol):
    """Protocol for event bus compatibility."""

    def publish(self, event_type: EventType, payload: Any) -> None:
        """Publish an event."""
        ...

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        ...


class IndicatorEngine:
    """
    Computes all registered indicators on bar close and publishes updates.

    The engine:
    1. Subscribes to BAR_CLOSE events
    2. Maintains bar history per (symbol, timeframe) for lookback calculations
    3. Calculates indicators in parallel using ThreadPoolExecutor
    4. Tracks previous states for transition detection in rules
    5. Publishes INDICATOR_UPDATE events for each indicator

    Example:
        engine = IndicatorEngine(event_bus, max_workers=4)
        engine.start()  # Begin processing bar events
    """

    def __init__(
        self,
        event_bus: EventBusProtocol,
        max_workers: int = 4,
        max_history: Optional[int] = None,
    ) -> None:
        """
        Initialize the indicator engine.

        Args:
            event_bus: Event bus for subscriptions and publishing
            max_workers: ThreadPool size for parallel indicator calculations
            max_history: Maximum bars to retain per symbol/timeframe (auto-calculated from warmup if None)
        """
        self._event_bus = event_bus
        self._registry = get_indicator_registry()
        self._indicators: List[Indicator] = self._registry.get_all()
        self._max_workers = max_workers

        # Calculate max warmup from all indicators
        warmup_periods = [ind.warmup_periods for ind in self._indicators]
        self._max_warmup = max(warmup_periods) if warmup_periods else 0
        self._max_history = max_history or max(50, self._max_warmup + 10)

        # Bar history per (symbol, timeframe)
        self._history: Dict[BarKey, Deque[Dict[str, Any]]] = {}

        # Previous indicator states for transition detection
        self._previous_states: Dict[StateKey, Dict[str, Any]] = {}

        # Thread safety for shared state
        self._lock = RLock()

        # Thread pool for parallel calculations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._started = False
        self._bars_processed = 0

    @property
    def bars_processed(self) -> int:
        """Total number of bars processed."""
        return self._bars_processed

    @property
    def indicator_count(self) -> int:
        """Number of registered indicators."""
        return len(self._indicators)

    def start(self) -> None:
        """Start the engine by subscribing to BAR_CLOSE events."""
        if self._started:
            return

        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)
        self._started = True
        logger.info(
            f"IndicatorEngine started: {len(self._indicators)} indicators, "
            f"max_workers={self._max_workers}, max_history={self._max_history}"
        )

    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        self._executor.shutdown(wait=False)
        self._started = False
        logger.info("IndicatorEngine stopped")

    def _on_bar_close(self, payload: Any) -> None:
        """Handle BAR_CLOSE event (sync entry point)."""
        if not self._started:
            return  # Ignore events after stop()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_bar_async(payload))
        except RuntimeError:
            # No event loop running - run synchronously
            asyncio.run(self._process_bar_async(payload))

    async def _process_bar_async(self, payload: Any) -> None:
        """Process a bar close event and compute indicators."""
        if not self._started:
            return  # Engine stopped, skip processing

        event = self._coerce_bar_close(payload)
        if event is None:
            return

        # Extract bar data from event
        bar_entry = {
            "timestamp": event.bar_end or event.timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }

        bar_key: BarKey = (event.symbol, event.timeframe)

        # Update history (thread-safe)
        with self._lock:
            if bar_key not in self._history:
                self._history[bar_key] = deque(maxlen=self._max_history)
            self._history[bar_key].append(bar_entry)
            bars = list(self._history[bar_key])

        self._bars_processed += 1

        if not self._indicators:
            return

        # Build tasks for indicators with sufficient warmup
        loop = asyncio.get_running_loop()
        tasks = []

        for indicator in self._indicators:
            if len(bars) < indicator.warmup_periods:
                continue

            # Re-check started flag before scheduling (race condition guard)
            if not self._started:
                return

            state_key: StateKey = (event.symbol, event.timeframe, indicator.name)
            prev_state = self._previous_states.get(state_key)

            try:
                task = loop.run_in_executor(
                    self._executor,
                    self._compute_indicator,
                    indicator,
                    bars,
                    event.symbol,
                    event.timeframe,
                    bar_entry["timestamp"],
                    prev_state,
                )
                tasks.append(task)
            except RuntimeError:
                # Executor was shutdown between check and call
                logger.debug("Executor shutdown during indicator processing")
                return

        if not tasks:
            return

        # Execute all indicator calculations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Publish results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Indicator calculation failed: {result}")
                continue

            if result is None:
                continue

            update_event, new_state = result
            self._publish_update(update_event, new_state)

    def _compute_indicator(
        self,
        indicator: Indicator,
        bars: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        prev_state: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[IndicatorUpdateEvent, Dict[str, Any]]]:
        """
        Compute a single indicator (runs in thread pool).

        Returns:
            Tuple of (IndicatorUpdateEvent, new_state) or None on failure
        """
        try:
            # Build DataFrame from bar history
            df = pd.DataFrame(bars)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")

            # Validate required fields
            for field in indicator.required_fields:
                if field not in df.columns:
                    logger.debug(
                        f"Indicator {indicator.name} missing field: {field}"
                    )
                    return None

            # Calculate indicator
            result_df = indicator.calculate(df, indicator.default_params)
            if result_df.empty:
                return None

            # Extract current and previous rows for state
            current = result_df.iloc[-1]
            previous = result_df.iloc[-2] if len(result_df) > 1 else None

            # Get state dictionary from indicator
            state = indicator.get_state(current, previous)
            value = state.get("value")

            prev_value = prev_state.get("value") if prev_state else None

            # Build event
            event = IndicatorUpdateEvent(
                timestamp=timestamp,
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator.name,
                value=value,
                state=state,
                previous_value=prev_value,
                previous_state=prev_state,
            )

            return event, state

        except Exception as e:
            logger.error(
                f"Indicator {indicator.name} failed for {symbol}/{timeframe}: {e}"
            )
            return None

    def _publish_update(
        self, event: IndicatorUpdateEvent, new_state: Dict[str, Any]
    ) -> None:
        """Publish INDICATOR_UPDATE event and update state cache."""
        try:
            self._event_bus.publish(EventType.INDICATOR_UPDATE, event)

            # Update previous state cache
            state_key: StateKey = (event.symbol, event.timeframe, event.indicator)
            self._previous_states[state_key] = new_state

        except Exception as e:
            logger.error(
                f"Failed to publish INDICATOR_UPDATE for "
                f"{event.symbol}/{event.timeframe}/{event.indicator}: {e}"
            )

    @staticmethod
    def _coerce_bar_close(payload: Any) -> Optional[BarCloseEvent]:
        """Coerce payload to BarCloseEvent."""
        if isinstance(payload, BarCloseEvent):
            return payload

        if isinstance(payload, dict):
            try:
                return BarCloseEvent.from_dict(payload)
            except Exception:
                return None

        return None
