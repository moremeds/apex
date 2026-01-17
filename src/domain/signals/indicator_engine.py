"""
IndicatorEngine - Computes indicators on bar close and publishes updates.

Subscribes to BAR_CLOSE events, calculates all registered indicators
using a ThreadPoolExecutor, and publishes INDICATOR_UPDATE events for
downstream rule evaluation.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from src.domain.events.domain_events import BarCloseEvent, IndicatorUpdateEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger

from .indicators.base import Indicator
from .indicators.registry import get_indicator_registry

if TYPE_CHECKING:
    from src.infrastructure.observability import SignalMetrics

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
        signal_metrics: Optional["SignalMetrics"] = None,
    ) -> None:
        """
        Initialize the indicator engine.

        Args:
            event_bus: Event bus for subscriptions and publishing
            max_workers: ThreadPool size for parallel indicator calculations
            max_history: Maximum bars to retain per symbol/timeframe (auto-calculated from warmup if None)
            signal_metrics: Metrics collector for instrumentation
        """
        self._event_bus = event_bus
        self._registry = get_indicator_registry()
        self._indicators: List[Indicator] = self._registry.get_all()
        self._max_workers = max_workers
        self._metrics = signal_metrics

        # Calculate max warmup from all indicators
        warmup_periods = [ind.warmup_periods for ind in self._indicators]
        self._max_warmup = max(warmup_periods) if warmup_periods else 0
        self._max_history = max_history or max(50, self._max_warmup + 10)

        # Bar history per (symbol, timeframe)
        self._history: Dict[BarKey, Deque[Dict[str, Any]]] = {}

        # Previous indicator states for transition detection
        self._previous_states: Dict[StateKey, Dict[str, Any]] = {}

        # PERF: Per-symbol locks to reduce contention (allows parallel processing of different symbols)
        self._locks: Dict[BarKey, RLock] = {}
        self._locks_lock = RLock()  # Meta-lock for creating new per-symbol locks

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

    # -------------------------------------------------------------------------
    # Introspection Methods (for SignalIntrospectionPort)
    # -------------------------------------------------------------------------

    def get_warmup_status(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get warmup progress for a symbol/timeframe.

        Per-symbol/timeframe granularity (cheap), not per-indicator.
        Reads from existing _history without copying.

        Args:
            symbol: Trading symbol (e.g., "AAPL").
            timeframe: Bar timeframe (e.g., "1h", "1d").

        Returns:
            Dict with warmup info.
        """
        key: BarKey = (symbol, timeframe)
        bar_count = len(self._history.get(key, []))
        bars_required = self._max_warmup
        progress_pct = min(1.0, bar_count / bars_required) if bars_required > 0 else 1.0

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_loaded": bar_count,
            "bars_required": bars_required,
            "progress_pct": progress_pct,
            "status": "ready" if bar_count >= bars_required else "warming_up",
        }

    def get_all_warmup_status(self) -> List[Dict[str, Any]]:
        """
        Get warmup status for all symbol/timeframe combinations.

        Returns:
            List of warmup status dicts.
        """
        return [self.get_warmup_status(sym, tf) for (sym, tf) in self._history.keys()]

    def get_indicator_state(
        self, symbol: str, timeframe: str, indicator: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current cached state for a specific indicator.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            indicator: Indicator name.

        Returns:
            Indicator state dict or None if not cached.
        """
        key: StateKey = (symbol, timeframe, indicator)
        return self._previous_states.get(key)

    def get_all_indicator_states(
        self, symbol: Optional[str] = None, timeframe: Optional[str] = None
    ) -> Dict[StateKey, Dict[str, Any]]:
        """
        Get all cached indicator states, optionally filtered.

        Args:
            symbol: Optional filter by symbol.
            timeframe: Optional filter by timeframe.

        Returns:
            Dict mapping (symbol, timeframe, indicator) -> state_dict.
        """
        if symbol is None and timeframe is None:
            return dict(self._previous_states)

        result: Dict[StateKey, Dict[str, Any]] = {}
        for key, state in self._previous_states.items():
            sym, tf, ind = key
            if symbol is not None and sym != symbol:
                continue
            if timeframe is not None and tf != timeframe:
                continue
            result[key] = state
        return result

    def start(self) -> None:
        """Start the engine by subscribing to BAR_CLOSE events."""
        if self._started:
            return

        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)
        self._started = True
        logger.info(
            "IndicatorEngine started",
            extra={
                "indicators": len(self._indicators),
                "indicator_names": [ind.name for ind in self._indicators],
                "max_workers": self._max_workers,
                "max_history": self._max_history,
                "max_warmup": self._max_warmup,
            },
        )

    def stop(self) -> None:
        """Stop the engine and clean up resources."""
        self._executor.shutdown(wait=False)
        self._started = False
        logger.info("IndicatorEngine stopped")

    def _get_lock(self, bar_key: BarKey) -> RLock:
        """
        Get or create a lock for a specific (symbol, timeframe) pair.

        Uses double-checked locking pattern for efficiency.
        """
        # Fast path: lock already exists
        lock = self._locks.get(bar_key)
        if lock is not None:
            return lock

        # Slow path: create new lock
        with self._locks_lock:
            # Double-check after acquiring meta-lock
            lock = self._locks.get(bar_key)
            if lock is None:
                lock = RLock()
                self._locks[bar_key] = lock
            return lock

    def inject_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        bar_dicts: List[Dict[str, Any]],
    ) -> int:
        """
        Inject historical bars for indicator warmup (IDEMPOTENT).

        Only injects bars newer than the latest bar in history.
        Safe to call multiple times - duplicates are skipped.

        Call this BEFORE live tick processing starts to warm up
        indicators with historical data from Parquet cache.

        Args:
            symbol: Trading symbol (e.g., "AAPL")
            timeframe: Bar timeframe (e.g., "1d")
            bar_dicts: List of bar dictionaries with keys:
                       {timestamp, open, high, low, close, volume}
                       Must be sorted ascending by timestamp.

        Returns:
            Number of NEW bars injected (duplicates are skipped)
        """
        bar_key: BarKey = (symbol, timeframe)

        with self._get_lock(bar_key):
            if bar_key not in self._history:
                self._history[bar_key] = deque(maxlen=self._max_history)

            # Get latest timestamp in existing history for idempotency check
            existing_history = self._history[bar_key]
            latest_ts = None
            if existing_history:
                latest_ts = existing_history[-1].get("timestamp")

            # Only inject bars NEWER than existing data
            new_bars = []
            for bar in bar_dicts:
                bar_ts = bar.get("timestamp")
                if latest_ts is None or bar_ts > latest_ts:
                    new_bars.append(bar)
                    latest_ts = bar_ts  # Update for next iteration

            for bar in new_bars:
                self._history[bar_key].append(bar)

            injected_count = len(new_bars)
            skipped_count = len(bar_dicts) - injected_count

            # Get latest bar timestamp for logging
            latest_bar_ts = None
            if self._history[bar_key]:
                latest_bar_ts = self._history[bar_key][-1].get("timestamp")

            logger.info(
                "Injected historical bars for indicator warmup",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars_injected": injected_count,
                    "bars_skipped_duplicate": skipped_count,
                    "history_size": len(self._history[bar_key]),
                    "max_history": self._max_history,
                    "max_warmup": self._max_warmup,
                    "latest_bar_ts": str(latest_bar_ts) if latest_bar_ts else None,
                },
            )
            return injected_count

    async def compute_on_history(self, symbol: str, timeframe: str) -> int:
        """
        Compute all indicators on existing history and publish updates.

        Call this AFTER inject_historical_bars() to immediately calculate
        indicator values instead of waiting for the next BAR_CLOSE event.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe

        Returns:
            Number of indicators computed
        """
        bar_key: BarKey = (symbol, timeframe)

        with self._get_lock(bar_key):
            if bar_key not in self._history:
                logger.debug(f"No history for {symbol}/{timeframe}, skipping compute")
                return 0
            bars = list(self._history[bar_key])

        if not bars:
            return 0

        if not self._indicators:
            return 0

        # Get latest bar timestamp for the computation
        latest_bar = bars[-1]
        timestamp: datetime = latest_bar.get("timestamp") or datetime.now(timezone.utc)

        # PERF: Create DataFrame ONCE and share across all indicator threads
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        bar_count = len(df)

        # Build tasks for indicators with sufficient warmup
        loop = asyncio.get_running_loop()
        tasks = []

        for indicator in self._indicators:
            if bar_count < indicator.warmup_periods:
                continue

            state_key: StateKey = (symbol, timeframe, indicator.name)
            prev_state = self._previous_states.get(state_key)

            try:
                task = loop.run_in_executor(
                    self._executor,
                    self._compute_indicator,
                    indicator,
                    df,  # Pass shared DataFrame instead of bars list
                    symbol,
                    timeframe,
                    timestamp,
                    prev_state,
                )
                tasks.append(task)
            except RuntimeError:
                logger.debug("Executor shutdown during history compute")
                return 0

        if not tasks:
            logger.debug(
                f"No indicators ready for {symbol}/{timeframe} "
                f"(history={bar_count}, max_warmup={self._max_warmup})"
            )
            return 0

        # Execute all indicator calculations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Publish results
        indicators_computed = 0
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Indicator computation error: {result}")
                continue

            if result is None:
                continue

            update_event, new_state = result
            self._publish_update(update_event, new_state)
            indicators_computed += 1

        logger.info(
            f"Computed {indicators_computed} indicators on history for {symbol}/{timeframe}",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators_computed": indicators_computed,
                "history_size": len(bars),
            },
        )

        return indicators_computed

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
        bar_timestamp = event.bar_end or event.timestamp
        bar_entry = {
            "timestamp": bar_timestamp,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }

        bar_key: BarKey = (event.symbol, event.timeframe)

        # Update history (thread-safe with per-symbol lock)
        with self._get_lock(bar_key):
            if bar_key not in self._history:
                self._history[bar_key] = deque(maxlen=self._max_history)
            self._history[bar_key].append(bar_entry)
            bars = list(self._history[bar_key])

        self._bars_processed += 1

        if not self._indicators:
            return

        # PERF: Create DataFrame ONCE and share across all indicator threads
        # This avoids creating 40 DataFrames (one per indicator) per bar close
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        bar_count = len(df)

        # Build tasks for indicators with sufficient warmup
        loop = asyncio.get_running_loop()
        tasks = []

        for indicator in self._indicators:
            if bar_count < indicator.warmup_periods:
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
                    df,  # Pass shared DataFrame instead of bars list
                    event.symbol,
                    event.timeframe,
                    bar_timestamp,
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
        batch_start = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Publish results and count successes
        indicators_computed = 0
        errors = 0
        for result in results:
            if isinstance(result, BaseException):
                errors += 1
                if self._metrics:
                    self._metrics.record_error("indicator_engine", "batch_compute")
                logger.error(
                    "Indicator calculation failed in batch",
                    extra={"error": str(result)},
                )
                continue

            if result is None:
                continue

            update_event, new_state = result
            self._publish_update(update_event, new_state)
            indicators_computed += 1

        # Log batch processing summary
        batch_duration_ms = (time.perf_counter() - batch_start) * 1000
        logger.debug(
            f"Bar processed: symbol={event.symbol} tf={event.timeframe} "
            f"indicators={indicators_computed} errors={errors} duration={batch_duration_ms:.1f}ms",
        )

    def _compute_indicator(
        self,
        indicator: Indicator,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        prev_state: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[IndicatorUpdateEvent, Dict[str, Any]]]:
        """
        Compute a single indicator (runs in thread pool).

        Args:
            indicator: The indicator to compute
            df: Pre-built DataFrame with OHLCV data (shared across threads, read-only)
            symbol: Trading symbol
            timeframe: Bar timeframe
            timestamp: Timestamp for the update event
            prev_state: Previous indicator state for transition detection

        Returns:
            Tuple of (IndicatorUpdateEvent, new_state) or None on failure
        """
        start_time = time.perf_counter()
        try:
            # Validate required fields
            for field in indicator.required_fields:
                if field not in df.columns:
                    logger.debug(
                        "Indicator missing required field",
                        extra={
                            "indicator": indicator.name,
                            "field": field,
                            "symbol": symbol,
                            "timeframe": timeframe,
                        },
                    )
                    return None

            # Calculate indicator (df is read-only, indicator.calculate returns new df)
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

            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            if self._metrics:
                self._metrics.record_indicator_compute_latency(duration_ms, indicator.name)
                self._metrics.record_indicator_computed(indicator.name)

            # Debug log with computation results - include state for rule debugging
            # Include numerical values for debugging signal generation issues
            state_summary = {
                k: v
                for k, v in state.items()
                if k
                in (
                    "zone",
                    "direction",
                    "trend",
                    "volatility",
                    "pressure",
                    "value",
                    "obv",
                    "cvd",
                    "ad",
                )
            }
            logger.debug(
                f"Indicator computed: {indicator.name} symbol={symbol} tf={timeframe} state={state_summary}",
            )

            return event, state

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            if self._metrics:
                self._metrics.record_error("indicator_engine", "compute")
            logger.error(
                "Indicator computation failed",
                extra={
                    "indicator": indicator.name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                },
            )
            return None

    def _publish_update(self, event: IndicatorUpdateEvent, new_state: Dict[str, Any]) -> None:
        """Publish INDICATOR_UPDATE event and update state cache."""
        try:
            self._event_bus.publish(EventType.INDICATOR_UPDATE, event)

            # Update previous state cache
            state_key: StateKey = (event.symbol, event.timeframe, event.indicator)
            self._previous_states[state_key] = new_state

        except Exception as e:
            if self._metrics:
                self._metrics.record_error("indicator_engine", "publish")
            logger.error(
                f"Failed to publish INDICATOR_UPDATE: {e!r} "
                f"(indicator={event.indicator}, symbol={event.symbol}, tf={event.timeframe})",
                exc_info=True,
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
