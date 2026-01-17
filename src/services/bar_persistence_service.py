"""
BarPersistenceService - Persists live BAR_CLOSE events to Parquet storage.

Batches writes to avoid excessive I/O, flushing when:
- Buffer reaches threshold (default: 10 bars per symbol/timeframe)
- Time since last flush exceeds threshold (default: 60 seconds)
- Explicit flush_all() on shutdown
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.domain.events.domain_events import BarCloseEvent, BarData
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_utc

if TYPE_CHECKING:
    from src.domain.interfaces.event_bus import EventBus
    from src.infrastructure.stores.parquet_historical_store import ParquetHistoricalStore

logger = get_logger(__name__)


class BarPersistenceService:
    """
    Persists live bars from BAR_CLOSE events to Parquet storage.

    Uses batching to minimize I/O overhead while ensuring data durability.
    Bars are buffered per (symbol, timeframe) and flushed when thresholds
    are reached or on graceful shutdown.
    """

    def __init__(
        self,
        event_bus: "EventBus",
        bar_store: "ParquetHistoricalStore",
        flush_threshold_bars: int = 10,
        flush_threshold_sec: float = 60.0,
    ) -> None:
        """
        Initialize the bar persistence service.

        Args:
            event_bus: Event bus to subscribe to BAR_CLOSE events.
            bar_store: Parquet store for writing bars.
            flush_threshold_bars: Flush buffer when this many bars accumulated.
            flush_threshold_sec: Flush buffer after this many seconds since last flush.
        """
        self._event_bus = event_bus
        self._bar_store = bar_store
        self._flush_threshold_bars = flush_threshold_bars
        self._flush_threshold_sec = flush_threshold_sec

        # Pending bars: (symbol, timeframe) -> List[BarData]
        self._pending: Dict[Tuple[str, str], List[BarData]] = defaultdict(list)
        self._last_flush: Dict[Tuple[str, str], datetime] = {}
        self._lock = threading.Lock()

        # Stats
        self._bars_received = 0
        self._bars_persisted = 0
        self._flushes = 0

        self._started = False

    def start(self) -> None:
        """Start the service and subscribe to BAR_CLOSE events."""
        if self._started:
            return

        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)
        self._started = True
        logger.info(
            "BarPersistenceService started",
            extra={
                "flush_threshold_bars": self._flush_threshold_bars,
                "flush_threshold_sec": self._flush_threshold_sec,
            },
        )

    def stop(self) -> None:
        """Stop the service, flush pending bars, and unsubscribe."""
        if not self._started:
            return

        self._started = False

        # Unsubscribe first
        try:
            self._event_bus.unsubscribe(EventType.BAR_CLOSE, self._on_bar_close)
        except Exception as e:
            logger.warning(f"Error unsubscribing from BAR_CLOSE: {e}")

        # Flush all pending bars
        flushed = self.flush_all()
        logger.info(
            "BarPersistenceService stopped",
            extra={
                "bars_flushed_on_stop": flushed,
                "total_bars_received": self._bars_received,
                "total_bars_persisted": self._bars_persisted,
                "total_flushes": self._flushes,
            },
        )

    def _on_bar_close(self, payload: Any) -> None:
        """Handle BAR_CLOSE event."""
        if not self._started:
            return

        # Coerce payload to BarCloseEvent
        if isinstance(payload, BarCloseEvent):
            event = payload
        elif isinstance(payload, dict):
            try:
                event = BarCloseEvent.from_dict(payload)
            except Exception as e:
                logger.warning(f"Failed to parse BAR_CLOSE payload: {e}")
                return
        else:
            logger.warning(f"Unexpected BAR_CLOSE payload type: {type(payload)}")
            return

        # Convert BarCloseEvent to BarData for storage
        bar = BarData(
            symbol=event.symbol,
            timeframe=event.timeframe,
            open=event.open,
            high=event.high,
            low=event.low,
            close=event.close,
            volume=int(event.volume) if event.volume else None,
            bar_start=event.bar_end,  # bar_end is the bar's close timestamp
            bar_end=event.bar_end,
            timestamp=event.timestamp,
            source="live",
        )

        key = (event.symbol, event.timeframe)

        with self._lock:
            self._pending[key].append(bar)
            self._bars_received += 1

            # Check flush conditions
            if self._should_flush(key):
                self._flush_key(key)

    def _should_flush(self, key: Tuple[str, str]) -> bool:
        """Check if buffer should be flushed for the given key."""
        # Bar count threshold
        if len(self._pending[key]) >= self._flush_threshold_bars:
            return True

        # Time threshold
        last = self._last_flush.get(key)
        if last is None:
            self._last_flush[key] = now_utc()
            return False

        elapsed = (now_utc() - last).total_seconds()
        return elapsed >= self._flush_threshold_sec

    def _flush_key(self, key: Tuple[str, str]) -> int:
        """
        Flush pending bars for a specific (symbol, timeframe).

        Returns:
            Number of bars written.
        """
        bars = self._pending.pop(key, [])
        if not bars:
            return 0

        symbol, timeframe = key

        try:
            count = self._bar_store.write_bars(symbol, timeframe, bars, mode="upsert")
            self._bars_persisted += count
            self._flushes += 1
            self._last_flush[key] = now_utc()

            logger.debug(
                "Flushed bars to Parquet",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars_written": count,
                    "total_persisted": self._bars_persisted,
                },
            )
            return count
        except Exception as e:
            logger.error(f"Failed to persist bars for {symbol}/{timeframe}: {e}")
            # Put bars back in queue for retry
            self._pending[key].extend(bars)
            return 0

    def flush_all(self) -> int:
        """
        Flush all pending bars immediately.

        Called on graceful shutdown to ensure no data loss.

        Returns:
            Total number of bars flushed.
        """
        with self._lock:
            keys = list(self._pending.keys())
            total = 0
            for key in keys:
                total += self._flush_key(key)

        if total > 0:
            logger.info(f"Flushed all pending bars: {total} bars written")
        return total

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        with self._lock:
            pending_count = sum(len(bars) for bars in self._pending.values())
            pending_keys = list(self._pending.keys())

        return {
            "bars_received": self._bars_received,
            "bars_persisted": self._bars_persisted,
            "bars_pending": pending_count,
            "pending_symbols": [f"{s}/{tf}" for s, tf in pending_keys],
            "flushes": self._flushes,
            "started": self._started,
        }
