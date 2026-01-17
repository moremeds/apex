"""
BarAggregator - Aggregates ticks into bars and publishes BAR_CLOSE events.

Designed for O(1) on_tick performance with per-symbol BarBuilder instances.
Supports multiple timeframes via separate aggregator instances.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

from src.domain.events.domain_events import BarCloseEvent, MarketDataTickEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_utc

from .bar_builder import TIMEFRAME_SECONDS, BarBuilder

if TYPE_CHECKING:
    from src.infrastructure.observability import SignalMetrics

logger = get_logger(__name__)


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus compatibility."""

    def publish(self, event_type: EventType, payload: Any, priority: Optional[int] = None) -> None:
        """Publish an event to the bus."""
        ...


class BarAggregator:
    """
    Aggregates ticks into OHLCV bars for a single timeframe.

    Creates one BarAggregator per timeframe to maintain O(1) on_tick performance.
    Each aggregator manages BarBuilders per symbol and publishes BAR_CLOSE
    events when bars complete.

    Example:
        aggregator_1m = BarAggregator("1m", event_bus)
        aggregator_5m = BarAggregator("5m", event_bus)

        def on_tick(tick):
            aggregator_1m.on_tick(tick)
            aggregator_5m.on_tick(tick)
    """

    def __init__(
        self,
        timeframe: str,
        event_bus: EventBusProtocol,
        signal_metrics: Optional["SignalMetrics"] = None,
    ) -> None:
        """
        Initialize aggregator for a specific timeframe.

        Args:
            timeframe: Bar timeframe (e.g., "1m", "5m", "1h")
            event_bus: Event bus for publishing BAR_CLOSE events
            signal_metrics: Metrics collector for instrumentation

        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(TIMEFRAME_SECONDS.keys())}"
            )

        self._timeframe = timeframe
        self._event_bus = event_bus
        self._metrics = signal_metrics
        self._builders: Dict[str, BarBuilder] = {}
        self._bars_emitted = 0

    @property
    def timeframe(self) -> str:
        """Current timeframe."""
        return self._timeframe

    @property
    def bars_emitted(self) -> int:
        """Total number of bars emitted since creation."""
        return self._bars_emitted

    def on_tick(self, tick: Any) -> Optional[BarCloseEvent]:
        """
        Process a single tick. O(1) time complexity.

        Accepts:
        - MarketDataTickEvent
        - QuoteTick
        - Dict with symbol/price/timestamp fields
        - Any object with symbol/last/timestamp attributes

        Args:
            tick: Tick data in any supported format

        Returns:
            BarCloseEvent if a bar was completed, None otherwise
        """
        symbol = self._extract_symbol(tick)
        if not symbol:
            logger.warning(f"Could not extract symbol from tick: type={type(tick).__name__}")
            return None

        price = self._extract_price(tick)
        if price is None:
            # Log at WARNING level to help diagnose missing ticks
            logger.warning(
                f"Could not extract price from tick: symbol={symbol}, "
                f"type={type(tick).__name__}, keys={list(tick.keys()) if isinstance(tick, dict) else 'N/A'}"
            )
            return None

        timestamp = self._extract_timestamp(tick)
        volume = self._extract_volume(tick)

        builder = self._builders.get(symbol)
        closed_event: Optional[BarCloseEvent] = None

        # Handle out-of-order tick (bump to WARNING for better visibility)
        if builder is not None and timestamp < builder.bar_start:
            logger.warning(
                f"Ignoring out-of-order tick for {symbol}: "
                f"tick_ts={timestamp}, bar_start={builder.bar_start}"
            )
            return None

        # Check if tick falls outside current bar window
        if builder is None or timestamp >= builder.bar_end:
            # Close existing bar if it has data
            if builder is not None and not builder.is_empty:
                closed_event = builder.to_close_event()
                self._publish_bar_close(closed_event)

            # Create new builder for current bar period
            builder = BarBuilder.create_for_timestamp(symbol, self._timeframe, timestamp)
            self._builders[symbol] = builder

            # Update active bar builders gauge
            if self._metrics:
                self._metrics.set_bar_builders_active(len(self._builders), self._timeframe)

        # Update current bar
        builder.update(price=price, volume=volume, timestamp=timestamp)

        return closed_event

    def flush(self, symbol: Optional[str] = None) -> list[BarCloseEvent]:
        """
        Force-close open bars and emit events.

        Useful for end-of-session cleanup or when switching modes.

        Args:
            symbol: Optional symbol to flush, or None for all symbols

        Returns:
            List of BarCloseEvents emitted
        """
        events: list[BarCloseEvent] = []

        if symbol is not None:
            builder = self._builders.pop(symbol, None)
            if builder is not None and not builder.is_empty:
                event = builder.to_close_event()
                self._publish_bar_close(event)
                events.append(event)
        else:
            for sym, builder in list(self._builders.items()):
                if not builder.is_empty:
                    event = builder.to_close_event()
                    self._publish_bar_close(event)
                    events.append(event)
            self._builders.clear()

        return events

    def _publish_bar_close(self, event: BarCloseEvent) -> None:
        """Publish BAR_CLOSE event to the bus."""
        try:
            self._event_bus.publish(EventType.BAR_CLOSE, event)
            self._bars_emitted += 1

            # Record bar emission metric
            if self._metrics:
                self._metrics.record_bar_emitted(self._timeframe)

            # Structured debug log for bar close
            logger.debug(
                "Bar closed",
                extra={
                    "symbol": event.symbol,
                    "timeframe": event.timeframe,
                    "open": event.open,
                    "high": event.high,
                    "low": event.low,
                    "close": event.close,
                    "volume": event.volume,
                    "bar_end": event.bar_end.isoformat() if event.bar_end else None,
                },
            )

        except Exception as e:
            if self._metrics:
                self._metrics.record_error("bar_aggregator", "publish")
            logger.error(
                f"Failed to publish BAR_CLOSE: {e!r} (symbol={event.symbol}, tf={event.timeframe})",
                exc_info=True,
            )

    @staticmethod
    def _extract_symbol(tick: Any) -> str:
        """Extract symbol from tick (O(1))."""
        if isinstance(tick, MarketDataTickEvent):
            return tick.symbol
        if isinstance(tick, dict):
            return tick.get("symbol", "")
        return getattr(tick, "symbol", "")

    @staticmethod
    def _extract_timestamp(tick: Any) -> datetime:
        """Extract timestamp from tick (O(1))."""
        if isinstance(tick, dict):
            ts = tick.get("timestamp")
            return ts if ts is not None else now_utc()
        ts = getattr(tick, "timestamp", None)
        return ts if ts is not None else now_utc()

    @staticmethod
    def _extract_volume(tick: Any) -> Optional[float]:
        """Extract volume from tick (O(1))."""
        if isinstance(tick, dict):
            return tick.get("volume")
        return getattr(tick, "volume", None)

    @staticmethod
    def _extract_price(tick: Any) -> Optional[float]:
        """
        Extract price from tick (O(1)).

        Priority: last > mid > computed mid from bid/ask
        """
        if isinstance(tick, dict):
            if "last" in tick and tick["last"] is not None:
                return float(tick["last"])
            if "mid" in tick and tick["mid"] is not None:
                return float(tick["mid"])
            if "price" in tick and tick["price"] is not None:
                return float(tick["price"])
            bid, ask = tick.get("bid"), tick.get("ask")
            if bid is not None and ask is not None:
                return (float(bid) + float(ask)) / 2
            return None

        # Object access
        last = getattr(tick, "last", None)
        if last is not None:
            return float(last)

        mid = getattr(tick, "mid", None)
        if mid is not None:
            return float(mid)

        bid = getattr(tick, "bid", None)
        ask = getattr(tick, "ask", None)
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2

        return None
