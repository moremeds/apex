"""
Market Data Router - Symbol registry with line limit enforcement.

A2: Centralized router for IB market data subscriptions with:
- Symbol registry with refcount tracking
- Line limit enforcement (80 API, 20 reserved for TWS)
- Snapshot vs streaming mode selection
- Internal pub-sub fanout to multiple consumers
- No duplicate subscriptions

Supersedes: C11

Usage:
    router = MarketDataRouter(ib_connection, event_bus)
    await router.start()

    # Subscribe to market data (refcounted)
    await router.subscribe("AAPL", consumer_id="risk_engine")
    await router.subscribe("AAPL", consumer_id="scanner")  # Reuses existing

    # Unsubscribe (only cancels when refcount reaches 0)
    await router.unsubscribe("AAPL", consumer_id="risk_engine")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ib_async import IB
    from ...domain.interfaces.event_bus import EventBus

logger = get_logger(__name__)


class SubscriptionMode(Enum):
    """Market data subscription mode."""
    STREAMING = "streaming"  # Continuous updates (uses line)
    SNAPSHOT = "snapshot"    # One-time fetch (no line usage)


@dataclass
class SymbolSubscription:
    """Subscription state for a single symbol."""
    symbol: str
    mode: SubscriptionMode
    consumers: Set[str] = field(default_factory=set)
    ticker: Any = None  # IB ticker object
    last_update: float = 0.0
    error_count: int = 0

    @property
    def refcount(self) -> int:
        """Number of consumers subscribed."""
        return len(self.consumers)


@dataclass
class RouterMetrics:
    """Metrics for market data router."""
    subscriptions_active: int = 0
    subscriptions_total: int = 0
    unsubscriptions_total: int = 0
    line_limit_rejections: int = 0
    duplicate_requests: int = 0
    fanout_total: int = 0


class MarketDataRouter:
    """
    Centralized market data subscription router.

    A2: Manages IB market data lines to prevent "line limit reached" errors
    while efficiently sharing subscriptions across multiple consumers.
    """

    # IB limits
    MAX_API_LINES = 80  # API-side limit
    RESERVED_TWS_LINES = 20  # Reserved for manual TWS usage
    EFFECTIVE_LIMIT = MAX_API_LINES - RESERVED_TWS_LINES  # 60 usable

    def __init__(
        self,
        ib: Optional["IB"] = None,
        event_bus: Optional["EventBus"] = None,
        max_lines: int = EFFECTIVE_LIMIT,
        snapshot_ttl_seconds: int = 5,
    ):
        """
        Initialize market data router.

        Args:
            ib: IB connection for market data.
            event_bus: Event bus for publishing updates.
            max_lines: Maximum streaming subscriptions (default: 60).
            snapshot_ttl_seconds: TTL for snapshot data freshness.
        """
        self._ib = ib
        self._event_bus = event_bus
        self._max_lines = max_lines
        self._snapshot_ttl = snapshot_ttl_seconds

        # Symbol registry
        self._subscriptions: Dict[str, SymbolSubscription] = {}
        self._lock = RLock()

        # Consumer callbacks
        self._callbacks: Dict[str, Callable[[str, Any], None]] = {}

        # State
        self._running = False
        self._metrics = RouterMetrics()

    def set_ib(self, ib: "IB") -> None:
        """Set IB connection (for late binding)."""
        self._ib = ib

    def set_event_bus(self, event_bus: "EventBus") -> None:
        """Set event bus (for late binding)."""
        self._event_bus = event_bus

    async def start(self) -> None:
        """Start the router."""
        if self._running:
            return

        self._running = True
        logger.info(
            "MarketDataRouter started (max_lines=%d, reserved=%d)",
            self._max_lines, self.RESERVED_TWS_LINES
        )

    async def stop(self) -> None:
        """Stop the router and cancel all subscriptions."""
        self._running = False

        # Cancel all active subscriptions
        with self._lock:
            for symbol in list(self._subscriptions.keys()):
                await self._cancel_subscription(symbol)
            # Clear all subscriptions after canceling
            self._subscriptions.clear()
            self._metrics.subscriptions_active = 0

        logger.info("MarketDataRouter stopped")

    @property
    def active_lines(self) -> int:
        """Number of active streaming subscriptions."""
        with self._lock:
            return sum(
                1 for sub in self._subscriptions.values()
                if sub.mode == SubscriptionMode.STREAMING and sub.ticker is not None
            )

    @property
    def available_lines(self) -> int:
        """Number of available streaming lines."""
        return self._max_lines - self.active_lines

    def register_callback(
        self,
        consumer_id: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        """
        Register callback for market data updates.

        Args:
            consumer_id: Unique consumer identifier.
            callback: Function to call with (symbol, market_data).
        """
        self._callbacks[consumer_id] = callback
        logger.debug("Registered callback for consumer: %s", consumer_id)

    def unregister_callback(self, consumer_id: str) -> None:
        """Unregister callback for a consumer."""
        self._callbacks.pop(consumer_id, None)

    async def subscribe(
        self,
        symbol: str,
        consumer_id: str,
        mode: SubscriptionMode = SubscriptionMode.STREAMING,
    ) -> bool:
        """
        Subscribe to market data for a symbol.

        If symbol already subscribed, adds consumer to refcount.
        If new subscription and at line limit, rejects or uses snapshot.

        Args:
            symbol: Stock/option symbol.
            consumer_id: Unique consumer identifier.
            mode: Subscription mode (streaming or snapshot).

        Returns:
            True if subscription succeeded, False if rejected.
        """
        with self._lock:
            # Check if already subscribed
            if symbol in self._subscriptions:
                sub = self._subscriptions[symbol]
                if consumer_id not in sub.consumers:
                    sub.consumers.add(consumer_id)
                    self._metrics.duplicate_requests += 1
                    logger.debug(
                        "Added consumer %s to existing subscription %s (refcount=%d)",
                        consumer_id, symbol, sub.refcount
                    )
                return True

            # Check line limit for streaming
            if mode == SubscriptionMode.STREAMING and self.active_lines >= self._max_lines:
                self._metrics.line_limit_rejections += 1
                logger.warning(
                    "Line limit reached (%d/%d), rejecting streaming subscription for %s",
                    self.active_lines, self._max_lines, symbol
                )
                # Fallback to snapshot mode
                mode = SubscriptionMode.SNAPSHOT

            # Create new subscription
            sub = SymbolSubscription(
                symbol=symbol,
                mode=mode,
                consumers={consumer_id},
            )
            self._subscriptions[symbol] = sub
            self._metrics.subscriptions_total += 1
            self._metrics.subscriptions_active += 1

        # Start IB subscription outside lock
        if self._ib and mode == SubscriptionMode.STREAMING:
            try:
                await self._start_subscription(symbol)
            except Exception as e:
                logger.error("Failed to start subscription for %s: %s", symbol, e)
                return False

        logger.debug(
            "Created %s subscription for %s (consumer=%s, lines=%d/%d)",
            mode.value, symbol, consumer_id, self.active_lines, self._max_lines
        )
        return True

    async def unsubscribe(self, symbol: str, consumer_id: str) -> None:
        """
        Unsubscribe a consumer from a symbol.

        Only cancels IB subscription when refcount reaches 0.

        Args:
            symbol: Symbol to unsubscribe from.
            consumer_id: Consumer to remove.
        """
        should_cancel = False

        with self._lock:
            if symbol not in self._subscriptions:
                return

            sub = self._subscriptions[symbol]
            sub.consumers.discard(consumer_id)
            self._metrics.unsubscriptions_total += 1

            if sub.refcount == 0:
                should_cancel = True
                del self._subscriptions[symbol]
                self._metrics.subscriptions_active -= 1

        # Cancel IB subscription outside lock
        if should_cancel:
            await self._cancel_subscription(symbol)
            logger.debug("Cancelled subscription for %s (no remaining consumers)", symbol)
        else:
            logger.debug(
                "Removed consumer %s from %s (refcount=%d)",
                consumer_id, symbol, sub.refcount
            )

    async def _start_subscription(self, symbol: str) -> None:
        """Start IB market data subscription for symbol."""
        if not self._ib:
            return

        # This would integrate with existing market_data_fetcher
        # For now, we track the intent
        with self._lock:
            if symbol in self._subscriptions:
                sub = self._subscriptions[symbol]
                sub.last_update = time.time()
                # sub.ticker = await self._ib.reqMktData(contract, ...)

    async def _cancel_subscription(self, symbol: str) -> None:
        """Cancel IB market data subscription for symbol."""
        if not self._ib:
            return

        # This would integrate with existing market_data_fetcher
        # For now, we track the intent
        pass

    def _fanout(self, symbol: str, data: Any) -> None:
        """
        Fanout market data to all registered consumers.

        Args:
            symbol: Symbol that received update.
            data: Market data to distribute.
        """
        with self._lock:
            if symbol not in self._subscriptions:
                return

            sub = self._subscriptions[symbol]
            sub.last_update = time.time()

            # Call each consumer's callback
            for consumer_id in sub.consumers:
                callback = self._callbacks.get(consumer_id)
                if callback:
                    try:
                        callback(symbol, data)
                        self._metrics.fanout_total += 1
                    except Exception as e:
                        logger.warning(
                            "Callback error for consumer %s on %s: %s",
                            consumer_id, symbol, e
                        )

    def get_subscription(self, symbol: str) -> Optional[SymbolSubscription]:
        """Get subscription info for a symbol."""
        with self._lock:
            return self._subscriptions.get(symbol)

    def get_all_symbols(self) -> List[str]:
        """Get all subscribed symbols."""
        with self._lock:
            return list(self._subscriptions.keys())

    def get_consumers(self, symbol: str) -> Set[str]:
        """Get all consumers for a symbol."""
        with self._lock:
            sub = self._subscriptions.get(symbol)
            return set(sub.consumers) if sub else set()

    def get_metrics(self) -> RouterMetrics:
        """Get router performance metrics."""
        return self._metrics

    def get_stats(self) -> dict:
        """Get router statistics for monitoring."""
        with self._lock:
            streaming_count = sum(
                1 for sub in self._subscriptions.values()
                if sub.mode == SubscriptionMode.STREAMING
            )
            snapshot_count = len(self._subscriptions) - streaming_count

            return {
                "running": self._running,
                "active_lines": self.active_lines,
                "max_lines": self._max_lines,
                "available_lines": self.available_lines,
                "streaming_subscriptions": streaming_count,
                "snapshot_subscriptions": snapshot_count,
                "total_consumers": sum(
                    sub.refcount for sub in self._subscriptions.values()
                ),
                "subscriptions_total": self._metrics.subscriptions_total,
                "line_limit_rejections": self._metrics.line_limit_rejections,
                "duplicate_requests": self._metrics.duplicate_requests,
                "fanout_total": self._metrics.fanout_total,
            }
