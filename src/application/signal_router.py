"""
Signal routing service for trading signals.

Routes trading signals from strategies to:
- Event bus (for TUI display)
- Structured logging
- Optional persistence
- Optional execution

Signal flow:
1. Strategy.emit_signal() -> TradingSignal
2. SignalRouter.route() wraps in TradingSignalEvent
3. Event bus publishes to subscribers (TUI, logging, etc.)
4. Optional: Persist to signal store
5. Optional: Forward to execution adapter

Usage:
    router = SignalRouter(event_bus)

    # Register with strategy
    strategy.on_signal_callback(router.route)

    # Or register in strategy runner
    def run_strategy(strategy, router):
        strategy.on_signal_callback(router.route)
        strategy.start()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from ..domain.events.domain_events import TradingSignalEvent
from ..domain.events.event_types import EventType

if TYPE_CHECKING:
    from ..domain.interfaces.event_bus import EventBus
    from ..domain.strategy.base import TradingSignal

logger = logging.getLogger(__name__)


@dataclass
class SignalRouterConfig:
    """Configuration for signal routing behavior."""

    # Rate limiting
    rate_limit_ms: int = 100  # Minimum ms between signals per symbol
    rate_limit_per_strategy: bool = False  # Also rate-limit per strategy

    # Deduplication
    dedupe_window_ms: int = 1000  # Ignore duplicate signals within window

    # Routing options
    persist: bool = False  # Save signals to store
    execute: bool = False  # Forward to execution adapter

    # Filtering
    min_strength: float = 0.0  # Ignore signals below this strength
    allowed_directions: Set[str] = field(default_factory=lambda: {"LONG", "SHORT", "FLAT"})


@dataclass
class SignalStats:
    """Statistics for signal routing."""

    total_received: int = 0
    total_published: int = 0
    rate_limited: int = 0
    deduplicated: int = 0
    filtered: int = 0
    errors: int = 0


class SignalRouter:
    """
    Routes trading signals with policy enforcement.

    Features:
    - Rate limiting per symbol (prevents signal spam)
    - Deduplication within time window
    - Strength filtering
    - Structured logging
    - Event bus publication for TUI
    - Optional persistence and execution

    Example:
        router = SignalRouter(event_bus)
        strategy.on_signal_callback(router.route)

        # Later: check stats
        print(f"Signals routed: {router.stats.total_published}")
    """

    def __init__(
        self,
        event_bus: "EventBus",
        config: Optional[SignalRouterConfig] = None,
        signal_store: Optional[Any] = None,
        execution_adapter: Optional[Any] = None,
    ):
        """
        Initialize signal router.

        Args:
            event_bus: Event bus for publishing signals
            config: Routing configuration
            signal_store: Optional store for signal persistence
            execution_adapter: Optional adapter for execution forwarding
        """
        self._event_bus = event_bus
        self._config = config or SignalRouterConfig()
        self._signal_store = signal_store
        self._execution_adapter = execution_adapter

        # Rate limiting state: {key: last_signal_time_ms}
        self._last_signal_time: Dict[str, float] = {}

        # Deduplication state: {hash: timestamp_ms}
        self._recent_signals: Dict[str, float] = {}

        # Statistics
        self._stats = SignalStats()

        # Callbacks for extensibility
        self._pre_route_callbacks: List[Callable[["TradingSignal"], bool]] = []
        self._post_route_callbacks: List[Callable[["TradingSignal"], None]] = []

    @property
    def stats(self) -> SignalStats:
        """Get routing statistics."""
        return self._stats

    def route(self, signal: "TradingSignal") -> bool:
        """
        Route a trading signal through the pipeline.

        Args:
            signal: Trading signal from strategy

        Returns:
            True if signal was published, False if filtered/rate-limited
        """
        self._stats.total_received += 1

        try:
            # Pre-route callbacks (can veto signal)
            for callback in self._pre_route_callbacks:
                if not callback(signal):
                    self._stats.filtered += 1
                    return False

            # Filter by strength
            if signal.strength < self._config.min_strength:
                self._stats.filtered += 1
                logger.debug(
                    f"Signal filtered: strength {signal.strength} < {self._config.min_strength}"
                )
                return False

            # Filter by direction
            if signal.direction not in self._config.allowed_directions:
                self._stats.filtered += 1
                logger.debug(f"Signal filtered: direction {signal.direction} not allowed")
                return False

            # Rate limiting
            if not self._check_rate_limit(signal):
                self._stats.rate_limited += 1
                logger.debug(f"Signal rate-limited: {signal.symbol}")
                return False

            # Deduplication
            if not self._check_dedupe(signal):
                self._stats.deduplicated += 1
                logger.debug(f"Signal deduplicated: {signal.symbol} {signal.direction}")
                return False

            # Publish to event bus
            self._publish_signal(signal)
            self._stats.total_published += 1

            # Structured logging
            self._log_signal(signal)

            # Optional persistence
            if self._config.persist and self._signal_store:
                self._persist_signal(signal)

            # Optional execution
            if self._config.execute and self._execution_adapter:
                self._execute_signal(signal)

            # Post-route callbacks
            for post_callback in self._post_route_callbacks:
                post_callback(signal)

            return True

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Signal routing error: {e}", exc_info=True)
            return False

    def _check_rate_limit(self, signal: "TradingSignal") -> bool:
        """Check if signal passes rate limiting."""
        now_ms = time.time() * 1000

        # Build rate limit key
        if self._config.rate_limit_per_strategy:
            key = f"{signal.strategy_id}:{signal.symbol}:{signal.direction}"
        else:
            key = f"{signal.symbol}:{signal.direction}"

        last_time = self._last_signal_time.get(key, 0)
        if now_ms - last_time < self._config.rate_limit_ms:
            return False

        self._last_signal_time[key] = now_ms
        return True

    def _check_dedupe(self, signal: "TradingSignal") -> bool:
        """Check if signal is a duplicate."""
        now_ms = time.time() * 1000

        # Clean old entries
        cutoff = now_ms - self._config.dedupe_window_ms
        self._recent_signals = {k: v for k, v in self._recent_signals.items() if v > cutoff}

        # Create signal hash
        signal_hash = f"{signal.symbol}:{signal.direction}:{signal.strength}"

        if signal_hash in self._recent_signals:
            return False

        self._recent_signals[signal_hash] = now_ms
        return True

    def _publish_signal(self, signal: "TradingSignal") -> None:
        """Publish signal to event bus."""
        event = TradingSignalEvent.from_signal(signal, source=signal.strategy_id)
        self._event_bus.publish(EventType.TRADING_SIGNAL, event)

    def _log_signal(self, signal: "TradingSignal") -> None:
        """Log signal with structured data."""
        logger.info(
            "trading_signal",
            extra={
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "strength": signal.strength,
                "strategy_id": signal.strategy_id,
                "reason": signal.reason,
                "target_quantity": signal.target_quantity,
                "target_price": signal.target_price,
            },
        )

    def _persist_signal(self, signal: "TradingSignal") -> None:
        """Persist signal to store."""
        assert self._signal_store is not None, "Signal store not configured"
        try:
            self._signal_store.save(signal)
        except Exception as e:
            logger.error(f"Signal persistence error: {e}")

    def _execute_signal(self, signal: "TradingSignal") -> None:
        """Forward signal to execution adapter."""
        assert self._execution_adapter is not None, "Execution adapter not configured"
        try:
            self._execution_adapter.submit(signal)
        except Exception as e:
            logger.error(f"Signal execution error: {e}")

    def add_pre_route_callback(self, callback: Callable[["TradingSignal"], bool]) -> None:
        """
        Add callback to run before routing.

        Callback returns True to allow signal, False to filter.
        """
        self._pre_route_callbacks.append(callback)

    def add_post_route_callback(self, callback: Callable[["TradingSignal"], None]) -> None:
        """Add callback to run after successful routing."""
        self._post_route_callbacks.append(callback)

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = SignalStats()

    def clear_state(self) -> None:
        """Clear rate limiting and deduplication state."""
        self._last_signal_time.clear()
        self._recent_signals.clear()
