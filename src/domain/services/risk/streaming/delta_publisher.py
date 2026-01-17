"""
Delta Publisher - Publishes position deltas to the event bus.

Bridges the RiskFacade to the event system, converting PositionDelta
objects to PositionDeltaEvents and publishing them on the fast lane.

Also handles position synchronization by subscribing to:
- POSITIONS_READY: Initial position load at startup
- POSITION_UPDATED: Trade deals and position changes during runtime

Thread Safety:
    - Tick handler may be called from IB callback thread
    - RiskFacade handles thread safety internally
    - Event bus handles cross-thread publishing

Usage:
    publisher = DeltaPublisher(risk_facade, event_bus, position_store)
    publisher.start()  # Subscribe to tick and position events

    # When shutting down:
    publisher.stop()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.domain.events.domain_events import MarketDataTickEvent
from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.domain.events.priority_event_bus import PriorityEventBus
    from src.infrastructure.stores import MarketDataStore, PositionStore
    from src.models.position import Position

    from ..risk_facade import RiskFacade

logger = get_logger(__name__)


class DeltaPublisher:
    """
    Subscribes to tick events and publishes position deltas.

    This is the glue between market data and the streaming risk system.

    Flow:
        MarketDataTickEvent (fast lane)
            → DeltaPublisher._on_tick()
            → RiskFacade.on_tick()
            → PositionDelta (if valid)
            → Event bus (POSITION_DELTA, fast lane)
            → TUI receives for immediate display

    Design:
        - Subscribes to MARKET_DATA_TICK on fast lane
        - Processes each tick through RiskFacade
        - Publishes valid deltas as POSITION_DELTA events
        - Logs statistics periodically
    """

    def __init__(
        self,
        risk_facade: RiskFacade,
        event_bus: PriorityEventBus,
        position_store: Optional["PositionStore"] = None,
        market_data_store: Optional["MarketDataStore"] = None,
    ) -> None:
        """
        Initialize DeltaPublisher.

        Args:
            risk_facade: Risk facade for tick processing.
            event_bus: Event bus for subscribing and publishing.
            position_store: Position store for syncing positions.
            market_data_store: Market data store for seeding initial state.
        """
        self._facade = risk_facade
        self._bus = event_bus
        self._position_store = position_store
        self._market_data_store = market_data_store
        self._started = False

        # Statistics
        self._ticks_received = 0
        self._deltas_published = 0
        self._ticks_filtered = 0
        self._positions_synced = 0

    def start(self) -> None:
        """
        Start listening for tick and position events.

        Subscribes to:
        - MARKET_DATA_TICK for streaming P&L updates
        - POSITIONS_READY for initial position synchronization
        - MARKET_DATA_READY for re-sync with market data (critical for daily P&L)
        - POSITION_UPDATED for ongoing position changes (trade deals)
        """
        if self._started:
            logger.warning("DeltaPublisher already started")
            return

        # Subscribe to tick events (fast path)
        self._bus.subscribe(EventType.MARKET_DATA_TICK, self._on_tick)

        # Subscribe to position events for synchronization
        self._bus.subscribe(EventType.POSITIONS_READY, self._on_positions_ready)
        self._bus.subscribe(EventType.POSITION_UPDATED, self._on_position_updated)

        # Subscribe to market data ready - re-sync positions with yesterday_close
        # This is critical because POSITIONS_READY fires before market data is fetched
        self._bus.subscribe(EventType.MARKET_DATA_READY, self._on_market_data_ready)

        self._started = True
        logger.info("DeltaPublisher started (shadow mode)")

    def stop(self) -> None:
        """
        Stop listening for events.

        Unsubscribes from all event types.
        """
        if not self._started:
            return

        self._bus.unsubscribe(EventType.MARKET_DATA_TICK, self._on_tick)
        self._bus.unsubscribe(EventType.POSITIONS_READY, self._on_positions_ready)
        self._bus.unsubscribe(EventType.POSITION_UPDATED, self._on_position_updated)
        self._bus.unsubscribe(EventType.MARKET_DATA_READY, self._on_market_data_ready)
        self._started = False

        logger.info(
            "DeltaPublisher stopped: ticks=%d, deltas=%d, filtered=%d, positions_synced=%d",
            self._ticks_received,
            self._deltas_published,
            self._ticks_filtered,
            self._positions_synced,
        )

    def _on_tick(self, event: MarketDataTickEvent) -> None:
        """
        Handle market data tick event.

        Called by event bus for each MARKET_DATA_TICK.
        Processes tick and publishes delta if valid.

        Args:
            event: Market data tick event.
        """
        self._ticks_received += 1

        # Process tick through facade
        delta = self._facade.on_tick(event)

        if delta is None:
            self._ticks_filtered += 1
            return

        # Convert to event and publish
        delta_event = delta.to_event()
        self._bus.publish(
            EventType.POSITION_DELTA,
            delta_event,
            source="DeltaPublisher",
        )
        self._deltas_published += 1

    def _on_positions_ready(self, payload: Any) -> None:
        """
        Handle POSITIONS_READY event (initial load at startup).

        Pulls positions from PositionStore and syncs to RiskFacade.

        Args:
            payload: Event payload (contains counts, not positions).
        """
        self._sync_positions_from_store("POSITIONS_READY")

    def _on_position_updated(self, payload: Any) -> None:
        """
        Handle POSITION_UPDATED event (trade deals, position changes).

        Resyncs positions from store to catch any changes.

        Args:
            payload: Event payload with position update info.
        """
        self._sync_positions_from_store("POSITION_UPDATED")

    def _on_market_data_ready(self, payload: Any) -> None:
        """
        Handle MARKET_DATA_READY event (market data fetch complete).

        Re-syncs positions to update state with yesterday_close values.
        This is critical because POSITIONS_READY fires before market data
        is fetched, so the initial sync has no reference prices for daily P&L.

        Args:
            payload: Event payload with market data status.
        """
        self._sync_positions_from_store("MARKET_DATA_READY")

    def _sync_positions_from_store(self, source: str) -> None:
        """
        Sync positions from PositionStore to RiskFacade.

        Creates synthetic ticks from MarketDataStore to seed initial state,
        which is critical for extended hours when options don't tick.

        Args:
            source: Event source for logging.
        """
        if self._position_store is None:
            logger.debug("DeltaPublisher: no position store, skipping sync")
            return

        # Pull positions from store
        positions = self._position_store.get_all()
        if not positions:
            return

        # Build synthetic ticks from stored market data
        initial_ticks = self._build_synthetic_ticks(positions)

        # Load positions into RiskFacade with initial state from synthetic ticks
        initialized = self._facade.load_positions(positions, initial_ticks=initial_ticks)
        self._positions_synced += len(positions)

        logger.info(
            "DeltaPublisher synced %d positions from store via %s (%d with initial state, %d synthetic ticks)",
            len(positions),
            source,
            initialized,
            len(initial_ticks),
        )

    def _build_synthetic_ticks(self, positions: List["Position"]) -> Dict[str, MarketDataTickEvent]:
        """
        Build synthetic tick events from stored market data.

        This allows positions to have initial state even during extended hours
        when no live ticks are flowing.

        Args:
            positions: List of positions to build ticks for.

        Returns:
            Dict mapping symbol to synthetic MarketDataTickEvent.
        """
        if self._market_data_store is None:
            return {}

        synthetic_ticks: Dict[str, MarketDataTickEvent] = {}

        for pos in positions:
            md = self._market_data_store.get(pos.symbol)
            if md is None:
                continue

            # If no live prices available, use yesterday_close as fallback for P&L
            # This ensures we can calculate daily P&L even without live data
            mid = md.mid
            last = md.last
            if mid is None and last is None and md.yesterday_close:
                mid = md.yesterday_close
                last = md.yesterday_close

            # Build synthetic tick from stored market data
            # Build kwargs dynamically to handle optional timestamp
            tick_kwargs = {
                "symbol": pos.symbol,
                "bid": md.bid,
                "ask": md.ask,
                "last": last,
                "mid": mid,
                "yesterday_close": md.yesterday_close,
                "underlying_price": md.underlying_price,
                # Greeks from stored data (if available)
                "delta": md.delta,
                "gamma": md.gamma,
                "vega": md.vega,
                "theta": md.theta,
                "iv": md.iv,
                # Mark as stale so is_reliable=False, avoiding false readiness
                "quality": "stale",
            }
            # Use actual timestamp from stored data if available (avoids false freshness)
            if md.timestamp is not None:
                tick_kwargs["timestamp"] = md.timestamp

            tick = MarketDataTickEvent(**tick_kwargs)
            synthetic_ticks[pos.symbol] = tick

        return synthetic_ticks

    @property
    def stats(self) -> dict:
        """Get publisher statistics."""
        return {
            "ticks_received": self._ticks_received,
            "deltas_published": self._deltas_published,
            "ticks_filtered": self._ticks_filtered,
            "positions_synced": self._positions_synced,
            "facade_position_count": self._facade.position_count,
            "filter_rate": (
                self._ticks_filtered / self._ticks_received if self._ticks_received > 0 else 0.0
            ),
        }
