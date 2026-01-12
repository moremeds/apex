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

from typing import Any, List, Optional, TYPE_CHECKING

from src.domain.events.event_types import EventType
from src.utils.logging_setup import get_logger

if TYPE_CHECKING:
    from src.domain.events.priority_event_bus import PriorityEventBus
    from src.domain.events.domain_events import MarketDataTickEvent
    from src.infrastructure.stores import PositionStore
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
    ) -> None:
        """
        Initialize DeltaPublisher.

        Args:
            risk_facade: Risk facade for tick processing.
            event_bus: Event bus for subscribing and publishing.
            position_store: Position store for syncing positions.
        """
        self._facade = risk_facade
        self._bus = event_bus
        self._position_store = position_store
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

    def _sync_positions_from_store(self, source: str) -> None:
        """
        Sync positions from PositionStore to RiskFacade.

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

        # Load positions into RiskFacade (state initialized on first tick)
        initialized = self._facade.load_positions(positions)
        self._positions_synced += len(positions)

        logger.info(
            "DeltaPublisher synced %d positions from store via %s (%d with initial state)",
            len(positions),
            source,
            initialized,
        )

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
                self._ticks_filtered / self._ticks_received
                if self._ticks_received > 0
                else 0.0
            ),
        }
