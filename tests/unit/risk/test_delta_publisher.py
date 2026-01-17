"""Unit tests for DeltaPublisher."""

from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import MarketDataTickEvent
from src.domain.events.event_types import EventType
from src.domain.services.risk.risk_facade import RiskFacade
from src.domain.services.risk.streaming.delta_publisher import DeltaPublisher
from src.models.position import AssetType, Position


class TestDeltaPublisher:
    """Tests for DeltaPublisher class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.subscribe = MagicMock(return_value=None)  # subscribe returns None
        bus.unsubscribe = MagicMock()
        bus.publish = MagicMock()
        return bus

    @pytest.fixture
    def mock_position_store(self):
        """Create mock position store."""
        store = MagicMock()
        store.get_all = MagicMock(return_value=[])
        return store

    @pytest.fixture
    def facade(self) -> RiskFacade:
        """Create RiskFacade instance."""
        return RiskFacade()

    @pytest.fixture
    def publisher(self, facade, mock_event_bus, mock_position_store) -> DeltaPublisher:
        """Create DeltaPublisher instance."""
        return DeltaPublisher(
            risk_facade=facade,
            event_bus=mock_event_bus,
            position_store=mock_position_store,
        )

    @pytest.fixture
    def stock_position(self) -> Position:
        """Create test stock position."""
        return Position(
            symbol="AAPL",
            underlying="AAPL",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_price=150.0,
            multiplier=1,
        )

    def test_start_subscribes_to_events(self, publisher, mock_event_bus):
        """start() should subscribe to tick and position events."""
        publisher.start()

        # Should subscribe to four event types (added MARKET_DATA_READY)
        assert mock_event_bus.subscribe.call_count == 4

        # Check subscription event types
        call_args = [call[0][0] for call in mock_event_bus.subscribe.call_args_list]
        assert EventType.MARKET_DATA_TICK in call_args
        assert EventType.POSITIONS_READY in call_args
        assert EventType.POSITION_UPDATED in call_args
        assert EventType.MARKET_DATA_READY in call_args

    def test_start_twice_warns(self, publisher, mock_event_bus):
        """start() should warn if called twice."""
        publisher.start()
        publisher.start()  # Second call

        # Should only subscribe once (4 event types)
        assert mock_event_bus.subscribe.call_count == 4

    def test_stop_unsubscribes(self, publisher, mock_event_bus):
        """stop() should unsubscribe from events."""
        publisher.start()
        publisher.stop()

        # Should unsubscribe from all four subscriptions
        assert mock_event_bus.unsubscribe.call_count == 4

    def test_stop_without_start(self, publisher, mock_event_bus):
        """stop() should be safe to call without start."""
        publisher.stop()  # Should not raise
        assert mock_event_bus.unsubscribe.call_count == 0

    def test_on_tick_without_position(self, publisher, facade):
        """_on_tick() should filter ticks for unknown symbols."""
        publisher.start()

        tick = MarketDataTickEvent(
            symbol="UNKNOWN",
            source="ib",
            bid=100.0,
            ask=100.10,
            quality="good",
        )

        publisher._on_tick(tick)

        assert publisher.stats["ticks_received"] == 1
        assert publisher.stats["ticks_filtered"] == 1
        assert publisher.stats["deltas_published"] == 0

    def test_on_tick_publishes_delta(self, publisher, facade, mock_event_bus, stock_position):
        """_on_tick() should publish delta for valid tick."""
        # Initialize with position
        initial_tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=154.0,
        )
        facade.load_positions([stock_position], {"AAPL": initial_tick})

        publisher.start()
        mock_event_bus.publish.reset_mock()  # Clear startup publishes

        # Process tick
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
            yesterday_close=154.0,
        )

        publisher._on_tick(tick)

        # Should publish POSITION_DELTA
        mock_event_bus.publish.assert_called_once()
        call_args = mock_event_bus.publish.call_args
        assert call_args[0][0] == EventType.POSITION_DELTA
        assert call_args[1]["source"] == "DeltaPublisher"

        assert publisher.stats["deltas_published"] == 1

    def test_on_positions_ready_syncs_positions(
        self, publisher, facade, mock_position_store, stock_position
    ):
        """_on_positions_ready() should sync positions from store to facade."""
        # Configure store to return position
        mock_position_store.get_all.return_value = [stock_position]
        publisher.start()

        # Trigger positions ready (payload is ignored, positions come from store)
        publisher._on_positions_ready({"count": 1})

        # Position should be registered
        assert facade.has_position("AAPL")
        assert publisher.stats["positions_synced"] == 1

    def test_on_positions_ready_empty_store(self, publisher, facade, mock_position_store):
        """_on_positions_ready() should handle empty store."""
        # Store returns no positions
        mock_position_store.get_all.return_value = []
        publisher.start()

        publisher._on_positions_ready({})

        assert facade.position_count == 0

    def test_on_positions_ready_no_store(self, facade, mock_event_bus):
        """_on_positions_ready() should skip sync when no store provided."""
        # Create publisher without position store
        publisher = DeltaPublisher(
            risk_facade=facade,
            event_bus=mock_event_bus,
            position_store=None,
        )
        publisher.start()

        # Should not raise, just skip sync
        publisher._on_positions_ready({})

        assert facade.position_count == 0

    def test_on_position_updated_syncs_positions(
        self, publisher, facade, mock_position_store, stock_position
    ):
        """_on_position_updated() should resync positions from store."""
        # Configure store to return position
        mock_position_store.get_all.return_value = [stock_position]
        publisher.start()

        # Trigger position update (e.g., from trade deal)
        publisher._on_position_updated({"symbol": "AAPL", "trade": {}})

        # Position should be registered
        assert facade.has_position("AAPL")
        assert publisher.stats["positions_synced"] == 1

    def test_stats_property(self, publisher, facade, stock_position):
        """stats property should return publisher statistics."""
        initial_tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        facade.load_positions([stock_position], {"AAPL": initial_tick})

        publisher.start()

        # Process some ticks
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
        )
        publisher._on_tick(tick)

        stats = publisher.stats
        assert "ticks_received" in stats
        assert "deltas_published" in stats
        assert "ticks_filtered" in stats
        assert "positions_synced" in stats
        assert "facade_position_count" in stats
        assert "filter_rate" in stats

    def test_filter_rate_calculation(self, publisher, facade):
        """Filter rate should be calculated correctly."""
        publisher.start()

        # Process several ticks without positions (all filtered)
        for _ in range(5):
            tick = MarketDataTickEvent(
                symbol="UNKNOWN",
                source="ib",
                bid=100.0,
                ask=100.10,
                quality="good",
            )
            publisher._on_tick(tick)

        assert publisher.stats["filter_rate"] == 1.0  # All filtered

    def test_filter_rate_zero_ticks(self, publisher):
        """Filter rate should handle zero ticks."""
        publisher.start()
        assert publisher.stats["filter_rate"] == 0.0
