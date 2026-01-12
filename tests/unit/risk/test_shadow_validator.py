"""Unit tests for ShadowValidator."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.domain.services.risk.streaming.shadow_validator import ShadowValidator
from src.domain.services.risk.risk_facade import RiskFacade
from src.domain.events.domain_events import MarketDataTickEvent
from src.domain.events.event_types import EventType
from src.models.risk_snapshot import RiskSnapshot
from src.models.position import Position, AssetType


class TestShadowValidator:
    """Tests for ShadowValidator class."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = MagicMock()
        bus.subscribe = MagicMock(return_value=None)  # subscribe returns None
        bus.unsubscribe = MagicMock()
        return bus

    @pytest.fixture
    def facade(self) -> RiskFacade:
        """Create RiskFacade instance."""
        return RiskFacade()

    @pytest.fixture
    def validator(self, facade, mock_event_bus) -> ShadowValidator:
        """Create ShadowValidator instance."""
        return ShadowValidator(
            risk_facade=facade,
            event_bus=mock_event_bus,
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

    def test_start_subscribes_to_snapshot_ready(self, validator, mock_event_bus):
        """start() should subscribe to SNAPSHOT_READY."""
        validator.start()

        mock_event_bus.subscribe.assert_called_once()
        call_args = mock_event_bus.subscribe.call_args
        assert call_args[0][0] == EventType.SNAPSHOT_READY

    def test_start_twice_is_idempotent(self, validator, mock_event_bus):
        """start() should be idempotent."""
        validator.start()
        validator.start()

        assert mock_event_bus.subscribe.call_count == 1

    def test_stop_unsubscribes(self, validator, mock_event_bus):
        """stop() should unsubscribe from events."""
        validator.start()
        validator.stop()

        mock_event_bus.unsubscribe.assert_called_once()

    def test_stop_without_start(self, validator, mock_event_bus):
        """stop() should be safe without start."""
        validator.stop()
        mock_event_bus.unsubscribe.assert_not_called()

    def test_on_snapshot_counts_match(self, validator, facade, stock_position):
        """_on_snapshot() should count as match when values agree."""
        # Setup facade with position
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=154.0,
        )
        facade.load_positions([stock_position], {"AAPL": tick})

        # Create matching snapshot
        streaming_snapshot = facade.get_snapshot()
        batch_snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_unrealized_pnl=streaming_snapshot.total_unrealized_pnl,
            total_daily_pnl=streaming_snapshot.total_daily_pnl,
            portfolio_delta=streaming_snapshot.portfolio_delta,
            total_positions=streaming_snapshot.total_positions,
        )

        validator._on_snapshot(batch_snapshot)

        assert validator.stats["comparisons"] == 1
        assert validator.stats["matches"] == 1
        assert validator.stats["mismatches"] == 0

    def test_on_snapshot_counts_mismatch(self, validator, facade, stock_position):
        """_on_snapshot() should count as mismatch when values differ."""
        # Setup facade with position
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=154.0,
        )
        facade.load_positions([stock_position], {"AAPL": tick})

        # Create mismatching snapshot (different P&L)
        batch_snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_unrealized_pnl=999999.0,  # Very different
            total_daily_pnl=0.0,
            portfolio_delta=100.0,
            total_positions=1,
        )

        validator._on_snapshot(batch_snapshot)

        assert validator.stats["comparisons"] == 1
        assert validator.stats["matches"] == 0
        assert validator.stats["mismatches"] == 1

    def test_values_match_within_absolute_tolerance(self, validator):
        """_values_match() should accept values within absolute tolerance."""
        # Difference of 0.005 < 0.01 tolerance
        assert validator._values_match(100.0, 100.005)
        assert validator._values_match(100.005, 100.0)

    def test_values_match_within_percentage_tolerance(self, validator):
        """_values_match() should accept values within percentage tolerance."""
        # 0.05% difference on 10000 = 5, which is > 0.01 absolute
        # but < 0.1% percentage tolerance
        assert validator._values_match(10000.0, 10005.0)

    def test_values_mismatch_exceeds_tolerance(self, validator):
        """_values_match() should reject values exceeding tolerance."""
        # Large difference
        assert not validator._values_match(100.0, 200.0)

    def test_values_match_zero_values(self, validator):
        """_values_match() should handle zero values."""
        assert validator._values_match(0.0, 0.0)
        assert validator._values_match(0.0, 0.005)

    def test_stats_property(self, validator):
        """stats property should return validation statistics."""
        stats = validator.stats

        assert "comparisons" in stats
        assert "matches" in stats
        assert "mismatches" in stats
        assert "match_rate" in stats

    def test_match_rate_calculation(self, validator, facade, stock_position):
        """Match rate should be calculated correctly."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        facade.load_positions([stock_position], {"AAPL": tick})

        streaming_snapshot = facade.get_snapshot()

        # One match
        match_snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_unrealized_pnl=streaming_snapshot.total_unrealized_pnl,
            total_daily_pnl=streaming_snapshot.total_daily_pnl,
            portfolio_delta=streaming_snapshot.portfolio_delta,
            total_positions=1,
        )
        validator._on_snapshot(match_snapshot)

        # One mismatch
        mismatch_snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_unrealized_pnl=999999.0,
            total_daily_pnl=0.0,
            portfolio_delta=100.0,
            total_positions=1,
        )
        validator._on_snapshot(mismatch_snapshot)

        assert validator.stats["match_rate"] == 0.5  # 1 match / 2 comparisons

    def test_match_rate_zero_comparisons(self, validator):
        """Match rate should handle zero comparisons."""
        assert validator.stats["match_rate"] == 0.0

    def test_position_count_mismatch(self, validator, facade, stock_position):
        """_on_snapshot() should detect position count mismatch."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        facade.load_positions([stock_position], {"AAPL": tick})

        # Snapshot with different position count
        batch_snapshot = RiskSnapshot(
            timestamp=datetime.now(),
            total_unrealized_pnl=500.0,
            total_daily_pnl=100.0,
            portfolio_delta=100.0,
            total_positions=99,  # Wrong count
        )

        validator._on_snapshot(batch_snapshot)

        assert validator.stats["mismatches"] == 1
