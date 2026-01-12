"""Unit tests for RiskFacade."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.domain.services.risk.risk_facade import RiskFacade
from src.domain.services.risk.state.portfolio_state import PortfolioState
from src.domain.services.risk.state.position_state import PositionState
from src.domain.events.domain_events import MarketDataTickEvent
from src.models.position import Position, AssetType


class TestRiskFacade:
    """Tests for RiskFacade class."""

    @pytest.fixture
    def facade(self) -> RiskFacade:
        """Create RiskFacade instance."""
        return RiskFacade()

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

    @pytest.fixture
    def option_position(self) -> Position:
        """Create test option position."""
        return Position(
            symbol="AAPL  240119C00150000",
            underlying="AAPL",
            asset_type=AssetType.OPTION,
            quantity=10,
            avg_price=5.0,
            multiplier=100,
            expiry="20240119",
            strike=150.0,
            right="C",
        )

    def test_initial_state(self, facade: RiskFacade):
        """RiskFacade should start with no positions."""
        assert facade.position_count == 0
        assert facade.symbols == []
        assert facade.has_position("AAPL") is False

    def test_load_positions(self, facade: RiskFacade, stock_position: Position):
        """load_positions() should register positions."""
        facade.load_positions([stock_position])

        assert facade.has_position("AAPL")
        assert "AAPL" in facade.symbols
        # Position count is 0 because no initial tick provided
        assert facade.position_count == 0

    def test_load_positions_with_ticks(
        self, facade: RiskFacade, stock_position: Position
    ):
        """load_positions() with ticks should create initial state."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=154.0,
        )

        initialized = facade.load_positions(
            [stock_position],
            initial_ticks={"AAPL": tick},
        )

        assert initialized == 1
        assert facade.position_count == 1

        state = facade.get_position_state("AAPL")
        assert state is not None
        assert state.mark_price == 155.0

    def test_on_tick_without_position(self, facade: RiskFacade):
        """on_tick() should return None for unknown symbol."""
        tick = MarketDataTickEvent(
            symbol="UNKNOWN",
            source="ib",
            bid=100.0,
            ask=100.10,
            quality="good",
        )

        delta = facade.on_tick(tick)
        assert delta is None

    def test_on_tick_without_state(
        self, facade: RiskFacade, stock_position: Position
    ):
        """on_tick() should return None if position has no state yet."""
        facade.load_positions([stock_position])  # No initial tick

        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            quality="good",
        )

        delta = facade.on_tick(tick)
        assert delta is None  # No state to update

    def test_on_tick_produces_delta(
        self, facade: RiskFacade, stock_position: Position
    ):
        """on_tick() should produce delta for valid tick."""
        # Initialize with tick
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

        # Process new tick
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
            yesterday_close=154.0,
        )

        delta = facade.on_tick(tick)

        assert delta is not None
        assert delta.symbol == "AAPL"
        assert delta.new_mark_price == 156.0
        # P&L change: (156 - 155) * 100 = 100
        assert delta.pnl_change == pytest.approx(100.0, rel=0.01)

    def test_on_tick_updates_state(
        self, facade: RiskFacade, stock_position: Position
    ):
        """on_tick() should update portfolio state."""
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

        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
            yesterday_close=154.0,
        )

        facade.on_tick(tick)

        state = facade.get_position_state("AAPL")
        assert state is not None
        assert state.mark_price == 156.0

    def test_on_tick_filters_bad_quality(
        self, facade: RiskFacade, stock_position: Position
    ):
        """on_tick() should filter bad quality ticks."""
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

        stale_tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=156.0,
            ask=156.10,
            quality="stale",
        )

        delta = facade.on_tick(stale_tick)
        assert delta is None

    def test_get_snapshot(self, facade: RiskFacade, stock_position: Position):
        """get_snapshot() should return portfolio metrics."""
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

        snapshot = facade.get_snapshot()

        assert snapshot is not None
        assert snapshot.total_positions == 1
        # Unrealized P&L: (155 - 150) * 100 = 500
        assert snapshot.total_unrealized_pnl == pytest.approx(500.0, rel=0.01)

    def test_add_position(self, facade: RiskFacade, stock_position: Position):
        """add_position() should add a single position."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )

        result = facade.add_position(stock_position, tick)

        assert result is True
        assert facade.has_position("AAPL")
        assert facade.position_count == 1

    def test_remove_position(self, facade: RiskFacade, stock_position: Position):
        """remove_position() should remove a position."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        facade.add_position(stock_position, tick)

        removed = facade.remove_position("AAPL")

        assert removed is not None
        assert removed.symbol == "AAPL"
        assert facade.has_position("AAPL") is False

    def test_clear(self, facade: RiskFacade, stock_position: Position):
        """clear() should remove all positions."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        facade.add_position(stock_position, tick)

        facade.clear()

        assert facade.position_count == 0
        assert facade.symbols == []

    def test_multiple_positions(
        self,
        facade: RiskFacade,
        stock_position: Position,
        option_position: Position,
    ):
        """RiskFacade should handle multiple positions."""
        stock_tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
        )
        option_tick = MarketDataTickEvent(
            symbol="AAPL  240119C00150000",
            source="ib",
            bid=5.40,
            ask=5.60,
            mid=5.50,
            quality="good",
        )

        facade.load_positions(
            [stock_position, option_position],
            {
                "AAPL": stock_tick,
                "AAPL  240119C00150000": option_tick,
            },
        )

        assert facade.position_count == 2
        assert len(facade.symbols) == 2
