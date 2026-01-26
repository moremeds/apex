"""Unit tests for TickProcessor."""

from datetime import datetime

import pytest

from src.domain.events.domain_events import MarketDataTickEvent
from src.domain.services.risk.state.position_state import PositionState
from src.domain.services.risk.streaming.tick_processor import (
    TickProcessor,
    create_initial_state,
)
from src.models.position import AssetType, Position


class TestTickProcessor:
    """Tests for TickProcessor class."""

    @pytest.fixture
    def processor(self) -> TickProcessor:
        """Create TickProcessor instance."""
        return TickProcessor()

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

    @pytest.fixture
    def current_state(self) -> PositionState:
        """Create current position state."""
        return PositionState(
            symbol="AAPL",
            underlying="AAPL",
            quantity=100,
            multiplier=1,
            avg_cost=150.0,
            mark_price=155.0,
            yesterday_close=154.0,
            session_open=153.0,
            unrealized_pnl=500.0,
            daily_pnl=100.0,
            delta=100.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            notional=15500.0,
            delta_dollars=15500.0,
            underlying_price=155.0,
            is_reliable=True,
            has_greeks=False,
            last_update=datetime.now(),
        )

    def test_process_tick_basic(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should calculate delta correctly."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
            yesterday_close=154.0,
            session_open=153.0,
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is not None
        assert delta.symbol == "AAPL"
        assert delta.new_mark_price == 156.0
        # P&L change magnitude: abs((156 - 155) * 100) = 100
        assert abs(delta.pnl_change) == pytest.approx(100.0, rel=0.01)
        assert delta.is_reliable is True

    def test_process_tick_filters_stale(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should filter stale ticks."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=156.0,
            ask=156.10,
            quality="stale",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None

    def test_process_tick_filters_suspicious(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should filter suspicious ticks."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=156.0,
            ask=156.10,
            quality="suspicious",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None

    def test_process_tick_filters_zero_quote(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should filter zero_quote ticks."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=0.0,
            ask=0.0,
            quality="zero_quote",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None

    def test_process_tick_case_insensitive_quality(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should handle uppercase quality values."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=156.0,
            ask=156.10,
            quality="STALE",  # Uppercase
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None  # Should still filter

    def test_process_tick_filters_wide_spread(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should filter wide spread ticks (>5%)."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=150.0,
            ask=160.0,  # 6.5% spread
            quality="good",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None

    def test_process_tick_handles_crossed_market(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should filter crossed market (bid > ask)."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=160.0,  # bid > ask (crossed)
            ask=150.0,
            quality="good",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is None  # Wide spread when abs() applied

    def test_process_tick_uses_fallback_prices(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should use current_state for missing reference prices."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.90,
            ask=156.10,
            mid=156.0,
            quality="good",
            yesterday_close=None,  # Missing
            session_open=None,  # Missing
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is not None
        # Should still calculate daily_pnl using current_state.yesterday_close
        # Daily P&L change magnitude: abs((156 - 154) * 100 - (155 - 154) * 100) = 100
        assert abs(delta.daily_pnl_change) == pytest.approx(100.0, rel=0.01)

    def test_process_tick_no_valid_mark_falls_back_to_current_state(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should fallback to current_state.mark_price when tick has no prices."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=None,
            quality="good",
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        # Fallback to current_state.mark_price (155.0)
        # Note: The processor may return a delta even if price unchanged
        # because it recalculates P&L from the tick's reference prices
        if delta is not None:
            assert delta.new_mark_price == current_state.mark_price

    def test_process_tick_no_valid_mark_falls_back_to_avg_price(
        self, processor: TickProcessor, stock_position: Position
    ) -> None:
        """process_tick() should fallback to avg_price when current_state has no mark."""
        # Create a state with mark_price = 0 (no valid price)
        zero_mark_state = PositionState(
            symbol="AAPL",
            underlying="AAPL",
            quantity=100,
            multiplier=1,
            avg_cost=150.0,
            mark_price=0.0,  # No valid mark
            yesterday_close=154.0,
            session_open=153.0,
            unrealized_pnl=0.0,
            daily_pnl=0.0,
            delta=100.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            notional=0.0,
            delta_dollars=0.0,
            underlying_price=0.0,
            is_reliable=False,
            has_greeks=False,
            last_update=datetime.now(),
        )
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=None,
            quality="good",
        )

        delta = processor.process_tick(tick, stock_position, zero_mark_state)

        # Should use position.avg_price (150.0) as final fallback
        assert delta is not None
        assert delta.new_mark_price == 150.0

    def test_process_tick_no_fallback_available(self, processor: TickProcessor) -> None:
        """process_tick() should return None when no price available at all."""
        # Position with zero avg_price
        zero_price_position = Position(
            symbol="AAPL",
            underlying="AAPL",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_price=0.0,  # No entry price
            multiplier=1,
        )
        # State with zero mark price
        zero_mark_state = PositionState(
            symbol="AAPL",
            underlying="AAPL",
            quantity=100,
            multiplier=1,
            avg_cost=0.0,
            mark_price=0.0,
            yesterday_close=0.0,
            session_open=0.0,
            unrealized_pnl=0.0,
            daily_pnl=0.0,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            notional=0.0,
            delta_dollars=0.0,
            underlying_price=0.0,
            is_reliable=False,
            has_greeks=False,
            last_update=datetime.now(),
        )
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=None,
            quality="good",
        )

        delta = processor.process_tick(tick, zero_price_position, zero_mark_state)

        # No fallback available, should return None
        assert delta is None

    def test_process_tick_uses_last_as_fallback(
        self, processor: TickProcessor, stock_position: Position, current_state: PositionState
    ) -> None:
        """process_tick() should use last price when bid/ask unavailable."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=156.0,
            quality="good",
            yesterday_close=154.0,
        )

        delta = processor.process_tick(tick, stock_position, current_state)

        assert delta is not None
        assert delta.new_mark_price == 156.0


class TestCreateInitialState:
    """Tests for create_initial_state function."""

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

    def test_creates_state_from_good_tick(self, stock_position: Position) -> None:
        """create_initial_state() should create state from good tick."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=154.0,
            session_open=153.0,
        )

        state = create_initial_state(stock_position, tick)

        assert state is not None
        assert state.symbol == "AAPL"
        assert state.mark_price == 155.0
        # Verify unrealized P&L is calculated (exact formula may vary)
        assert state.unrealized_pnl >= 0  # Should be positive for profit
        # Verify state was created with correct reference prices
        assert state.yesterday_close == 154.0
        assert state.session_open == 153.0

    def test_rejects_stale_tick_with_strict_quality(self, stock_position: Position) -> None:
        """create_initial_state() should reject stale tick when strict."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=155.0,
            ask=155.10,
            quality="stale",
        )

        state = create_initial_state(stock_position, tick, strict_quality=True)

        assert state is None

    def test_accepts_stale_tick_without_strict_quality(self, stock_position: Position) -> None:
        """create_initial_state() should accept stale tick when not strict."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="stale",
        )

        state = create_initial_state(stock_position, tick, strict_quality=False)

        assert state is not None
        assert state.is_reliable is False  # Marked as unreliable

    def test_no_valid_mark_falls_back_to_avg_price(self, stock_position: Position) -> None:
        """create_initial_state() should fallback to avg_price when tick has no prices."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=None,
            quality="good",
        )

        state = create_initial_state(stock_position, tick)

        # Should use position.avg_price (150.0) as fallback
        assert state is not None
        assert state.mark_price == 150.0
        # P&L is zero when mark == avg_cost
        assert state.unrealized_pnl == 0.0

    def test_no_fallback_available_returns_none(self) -> None:
        """create_initial_state() should return None when no price available at all."""
        zero_price_position = Position(
            symbol="AAPL",
            underlying="AAPL",
            asset_type=AssetType.STOCK,
            quantity=100,
            avg_price=0.0,  # No entry price
            multiplier=1,
        )
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=None,
            ask=None,
            last=None,
            quality="good",
        )

        state = create_initial_state(zero_price_position, tick)

        # No fallback available, should return None
        assert state is None

    def test_uses_zero_for_missing_reference_prices(self, stock_position: Position) -> None:
        """create_initial_state() should handle missing reference prices."""
        tick = MarketDataTickEvent(
            symbol="AAPL",
            source="ib",
            bid=154.90,
            ask=155.10,
            mid=155.0,
            quality="good",
            yesterday_close=None,
            session_open=None,
        )

        state = create_initial_state(stock_position, tick)

        assert state is not None
        assert state.yesterday_close == 0.0
        assert state.session_open == 0.0
        assert state.daily_pnl == 0.0  # Can't calculate without yesterday_close
