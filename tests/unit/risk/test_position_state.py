"""Unit tests for PositionState and PositionDelta."""

import pytest
from datetime import datetime

from src.domain.services.risk.state.position_state import (
    PositionState,
    PositionDelta,
)
from src.domain.events.domain_events import PositionDeltaEvent


class TestPositionDelta:
    """Tests for PositionDelta dataclass."""

    def test_delta_is_frozen(self):
        """PositionDelta should be immutable."""
        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=datetime.now(),
            new_mark_price=150.0,
            pnl_change=10.0,
            daily_pnl_change=5.0,
            delta_change=1.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=100.0,
            delta_dollars_change=150.0,
            underlying_price=150.0,
            is_reliable=True,
            has_greeks=False,
        )

        with pytest.raises(AttributeError):
            delta.pnl_change = 20.0  # type: ignore

    def test_to_event_conversion(self):
        """to_event() should convert to PositionDeltaEvent."""
        timestamp = datetime.now()
        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=timestamp,
            new_mark_price=155.0,
            pnl_change=100.0,
            daily_pnl_change=50.0,
            delta_change=5.0,
            gamma_change=0.1,
            vega_change=0.2,
            theta_change=-0.3,
            notional_change=500.0,
            delta_dollars_change=775.0,
            underlying_price=155.0,
            is_reliable=True,
            has_greeks=True,
        )

        event = delta.to_event()

        assert isinstance(event, PositionDeltaEvent)
        assert event.symbol == "AAPL"
        assert event.underlying == "AAPL"
        assert event.timestamp == timestamp
        assert event.new_mark_price == 155.0
        assert event.pnl_change == 100.0
        assert event.daily_pnl_change == 50.0
        assert event.delta_change == 5.0
        assert event.gamma_change == 0.1
        assert event.vega_change == 0.2
        assert event.theta_change == -0.3
        assert event.notional_change == 500.0
        assert event.delta_dollars_change == 775.0
        assert event.underlying_price == 155.0
        assert event.is_reliable is True
        assert event.has_greeks is True

    def test_to_event_preserves_has_greeks_false(self):
        """has_greeks=False should be preserved in event."""
        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=datetime.now(),
            new_mark_price=150.0,
            pnl_change=0.0,
            daily_pnl_change=0.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=0.0,
            delta_dollars_change=0.0,
            underlying_price=150.0,
            is_reliable=True,
            has_greeks=False,
        )

        event = delta.to_event()
        assert event.has_greeks is False


class TestPositionState:
    """Tests for PositionState dataclass."""

    def test_state_is_frozen(self):
        """PositionState should be immutable."""
        state = PositionState(
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

        with pytest.raises(AttributeError):
            state.unrealized_pnl = 1000.0  # type: ignore

    def test_with_update_creates_new_state(self):
        """with_update() should return a new PositionState."""
        timestamp = datetime.now()
        old_state = PositionState(
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
            last_update=timestamp,
        )

        new_timestamp = datetime.now()
        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=new_timestamp,
            new_mark_price=156.0,
            pnl_change=100.0,  # P&L increased
            daily_pnl_change=50.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=100.0,
            delta_dollars_change=100.0,
            underlying_price=156.0,
            is_reliable=True,
            has_greeks=True,  # Greeks became available
        )

        new_state = old_state.with_update(delta)

        # New state should have updated values
        assert new_state.mark_price == 156.0
        assert new_state.unrealized_pnl == 600.0  # 500 + 100
        assert new_state.daily_pnl == 150.0  # 100 + 50
        assert new_state.notional == 15600.0  # 15500 + 100
        assert new_state.has_greeks is True  # Updated
        assert new_state.last_update == new_timestamp

        # Old state should be unchanged
        assert old_state.mark_price == 155.0
        assert old_state.unrealized_pnl == 500.0
        assert old_state.has_greeks is False

    def test_with_update_preserves_static_fields(self):
        """with_update() should preserve quantity, avg_cost, etc."""
        old_state = PositionState(
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

        delta = PositionDelta(
            symbol="AAPL",
            underlying="AAPL",
            timestamp=datetime.now(),
            new_mark_price=156.0,
            pnl_change=100.0,
            daily_pnl_change=50.0,
            delta_change=0.0,
            gamma_change=0.0,
            vega_change=0.0,
            theta_change=0.0,
            notional_change=100.0,
            delta_dollars_change=100.0,
            underlying_price=156.0,
            is_reliable=True,
            has_greeks=False,
        )

        new_state = old_state.with_update(delta)

        # Static fields should be preserved
        assert new_state.symbol == "AAPL"
        assert new_state.underlying == "AAPL"
        assert new_state.quantity == 100
        assert new_state.multiplier == 1
        assert new_state.avg_cost == 150.0
        assert new_state.yesterday_close == 154.0
        assert new_state.session_open == 153.0

    def test_to_dict_serialization(self):
        """to_dict() should serialize all fields."""
        timestamp = datetime.now()
        state = PositionState(
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
            gamma=0.5,
            vega=1.0,
            theta=-0.5,
            notional=15500.0,
            delta_dollars=15500.0,
            underlying_price=155.0,
            is_reliable=True,
            has_greeks=True,
            last_update=timestamp,
        )

        d = state.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["quantity"] == 100
        assert d["unrealized_pnl"] == 500.0
        assert d["delta_dollars"] == 15500.0
        assert d["underlying_price"] == 155.0
        assert d["has_greeks"] is True
        assert d["last_update"] == timestamp.isoformat()
