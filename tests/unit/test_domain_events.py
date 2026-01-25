"""Unit tests for domain events."""

import json

import pytest

from src.domain.events.domain_events import (
    EVENT_REGISTRY,
    AccountSnapshot,
    BarData,
    ConnectionEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdate,
    PositionSnapshot,
    QuoteTick,
    RiskBreachEvent,
    Timeframe,
    TradeFill,
    deserialize_event,
    deserialize_events,
)
from src.domain.events.event_types import (
    EventType,
    get_event_type_mapping,
    validate_event_payload,
)


class TestQuoteTick:
    """Tests for QuoteTick domain event."""

    def test_create_quote_tick(self) -> None:
        """Test creating a QuoteTick."""
        tick = QuoteTick(
            symbol="AAPL",
            bid=150.0,
            ask=150.10,
            last=150.05,
            bid_size=100,
            ask_size=200,
            source="IB",
        )
        assert tick.symbol == "AAPL"
        assert tick.bid == 150.0
        assert tick.ask == 150.10
        assert tick.last == 150.05
        assert tick.source == "IB"

    def test_quote_tick_mid_price(self) -> None:
        """Test mid price calculation."""
        tick = QuoteTick(symbol="AAPL", bid=150.0, ask=150.10)
        assert tick.mid == 150.05

    def test_quote_tick_mid_fallback_to_last(self) -> None:
        """Test mid price fallback to last when bid/ask missing."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        assert tick.mid == 150.0

    def test_quote_tick_spread(self) -> None:
        """Test spread calculation."""
        tick = QuoteTick(symbol="AAPL", bid=150.0, ask=150.10)
        assert tick.spread == pytest.approx(0.10)

    def test_quote_tick_spread_pct(self) -> None:
        """Test spread percentage calculation."""
        tick = QuoteTick(symbol="AAPL", bid=100.0, ask=101.0)
        assert tick.spread_pct == pytest.approx(0.99, rel=0.01)

    def test_quote_tick_immutable(self) -> None:
        """Test that QuoteTick is immutable."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        with pytest.raises(AttributeError):
            tick.symbol = "MSFT"

    def test_quote_tick_to_dict(self) -> None:
        """Test serialization to dict."""
        tick = QuoteTick(symbol="AAPL", last=150.0, source="IB")
        d = tick.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["last"] == 150.0
        assert d["source"] == "IB"
        assert d["_event_type"] == "QuoteTick"
        assert "timestamp" in d

    def test_quote_tick_from_dict(self) -> None:
        """Test deserialization from dict."""
        tick = QuoteTick(symbol="AAPL", last=150.0, source="IB")
        d = tick.to_dict()
        restored = QuoteTick.from_dict(d)
        assert restored.symbol == tick.symbol
        assert restored.last == tick.last
        assert restored.source == tick.source

    def test_quote_tick_to_json(self) -> None:
        """Test JSON serialization."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        json_str = tick.to_json()
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"
        assert parsed["last"] == 150.0

    def test_quote_tick_from_json(self) -> None:
        """Test JSON deserialization."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        json_str = tick.to_json()
        restored = QuoteTick.from_json(json_str)
        assert restored.symbol == tick.symbol
        assert restored.last == tick.last


class TestBarData:
    """Tests for BarData domain event."""

    def test_create_bar_data(self) -> None:
        """Test creating BarData."""
        bar = BarData(
            symbol="AAPL",
            timeframe="1m",
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=10000,
            source="IB",
        )
        assert bar.symbol == "AAPL"
        assert bar.timeframe == "1m"
        assert bar.close == 150.5

    def test_bar_data_range(self) -> None:
        """Test range calculation."""
        bar = BarData(symbol="AAPL", high=151.0, low=149.0)
        assert bar.range == 2.0

    def test_bar_data_body(self) -> None:
        """Test body calculation."""
        bar = BarData(symbol="AAPL", open=150.0, close=151.5)
        assert bar.body == 1.5

    def test_bar_data_is_bullish(self) -> None:
        """Test bullish candle detection."""
        bullish = BarData(symbol="AAPL", open=150.0, close=151.0)
        bearish = BarData(symbol="AAPL", open=151.0, close=150.0)
        assert bullish.is_bullish is True
        assert bearish.is_bullish is False

    def test_bar_data_serialization(self) -> None:
        """Test BarData serialization round trip."""
        bar = BarData(
            symbol="AAPL",
            timeframe="5m",
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
        )
        d = bar.to_dict()
        restored = BarData.from_dict(d)
        assert restored.symbol == bar.symbol
        assert restored.timeframe == bar.timeframe
        assert restored.close == bar.close


class TestTradeFill:
    """Tests for TradeFill domain event."""

    def test_create_trade_fill(self) -> None:
        """Test creating TradeFill."""
        fill = TradeFill(
            symbol="AAPL",
            underlying="AAPL",
            side="BUY",
            quantity=100.0,
            price=150.0,
            commission=1.0,
            exec_id="exec123",
            order_id="order456",
            source="IB",
        )
        assert fill.symbol == "AAPL"
        assert fill.side == "BUY"
        assert fill.quantity == 100.0
        assert fill.price == 150.0

    def test_trade_fill_notional(self) -> None:
        """Test notional calculation."""
        fill = TradeFill(
            symbol="AAPL",
            quantity=100.0,
            price=150.0,
            multiplier=1,
        )
        assert fill.notional == 15000.0

    def test_trade_fill_notional_options(self) -> None:
        """Test notional calculation for options."""
        fill = TradeFill(
            symbol="AAPL 240315C180",
            quantity=10.0,
            price=5.0,
            multiplier=100,
            asset_type="OPTION",
        )
        assert fill.notional == 5000.0

    def test_trade_fill_net_amount_buy(self) -> None:
        """Test net amount for buy order."""
        fill = TradeFill(
            symbol="AAPL",
            side="BUY",
            quantity=100.0,
            price=150.0,
            commission=1.0,
            multiplier=1,
        )
        # Buy: -quantity * price - commission
        assert fill.net_amount == -15001.0

    def test_trade_fill_net_amount_sell(self) -> None:
        """Test net amount for sell order."""
        fill = TradeFill(
            symbol="AAPL",
            side="SELL",
            quantity=100.0,
            price=150.0,
            commission=1.0,
            multiplier=1,
        )
        # Sell: quantity * price - commission
        assert fill.net_amount == 14999.0


class TestOrderUpdate:
    """Tests for OrderUpdate domain event."""

    def test_create_order_update(self) -> None:
        """Test creating OrderUpdate."""
        order = OrderUpdate(
            order_id="order123",
            symbol="AAPL",
            side="BUY",
            order_type="LIMIT",
            status="SUBMITTED",
            quantity=100.0,
            limit_price=150.0,
        )
        assert order.order_id == "order123"
        assert order.status == "SUBMITTED"

    def test_order_is_active(self) -> None:
        """Test active order detection."""
        submitted = OrderUpdate(order_id="1", status="SUBMITTED")
        partial = OrderUpdate(order_id="2", status="PARTIALLY_FILLED")
        filled = OrderUpdate(order_id="3", status="FILLED")
        cancelled = OrderUpdate(order_id="4", status="CANCELLED")

        assert submitted.is_active is True
        assert partial.is_active is True
        assert filled.is_active is False
        assert cancelled.is_active is False

    def test_order_is_terminal(self) -> None:
        """Test terminal state detection."""
        submitted = OrderUpdate(order_id="1", status="SUBMITTED")
        filled = OrderUpdate(order_id="2", status="FILLED")
        rejected = OrderUpdate(order_id="3", status="REJECTED")

        assert submitted.is_terminal is False
        assert filled.is_terminal is True
        assert rejected.is_terminal is True

    def test_order_fill_pct(self) -> None:
        """Test fill percentage calculation."""
        order = OrderUpdate(
            order_id="1",
            quantity=100.0,
            filled_quantity=75.0,
        )
        assert order.fill_pct == 75.0


class TestPositionSnapshot:
    """Tests for PositionSnapshot domain event."""

    def test_create_position_snapshot(self) -> None:
        """Test creating PositionSnapshot."""
        pos = PositionSnapshot(
            symbol="AAPL",
            underlying="AAPL",
            quantity=100.0,
            avg_price=150.0,
            mark_price=155.0,
            unrealized_pnl=500.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100.0

    def test_position_is_long_short(self) -> None:
        """Test long/short detection."""
        long_pos = PositionSnapshot(symbol="AAPL", quantity=100.0)
        short_pos = PositionSnapshot(symbol="AAPL", quantity=-100.0)

        assert long_pos.is_long is True
        assert long_pos.is_short is False
        assert short_pos.is_long is False
        assert short_pos.is_short is True

    def test_position_cost_basis(self) -> None:
        """Test cost basis calculation."""
        pos = PositionSnapshot(
            symbol="AAPL",
            quantity=100.0,
            avg_price=150.0,
            multiplier=1,
        )
        assert pos.cost_basis == 15000.0


class TestAccountSnapshot:
    """Tests for AccountSnapshot domain event."""

    def test_create_account_snapshot(self) -> None:
        """Test creating AccountSnapshot."""
        acct = AccountSnapshot(
            account_id="U12345",
            net_liquidation=1000000.0,
            total_cash=500000.0,
            buying_power=2000000.0,
            margin_used=100000.0,
            margin_available=900000.0,
        )
        assert acct.account_id == "U12345"
        assert acct.net_liquidation == 1000000.0

    def test_account_margin_utilization(self) -> None:
        """Test margin utilization calculation."""
        acct = AccountSnapshot(
            net_liquidation=1000000.0,
            margin_used=200000.0,
        )
        assert acct.margin_utilization == 20.0


class TestConnectionEvent:
    """Tests for ConnectionEvent domain event."""

    def test_create_connection_event(self) -> None:
        """Test creating ConnectionEvent."""
        event = ConnectionEvent(
            adapter_name="ib_live",
            adapter_type="live",
            connected=True,
            host="127.0.0.1",
            port=7497,
            client_id=1,
        )
        assert event.adapter_name == "ib_live"
        assert event.connected is True


class TestRiskBreachEvent:
    """Tests for RiskBreachEvent domain event."""

    def test_create_risk_breach_event(self) -> None:
        """Test creating RiskBreachEvent."""
        event = RiskBreachEvent(
            rule_name="max_delta",
            breach_level="HARD",
            current_value=60000.0,
            limit_value=50000.0,
            breach_pct=20.0,
            metric="delta",
            message="Portfolio delta exceeded hard limit",
        )
        assert event.rule_name == "max_delta"
        assert event.breach_level == "HARD"


class TestEventRegistry:
    """Tests for event registry and deserialization."""

    def test_event_registry_contains_all_events(self) -> None:
        """Test that registry contains all domain events."""
        assert "QuoteTick" in EVENT_REGISTRY
        assert "BarData" in EVENT_REGISTRY
        assert "TradeFill" in EVENT_REGISTRY
        assert "OrderUpdate" in EVENT_REGISTRY
        assert "PositionSnapshot" in EVENT_REGISTRY
        assert "AccountSnapshot" in EVENT_REGISTRY
        assert "ConnectionEvent" in EVENT_REGISTRY
        assert "RiskBreachEvent" in EVENT_REGISTRY

    def test_deserialize_event(self) -> None:
        """Test generic event deserialization."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        d = tick.to_dict()
        restored = deserialize_event(d)
        assert isinstance(restored, QuoteTick)
        assert restored.symbol == "AAPL"

    def test_deserialize_events_list(self) -> None:
        """Test deserializing list of events."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        bar = BarData(symbol="AAPL", close=151.0)
        events = deserialize_events([tick.to_dict(), bar.to_dict()])
        assert len(events) == 2
        assert isinstance(events[0], QuoteTick)
        assert isinstance(events[1], BarData)

    def test_deserialize_unknown_event_raises(self) -> None:
        """Test that unknown event type raises ValueError."""
        data = {"_event_type": "UnknownEvent", "foo": "bar"}
        with pytest.raises(ValueError, match="Unknown event type"):
            deserialize_event(data)

    def test_deserialize_missing_type_raises(self) -> None:
        """Test that missing _event_type raises ValueError."""
        data = {"symbol": "AAPL"}
        with pytest.raises(ValueError, match="Missing _event_type"):
            deserialize_event(data)


class TestEventTypeMapping:
    """Tests for EventType to DomainEvent mapping."""

    def test_get_event_type_mapping(self) -> None:
        """Test that mapping returns expected types."""
        from src.domain.events.domain_events import MarketDataTickEvent

        mapping = get_event_type_mapping()
        # Note: MARKET_DATA_TICK now uses MarketDataTickEvent, not QuoteTick
        assert mapping[EventType.MARKET_DATA_TICK] == MarketDataTickEvent
        assert mapping[EventType.MARKET_DATA_BATCH] == BarData
        assert mapping[EventType.ORDER_FILLED] == TradeFill
        assert mapping[EventType.POSITION_UPDATED] == PositionSnapshot
        assert mapping[EventType.ACCOUNT_UPDATED] == AccountSnapshot

    def test_validate_event_payload_valid(self) -> None:
        """Test validation passes for correct payload type."""
        from src.domain.events.domain_events import MarketDataTickEvent

        # Use MarketDataTickEvent instead of QuoteTick for MARKET_DATA_TICK
        tick = MarketDataTickEvent(symbol="AAPL", bid=150.0, ask=150.10, quality="good")
        assert validate_event_payload(EventType.MARKET_DATA_TICK, tick) is True

    def test_validate_event_payload_invalid(self) -> None:
        """Test validation fails for incorrect payload type."""
        tick = QuoteTick(symbol="AAPL", last=150.0)
        # Using tick for ACCOUNT_UPDATED (expects AccountSnapshot)
        assert validate_event_payload(EventType.ACCOUNT_UPDATED, tick) is False

    def test_validate_event_payload_dict_backward_compat(self) -> None:
        """Test that dicts fail validation for mapped types."""
        payload = {"symbol": "AAPL", "last": 150.0}
        assert validate_event_payload(EventType.MARKET_DATA_TICK, payload) is False

    def test_validate_event_payload_unmapped_type(self) -> None:
        """Test that unmapped types allow any payload."""
        # TIMER_TICK is not mapped
        payload = {"tick": 1}
        assert validate_event_payload(EventType.TIMER_TICK, payload) is True


class TestEnums:
    """Tests for domain event enums."""

    def test_timeframe_enum(self) -> None:
        """Test Timeframe enum values."""
        assert Timeframe.M1.value == "1m"
        assert Timeframe.H1.value == "1h"
        assert Timeframe.D1.value == "1d"

    def test_order_side_enum(self) -> None:
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_status_enum(self) -> None:
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"

    def test_order_type_enum(self) -> None:
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
