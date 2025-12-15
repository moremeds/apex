"""
Domain events for the Apex trading system.

These typed events replace dict payloads in the event bus and provide:
- Type safety with dataclasses
- Serialization for persistence (JSON/msgpack)
- Clear contracts between components

Usage:
    from src.domain.events.domain_events import QuoteTick, BarData

    # Publish typed event
    event_bus.publish(EventType.MARKET_DATA_TICK, QuoteTick(symbol="AAPL", last=150.0, ...))

    # Serialize for persistence
    tick_dict = quote_tick.to_dict()
    restored = QuoteTick.from_dict(tick_dict)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Literal, Dict, Any, List, Type, TypeVar
from enum import Enum
import json

from ...utils.timezone import now_local


# Type variable for generic from_dict
T = TypeVar('T', bound='DomainEvent')


class Timeframe(Enum):
    """Bar timeframe enumeration."""
    TICK = "tick"
    S1 = "1s"
    S5 = "5s"
    S15 = "15s"
    S30 = "30s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


# =============================================================================
# Base Domain Event
# =============================================================================

@dataclass(frozen=True, slots=True)
class DomainEvent:
    """
    Base class for all domain events.

    All domain events are immutable (frozen) and use slots for memory efficiency.
    """
    timestamp: datetime = field(default_factory=now_local)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize event to dictionary for JSON persistence.

        Handles datetime conversion and nested objects.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            else:
                result[key] = value
        result['_event_type'] = self.__class__.__name__
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Deserialize event from dictionary.

        Args:
            data: Dictionary with event data.

        Returns:
            Reconstructed event instance.
        """
        # Remove metadata
        data = data.copy()
        data.pop('_event_type', None)

        # Convert timestamp back to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        return cls(**data)

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize event from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Market Data Events
# =============================================================================

@dataclass(frozen=True, slots=True)
class QuoteTick(DomainEvent):
    """
    Single price update event.

    Represents a point-in-time quote with bid/ask/last prices.
    Published on EventType.MARKET_DATA_TICK.
    """
    symbol: str = ""
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None

    # Greeks (for options)
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None

    # Underlying price (for options)
    underlying_price: Optional[float] = None

    # Source tracking
    source: str = ""  # "IB", "FUTU", "YAHOO", etc.

    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate spread as percentage of mid."""
        mid = self.mid
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100
        return None


@dataclass(frozen=True, slots=True)
class BarData(DomainEvent):
    """
    OHLCV candle/bar data event.

    Represents aggregated price data over a time period.
    Published on EventType.MARKET_DATA_BATCH (for bar updates).
    """
    symbol: str = ""
    timeframe: str = "1m"  # Timeframe enum value
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None  # Volume-weighted average price
    trade_count: Optional[int] = None  # Number of trades in bar

    # Bar time boundaries
    bar_start: Optional[datetime] = None
    bar_end: Optional[datetime] = None

    # Source tracking
    source: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarData":
        """Deserialize with datetime handling."""
        data = data.copy()
        data.pop('_event_type', None)

        for dt_field in ['timestamp', 'bar_start', 'bar_end']:
            if dt_field in data and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field])

        return cls(**data)

    @property
    def range(self) -> Optional[float]:
        """Calculate high-low range."""
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None

    @property
    def body(self) -> Optional[float]:
        """Calculate candle body (close - open)."""
        if self.close is not None and self.open is not None:
            return self.close - self.open
        return None

    @property
    def is_bullish(self) -> Optional[bool]:
        """Check if candle is bullish (green)."""
        if self.close is not None and self.open is not None:
            return self.close >= self.open
        return None


# =============================================================================
# Trading Events
# =============================================================================

@dataclass(frozen=True, slots=True)
class TradeFill(DomainEvent):
    """
    Execution/fill notification event.

    Represents a completed trade execution.
    Published on EventType.ORDER_FILLED.
    """
    symbol: str = ""
    underlying: str = ""
    side: str = "BUY"  # OrderSide enum value
    quantity: float = 0.0
    price: float = 0.0
    commission: float = 0.0

    # Identifiers
    exec_id: str = ""
    order_id: str = ""
    account_id: str = ""

    # Position effect
    is_opening: bool = True  # True if opening position, False if closing

    # Asset details
    asset_type: str = "STOCK"  # "STOCK", "OPTION", "FUTURE"
    multiplier: int = 1

    # Option details (if applicable)
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None  # "C" or "P"

    # Source tracking
    source: str = ""  # "IB", "FUTU"

    @property
    def notional(self) -> float:
        """Calculate trade notional value."""
        return abs(self.quantity) * self.price * self.multiplier

    @property
    def net_amount(self) -> float:
        """Calculate net amount (including commission)."""
        base = self.quantity * self.price * self.multiplier
        if self.side == "BUY":
            return -base - self.commission
        else:
            return base - self.commission


@dataclass(frozen=True, slots=True)
class OrderUpdate(DomainEvent):
    """
    Order status change event.

    Represents a change in order state.
    Published on EventType.ORDER_SUBMITTED, ORDER_CANCELLED, etc.
    """
    order_id: str = ""
    symbol: str = ""
    underlying: str = ""
    side: str = "BUY"  # OrderSide enum value
    order_type: str = "LIMIT"  # OrderType enum value
    status: str = "PENDING"  # OrderStatus enum value

    # Quantities
    quantity: float = 0.0
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    avg_fill_price: Optional[float] = None

    # Asset details
    asset_type: str = "STOCK"
    multiplier: int = 1

    # Option details
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None

    # Metadata
    account_id: str = ""
    client_order_id: str = ""  # User-defined order ID
    parent_order_id: Optional[str] = None  # For bracket orders

    # Rejection details
    reject_reason: Optional[str] = None

    # Source tracking
    source: str = ""

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in ("PENDING", "SUBMITTED", "PARTIALLY_FILLED")

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in ("FILLED", "CANCELLED", "REJECTED", "EXPIRED")

    @property
    def fill_pct(self) -> float:
        """Calculate fill percentage."""
        if self.quantity > 0:
            return (self.filled_quantity / self.quantity) * 100
        return 0.0


# =============================================================================
# Position Events
# =============================================================================

@dataclass(frozen=True, slots=True)
class PositionSnapshot(DomainEvent):
    """
    Point-in-time position state event.

    Represents a position at a specific moment.
    Published on EventType.POSITION_UPDATED.
    """
    symbol: str = ""
    underlying: str = ""
    asset_type: str = "STOCK"

    # Position details
    quantity: float = 0.0
    avg_price: float = 0.0
    multiplier: int = 1

    # Current valuation
    mark_price: Optional[float] = None
    market_value: Optional[float] = None
    notional: Optional[float] = None

    # P&L
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    daily_pnl: Optional[float] = None

    # Greeks (for options)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None

    # Option details
    expiry: Optional[str] = None
    strike: Optional[float] = None
    right: Optional[str] = None
    days_to_expiry: Optional[int] = None

    # Metadata
    account_id: str = ""
    source: str = ""

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        return abs(self.quantity) * self.avg_price * self.multiplier


# =============================================================================
# Account Events
# =============================================================================

@dataclass(frozen=True, slots=True)
class AccountSnapshot(DomainEvent):
    """
    Account state event.

    Represents account balances and margin at a specific moment.
    Published on EventType.ACCOUNT_UPDATED.
    """
    account_id: str = ""

    # Balances
    net_liquidation: float = 0.0
    total_cash: float = 0.0
    buying_power: float = 0.0

    # Margin
    margin_used: float = 0.0
    margin_available: float = 0.0
    maintenance_margin: float = 0.0
    init_margin_req: float = 0.0
    excess_liquidity: float = 0.0

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Position counts
    position_count: int = 0
    open_order_count: int = 0

    # Source tracking
    source: str = ""

    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization percentage."""
        if self.net_liquidation > 0:
            return (self.margin_used / self.net_liquidation) * 100
        return 0.0

    @property
    def available_pct(self) -> float:
        """Calculate available margin percentage."""
        if self.net_liquidation > 0:
            return (self.margin_available / self.net_liquidation) * 100
        return 0.0


# =============================================================================
# System Events
# =============================================================================

@dataclass(frozen=True, slots=True)
class ConnectionEvent(DomainEvent):
    """
    Broker/adapter connection state change event.

    Published on EventType.BROKER_CONNECTED, BROKER_DISCONNECTED.
    """
    adapter_name: str = ""
    adapter_type: str = ""  # "live", "historical", "execution"
    connected: bool = False
    error_message: Optional[str] = None
    reconnect_attempt: int = 0

    # Connection details
    host: str = ""
    port: int = 0
    client_id: Optional[int] = None


@dataclass(frozen=True, slots=True)
class RiskBreachEvent(DomainEvent):
    """
    Risk limit breach event.

    Published on EventType.RISK_BREACH.
    """
    rule_name: str = ""
    breach_level: str = "SOFT"  # "SOFT" or "HARD"
    current_value: float = 0.0
    limit_value: float = 0.0
    breach_pct: float = 0.0  # How much over limit (percentage)

    # Context
    symbol: Optional[str] = None
    underlying: Optional[str] = None
    metric: str = ""  # "notional", "delta", "gamma", etc.

    message: str = ""


# =============================================================================
# Event Registry
# =============================================================================

# Map event type names to classes for deserialization
EVENT_REGISTRY: Dict[str, Type[DomainEvent]] = {
    "QuoteTick": QuoteTick,
    "BarData": BarData,
    "TradeFill": TradeFill,
    "OrderUpdate": OrderUpdate,
    "PositionSnapshot": PositionSnapshot,
    "AccountSnapshot": AccountSnapshot,
    "ConnectionEvent": ConnectionEvent,
    "RiskBreachEvent": RiskBreachEvent,
}


def deserialize_event(data: Dict[str, Any]) -> DomainEvent:
    """
    Deserialize any domain event from dictionary.

    Uses _event_type field to determine the correct class.

    Args:
        data: Dictionary with event data and _event_type field.

    Returns:
        Reconstructed event instance.

    Raises:
        ValueError: If event type is unknown.
    """
    event_type = data.get('_event_type')
    if not event_type:
        raise ValueError("Missing _event_type field in event data")

    event_class = EVENT_REGISTRY.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.from_dict(data)


def deserialize_events(data_list: List[Dict[str, Any]]) -> List[DomainEvent]:
    """Deserialize a list of domain events."""
    return [deserialize_event(d) for d in data_list]
