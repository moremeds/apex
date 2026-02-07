"""
Shared pytest fixtures for signal pipeline unit tests.

Provides mock event bus, sample OHLCV data, and domain event factories
for testing IndicatorEngine, RuleEngine, ConfluenceCalculator, and SignalCoordinator.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from src.domain.events.domain_events import (
    BarCloseEvent,
    IndicatorUpdateEvent,
)
from src.domain.events.event_types import EventType
from src.domain.signals.models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
    TradingSignal,
)

# =============================================================================
# Mock Event Bus
# =============================================================================


class MockEventBus:
    """
    Mock event bus for testing signal pipeline components.

    Tracks published events and subscribed callbacks for verification.
    Thread-safe for basic test scenarios.
    """

    def __init__(self) -> None:
        self.events: Dict[EventType, List[Any]] = defaultdict(list)
        self.subscriptions: Dict[EventType, List[Callable[[Any], None]]] = defaultdict(list)
        self.publish_count = 0
        self.subscribe_count = 0

    def publish(self, event_type: EventType, payload: Any) -> None:
        """Publish an event and invoke all subscribers."""
        self.events[event_type].append(payload)
        self.publish_count += 1

        # Invoke subscribers (synchronously for testing)
        for callback in self.subscriptions.get(event_type, []):
            try:
                callback(payload)
            except Exception:
                pass  # Ignore errors in callbacks for testing

    def subscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        self.subscriptions[event_type].append(callback)
        self.subscribe_count += 1

    def unsubscribe(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """Unsubscribe from an event type."""
        if callback in self.subscriptions.get(event_type, []):
            self.subscriptions[event_type].remove(callback)

    def get_events(self, event_type: EventType) -> List[Any]:
        """Get all published events of a type."""
        return list(self.events.get(event_type, []))

    def get_last_event(self, event_type: EventType) -> Optional[Any]:
        """Get the most recent event of a type."""
        events = self.events.get(event_type, [])
        return events[-1] if events else None

    def clear(self) -> None:
        """Clear all events and subscriptions."""
        self.events.clear()
        self.subscriptions.clear()
        self.publish_count = 0
        self.subscribe_count = 0


# =============================================================================
# Event Bus Fixture
# =============================================================================


@pytest.fixture
def mock_event_bus() -> MockEventBus:
    """Create a fresh mock event bus for each test."""
    return MockEventBus()


# =============================================================================
# Sample OHLCV Data Generation
# =============================================================================


def generate_ohlcv_data(
    n_bars: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
    start_time: Optional[datetime] = None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for testing.

    Args:
        n_bars: Number of bars to generate
        start_price: Starting price
        volatility: Price volatility (daily returns std)
        seed: Random seed for reproducibility
        start_time: Starting timestamp
        timeframe: Timeframe for bar spacing

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    np.random.seed(seed)

    if start_time is None:
        start_time = datetime(2024, 1, 1, 9, 30, 0, tzinfo=timezone.utc)

    # Generate returns and prices
    returns = np.random.normal(0, volatility, n_bars)
    prices = start_price * np.exp(np.cumsum(returns))

    # Generate OHLC from prices
    opens = np.roll(prices, 1)
    opens[0] = start_price
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, volatility / 2, n_bars))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, volatility / 2, n_bars))
    closes = prices

    # Generate volume
    avg_volume = 1_000_000
    volumes = np.random.lognormal(np.log(avg_volume), 0.5, n_bars).astype(int)

    # Generate timestamps based on timeframe
    timeframe_deltas = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    delta = timeframe_deltas.get(timeframe, timedelta(days=1))
    timestamps = [start_time + delta * i for i in range(n_bars)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate 100 bars of sample OHLCV data."""
    return generate_ohlcv_data(n_bars=100)


@pytest.fixture
def short_ohlcv_data() -> pd.DataFrame:
    """Generate 20 bars of sample OHLCV data (for warmup tests)."""
    return generate_ohlcv_data(n_bars=20)


@pytest.fixture
def long_ohlcv_data() -> pd.DataFrame:
    """Generate 500 bars of sample OHLCV data (for history tests)."""
    return generate_ohlcv_data(n_bars=500)


# =============================================================================
# Bar Event Factories
# =============================================================================


def make_bar_close_event(
    symbol: str = "AAPL",
    timeframe: str = "1d",
    open_price: float = 100.0,
    high_price: float = 102.0,
    low_price: float = 99.0,
    close_price: float = 101.0,
    volume: float = 1_000_000,
    timestamp: Optional[datetime] = None,
) -> BarCloseEvent:
    """Create a BarCloseEvent for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return BarCloseEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        timestamp=timestamp,
        bar_end=timestamp,
    )


@pytest.fixture
def bar_close_event() -> BarCloseEvent:
    """Create a sample BarCloseEvent."""
    return make_bar_close_event()


@pytest.fixture
def bar_close_event_factory() -> Callable[..., BarCloseEvent]:
    """Factory fixture for creating BarCloseEvents."""
    return make_bar_close_event


# =============================================================================
# Indicator Event Factories
# =============================================================================


def make_indicator_update_event(
    symbol: str = "AAPL",
    timeframe: str = "1d",
    indicator: str = "rsi",
    value: float = 50.0,
    state: Optional[Dict[str, Any]] = None,
    previous_value: Optional[float] = None,
    previous_state: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
) -> IndicatorUpdateEvent:
    """Create an IndicatorUpdateEvent for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if state is None:
        state = {"value": value, "zone": "neutral"}

    return IndicatorUpdateEvent(
        symbol=symbol,
        timeframe=timeframe,
        indicator=indicator,
        value=value,
        state=state,
        previous_value=previous_value,
        previous_state=previous_state,
        timestamp=timestamp,
    )


@pytest.fixture
def indicator_update_event() -> IndicatorUpdateEvent:
    """Create a sample IndicatorUpdateEvent."""
    return make_indicator_update_event()


@pytest.fixture
def indicator_update_event_factory() -> Callable[..., IndicatorUpdateEvent]:
    """Factory fixture for creating IndicatorUpdateEvents."""
    return make_indicator_update_event


# =============================================================================
# Signal Rule Factories
# =============================================================================


def make_signal_rule(
    name: str = "test_rule",
    indicator: str = "rsi",
    category: SignalCategory = SignalCategory.MOMENTUM,
    direction: SignalDirection = SignalDirection.BUY,
    strength: int = 70,
    priority: SignalPriority = SignalPriority.MEDIUM,
    condition_type: ConditionType = ConditionType.STATE_CHANGE,
    condition_config: Optional[Dict[str, Any]] = None,
    timeframes: Tuple[str, ...] = ("1d",),
    cooldown_seconds: int = 3600,
    enabled: bool = True,
    message_template: str = "{symbol} {indicator} signal",
) -> SignalRule:
    """Create a SignalRule for testing."""
    if condition_config is None:
        condition_config = {"field": "zone", "from": ["oversold"], "to": ["neutral"]}

    return SignalRule(
        name=name,
        indicator=indicator,
        category=category,
        direction=direction,
        strength=strength,
        priority=priority,
        condition_type=condition_type,
        condition_config=condition_config,
        timeframes=timeframes,
        cooldown_seconds=cooldown_seconds,
        enabled=enabled,
        message_template=message_template,
    )


@pytest.fixture
def sample_signal_rule() -> SignalRule:
    """Create a sample SignalRule."""
    return make_signal_rule()


@pytest.fixture
def signal_rule_factory() -> Callable[..., SignalRule]:
    """Factory fixture for creating SignalRules."""
    return make_signal_rule


# =============================================================================
# Trading Signal Factories
# =============================================================================


def make_trading_signal(
    signal_id: str = "momentum:rsi:AAPL:1d",
    symbol: str = "AAPL",
    category: SignalCategory = SignalCategory.MOMENTUM,
    indicator: str = "rsi",
    direction: SignalDirection = SignalDirection.BUY,
    strength: int = 70,
    priority: SignalPriority = SignalPriority.MEDIUM,
    timeframe: str = "1d",
    trigger_rule: str = "test_rule",
    current_value: float = 30.0,
    threshold: Optional[float] = None,
    timestamp: Optional[datetime] = None,
    message: str = "Test signal",
) -> TradingSignal:
    """Create a TradingSignal for testing."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    return TradingSignal(
        signal_id=signal_id,
        symbol=symbol,
        category=category,
        indicator=indicator,
        direction=direction,
        strength=strength,
        priority=priority,
        timeframe=timeframe,
        trigger_rule=trigger_rule,
        current_value=current_value,
        threshold=threshold,
        timestamp=timestamp,
        message=message,
    )


@pytest.fixture
def trading_signal() -> TradingSignal:
    """Create a sample TradingSignal."""
    return make_trading_signal()


@pytest.fixture
def trading_signal_factory() -> Callable[..., TradingSignal]:
    """Factory fixture for creating TradingSignals."""
    return make_trading_signal


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Provide a base timestamp for tests."""
    return datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def bar_dicts_from_ohlcv(sample_ohlcv_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert OHLCV DataFrame to list of bar dictionaries."""
    return sample_ohlcv_data.to_dict("records")
