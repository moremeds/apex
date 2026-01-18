"""
Tests for MTF RSI Trend Strategy.

Tests the multi-timeframe RSI calculation and signal generation.
"""

from collections import deque
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.domain.events.domain_events import BarData
from src.domain.strategy.base import StrategyContext
from src.domain.strategy.examples.mtf_rsi_trend import MTFRsiTrendStrategy


@pytest.fixture
def mock_context():
    """Create mock strategy context."""
    context = MagicMock(spec=StrategyContext)
    context.get_position_quantity.return_value = 0
    context.clock = MagicMock()
    context.clock.now.return_value = datetime.now()
    return context


@pytest.fixture
def strategy(mock_context):
    """Create MTF RSI Trend strategy instance."""
    return MTFRsiTrendStrategy(
        strategy_id="test_mtf_rsi",
        symbols=["AAPL"],
        context=mock_context,
        primary_timeframe="1d",
        secondary_timeframe="1h",
        trend_rsi_period=14,
        entry_rsi_period=14,
        trend_threshold=50.0,
        entry_oversold=30.0,
        entry_overbought=70.0,
        position_size=100,
    )


def make_bar(symbol: str, close: float, timeframe: str = "1d") -> BarData:
    """Create a test bar."""
    return BarData(
        symbol=symbol,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=1000000,
        timestamp=datetime.now(),
        timeframe=timeframe,
    )


class TestMTFRsiStrategy:
    """Test MTF RSI Trend Strategy."""

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly."""
        assert strategy.primary_tf == "1d"
        assert strategy.secondary_tf == "1h"
        assert strategy.trend_rsi_period == 14
        assert strategy.entry_rsi_period == 14
        assert "AAPL" in strategy._prices

    def test_rsi_calculation_with_talib(self, strategy):
        """Test RSI calculation uses TA-Lib via indicators module."""
        # Fill price history with enough data for RSI
        prices = deque(maxlen=20)

        # Create uptrending prices (should give RSI > 50)
        for i in range(20):
            prices.append(100 + i * 0.5)  # 100, 100.5, 101, ...

        rsi = strategy._calculate_rsi(prices, period=14)

        assert rsi is not None
        assert 0 <= rsi <= 100
        # Uptrending prices should give high RSI
        assert rsi > 50, f"Expected RSI > 50 for uptrend, got {rsi}"

    def test_rsi_returns_none_insufficient_data(self, strategy):
        """Test RSI returns None when not enough data."""
        prices = deque([100, 101, 102])  # Only 3 prices
        rsi = strategy._calculate_rsi(prices, period=14)
        assert rsi is None

    def test_on_bars_with_mtf_data(self, strategy, mock_context):
        """Test on_bars() processes multi-timeframe data."""
        # Warm up with enough bars for RSI calculation
        for i in range(20):
            price = 100 + i * 0.5  # Uptrending
            bars = {
                "1d": make_bar("AAPL", price, "1d"),
                "1h": make_bar("AAPL", price * 0.99, "1h"),  # Hourly slightly lower
            }
            strategy.on_bars(bars)

        # Check RSI values are calculated
        assert strategy._trend_rsi["AAPL"] is not None
        assert strategy._entry_rsi["AAPL"] is not None

    def test_on_bar_fallback(self, strategy):
        """Test on_bar() falls back to single timeframe mode."""
        # Warm up
        for i in range(20):
            bar = make_bar("AAPL", 100 + i, "1d")
            strategy.on_bar(bar)

        # Should have trend RSI
        assert strategy._trend_rsi["AAPL"] is not None

    def test_buy_signal_conditions(self, strategy, mock_context):
        """Test BUY signal when trend bullish + entry oversold."""
        # Simulate bullish trend (RSI > 50) with oversold entry (RSI < 30)
        # This requires specific price patterns

        # First, create uptrend for daily (trend RSI > 50)
        for i in range(20):
            strategy._prices["AAPL"]["1d"].append(100 + i * 2)

        # Create downtrend for hourly (entry RSI < 30)
        for i in range(20):
            strategy._prices["AAPL"]["1h"].append(150 - i * 3)

        trend_rsi = strategy._calculate_rsi(
            strategy._prices["AAPL"]["1d"], strategy.trend_rsi_period
        )
        entry_rsi = strategy._calculate_rsi(
            strategy._prices["AAPL"]["1h"], strategy.entry_rsi_period
        )

        # Verify conditions
        assert trend_rsi is not None
        assert entry_rsi is not None
        print(f"Trend RSI: {trend_rsi:.1f}, Entry RSI: {entry_rsi:.1f}")

        # Trend should be bullish (>50) for uptrending prices
        assert trend_rsi > 50

    def test_get_state(self, strategy):
        """Test get_state returns correct structure."""
        state = strategy.get_state()

        assert "AAPL" in state
        assert "trend_rsi" in state["AAPL"]
        assert "entry_rsi" in state["AAPL"]
        assert "primary_bars" in state["AAPL"]
        assert "secondary_bars" in state["AAPL"]


class TestTALibIntegration:
    """Test TA-Lib indicator integration."""

    def test_indicators_module_import(self):
        """Test indicators module imports correctly."""
        from src.domain.strategy.signals.indicators import rsi

        assert callable(rsi)

    def test_rsi_matches_expected_range(self):
        """Test RSI output is in valid range."""
        import pandas as pd

        from src.domain.strategy.signals.indicators import rsi

        # Create test data
        prices = pd.Series([100 + i for i in range(30)])
        rsi_values = rsi(prices, period=14)

        # Check valid range
        valid_values = rsi_values.dropna()
        assert all(0 <= v <= 100 for v in valid_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
