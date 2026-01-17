"""
Unit tests for trading signal domain models.

Tests:
- TradingSignal creation and serialization
- SignalRule condition checking (all types)
- ConditionType.CUSTOM raises NotImplementedError
- Divergence and ConfluenceScore models
"""

from datetime import datetime

import pytest

from src.domain.signals.models import (
    ConditionType,
    ConfluenceScore,
    Divergence,
    DivergenceType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
    TradingSignal,
)


class TestTradingSignal:
    """Tests for TradingSignal dataclass."""

    def test_create_signal(self) -> None:
        """Test basic signal creation."""
        signal = TradingSignal(
            signal_id="momentum:rsi:AAPL:1h",
            symbol="AAPL",
            category=SignalCategory.MOMENTUM,
            indicator="rsi",
            direction=SignalDirection.BUY,
            strength=70,
            priority=SignalPriority.HIGH,
            timeframe="1h",
            trigger_rule="rsi_oversold_exit",
            current_value=35.5,
            message="RSI exiting oversold",
        )

        assert signal.signal_id == "momentum:rsi:AAPL:1h"
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == 70

    def test_signal_to_dict(self) -> None:
        """Test signal serialization."""
        signal = TradingSignal(
            signal_id="trend:supertrend:MSFT:4h",
            symbol="MSFT",
            category=SignalCategory.TREND,
            indicator="supertrend",
            direction=SignalDirection.SELL,
            strength=80,
            priority=SignalPriority.HIGH,
            timeframe="4h",
            trigger_rule="supertrend_bearish",
            current_value=150.0,
            threshold=145.0,
        )

        data = signal.to_dict()
        assert data["symbol"] == "MSFT"
        assert data["category"] == "trend"
        assert data["direction"] == "sell"
        assert data["threshold"] == 145.0

    def test_signal_str_representation(self) -> None:
        """Test human-readable string output."""
        signal = TradingSignal(
            signal_id="test",
            symbol="GOOGL",
            category=SignalCategory.MOMENTUM,
            indicator="macd",
            direction=SignalDirection.BUY,
            strength=60,
            priority=SignalPriority.MEDIUM,
            timeframe="1d",
            trigger_rule="macd_cross",
            current_value=0.5,
            message="MACD bullish cross",
        )

        s = str(signal)
        assert "GOOGL" in s
        assert "macd" in s
        assert "MACD bullish cross" in s


class TestSignalRule:
    """Tests for SignalRule condition checking."""

    def test_threshold_cross_up(self) -> None:
        """Test THRESHOLD_CROSS_UP condition."""
        rule = SignalRule(
            name="rsi_above_50",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=50,
            priority=SignalPriority.LOW,
            condition_type=ConditionType.THRESHOLD_CROSS_UP,
            condition_config={"field": "value", "threshold": 50},
        )

        # Crossing up from below
        assert (
            rule.check_condition(
                {"value": 45},
                {"value": 55},
            )
            is True
        )

        # Already above - no cross
        assert (
            rule.check_condition(
                {"value": 55},
                {"value": 60},
            )
            is False
        )

        # No previous state
        assert (
            rule.check_condition(
                None,
                {"value": 55},
            )
            is False
        )

    def test_threshold_cross_down(self) -> None:
        """Test THRESHOLD_CROSS_DOWN condition."""
        rule = SignalRule(
            name="rsi_below_30",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.SELL,
            strength=60,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
            condition_config={"field": "value", "threshold": 30},
        )

        # Crossing down from above
        assert (
            rule.check_condition(
                {"value": 35},
                {"value": 25},
            )
            is True
        )

        # Already below - no cross
        assert (
            rule.check_condition(
                {"value": 25},
                {"value": 20},
            )
            is False
        )

    def test_state_change(self) -> None:
        """Test STATE_CHANGE condition."""
        rule = SignalRule(
            name="rsi_oversold_exit",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=70,
            priority=SignalPriority.HIGH,
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={
                "field": "zone",
                "from": ["oversold"],
                "to": ["neutral"],
            },
        )

        # Valid state change
        assert (
            rule.check_condition(
                {"zone": "oversold", "value": 25},
                {"zone": "neutral", "value": 35},
            )
            is True
        )

        # Wrong transition
        assert (
            rule.check_condition(
                {"zone": "neutral", "value": 50},
                {"zone": "overbought", "value": 75},
            )
            is False
        )

        # No previous state
        assert (
            rule.check_condition(
                None,
                {"zone": "neutral", "value": 35},
            )
            is False
        )

    def test_cross_up(self) -> None:
        """Test CROSS_UP condition (line A crosses above line B)."""
        rule = SignalRule(
            name="macd_bullish",
            indicator="macd",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=60,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.CROSS_UP,
            condition_config={"line_a": "macd", "line_b": "signal"},
        )

        # MACD crosses above signal
        assert (
            rule.check_condition(
                {"macd": -0.5, "signal": 0.0},
                {"macd": 0.5, "signal": 0.0},
            )
            is True
        )

        # MACD already above
        assert (
            rule.check_condition(
                {"macd": 0.5, "signal": 0.0},
                {"macd": 0.8, "signal": 0.0},
            )
            is False
        )

    def test_cross_down(self) -> None:
        """Test CROSS_DOWN condition (line A crosses below line B)."""
        rule = SignalRule(
            name="macd_bearish",
            indicator="macd",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.SELL,
            strength=60,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.CROSS_DOWN,
            condition_config={"line_a": "macd", "line_b": "signal"},
        )

        # MACD crosses below signal
        assert (
            rule.check_condition(
                {"macd": 0.5, "signal": 0.0},
                {"macd": -0.5, "signal": 0.0},
            )
            is True
        )

    def test_range_entry(self) -> None:
        """Test RANGE_ENTRY condition."""
        rule = SignalRule(
            name="rsi_neutral_entry",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.ALERT,
            strength=30,
            priority=SignalPriority.LOW,
            condition_type=ConditionType.RANGE_ENTRY,
            condition_config={"field": "value", "lower": 30, "upper": 70},
        )

        # Entering range from below
        assert (
            rule.check_condition(
                {"value": 25},
                {"value": 35},
            )
            is True
        )

        # Entering range from above
        assert (
            rule.check_condition(
                {"value": 75},
                {"value": 65},
            )
            is True
        )

        # Already inside
        assert (
            rule.check_condition(
                {"value": 50},
                {"value": 55},
            )
            is False
        )

    def test_range_exit(self) -> None:
        """Test RANGE_EXIT condition."""
        rule = SignalRule(
            name="rsi_extreme_exit",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.ALERT,
            strength=50,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.RANGE_EXIT,
            condition_config={"field": "value", "lower": 30, "upper": 70},
        )

        # Exiting range below
        assert (
            rule.check_condition(
                {"value": 35},
                {"value": 25},
            )
            is True
        )

        # Exiting range above
        assert (
            rule.check_condition(
                {"value": 65},
                {"value": 75},
            )
            is True
        )

        # Still inside
        assert (
            rule.check_condition(
                {"value": 50},
                {"value": 60},
            )
            is False
        )

    def test_custom_raises_not_implemented(self) -> None:
        """Test that CUSTOM condition type raises NotImplementedError."""
        rule = SignalRule(
            name="custom_rule",
            indicator="custom",
            category=SignalCategory.PATTERN,
            direction=SignalDirection.ALERT,
            strength=50,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.CUSTOM,
            condition_config={},
        )

        with pytest.raises(NotImplementedError) as exc_info:
            rule.check_condition({"value": 1}, {"value": 2})

        assert "CUSTOM condition type requires" in str(exc_info.value)
        assert "custom_rule" in str(exc_info.value)

    def test_format_message(self) -> None:
        """Test message template formatting."""
        rule = SignalRule(
            name="test",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=70,
            priority=SignalPriority.HIGH,
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={},
            message_template="{symbol} RSI at {value} (threshold: {threshold})",
        )

        msg = rule.format_message("AAPL", value=35.5, threshold=30.0)
        assert "AAPL" in msg
        assert "35.50" in msg
        assert "30.00" in msg


class TestDivergence:
    """Tests for Divergence dataclass."""

    def test_create_divergence(self) -> None:
        """Test divergence creation."""
        now = datetime.utcnow()
        div = Divergence(
            type=DivergenceType.BULLISH,
            indicator="rsi",
            symbol="AAPL",
            timeframe="1h",
            price_point1=(now, 150.0),
            price_point2=(now, 145.0),
            indicator_point1=(now, 25.0),
            indicator_point2=(now, 30.0),
            strength=75,
            bars_apart=10,
        )

        assert div.type == DivergenceType.BULLISH
        assert div.strength == 75
        assert "bullish" in str(div).lower()

    def test_divergence_to_dict(self) -> None:
        """Test divergence serialization."""
        now = datetime.utcnow()
        div = Divergence(
            type=DivergenceType.BEARISH,
            indicator="macd",
            symbol="MSFT",
            timeframe="4h",
            price_point1=(now, 300.0),
            price_point2=(now, 310.0),
            indicator_point1=(now, 5.0),
            indicator_point2=(now, 3.0),
            strength=60,
            bars_apart=8,
        )

        data = div.to_dict()
        assert data["type"] == "bearish"
        assert data["indicator"] == "macd"
        assert data["strength"] == 60


class TestConfluenceScore:
    """Tests for ConfluenceScore dataclass."""

    def test_create_confluence_score(self) -> None:
        """Test confluence score creation."""
        score = ConfluenceScore(
            symbol="AAPL",
            timeframe="1h",
            bullish_count=5,
            bearish_count=2,
            neutral_count=3,
            alignment_score=30,
            diverging_pairs=[("rsi", "macd", "rsi bullish, macd bearish")],
            strongest_signal="bullish",
        )

        assert score.bullish_count == 5
        assert score.alignment_score == 30
        assert len(score.diverging_pairs) == 1

    def test_confluence_to_dict(self) -> None:
        """Test confluence serialization."""
        score = ConfluenceScore(
            symbol="SPY",
            timeframe="1d",
            bullish_count=3,
            bearish_count=5,
            neutral_count=2,
            alignment_score=-20,
            diverging_pairs=[],
            strongest_signal="bearish",
        )

        data = score.to_dict()
        assert data["alignment_score"] == -20
        assert data["strongest_signal"] == "bearish"
