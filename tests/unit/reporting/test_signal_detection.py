"""
Tests for signal_detection.py — cross up/down, threshold crosses, state changes,
MACD crosses, detect_historical_signals dispatch, signal outcomes, aggregate_rule_frequency.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.domain.signals.models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)
from src.infrastructure.reporting.signal_report.signal_detection import (
    _detect_cross_down_signals,
    _detect_cross_up_signals,
    _detect_macd_crosses,
    _detect_state_change_signals,
    _detect_threshold_cross_down_signals,
    _detect_threshold_cross_up_signals,
    aggregate_rule_frequency,
    calculate_signal_outcomes,
    detect_historical_signals,
    detect_signals_with_frequency,
)

# =============================================================================
# Helpers
# =============================================================================


def _rule(
    name: str = "test_rule",
    indicator: str = "macd",
    condition_type: ConditionType = ConditionType.CROSS_UP,
    direction: SignalDirection = SignalDirection.BUY,
    config: Dict[str, Any] | None = None,
    timeframes: tuple = ("1d",),
    message: str = "{symbol} test signal",
) -> SignalRule:
    return SignalRule(
        name=name,
        indicator=indicator,
        category=SignalCategory.MOMENTUM,
        direction=direction,
        strength=70,
        priority=SignalPriority.HIGH,
        condition_type=condition_type,
        condition_config=config or {},
        timeframes=timeframes,
        message_template=message,
    )


def _df_with_cross(n: int = 20) -> pd.DataFrame:
    """DataFrame where macd crosses signal at bar 10 (bullish) and 15 (bearish)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    macd = np.zeros(n)
    signal = np.zeros(n)

    # Before cross: macd below signal
    for i in range(10):
        macd[i] = -1.0 + i * 0.1
        signal[i] = 0.0

    # At bar 10: macd crosses above signal (bullish)
    macd[10] = 0.5
    signal[10] = 0.0

    # Between crosses: macd above signal
    for i in range(11, 15):
        macd[i] = 1.0 - (i - 10) * 0.3
        signal[i] = 0.0

    # At bar 15: macd crosses below signal (bearish)
    macd[15] = -0.5
    signal[15] = 0.0

    # After
    for i in range(16, n):
        macd[i] = -1.0
        signal[i] = 0.0

    return pd.DataFrame(
        {
            "macd_macd": macd,
            "macd_signal": signal,
            "close": [100.0 + i for i in range(n)],
            "high": [105.0 + i for i in range(n)],
            "low": [95.0 + i for i in range(n)],
            "volume": [1_000_000] * n,
        },
        index=dates,
    )


# =============================================================================
# Cross Up
# =============================================================================


class TestCrossUp:
    def test_detects_bullish_cross(self) -> None:
        df = _df_with_cross()
        rule = _rule(
            config={"line_a": "macd", "line_b": "signal"},
            condition_type=ConditionType.CROSS_UP,
        )
        ind_cols = ["macd_macd", "macd_signal"]
        timestamps = df.index.tolist()
        signals = _detect_cross_up_signals(df, rule, ind_cols, timestamps, "AAPL")
        assert len(signals) >= 1
        assert signals[0]["direction"] == "buy"

    def test_no_cross_returns_empty(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {"macd_macd": [1, 2, 3, 4, 5], "macd_signal": [10, 10, 10, 10, 10]},
            index=dates,
        )
        rule = _rule(config={"line_a": "macd", "line_b": "signal"})
        signals = _detect_cross_up_signals(
            df, rule, ["macd_macd", "macd_signal"], dates.tolist(), "AAPL"
        )
        assert signals == []

    def test_missing_column_returns_empty(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"macd_macd": [1, 2, 3, 4, 5]}, index=dates)
        rule = _rule(config={"line_a": "macd", "line_b": "signal"})
        signals = _detect_cross_up_signals(df, rule, ["macd_macd"], dates.tolist(), "X")
        assert signals == []

    def test_nan_at_crossover_skipped(self) -> None:
        """NaN values at crossover points are handled gracefully."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "macd_macd": [0, np.nan, 2, 3, 4],
                "macd_signal": [1, 1, 1, 1, 1],
            },
            index=dates,
        )
        rule = _rule(config={"line_a": "macd", "line_b": "signal"})
        signals = _detect_cross_up_signals(
            df, rule, ["macd_macd", "macd_signal"], dates.tolist(), "X"
        )
        # Bar 1 has NaN so cross from bar 0->1 and 1->2 should be skipped
        for s in signals:
            assert s["timestamp"] != dates[1]
            assert s["timestamp"] != dates[2]


# =============================================================================
# Cross Down
# =============================================================================


class TestCrossDown:
    def test_detects_bearish_cross(self) -> None:
        df = _df_with_cross()
        rule = _rule(
            config={"line_a": "macd", "line_b": "signal"},
            condition_type=ConditionType.CROSS_DOWN,
            direction=SignalDirection.SELL,
        )
        ind_cols = ["macd_macd", "macd_signal"]
        timestamps = df.index.tolist()
        signals = _detect_cross_down_signals(df, rule, ind_cols, timestamps, "AAPL")
        assert len(signals) >= 1
        assert signals[0]["direction"] == "sell"

    def test_exact_equality_then_cross(self) -> None:
        """When prev a == b and curr a < b, should trigger."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"macd_macd": [1.0, 1.0, 0.5], "macd_signal": [1.0, 1.0, 1.0]},
            index=dates,
        )
        rule = _rule(
            config={"line_a": "macd", "line_b": "signal"},
            condition_type=ConditionType.CROSS_DOWN,
            direction=SignalDirection.SELL,
        )
        signals = _detect_cross_down_signals(
            df, rule, ["macd_macd", "macd_signal"], dates.tolist(), "X"
        )
        assert len(signals) == 1
        assert signals[0]["timestamp"] == dates[2]


# =============================================================================
# Threshold Cross Up
# =============================================================================


class TestThresholdCrossUp:
    def test_crosses_above_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"rsi_rsi": [25, 28, 29, 31, 35]}, index=dates)
        rule = _rule(
            name="rsi_overbought",
            indicator="rsi",
            condition_type=ConditionType.THRESHOLD_CROSS_UP,
            config={"field": "rsi", "threshold": 30},
            message="{symbol} RSI > {threshold}",
        )
        signals = _detect_threshold_cross_up_signals(df, rule, ["rsi_rsi"], dates.tolist(), "AAPL")
        assert len(signals) == 1
        assert signals[0]["timestamp"] == dates[3]
        assert signals[0]["threshold"] == 30

    def test_already_above_no_signal(self) -> None:
        """If value is already above threshold, no cross detected."""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"rsi_rsi": [35, 40, 45]}, index=dates)
        rule = _rule(
            indicator="rsi",
            condition_type=ConditionType.THRESHOLD_CROSS_UP,
            config={"field": "rsi", "threshold": 30},
        )
        signals = _detect_threshold_cross_up_signals(df, rule, ["rsi_rsi"], dates.tolist(), "X")
        assert signals == []

    def test_nan_before_cross(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame({"rsi_rsi": [np.nan, 25, 35, 40]}, index=dates)
        rule = _rule(
            indicator="rsi",
            condition_type=ConditionType.THRESHOLD_CROSS_UP,
            config={"field": "rsi", "threshold": 30},
        )
        signals = _detect_threshold_cross_up_signals(df, rule, ["rsi_rsi"], dates.tolist(), "X")
        # Bar 0->1: NaN skip; Bar 1->2: 25->35 crosses 30
        assert len(signals) == 1
        assert signals[0]["timestamp"] == dates[2]


# =============================================================================
# Threshold Cross Down
# =============================================================================


class TestThresholdCrossDown:
    def test_crosses_below_threshold(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"rsi_rsi": [40, 35, 31, 29, 25]}, index=dates)
        rule = _rule(
            indicator="rsi",
            condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
            direction=SignalDirection.BUY,
            config={"field": "rsi", "threshold": 30},
            message="{symbol} RSI < {threshold}",
        )
        signals = _detect_threshold_cross_down_signals(
            df, rule, ["rsi_rsi"], dates.tolist(), "AAPL"
        )
        assert len(signals) == 1
        assert signals[0]["timestamp"] == dates[3]

    def test_missing_threshold_returns_empty(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"rsi_rsi": [40, 30, 20]}, index=dates)
        rule = _rule(
            indicator="rsi",
            condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
            config={"field": "rsi"},  # No threshold
        )
        signals = _detect_threshold_cross_down_signals(df, rule, ["rsi_rsi"], dates.tolist(), "X")
        assert signals == []


# =============================================================================
# State Change
# =============================================================================


class TestStateChange:
    def test_detects_state_transition(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {"supertrend_direction": ["bearish", "bearish", "bullish", "bullish"]},
            index=dates,
        )
        rule = _rule(
            name="st_flip",
            indicator="supertrend",
            condition_type=ConditionType.STATE_CHANGE,
            config={"field": "direction", "from": ["bearish"], "to": ["bullish"]},
        )
        signals = _detect_state_change_signals(
            df, rule, ["supertrend_direction"], dates.tolist(), "AAPL"
        )
        assert len(signals) == 1
        assert signals[0]["timestamp"] == dates[2]

    def test_no_matching_transition(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"supertrend_direction": ["bullish", "bullish", "bullish"]}, index=dates)
        rule = _rule(
            indicator="supertrend",
            condition_type=ConditionType.STATE_CHANGE,
            config={"field": "direction", "from": ["bearish"], "to": ["bullish"]},
        )
        signals = _detect_state_change_signals(
            df, rule, ["supertrend_direction"], dates.tolist(), "X"
        )
        assert signals == []

    def test_missing_column_returns_empty(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102]}, index=dates)
        rule = _rule(
            indicator="supertrend",
            condition_type=ConditionType.STATE_CHANGE,
            config={"field": "direction", "from": ["bearish"], "to": ["bullish"]},
        )
        signals = _detect_state_change_signals(df, rule, [], dates.tolist(), "X")
        assert signals == []


# =============================================================================
# MACD Crosses
# =============================================================================


class TestMACDCrosses:
    def test_detects_bullish_and_bearish(self) -> None:
        df = _df_with_cross()
        rule = _rule(name="macd_base")
        timestamps = df.index.tolist()
        signals = _detect_macd_crosses(df, rule, timestamps, "AAPL", [])
        rules_found = {s["rule"] for s in signals}
        assert "macd_bullish_cross" in rules_found
        assert "macd_bearish_cross" in rules_found

    def test_skips_duplicate_timestamps(self) -> None:
        """Existing MACD signals at same timestamp prevent duplicates."""
        df = _df_with_cross()
        rule = _rule(name="macd_base")
        timestamps = df.index.tolist()
        # Pre-populate existing signals at the cross point
        existing = [{"timestamp": timestamps[10], "rule": "macd_bullish_cross"}]
        signals = _detect_macd_crosses(df, rule, timestamps, "AAPL", existing)
        # Should not have another signal at timestamp[10]
        ts_10_signals = [s for s in signals if s["timestamp"] == timestamps[10]]
        assert len(ts_10_signals) == 0

    def test_skips_when_cross_in_name(self) -> None:
        """Rules with 'cross' in name are skipped (handled by CROSS_UP/DOWN)."""
        df = _df_with_cross()
        rule = _rule(name="macd_cross_special")
        signals = _detect_macd_crosses(df, rule, df.index.tolist(), "AAPL", [])
        assert signals == []

    def test_missing_columns(self) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        rule = _rule(name="macd_base")
        signals = _detect_macd_crosses(df, rule, dates.tolist(), "X", [])
        assert signals == []


# =============================================================================
# detect_historical_signals (dispatch)
# =============================================================================


class TestDetectHistoricalSignals:
    def test_dispatches_by_condition_type(self, sample_signal_rules: List[SignalRule]) -> None:
        df = _df_with_cross(50)
        # Add RSI column that crosses 30 and 70
        rsi_vals = np.linspace(25, 75, 50)
        df["rsi_rsi"] = rsi_vals
        # Add supertrend direction
        df["supertrend_direction"] = ["bearish"] * 25 + ["bullish"] * 25

        signals = detect_historical_signals(
            df, sample_signal_rules, "AAPL", "1d", calculate_outcomes=False
        )
        assert len(signals) > 0
        # Should be sorted by timestamp
        for i in range(1, len(signals)):
            assert signals[i]["timestamp"] >= signals[i - 1]["timestamp"]

    def test_disabled_rules_skipped(self) -> None:
        df = _df_with_cross()
        rule = _rule(config={"line_a": "macd", "line_b": "signal"})
        rule.enabled = False
        signals = detect_historical_signals(df, [rule], "AAPL", "1d")
        assert signals == []

    def test_wrong_timeframe_skipped(self) -> None:
        df = _df_with_cross()
        rule = _rule(config={"line_a": "macd", "line_b": "signal"}, timeframes=("1h",))
        signals = detect_historical_signals(df, [rule], "AAPL", "1d")
        assert signals == []


# =============================================================================
# calculate_signal_outcomes
# =============================================================================


class TestSignalOutcomes:
    def test_calculates_price_change(self) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0 + i for i in range(20)],
                "high": [102.0 + i for i in range(20)],
                "low": [98.0 + i for i in range(20)],
            },
            index=dates,
        )
        signals = [{"timestamp": dates[5], "direction": "buy", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals, forward_bars=5)
        outcome = result[0]["outcome"]
        assert outcome["status"] == "completed"
        assert outcome["price_change_pct"] > 0
        assert outcome["correct"] == True  # noqa: E712 (np.bool_ vs bool)
        assert "mfe_pct" in outcome

    def test_sell_signal_correct_when_price_drops(self) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "close": [120.0 - i for i in range(20)],
                "high": [122.0 - i for i in range(20)],
                "low": [118.0 - i for i in range(20)],
            },
            index=dates,
        )
        signals = [{"timestamp": dates[5], "direction": "sell", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals, forward_bars=5)
        assert result[0]["outcome"]["correct"] == True  # noqa: E712

    def test_forming_signal_near_end(self) -> None:
        """Signals near the end of data get 'forming' status."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0 + i for i in range(10)],
                "high": [102.0 + i for i in range(10)],
                "low": [98.0 + i for i in range(10)],
            },
            index=dates,
        )
        signals = [{"timestamp": dates[9], "direction": "buy", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals, forward_bars=5)
        assert result[0]["outcome"]["status"] == "forming"
        assert result[0]["outcome"]["is_current"] is True

    def test_alert_direction(self) -> None:
        """Alert signals have correct=None."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "close": [100.0 + i for i in range(20)],
                "high": [102.0 + i for i in range(20)],
                "low": [98.0 + i for i in range(20)],
            },
            index=dates,
        )
        signals = [{"timestamp": dates[5], "direction": "alert", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals, forward_bars=5)
        assert result[0]["outcome"]["correct"] is None

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["close", "high", "low"])
        signals = [{"timestamp": "2024-01-01", "direction": "buy", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals)
        # Should return signals unchanged (no outcome added)
        assert "outcome" not in result[0]

    def test_zero_entry_price_skipped(self) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        close = [0.0] + [100.0 + i for i in range(19)]
        df = pd.DataFrame(
            {
                "close": close,
                "high": [c + 2 for c in close],
                "low": [max(c - 2, 0) for c in close],
            },
            index=dates,
        )
        signals = [{"timestamp": dates[0], "direction": "buy", "rule": "test"}]
        result = calculate_signal_outcomes(df, signals, forward_bars=5)
        assert "outcome" not in result[0]


# =============================================================================
# detect_signals_with_frequency
# =============================================================================


class TestDetectSignalsWithFrequency:
    def test_returns_frequency_dict(self) -> None:
        df = _df_with_cross(30)
        rule = _rule(
            config={"line_a": "macd", "line_b": "signal"},
            condition_type=ConditionType.CROSS_UP,
        )
        signals, frequency = detect_signals_with_frequency(df, [rule], "AAPL", "1d")
        assert isinstance(frequency, dict)
        if signals:
            assert sum(frequency.values()) == len(signals)

    def test_lookback_filters(self) -> None:
        df = _df_with_cross(30)
        rule = _rule(
            config={"line_a": "macd", "line_b": "signal"},
            condition_type=ConditionType.CROSS_UP,
        )
        signals_all, _ = detect_signals_with_frequency(df, [rule], "AAPL", "1d", lookback_bars=None)
        signals_short, _ = detect_signals_with_frequency(df, [rule], "AAPL", "1d", lookback_bars=5)
        assert len(signals_short) <= len(signals_all)


# =============================================================================
# aggregate_rule_frequency
# =============================================================================


class TestAggregateRuleFrequency:
    def test_aggregates_across_symbols(self) -> None:
        all_signals = {
            "AAPL_1d": [
                {"rule": "rsi_oversold", "direction": "buy"},
                {"rule": "macd_cross", "direction": "buy"},
            ],
            "SPY_1d": [
                {"rule": "rsi_oversold", "direction": "sell"},
            ],
        }
        result = aggregate_rule_frequency(all_signals)
        assert result["total_signals"] == 3
        assert result["by_symbol"]["AAPL"] == 2
        assert result["by_symbol"]["SPY"] == 1
        assert result["by_rule"]["rsi_oversold"] == 2
        assert result["buy_by_symbol"]["AAPL"] == 2
        assert result["sell_by_symbol"]["SPY"] == 1

    def test_empty_input(self) -> None:
        result = aggregate_rule_frequency({})
        assert result["total_signals"] == 0
        assert result["top_symbols"] == []
        assert result["top_rules"] == []

    def test_top_symbols_sorted(self) -> None:
        all_signals = {
            "A_1d": [{"rule": "r1", "direction": "buy"}] * 5,
            "B_1d": [{"rule": "r1", "direction": "buy"}] * 10,
            "C_1d": [{"rule": "r1", "direction": "buy"}] * 3,
        }
        result = aggregate_rule_frequency(all_signals)
        assert result["top_symbols"][0][0] == "B"
        assert result["top_symbols"][0][1] == 10
