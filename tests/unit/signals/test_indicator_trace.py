"""
Tests for IndicatorTrace (Phase 3: Indicator Observability).

Tests cover:
1. IndicatorTrace creation and serialization
2. Delta calculation between current and previous values
3. Integration with RegimeOutput
"""

from datetime import datetime

import pytest

from src.domain.signals.models import IndicatorTrace


class TestIndicatorTrace:
    """Tests for IndicatorTrace dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating an IndicatorTrace with basic fields."""
        trace = IndicatorTrace(
            indicator_name="rsi",
            timeframe="5m",
            bar_ts=datetime(2024, 1, 15, 10, 30),
            symbol="AAPL",
        )

        assert trace.indicator_name == "rsi"
        assert trace.timeframe == "5m"
        assert trace.symbol == "AAPL"
        assert trace.raw == {}
        assert trace.state == {}
        assert trace.rules_triggered_now == []

    def test_creation_with_values(self) -> None:
        """Test creating an IndicatorTrace with raw values and state."""
        trace = IndicatorTrace(
            indicator_name="macd",
            timeframe="1h",
            bar_ts=datetime(2024, 1, 15, 10, 0),
            symbol="TSLA",
            raw={"macd": -1.2, "signal": -0.8, "histogram": -0.4},
            state={"direction": "bearish", "crossover": "below_signal"},
            rules_triggered_now=["macd_bearish_cross", "macd_below_zero"],
            lookback=26,
        )

        assert trace.raw["macd"] == -1.2
        assert trace.state["direction"] == "bearish"
        assert "macd_bearish_cross" in trace.rules_triggered_now
        assert trace.lookback == 26

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        ts = datetime(2024, 1, 15, 10, 30)
        trace = IndicatorTrace(
            indicator_name="rsi",
            timeframe="5m",
            bar_ts=ts,
            symbol="AAPL",
            raw={"rsi": 28.5},
            state={"zone": "oversold"},
            rules_triggered_now=["rsi_oversold"],
            lookback=14,
        )

        data = trace.to_dict()

        assert data["indicator_name"] == "rsi"
        assert data["timeframe"] == "5m"
        assert data["bar_ts"] == ts.isoformat()
        assert data["symbol"] == "AAPL"
        assert data["raw"]["rsi"] == 28.5
        assert data["state"]["zone"] == "oversold"
        assert data["rules_triggered_now"] == ["rsi_oversold"]
        assert data["lookback"] == 14

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "indicator_name": "bollinger",
            "timeframe": "15m",
            "bar_ts": "2024-01-15T10:45:00",
            "symbol": "SPY",
            "raw": {"upper": 485.0, "middle": 480.0, "lower": 475.0},
            "state": {"band_width": "normal", "price_position": "middle"},
            "rules_triggered_now": [],
            "lookback": 20,
        }

        trace = IndicatorTrace.from_dict(data)

        assert trace.indicator_name == "bollinger"
        assert trace.timeframe == "15m"
        assert trace.bar_ts == datetime(2024, 1, 15, 10, 45)
        assert trace.symbol == "SPY"
        assert trace.raw["middle"] == 480.0
        assert trace.state["band_width"] == "normal"
        assert trace.lookback == 20

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = IndicatorTrace(
            indicator_name="atr",
            timeframe="1d",
            bar_ts=datetime(2024, 1, 15, 16, 0),
            symbol="QQQ",
            raw={"atr": 5.25, "atr_pct": 1.2},
            state={"volatility": "high"},
            rules_triggered_now=["high_volatility"],
            lookback=14,
            prev_raw={"atr": 4.85, "atr_pct": 1.1},
        )

        data = original.to_dict()
        restored = IndicatorTrace.from_dict(data)

        assert restored.indicator_name == original.indicator_name
        assert restored.timeframe == original.timeframe
        assert restored.symbol == original.symbol
        assert restored.raw == original.raw
        assert restored.state == original.state
        assert restored.rules_triggered_now == original.rules_triggered_now
        assert restored.prev_raw == original.prev_raw

    def test_get_delta_with_prev_raw(self) -> None:
        """Test delta calculation when prev_raw is available."""
        trace = IndicatorTrace(
            indicator_name="rsi",
            timeframe="5m",
            bar_ts=datetime(2024, 1, 15, 10, 30),
            raw={"rsi": 35.0},
            prev_raw={"rsi": 28.5},
        )

        delta = trace.get_delta("rsi")

        assert delta == pytest.approx(6.5)

    def test_get_delta_without_prev_raw(self) -> None:
        """Test delta returns None when prev_raw is not available."""
        trace = IndicatorTrace(
            indicator_name="rsi",
            timeframe="5m",
            bar_ts=datetime(2024, 1, 15, 10, 30),
            raw={"rsi": 35.0},
        )

        delta = trace.get_delta("rsi")

        assert delta is None

    def test_get_delta_missing_key(self) -> None:
        """Test delta returns None when key is missing from either raw dict."""
        trace = IndicatorTrace(
            indicator_name="macd",
            timeframe="1h",
            bar_ts=datetime(2024, 1, 15, 10, 0),
            raw={"macd": -1.2},
            prev_raw={"signal": -0.8},  # Different key
        )

        delta = trace.get_delta("macd")

        assert delta is None

    def test_str_representation(self) -> None:
        """Test human-readable string representation."""
        trace = IndicatorTrace(
            indicator_name="regime",
            timeframe="1d",
            bar_ts=datetime(2024, 1, 15),
            raw={"confidence": 75},
            state={"regime": "R0"},
            rules_triggered_now=["trend_up", "vol_normal"],
        )

        str_repr = str(trace)

        assert "regime" in str_repr
        assert "1d" in str_repr
        assert "trend_up" in str_repr

    def test_str_no_rules(self) -> None:
        """Test string representation with no rules triggered."""
        trace = IndicatorTrace(
            indicator_name="rsi",
            timeframe="5m",
            bar_ts=datetime(2024, 1, 15, 10, 30),
            raw={"rsi": 45.0},
            state={"zone": "neutral"},
        )

        str_repr = str(trace)

        assert "none" in str_repr.lower()


class TestIndicatorTraceWithRegimeOutput:
    """Tests for IndicatorTrace integration with RegimeOutput."""

    def test_regime_output_with_indicator_traces(self) -> None:
        """Test that RegimeOutput can store and serialize indicator traces."""
        from src.domain.signals.indicators.regime.models import (
            MarketRegime,
            RegimeOutput,
        )

        traces = [
            IndicatorTrace(
                indicator_name="trend",
                timeframe="1d",
                bar_ts=datetime(2024, 1, 15),
                raw={"ma50_slope": 0.002, "price_vs_ma200": 1.05},
                state={"trend_state": "trend_up"},
            ),
            IndicatorTrace(
                indicator_name="volatility",
                timeframe="1d",
                bar_ts=datetime(2024, 1, 15),
                raw={"atr_pct_63": 45.0},
                state={"vol_state": "vol_normal"},
            ),
            IndicatorTrace(
                indicator_name="chop",
                timeframe="1d",
                bar_ts=datetime(2024, 1, 15),
                raw={"chop_pct": 35.0},
                state={"chop_state": "trending"},
            ),
        ]

        output = RegimeOutput(
            symbol="AAPL",
            asof_ts=datetime(2024, 1, 15),
            final_regime=MarketRegime.R0_HEALTHY_UPTREND,
            indicator_traces=traces,
        )

        assert len(output.indicator_traces) == 3
        assert output.indicator_traces[0].indicator_name == "trend"

        # Test serialization
        data = output.to_dict()
        assert len(data["indicator_traces"]) == 3
        assert data["indicator_traces"][0]["indicator_name"] == "trend"

    def test_regime_output_roundtrip_with_traces(self) -> None:
        """Test RegimeOutput roundtrip preserves indicator traces."""
        from src.domain.signals.indicators.regime.models import (
            MarketRegime,
            RegimeOutput,
        )

        traces = [
            IndicatorTrace(
                indicator_name="regime",
                timeframe="1d",
                bar_ts=datetime(2024, 1, 15),
                symbol="AAPL",
                raw={"confidence": 80, "atr_pct_63": 45.0},
                state={"final_regime": "R0"},
                rules_triggered_now=["trend_confirmed", "vol_normal"],
            ),
        ]

        original = RegimeOutput(
            symbol="AAPL",
            asof_ts=datetime(2024, 1, 15),
            final_regime=MarketRegime.R0_HEALTHY_UPTREND,
            indicator_traces=traces,
        )

        data = original.to_dict()
        restored = RegimeOutput.from_dict(data)

        assert len(restored.indicator_traces) == 1
        trace = restored.indicator_traces[0]
        assert trace.indicator_name == "regime"
        assert trace.raw["confidence"] == 80
        assert "trend_confirmed" in trace.rules_triggered_now

    def test_regime_output_empty_traces(self) -> None:
        """Test RegimeOutput with empty indicator traces."""
        from src.domain.signals.indicators.regime.models import RegimeOutput

        output = RegimeOutput(symbol="AAPL")

        assert output.indicator_traces == []

        data = output.to_dict()
        assert data["indicator_traces"] == []

        restored = RegimeOutput.from_dict(data)
        assert restored.indicator_traces == []
