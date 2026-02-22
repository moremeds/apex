"""
Tests for confluence_analyzer.py — _derive_*_state() functions,
_get_indicator_direction(), calculate_confluence, calculate_mtf_confluence.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.reporting.package.confluence_analyzer import (
    _derive_adx_state,
    _derive_bollinger_state,
    _derive_kdj_state,
    _derive_macd_state,
    _derive_rsi_state,
    _derive_supertrend_state,
    _get_indicator_direction,
    calculate_confluence,
    calculate_mtf_confluence,
    derive_indicator_states,
)

# =============================================================================
# _derive_rsi_state
# =============================================================================


class TestDeriveRSIState:
    def test_oversold(self) -> None:
        df = pd.DataFrame({"rsi_rsi": [25.0]})
        result = _derive_rsi_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "oversold"
        assert result["value"] == 25.0

    def test_overbought(self) -> None:
        df = pd.DataFrame({"rsi_rsi": [75.0]})
        result = _derive_rsi_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "overbought"

    def test_neutral(self) -> None:
        df = pd.DataFrame({"rsi_rsi": [50.0]})
        result = _derive_rsi_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "neutral"

    def test_boundary_30(self) -> None:
        """RSI == 30 is neutral (not oversold, < 30 is oversold)."""
        df = pd.DataFrame({"rsi_rsi": [30.0]})
        result = _derive_rsi_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "neutral"

    def test_boundary_70(self) -> None:
        """RSI == 70 is neutral (not overbought, > 70 is overbought)."""
        df = pd.DataFrame({"rsi_rsi": [70.0]})
        result = _derive_rsi_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "neutral"

    def test_nan_returns_none(self) -> None:
        df = pd.DataFrame({"rsi_rsi": [np.nan]})
        assert _derive_rsi_state(df, df.iloc[0]) is None

    def test_missing_column_returns_none(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        assert _derive_rsi_state(df, df.iloc[0]) is None


# =============================================================================
# _derive_macd_state
# =============================================================================


class TestDeriveMACDState:
    def test_basic_macd_state(self) -> None:
        df = pd.DataFrame(
            {
                "macd_macd": [0.5, 1.0],
                "macd_signal": [0.3, 0.8],
                "macd_histogram": [0.2, 0.2],
            }
        )
        result = _derive_macd_state(df, df.iloc[-1])
        assert result is not None
        assert result["histogram"] == 0.2
        assert result["macd"] == 1.0
        assert result["signal"] == 0.8

    def test_bullish_cross(self) -> None:
        df = pd.DataFrame(
            {
                "macd_macd": [-0.5, 0.5],
                "macd_signal": [0.0, 0.0],
                "macd_histogram": [-0.5, 0.5],
            }
        )
        result = _derive_macd_state(df, df.iloc[-1])
        assert result is not None
        assert result["cross"] == "bullish"

    def test_bearish_cross(self) -> None:
        df = pd.DataFrame(
            {
                "macd_macd": [0.5, -0.5],
                "macd_signal": [0.0, 0.0],
                "macd_histogram": [0.5, -0.5],
            }
        )
        result = _derive_macd_state(df, df.iloc[-1])
        assert result is not None
        assert result["cross"] == "bearish"

    def test_missing_histogram_returns_none(self) -> None:
        df = pd.DataFrame({"macd_macd": [1.0], "macd_signal": [0.5]})
        assert _derive_macd_state(df, df.iloc[0]) is None


# =============================================================================
# _derive_supertrend_state
# =============================================================================


class TestDeriveSuperTrendState:
    def test_direction_column(self) -> None:
        df = pd.DataFrame({"supertrend_direction": ["Bullish"], "close": [100.0]})
        result = _derive_supertrend_state(df, df.iloc[0])
        assert result is not None
        assert result["direction"] == "bullish"

    def test_infers_from_price(self) -> None:
        df = pd.DataFrame({"supertrend_supertrend": [95.0], "close": [100.0]})
        result = _derive_supertrend_state(df, df.iloc[0])
        assert result is not None
        assert result["direction"] == "bullish"

    def test_bearish_inferred(self) -> None:
        df = pd.DataFrame({"supertrend_supertrend": [105.0], "close": [100.0]})
        result = _derive_supertrend_state(df, df.iloc[0])
        assert result is not None
        assert result["direction"] == "bearish"

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        assert _derive_supertrend_state(df, df.iloc[0]) is None


# =============================================================================
# _derive_bollinger_state
# =============================================================================


class TestDeriveBollingerState:
    def test_below_lower(self) -> None:
        df = pd.DataFrame(
            {
                "bollinger_bb_upper": [110.0],
                "bollinger_bb_lower": [90.0],
                "close": [85.0],
            }
        )
        result = _derive_bollinger_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "below_lower"

    def test_above_upper(self) -> None:
        df = pd.DataFrame(
            {
                "bollinger_bb_upper": [110.0],
                "bollinger_bb_lower": [90.0],
                "close": [115.0],
            }
        )
        result = _derive_bollinger_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "above_upper"

    def test_middle(self) -> None:
        df = pd.DataFrame(
            {
                "bollinger_bb_upper": [110.0],
                "bollinger_bb_lower": [90.0],
                "close": [100.0],
            }
        )
        result = _derive_bollinger_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "middle"

    def test_at_boundary_lower(self) -> None:
        """Close == lower band is below_lower."""
        df = pd.DataFrame(
            {
                "bollinger_bb_upper": [110.0],
                "bollinger_bb_lower": [90.0],
                "close": [90.0],
            }
        )
        result = _derive_bollinger_state(df, df.iloc[0])
        assert result is not None
        assert result["zone"] == "below_lower"

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        assert _derive_bollinger_state(df, df.iloc[0]) is None


# =============================================================================
# _derive_kdj_state
# =============================================================================


class TestDeriveKDJState:
    def test_oversold(self) -> None:
        df = pd.DataFrame({"kdj_k": [15.0, 15.0], "kdj_d": [18.0, 18.0]})
        result = _derive_kdj_state(df, df.iloc[-1])
        assert result is not None
        assert result["zone"] == "oversold"

    def test_overbought(self) -> None:
        df = pd.DataFrame({"kdj_k": [85.0, 85.0], "kdj_d": [82.0, 82.0]})
        result = _derive_kdj_state(df, df.iloc[-1])
        assert result is not None
        assert result["zone"] == "overbought"

    def test_neutral(self) -> None:
        df = pd.DataFrame({"kdj_k": [50.0, 50.0], "kdj_d": [48.0, 48.0]})
        result = _derive_kdj_state(df, df.iloc[-1])
        assert result is not None
        assert result["zone"] == "neutral"

    def test_cross_detection(self) -> None:
        df = pd.DataFrame({"kdj_k": [40.0, 55.0], "kdj_d": [50.0, 50.0]})
        result = _derive_kdj_state(df, df.iloc[-1])
        assert result is not None
        assert result["cross"] == "bullish"


# =============================================================================
# _derive_adx_state
# =============================================================================


class TestDeriveADXState:
    def test_basic_adx(self) -> None:
        df = pd.DataFrame({"adx_adx": [25.0], "adx_di_plus": [30.0], "adx_di_minus": [20.0]})
        result = _derive_adx_state(df, df.iloc[0])
        assert result is not None
        assert result["adx"] == 25.0
        assert result["di_plus"] == 30.0
        assert result["di_minus"] == 20.0

    def test_missing_di_columns(self) -> None:
        df = pd.DataFrame({"adx_adx": [25.0]})
        assert _derive_adx_state(df, df.iloc[0]) is None


# =============================================================================
# derive_indicator_states (integration)
# =============================================================================


class TestDeriveIndicatorStates:
    def test_empty_df(self) -> None:
        df = pd.DataFrame()
        assert derive_indicator_states(df) == {}

    def test_all_indicators(self, sample_ohlcv_df_with_indicators: pd.DataFrame) -> None:
        states = derive_indicator_states(sample_ohlcv_df_with_indicators)
        assert "rsi" in states
        assert "macd" in states
        assert "supertrend" in states
        assert "bollinger" in states
        assert "kdj" in states
        assert "adx" in states


# =============================================================================
# _get_indicator_direction (parametrized)
# =============================================================================


class TestGetIndicatorDirection:
    @pytest.mark.parametrize(
        "indicator, state, expected",
        [
            # Explicit direction
            ("supertrend", {"direction": "bullish"}, "bullish"),
            ("supertrend", {"direction": "bearish"}, "bearish"),
            ("supertrend", {"direction": "up"}, "bullish"),
            ("supertrend", {"direction": "down"}, "bearish"),
            ("any", {"direction": "long"}, "bullish"),
            ("any", {"direction": "short"}, "bearish"),
            # Cross field
            ("kdj", {"cross": "bullish", "zone": "neutral"}, "bullish"),
            ("kdj", {"cross": "bearish", "zone": "neutral"}, "bearish"),
            ("kdj", {"cross": "neutral", "zone": "neutral"}, "neutral"),
            # Zone
            ("rsi", {"zone": "oversold", "value": 25}, "bullish"),
            ("rsi", {"zone": "overbought", "value": 75}, "bearish"),
            ("rsi", {"zone": "neutral", "value": 50}, "neutral"),
            ("bollinger", {"zone": "below_lower"}, "bullish"),
            ("bollinger", {"zone": "above_upper"}, "bearish"),
            ("bollinger", {"zone": "middle"}, "neutral"),
            # MACD histogram
            ("macd", {"histogram": 0.5}, "bullish"),
            ("macd", {"histogram": -0.5}, "bearish"),
            ("macd", {"histogram": 0.0}, "neutral"),
            # ADX DI comparison
            ("adx", {"di_plus": 30, "di_minus": 20}, "bullish"),
            ("adx", {"di_plus": 20, "di_minus": 30}, "bearish"),
            ("adx", {"di_plus": 25, "di_minus": 25}, "neutral"),
            # Unknown indicator with no usable fields
            ("unknown", {"foo": "bar"}, "neutral"),
        ],
    )
    def test_direction(self, indicator: str, state: Dict[str, Any], expected: str) -> None:
        assert _get_indicator_direction(indicator, state) == expected


# =============================================================================
# calculate_confluence
# =============================================================================


class TestCalculateConfluence:
    def test_returns_scores_for_each_key(
        self,
        sample_data_dict_with_indicators: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        scores = calculate_confluence(sample_data_dict_with_indicators)
        assert "AAPL_1d" in scores
        assert "SPY_1d" in scores

    def test_score_has_expected_fields(
        self,
        sample_data_dict_with_indicators: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        scores = calculate_confluence(sample_data_dict_with_indicators)
        score = scores["AAPL_1d"]
        assert hasattr(score, "alignment_score")
        assert hasattr(score, "bullish_count")
        assert hasattr(score, "bearish_count")

    def test_empty_data(self) -> None:
        scores = calculate_confluence({})
        assert scores == {}


# =============================================================================
# calculate_mtf_confluence
# =============================================================================


class TestCalculateMTFConfluence:
    def test_multi_timeframe(
        self,
        sample_data_dict_multi_tf: Dict[Tuple[str, str], pd.DataFrame],
    ) -> None:
        result = calculate_mtf_confluence(sample_data_dict_multi_tf, "AAPL", ("1h", "4h", "1d"))
        assert "timeframes" in result
        assert len(result["timeframes"]) == 3
        assert "alignment_score" in result
        assert "primary_direction" in result
        assert "confidence" in result
        assert "by_timeframe" in result

    def test_insufficient_timeframes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"rsi_rsi": [50.0] * 10, "close": [100.0] * 10}, index=dates)
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): df}
        result = calculate_mtf_confluence(data, "AAPL", ("1h", "4h", "1d"))
        assert result["aligned"] is False
        assert "Insufficient" in result["message"]

    def test_all_aligned_bullish(self) -> None:
        """All timeframes bullish -> aligned=True, confidence=1.0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "rsi_rsi": [25.0] * 10,  # oversold -> bullish
                "macd_macd": [1.0] * 10,
                "macd_signal": [0.5] * 10,
                "macd_histogram": [0.5] * 10,
                "close": [100.0] * 10,
            },
            index=dates,
        )
        data: Dict[Tuple[str, str], pd.DataFrame] = {
            ("AAPL", "1h"): df.copy(),
            ("AAPL", "4h"): df.copy(),
            ("AAPL", "1d"): df.copy(),
        }
        result = calculate_mtf_confluence(data, "AAPL", ("1h", "4h", "1d"))
        assert result["aligned"] is True
        assert result["confidence"] == 1.0
        assert result["alignment_score"] > 0
