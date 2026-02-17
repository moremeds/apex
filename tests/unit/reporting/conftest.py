"""
Shared fixtures for reporting module tests.

Provides reusable data structures (DataFrames, mock models, signal rules)
used across test_score_history, test_signal_detection, test_confluence_analyzer,
test_summary_builder, test_heatmap, and test_strategy_comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.models import (
    ConditionType,
    SignalCategory,
    SignalDirection,
    SignalPriority,
    SignalRule,
)

# =============================================================================
# OHLCV DataFrames
# =============================================================================


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """100-bar DatetimeIndex DataFrame with OHLCV columns."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(100)],
            "high": [105 + i * 0.5 for i in range(100)],
            "low": [95 + i * 0.5 for i in range(100)],
            "close": [102 + i * 0.5 for i in range(100)],
            "volume": [1_000_000 + i * 10_000 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def sample_ohlcv_df_with_indicators(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame extended with realistic indicator columns."""
    df = sample_ohlcv_df.copy()

    # RSI oscillates between 25 and 75
    df["rsi_rsi"] = 50 + 25 * np.sin(np.linspace(0, 4 * np.pi, 100))

    # MACD with crossovers
    df["macd_macd"] = np.sin(np.linspace(0, 6 * np.pi, 100)) * 2
    df["macd_signal"] = np.sin(np.linspace(0.3, 6 * np.pi + 0.3, 100)) * 1.8
    df["macd_histogram"] = df["macd_macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bollinger_bb_upper"] = df["close"] + 8
    df["bollinger_bb_middle"] = df["close"]
    df["bollinger_bb_lower"] = df["close"] - 8

    # SuperTrend
    df["supertrend_supertrend"] = df["close"] - 3  # Bullish (close > supertrend)
    df["supertrend_direction"] = "bullish"

    # KDJ
    df["kdj_k"] = 50 + 30 * np.sin(np.linspace(0, 3 * np.pi, 100))
    df["kdj_d"] = 50 + 28 * np.sin(np.linspace(0.2, 3 * np.pi + 0.2, 100))

    # ADX
    df["adx_adx"] = 25 + 10 * np.sin(np.linspace(0, 2 * np.pi, 100))
    df["adx_di_plus"] = 30 + 10 * np.sin(np.linspace(0, 3 * np.pi, 100))
    df["adx_di_minus"] = 20 + 10 * np.cos(np.linspace(0, 3 * np.pi, 100))

    return df


# =============================================================================
# Data dictionaries (for summary_builder, confluence)
# =============================================================================


@pytest.fixture
def sample_data_dict(
    sample_ohlcv_df: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Dict mapping (symbol, timeframe) to DataFrame for AAPL+SPY 1d."""
    return {
        ("AAPL", "1d"): sample_ohlcv_df.copy(),
        ("SPY", "1d"): sample_ohlcv_df.copy(),
    }


@pytest.fixture
def sample_data_dict_with_indicators(
    sample_ohlcv_df_with_indicators: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Dict with indicator columns for AAPL+SPY 1d."""
    return {
        ("AAPL", "1d"): sample_ohlcv_df_with_indicators.copy(),
        ("SPY", "1d"): sample_ohlcv_df_with_indicators.copy(),
    }


@pytest.fixture
def sample_data_dict_multi_tf(
    sample_ohlcv_df_with_indicators: pd.DataFrame,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Dict with indicator columns for AAPL across 1h, 4h, 1d timeframes."""
    return {
        ("AAPL", "1h"): sample_ohlcv_df_with_indicators.copy(),
        ("AAPL", "4h"): sample_ohlcv_df_with_indicators.copy(),
        ("AAPL", "1d"): sample_ohlcv_df_with_indicators.copy(),
    }


# =============================================================================
# Mock RegimeOutput
# =============================================================================


@pytest.fixture
def mock_regime_output():
    """Factory returning a MagicMock matching RegimeOutput interface."""

    def _factory(
        final_regime: str = "R0",
        regime_name: str = "Healthy Uptrend",
        confidence: float = 0.85,
        composite_score: float = 72.5,
        regime_changed: bool = False,
        close: float = 150.0,
    ) -> MagicMock:
        mock = MagicMock()
        mock.final_regime.value = final_regime
        mock.confidence = confidence
        mock.composite_score = composite_score
        mock.regime_changed = regime_changed
        mock.quality = MagicMock()
        mock.quality.component_validity = {"close": True}
        mock.quality.component_issues = {}

        mock.to_dict.return_value = {
            "final_regime": final_regime,
            "regime_name": regime_name,
            "confidence": confidence,
            "regime_changed": regime_changed,
            "decision_regime": final_regime,
            "component_states": {"trend_state": "uptrend"},
            "component_values": {"close": close, "ma50_slope": 0.02},
            "derived_metrics": {"atr_pct": 1.5},
            "transition": {"holding_bars": 5},
            "quality": {"bars_used": 100, "component_validity": {"close": True}},
            "rules_fired_decision": ["R0_default"],
            "turning_point": None,
            "data_window": {"start": "2024-01-01", "end": "2024-04-10"},
            "asof_ts": "2024-04-10T16:00:00",
        }
        return mock

    return _factory


# =============================================================================
# Sample ticker summary (for heatmap, strategy_comparison)
# =============================================================================


@pytest.fixture
def sample_ticker_summary() -> Dict[str, Any]:
    """Dict matching _build_ticker_summary output shape."""
    return {
        "symbol": "AAPL",
        "regime": "R0",
        "regime_name": "Healthy Uptrend",
        "confidence": 0.85,
        "daily_change_pct": 1.25,
        "close": 175.50,
        "volume": 45_000_000,
        "composite_score_avg": 72.5,
        "component_states": {"trend_state": "uptrend"},
        "component_values": {"close": 175.50, "ma50_slope": 0.02},
        "signal_count": 5,
        "buy_signal_count": 3,
        "sell_signal_count": 2,
    }


@pytest.fixture
def sample_summary_data(sample_ticker_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Full summary.json structure."""
    spy = sample_ticker_summary.copy()
    spy["symbol"] = "SPY"
    spy["regime"] = "R1"
    spy["composite_score_avg"] = 55.0
    spy["daily_change_pct"] = -0.5

    return {
        "version": "1.0",
        "generated_at": "2024-04-10T16:00:00",
        "symbols": ["AAPL", "SPY"],
        "timeframes": ["1d"],
        "symbol_count": 2,
        "timeframe_count": 1,
        "tickers": [sample_ticker_summary, spy],
        "market": {"benchmarks": {"SPY": {"regime": "R1", "confidence": 0.7}}},
        "confluence": {},
        "rule_frequency": {
            "by_symbol": {"AAPL": 5, "SPY": 3},
            "buy_by_symbol": {"AAPL": 3, "SPY": 1},
            "sell_by_symbol": {"AAPL": 2, "SPY": 2},
            "total_signals": 8,
        },
        "dual_macd": {
            "1d": {
                "alerts": {"dip_buy": ["AAPL"], "rally_sell": []},
                "trends": [
                    {
                        "symbol": "AAPL",
                        "slow_hist_delta": 0.5,
                        "fast_hist_delta": 0.3,
                        "trend_state": "uptrend",
                        "tactical_signal": "DIP_BUY",
                        "confidence": 0.8,
                    }
                ],
            }
        },
    }


# =============================================================================
# Signal Rules
# =============================================================================


@pytest.fixture
def sample_signal_rules() -> List[SignalRule]:
    """List of SignalRules covering all 5 main condition types."""
    return [
        SignalRule(
            name="macd_bullish_cross",
            indicator="macd",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=70,
            priority=SignalPriority.HIGH,
            condition_type=ConditionType.CROSS_UP,
            condition_config={"line_a": "macd", "line_b": "signal"},
            timeframes=("1h", "4h", "1d"),
            message_template="{symbol} MACD bullish crossover",
        ),
        SignalRule(
            name="macd_bearish_cross",
            indicator="macd",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.SELL,
            strength=70,
            priority=SignalPriority.HIGH,
            condition_type=ConditionType.CROSS_DOWN,
            condition_config={"line_a": "macd", "line_b": "signal"},
            timeframes=("1h", "4h", "1d"),
            message_template="{symbol} MACD bearish crossover",
        ),
        SignalRule(
            name="rsi_oversold",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.BUY,
            strength=60,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.THRESHOLD_CROSS_DOWN,
            condition_config={"field": "rsi", "threshold": 30},
            timeframes=("1h", "4h", "1d"),
            message_template="{symbol} RSI oversold at {value:.1f} (< {threshold})",
        ),
        SignalRule(
            name="rsi_overbought",
            indicator="rsi",
            category=SignalCategory.MOMENTUM,
            direction=SignalDirection.SELL,
            strength=60,
            priority=SignalPriority.MEDIUM,
            condition_type=ConditionType.THRESHOLD_CROSS_UP,
            condition_config={"field": "rsi", "threshold": 70},
            timeframes=("1h", "4h", "1d"),
            message_template="{symbol} RSI overbought at {value:.1f} (> {threshold})",
        ),
        SignalRule(
            name="supertrend_flip_bullish",
            indicator="supertrend",
            category=SignalCategory.TREND,
            direction=SignalDirection.BUY,
            strength=75,
            priority=SignalPriority.HIGH,
            condition_type=ConditionType.STATE_CHANGE,
            condition_config={
                "field": "direction",
                "from": ["bearish"],
                "to": ["bullish"],
            },
            timeframes=("1h", "4h", "1d"),
            message_template="{symbol} SuperTrend flipped bullish",
        ),
    ]
