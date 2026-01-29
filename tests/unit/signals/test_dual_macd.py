"""
Tests for DualMACD v1.0 indicator.

Covers:
1. Contract: output contains all required state keys
2. Logic: Synthetic series triggering DIP_BUY and RALLY_SELL
3. Isolation: CompositeScore payload has zero macd_ keys
4. No-lookahead: Rolling percentile uses only past window
5. Behavioral consistency: DETERIORATING trend state
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dual_macd_indicator():
    """Get DualMACD indicator instance."""
    from src.domain.signals.indicators.momentum.dual_macd import DualMACDIndicator

    return DualMACDIndicator()


@pytest.fixture
def long_ohlcv_data() -> pd.DataFrame:
    """Generate 300-bar OHLCV data for full warmup."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    base_price = 100.0
    returns = np.random.randn(n) * 0.01
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame(
        {"open": (high + low) / 2, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestDualMACDContract:
    """Output contains all required keys from spec."""

    def test_state_keys_present(
        self, dual_macd_indicator: Any, long_ohlcv_data: pd.DataFrame
    ) -> None:
        """State dict must contain all 8 required keys."""
        result = dual_macd_indicator.calculate(
            long_ohlcv_data[["close"]], dual_macd_indicator.default_params
        )
        current = result.iloc[-1]
        previous = result.iloc[-2]
        state = dual_macd_indicator._get_state(
            current, previous, dual_macd_indicator.default_params
        )

        required_keys = {
            "slow_histogram",
            "fast_histogram",
            "slow_hist_delta",
            "fast_hist_delta",
            "trend_state",
            "tactical_signal",
            "momentum_balance",
            "confidence",
        }
        assert required_keys == set(
            state.keys()
        ), f"Missing keys: {required_keys - set(state.keys())}"

    def test_no_none_values_in_state(
        self, dual_macd_indicator: Any, long_ohlcv_data: pd.DataFrame
    ) -> None:
        """No state values should be None after warmup."""
        result = dual_macd_indicator.calculate(
            long_ohlcv_data[["close"]], dual_macd_indicator.default_params
        )
        current = result.iloc[-1]
        previous = result.iloc[-2]
        state = dual_macd_indicator._get_state(
            current, previous, dual_macd_indicator.default_params
        )

        for key, value in state.items():
            assert value is not None, f"State[{key}] is None"

    def test_trend_state_enum_values(
        self, dual_macd_indicator: Any, long_ohlcv_data: pd.DataFrame
    ) -> None:
        """trend_state must be one of the 4 valid values."""
        result = dual_macd_indicator.calculate(
            long_ohlcv_data[["close"]], dual_macd_indicator.default_params
        )
        current = result.iloc[-1]
        state = dual_macd_indicator._get_state(current, None, dual_macd_indicator.default_params)
        valid = {"BULLISH", "BEARISH", "IMPROVING", "DETERIORATING"}
        assert state["trend_state"] in valid

    def test_tactical_signal_enum_values(
        self, dual_macd_indicator: Any, long_ohlcv_data: pd.DataFrame
    ) -> None:
        """tactical_signal must be one of 3 valid values."""
        result = dual_macd_indicator.calculate(
            long_ohlcv_data[["close"]], dual_macd_indicator.default_params
        )
        current = result.iloc[-1]
        state = dual_macd_indicator._get_state(current, None, dual_macd_indicator.default_params)
        valid = {"DIP_BUY", "RALLY_SELL", "NONE"}
        assert state["tactical_signal"] in valid

    def test_confidence_range(
        self, dual_macd_indicator: Any, long_ohlcv_data: pd.DataFrame
    ) -> None:
        """Confidence must be in [0, 1]."""
        result = dual_macd_indicator.calculate(
            long_ohlcv_data[["close"]], dual_macd_indicator.default_params
        )
        current = result.iloc[-1]
        state = dual_macd_indicator._get_state(current, None, dual_macd_indicator.default_params)
        assert 0.0 <= state["confidence"] <= 1.0


class TestDualMACDLogic:
    """Synthetic series triggering specific signals."""

    def test_dip_buy_conditions(self, dual_macd_indicator: Any) -> None:
        """DIP_BUY fires when H_slow>0, H_fast<0, |ΔH_fast|>|ΔH_slow|, ΔH_fast>=0."""
        # Directly construct a row satisfying DIP_BUY conditions
        row = pd.Series(
            {
                "slow_histogram": 5.0,  # H_slow > 0
                "fast_histogram": -2.0,  # H_fast < 0
                "slow_hist_delta": 0.1,  # small ΔH_slow
                "fast_hist_delta": 0.5,  # |0.5| > |0.1| and >= 0
                "fast_hist_delta2": 0.3,  # positive curvature
                "slow_hist_norm": 0.5,
                "fast_hist_norm": 0.5,
            }
        )
        state = dual_macd_indicator._get_state(row, None, dual_macd_indicator.default_params)
        assert state["tactical_signal"] == "DIP_BUY"
        assert state["confidence"] > 0.0

    def test_rally_sell_conditions(self, dual_macd_indicator: Any) -> None:
        """RALLY_SELL fires when H_slow<0, H_fast>0, |ΔH_fast|>|ΔH_slow|, ΔH_fast<=0."""
        row = pd.Series(
            {
                "slow_histogram": -5.0,  # H_slow < 0
                "fast_histogram": 2.0,  # H_fast > 0
                "slow_hist_delta": -0.1,  # small ΔH_slow
                "fast_hist_delta": -0.5,  # |0.5| > |0.1| and <= 0
                "fast_hist_delta2": -0.3,  # negative curvature
                "slow_hist_norm": 0.5,
                "fast_hist_norm": 0.5,
            }
        )
        state = dual_macd_indicator._get_state(row, None, dual_macd_indicator.default_params)
        assert state["tactical_signal"] == "RALLY_SELL"
        assert state["confidence"] > 0.0

    def test_deteriorating_state(self, dual_macd_indicator: Any) -> None:
        """DETERIORATING fires when H_slow>0 and ΔH_slow<0."""
        row = pd.Series(
            {
                "slow_histogram": 3.0,  # H_slow > 0
                "fast_histogram": -1.0,
                "slow_hist_delta": -0.5,  # ΔH_slow < 0 → deteriorating
                "fast_hist_delta": 0.0,
                "fast_hist_delta2": 0.0,
                "slow_hist_norm": 0.5,
                "fast_hist_norm": 0.5,
            }
        )
        state = dual_macd_indicator._get_state(row, None, dual_macd_indicator.default_params)
        assert state["trend_state"] == "DETERIORATING"

    def test_improving_state(self, dual_macd_indicator: Any) -> None:
        """IMPROVING fires when H_slow<0 and ΔH_slow>0."""
        row = pd.Series(
            {
                "slow_histogram": -3.0,
                "fast_histogram": 1.0,
                "slow_hist_delta": 0.5,  # ΔH_slow > 0 → improving
                "fast_hist_delta": 0.0,
                "fast_hist_delta2": 0.0,
                "slow_hist_norm": 0.5,
                "fast_hist_norm": 0.5,
            }
        )
        state = dual_macd_indicator._get_state(row, None, dual_macd_indicator.default_params)
        assert state["trend_state"] == "IMPROVING"


class TestCompositeScoreIsolation:
    """CompositeScore should have zero macd_ keys."""

    def test_no_macd_in_composite_weights(self) -> None:
        """CompositeWeights should not contain macd fields."""
        from src.domain.signals.indicators.regime.composite_scorer import CompositeWeights

        w = CompositeWeights()
        d = w.to_dict()
        macd_keys = [k for k in d if "macd" in k]
        assert macd_keys == [], f"Found macd keys in CompositeWeights: {macd_keys}"

    def test_no_macd_in_normalized_factors(self) -> None:
        """NormalizedFactors should not have macd fields."""
        import dataclasses

        from src.domain.signals.indicators.regime.factor_normalizer import NormalizedFactors

        field_names = [f.name for f in dataclasses.fields(NormalizedFactors)]
        macd_fields = [f for f in field_names if "macd" in f]
        assert macd_fields == [], f"Found macd fields in NormalizedFactors: {macd_fields}"

    def test_composite_score_output_no_macd(self) -> None:
        """score_and_classify output should have no macd_ columns."""
        from src.domain.signals.indicators.regime.composite_scorer import CompositeRegimeScorer

        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        volume = np.random.randint(1000, 10000, n).astype(float)
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="1d"),
        )

        scorer = CompositeRegimeScorer()
        result = scorer.score_and_classify(df)
        macd_cols = [c for c in result.columns if "macd" in c]
        assert macd_cols == [], f"Found macd columns in output: {macd_cols}"


class TestNoLookahead:
    """Rolling percentile uses only past window."""

    def test_rolling_percentile_causal(self, dual_macd_indicator: Any) -> None:
        """Modifying future data should not change past percentile values."""
        n = 300
        close_a = 100 + np.cumsum(np.random.RandomState(42).randn(n) * 0.5)
        close_b = close_a.copy()
        close_b[250:] = close_b[250:] + 50  # Spike future data

        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        df_a = pd.DataFrame({"close": close_a}, index=dates)
        df_b = pd.DataFrame({"close": close_b}, index=dates)

        result_a = dual_macd_indicator.calculate(df_a, dual_macd_indicator.default_params)
        result_b = dual_macd_indicator.calculate(df_b, dual_macd_indicator.default_params)

        # Check that slow_hist_norm values before bar 250 are identical
        norm_a = result_a["slow_hist_norm"].iloc[:250].dropna()
        norm_b = result_b["slow_hist_norm"].iloc[:250].dropna()
        if len(norm_a) > 0:
            np.testing.assert_array_equal(norm_a.values, norm_b.values)


class TestWarmup:
    """Warmup period is correct."""

    def test_warmup_value(self, dual_macd_indicator: Any) -> None:
        """Warmup should be 255 = max(89, 252) + 3."""
        assert dual_macd_indicator.warmup_periods == 255
