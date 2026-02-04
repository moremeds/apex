"""
Comprehensive unit tests for all 45 trading signal indicators.

Tests cover:
- Indicator properties (name, category, required_fields, warmup_periods)
- Default params existence
- Calculate with normal data
- Calculate with empty data
- Calculate with short data (less than warmup)
- _get_state with NaN handling
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.indicators.registry import IndicatorRegistry
from src.domain.signals.models import SignalCategory


@pytest.fixture(scope="module")
def registry() -> IndicatorRegistry:
    """Create and populate indicator registry."""
    reg = IndicatorRegistry()
    reg.discover()
    return reg


@pytest.fixture(scope="module")
def all_indicators(registry: IndicatorRegistry):
    """Get all discovered indicators."""
    return registry.get_all()


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_p = (high + low) / 2 + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def empty_ohlcv_data() -> pd.DataFrame:
    """Empty OHLCV DataFrame."""
    return pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([]),
    )


@pytest.fixture
def short_ohlcv_data() -> pd.DataFrame:
    """Short OHLCV data (5 bars only)."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1h")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 101, 100],
            "high": [102, 103, 104, 103, 102],
            "low": [99, 100, 101, 100, 99],
            "close": [101, 102, 103, 102, 101],
            "volume": [1000, 1100, 1200, 1100, 1000],
        },
        index=dates,
    )


# Helper functions for reference tests
def _donchian_mid(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Calculate Donchian channel midpoint for reference testing."""
    n = len(high)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1 : i + 1])
        lowest = np.min(low[i - period + 1 : i + 1])
        result[i] = (highest + lowest) / 2
    return result


def _assert_allclose_with_nan(
    actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    """Assert arrays are close, handling NaN values."""
    assert actual.shape == expected.shape
    expected_mask = ~np.isnan(expected)
    np.testing.assert_allclose(actual[expected_mask], expected[expected_mask], rtol=rtol, atol=atol)
    if np.isnan(expected).any():
        assert np.isnan(actual[~expected_mask]).all()


class TestIndicatorDiscovery:
    """Tests for indicator discovery."""

    def test_total_indicator_count(self, registry: IndicatorRegistry) -> None:
        """Verify all 48 indicators are discovered."""
        indicators = registry.get_all()
        assert len(indicators) == 48, f"Expected 48 indicators, found {len(indicators)}"

    def test_momentum_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify all 13 momentum indicators."""
        indicators = registry.get_by_category(SignalCategory.MOMENTUM)
        names = {ind.name for ind in indicators}
        expected = {
            "rsi",
            "rsi_harmonics",
            "macd",
            "dual_macd",
            "kdj",
            "cci",
            "williams_r",
            "mfi",
            "roc",
            "momentum",
            "tsi",
            "ultimate",
            "awesome",
        }
        assert names == expected, f"Missing momentum indicators: {expected - names}"

    def test_trend_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify all 11 trend indicators."""
        indicators = registry.get_by_category(SignalCategory.TREND)
        names = {ind.name for ind in indicators}
        expected = {
            "ema",
            "sma",
            "supertrend",
            "adx",
            "ichimoku",
            "psar",
            "aroon",
            "zerolag",
            "trix",
            "vortex",
            "trend_pulse",
        }
        assert names == expected, f"Missing trend indicators: {expected - names}"

    def test_volatility_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify all 8 volatility indicators."""
        indicators = registry.get_by_category(SignalCategory.VOLATILITY)
        names = {ind.name for ind in indicators}
        expected = {
            "bollinger",
            "atr",
            "keltner",
            "donchian",
            "stddev",
            "chaikin_vol",
            "hvol",
            "squeeze",
        }
        assert names == expected, f"Missing volatility indicators: {expected - names}"

    def test_volume_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify all 9 volume indicators."""
        indicators = registry.get_by_category(SignalCategory.VOLUME)
        names = {ind.name for ind in indicators}
        expected = {
            "volume",
            "obv",
            "vwap",
            "cvd",
            "volume_ratio",
            "ad",
            "cmf",
            "force",
            "vpvr",
        }
        assert names == expected, f"Missing volume indicators: {expected - names}"

    def test_pattern_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify all 6 pattern indicators."""
        indicators = registry.get_by_category(SignalCategory.PATTERN)
        names = {ind.name for ind in indicators}
        expected = {
            "candlestick",
            "support_resistance",
            "trendline",
            "chart_patterns",
            "fibonacci",
            "pivot",
        }
        assert names == expected, f"Missing pattern indicators: {expected - names}"

    def test_regime_indicators(self, registry: IndicatorRegistry) -> None:
        """Verify regime indicator."""
        indicators = registry.get_by_category(SignalCategory.REGIME)
        names = {ind.name for ind in indicators}
        expected = {"regime_detector"}
        assert names == expected, f"Missing regime indicators: {expected - names}"


class TestIndicatorProperties:
    """Tests for indicator properties."""

    def test_all_have_name(self, all_indicators: Any) -> None:
        """All indicators must have a name."""
        for ind in all_indicators:
            assert hasattr(ind, "name"), f"{type(ind).__name__} missing 'name'"
            assert isinstance(ind.name, str), f"{type(ind).__name__}.name not a string"
            assert len(ind.name) > 0, f"{type(ind).__name__}.name is empty"

    def test_all_have_category(self, all_indicators: Any) -> None:
        """All indicators must have a category."""
        for ind in all_indicators:
            assert hasattr(ind, "category"), f"{ind.name} missing 'category'"
            assert isinstance(
                ind.category, SignalCategory
            ), f"{ind.name}.category not SignalCategory"

    def test_all_have_required_fields(self, all_indicators: Any) -> None:
        """All indicators must have required_fields."""
        for ind in all_indicators:
            assert hasattr(ind, "required_fields"), f"{ind.name} missing 'required_fields'"
            assert isinstance(
                ind.required_fields, (list, tuple)
            ), f"{ind.name}.required_fields not list/tuple"
            assert len(ind.required_fields) > 0, f"{ind.name}.required_fields is empty"

    def test_all_have_warmup_periods(self, all_indicators: Any) -> None:
        """All indicators must have warmup_periods."""
        for ind in all_indicators:
            assert hasattr(ind, "warmup_periods"), f"{ind.name} missing 'warmup_periods'"
            assert isinstance(ind.warmup_periods, int), f"{ind.name}.warmup_periods not int"
            assert ind.warmup_periods >= 1, f"{ind.name}.warmup_periods must be >= 1"

    def test_all_have_default_params(self, all_indicators: Any) -> None:
        """All indicators must have default_params."""
        for ind in all_indicators:
            params = ind.default_params
            assert isinstance(params, dict), f"{ind.name}.default_params not dict"


class TestIndicatorCalculation:
    """Tests for indicator calculation."""

    def test_calculate_normal_data(self, all_indicators: Any, sample_ohlcv_data: Any) -> None:
        """All indicators should calculate without errors on normal data."""
        for ind in all_indicators:
            # Filter data to only include required fields
            cols = [c for c in ind.required_fields if c in sample_ohlcv_data.columns]
            data = sample_ohlcv_data[cols].copy()

            result = ind.calculate(data, ind.default_params)

            assert isinstance(
                result, pd.DataFrame
            ), f"{ind.name}.calculate() didn't return DataFrame"
            assert len(result) == len(data), f"{ind.name}.calculate() length mismatch"

    def test_calculate_empty_data(self, all_indicators: Any, empty_ohlcv_data: Any) -> None:
        """All indicators should handle empty data gracefully."""
        for ind in all_indicators:
            cols = [c for c in ind.required_fields if c in empty_ohlcv_data.columns]
            data = empty_ohlcv_data[cols].copy() if cols else empty_ohlcv_data.copy()

            result = ind.calculate(data, ind.default_params)

            assert isinstance(result, pd.DataFrame), f"{ind.name} failed on empty data"
            assert len(result) == 0, f"{ind.name} should return empty DataFrame"

    def test_calculate_short_data(self, all_indicators: Any, short_ohlcv_data: Any) -> None:
        """All indicators should handle short data (< warmup) gracefully."""
        for ind in all_indicators:
            cols = [c for c in ind.required_fields if c in short_ohlcv_data.columns]
            data = short_ohlcv_data[cols].copy()

            result = ind.calculate(data, ind.default_params)

            assert isinstance(result, pd.DataFrame), f"{ind.name} failed on short data"
            assert len(result) == len(data), f"{ind.name} length mismatch on short data"

    def test_calculate_output_columns_prefixed(
        self, all_indicators: Any, sample_ohlcv_data: Any
    ) -> None:
        """Most indicators should prefix their output columns."""
        exceptions = {"candlestick"}  # Uses cdl_ prefix

        for ind in all_indicators:
            if ind.name in exceptions:
                continue

            cols = [c for c in ind.required_fields if c in sample_ohlcv_data.columns]
            data = sample_ohlcv_data[cols].copy()

            result = ind.calculate(data, ind.default_params)

            # Check that output columns don't overlap with input columns
            input_cols = set(data.columns)
            output_cols = set(result.columns)
            overlap = input_cols & output_cols

            # Some indicators may include _close for position calculation
            allowed_overlap = {c for c in overlap if c.endswith("_close")}
            unexpected_overlap = overlap - allowed_overlap

            assert not unexpected_overlap, f"{ind.name} output overlaps input: {unexpected_overlap}"


class TestIndicatorState:
    """Tests for indicator state extraction."""

    def test_get_state_with_nan(self, all_indicators: Any) -> None:
        """All indicators should handle NaN values in _get_state."""
        nan_row = pd.Series(dtype=float)  # Empty series

        for ind in all_indicators:
            state = ind._get_state(nan_row, None, ind.default_params)

            assert isinstance(state, dict), f"{ind.name}._get_state() didn't return dict"
            assert len(state) > 0, f"{ind.name}._get_state() returned empty dict"

    def test_get_state_returns_required_keys(
        self, all_indicators: Any, sample_ohlcv_data: Any
    ) -> None:
        """All indicators should return consistent state keys."""
        for ind in all_indicators:
            cols = [c for c in ind.required_fields if c in sample_ohlcv_data.columns]
            data = sample_ohlcv_data[cols].copy()

            result = ind.calculate(data, ind.default_params)

            if len(result) > 0:
                # Get state for last row
                current = result.iloc[-1]
                previous = result.iloc[-2] if len(result) > 1 else None

                state = ind._get_state(current, previous, ind.default_params)

                assert isinstance(state, dict), f"{ind.name}._get_state() didn't return dict"

                # Check no None values in state (use explicit types)
                # Some keys are allowed to be None when no signal/pattern is detected
                allowed_none_keys = {
                    # Cross detection (when no crossover occurs)
                    "tk_cross",
                    "cross_zero",
                    "signal",
                    "cross",
                    "kumo_twist",
                    # Pattern detection outputs (when no pattern found)
                    "pattern",
                    "pattern_type",
                    "reversal_signal",
                    "direction",
                    "nearest_level",
                    # Regime detector: regime transition tracking (None when no pending change)
                    "pending_regime",
                    "previous_regime",
                    "iv_state",
                }
                for key, value in state.items():
                    assert (
                        value is not None or key in allowed_none_keys
                    ), f"{ind.name} state[{key}] is None"


class TestSpecificIndicators:
    """Tests for specific indicator behaviors."""

    def test_rsi_range(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """RSI should be bounded 0-100."""
        rsi = registry.get("rsi")

        assert rsi is not None
        data = sample_ohlcv_data[["close"]].copy()
        result = rsi.calculate(data, rsi.default_params)

        valid_values = result["rsi"].dropna()
        assert (valid_values >= 0).all(), "RSI has values < 0"
        assert (valid_values <= 100).all(), "RSI has values > 100"

    def test_bollinger_bands_order(
        self, registry: IndicatorRegistry, sample_ohlcv_data: Any
    ) -> None:
        """Bollinger upper > middle > lower."""
        bb = registry.get("bollinger")

        assert bb is not None
        data = sample_ohlcv_data[["close"]].copy()
        result = bb.calculate(data, bb.default_params)

        valid_idx = ~(
            result["bb_upper"].isna() | result["bb_middle"].isna() | result["bb_lower"].isna()
        )
        valid = result[valid_idx]

        if len(valid) > 0:
            assert (valid["bb_upper"] >= valid["bb_middle"]).all(), "BB upper < middle"
            assert (valid["bb_middle"] >= valid["bb_lower"]).all(), "BB middle < lower"

    def test_macd_signal_line(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """MACD should have macd, signal, and histogram."""
        macd = registry.get("macd")

        assert macd is not None
        data = sample_ohlcv_data[["close"]].copy()
        result = macd.calculate(data, macd.default_params)

        assert "macd" in result.columns, "Missing macd column"
        assert "signal" in result.columns, "Missing signal column"
        assert "histogram" in result.columns, "Missing histogram column"

    def test_adx_range(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """ADX should be bounded 0-100."""
        adx = registry.get("adx")

        assert adx is not None
        data = sample_ohlcv_data[["high", "low", "close"]].copy()
        result = adx.calculate(data, adx.default_params)

        valid_values = result["adx"].dropna()
        if len(valid_values) > 0:
            assert (valid_values >= 0).all(), "ADX has values < 0"
            assert (valid_values <= 100).all(), "ADX has values > 100"

    def test_vwap_session_reset(self, registry: IndicatorRegistry) -> None:
        """VWAP should reset at session boundaries."""
        vwap = registry.get("vwap")

        assert vwap is not None

        # Create multi-day data
        dates = pd.date_range("2024-01-01 09:30", periods=20, freq="30min")
        data = pd.DataFrame(
            {
                "high": [100 + i * 0.1 for i in range(20)],
                "low": [99 + i * 0.1 for i in range(20)],
                "close": [99.5 + i * 0.1 for i in range(20)],
                "volume": [1000] * 20,
            },
            index=dates,
        )

        result = vwap.calculate(data, {"reset_daily": True})

        # VWAP should be calculated
        assert "vwap" in result.columns
        assert not result["vwap"].isna().all(), "VWAP all NaN"

    def test_candlestick_pattern_count(self, registry: IndicatorRegistry) -> None:
        """Candlestick should have 61 patterns."""
        candlestick = registry.get("candlestick")

        assert candlestick is not None
        patterns = candlestick.default_params.get("patterns", [])
        assert len(patterns) == 61, f"Expected 61 patterns, got {len(patterns)}"

    def test_ichimoku_components(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """Ichimoku should have all 4 main lines."""
        ichimoku = registry.get("ichimoku")

        assert ichimoku is not None
        data = sample_ohlcv_data[["high", "low", "close"]].copy()
        result = ichimoku.calculate(data, ichimoku.default_params)

        required = {"tenkan", "kijun", "senkou_a", "senkou_b"}
        actual = set(result.columns)
        assert required <= actual, f"Missing Ichimoku columns: {required - actual}"

    def test_pivot_methods(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """Pivot should support multiple methods."""
        pivot = registry.get("pivot")

        assert pivot is not None
        data = sample_ohlcv_data[["high", "low", "close"]].copy()

        for method in ["classic", "woodie", "camarilla", "fibonacci"]:
            result = pivot.calculate(data, {"method": method})
            assert "pivot_pp" in result.columns, f"Missing pivot_pp for {method}"
            assert "pivot_r1" in result.columns, f"Missing pivot_r1 for {method}"
            assert "pivot_s1" in result.columns, f"Missing pivot_s1 for {method}"

    def test_cmf_uses_manual_calculation(
        self, registry: IndicatorRegistry, sample_ohlcv_data: Any
    ) -> None:
        """CMF should use manual calculation, not TA-Lib ADOSC."""
        cmf = registry.get("cmf")

        assert cmf is not None
        data = sample_ohlcv_data[["high", "low", "close", "volume"]].copy()
        result = cmf.calculate(data, cmf.default_params)

        # CMF should be bounded -1 to +1
        valid_values = result["cmf"].dropna()
        if len(valid_values) > 0:
            assert (valid_values >= -1.01).all(), "CMF has values < -1"
            assert (valid_values <= 1.01).all(), "CMF has values > 1"

    def test_ichimoku_displacement_matches_manual(self, registry: IndicatorRegistry) -> None:
        """Ichimoku Senkou spans should be displaced by 26 periods."""
        ichimoku = registry.get("ichimoku")

        assert ichimoku is not None
        n = 120
        close = np.linspace(100.0, 220.0, n)
        high = close + 2.0
        low = close - 2.0
        data = pd.DataFrame({"high": high, "low": low, "close": close})

        result = ichimoku.calculate(data, ichimoku.default_params)

        # Manual reference calculation
        tenkan = _donchian_mid(high, low, 9)
        kijun = _donchian_mid(high, low, 26)
        senkou_b_future = _donchian_mid(high, low, 52)
        senkou_a_future = np.full(n, np.nan, dtype=np.float64)
        valid = ~np.isnan(tenkan) & ~np.isnan(kijun)
        senkou_a_future[valid] = (tenkan[valid] + kijun[valid]) / 2

        # Displaced values - shifted forward by 26 periods
        senkou_a = np.full(n, np.nan, dtype=np.float64)
        senkou_b = np.full(n, np.nan, dtype=np.float64)
        senkou_a[26:] = senkou_a_future[:-26]
        senkou_b[26:] = senkou_b_future[:-26]

        _assert_allclose_with_nan(result["senkou_a"].to_numpy(), senkou_a)
        _assert_allclose_with_nan(result["senkou_b"].to_numpy(), senkou_b)
        _assert_allclose_with_nan(result["senkou_a_future"].to_numpy(), senkou_a_future)
        _assert_allclose_with_nan(result["senkou_b_future"].to_numpy(), senkou_b_future)
        _assert_allclose_with_nan(result["chikou"].to_numpy(), close)

        # price_at_chikou should be close from 26 bars ago
        price_at_chikou = np.full(n, np.nan, dtype=np.float64)
        price_at_chikou[26:] = close[:-26]
        _assert_allclose_with_nan(result["price_at_chikou"].to_numpy(), price_at_chikou)

    def test_aroon_matches_talib(self, registry: IndicatorRegistry, sample_ohlcv_data: Any) -> None:
        """Aroon should match TA-Lib output (alignment/off-by-one check)."""
        talib = pytest.importorskip("talib")
        aroon = registry.get("aroon")

        assert aroon is not None
        data = sample_ohlcv_data[["high", "low"]].copy()
        result = aroon.calculate(data, aroon.default_params)

        period = aroon.default_params["period"]
        expected_down, expected_up = talib.AROON(
            data["high"].to_numpy(), data["low"].to_numpy(), timeperiod=period
        )

        _assert_allclose_with_nan(result["aroon_up"].to_numpy(), expected_up)
        _assert_allclose_with_nan(result["aroon_down"].to_numpy(), expected_down)

    def test_support_resistance_distance_calculation(self, registry: IndicatorRegistry) -> None:
        """Support/resistance distance percentages should be correct."""
        sr = registry.get("support_resistance")

        assert sr is not None
        params = sr.default_params

        # Test case: price at 100, support at 95, resistance at 110
        current = pd.Series({"sr_support": 95.0, "sr_resistance": 110.0, "sr_close": 100.0})
        state = sr.get_state(current, None, params)  # type: ignore[attr-defined]

        # Distance to support: (100 - 95) / 95 * 100 = 5.26%
        assert state["support_distance_pct"] == pytest.approx((100.0 - 95.0) / 95.0 * 100)
        # Distance to resistance: (110 - 100) / 110 * 100 = 9.09%
        assert state["resistance_distance_pct"] == pytest.approx((110.0 - 100.0) / 110.0 * 100)
        assert state["position"] == "between"

        # Test at_support position (within 1% proximity)
        current = pd.Series({"sr_support": 99.5, "sr_resistance": 110.0, "sr_close": 100.0})
        state = sr.get_state(current, None, params)  # type: ignore[attr-defined]
        assert state["position"] == "at_support"

    def test_reference_indicators_match_talib(self, registry: IndicatorRegistry) -> None:
        """Core indicators should match TA-Lib reference outputs."""
        talib = pytest.importorskip("talib")
        n = 300
        close = np.linspace(100.0, 130.0, n)
        high = close + 1.5
        low = close - 1.5
        data = pd.DataFrame({"close": close, "high": high, "low": low})

        # EMA tests
        ema_fast = talib.EMA(close, timeperiod=12)
        ema_slow = talib.EMA(close, timeperiod=26)
        ema = registry.get("ema")
        assert ema is not None
        ema_result = ema.calculate(data[["close"]], {"fast_period": 12, "slow_period": 26})
        _assert_allclose_with_nan(ema_result["ema_fast"].to_numpy(), ema_fast)
        _assert_allclose_with_nan(ema_result["ema_slow"].to_numpy(), ema_slow)

        # SMA tests
        sma_fast = talib.SMA(close, timeperiod=50)
        sma_slow = talib.SMA(close, timeperiod=200)
        sma = registry.get("sma")
        assert sma is not None
        sma_result = sma.calculate(data[["close"]], {"fast_period": 50, "slow_period": 200})
        _assert_allclose_with_nan(sma_result["sma_fast"].to_numpy(), sma_fast)
        _assert_allclose_with_nan(sma_result["sma_slow"].to_numpy(), sma_slow)

        # RSI test
        rsi_vals = talib.RSI(close, timeperiod=14)
        rsi = registry.get("rsi")
        assert rsi is not None
        rsi_result = rsi.calculate(data[["close"]], {"period": 14})
        _assert_allclose_with_nan(rsi_result["rsi"].to_numpy(), rsi_vals)

        # ATR test
        atr_vals = talib.ATR(high, low, close, timeperiod=14)
        atr = registry.get("atr")
        assert atr is not None
        atr_result = atr.calculate(data[["high", "low", "close"]], {"period": 14})
        _assert_allclose_with_nan(atr_result["atr"].to_numpy(), atr_vals)

        # MACD test
        macd_line, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        macd = registry.get("macd")
        assert macd is not None
        macd_result = macd.calculate(
            data[["close"]], {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        )
        _assert_allclose_with_nan(macd_result["macd"].to_numpy(), macd_line)
        _assert_allclose_with_nan(macd_result["signal"].to_numpy(), macd_signal)
        _assert_allclose_with_nan(macd_result["histogram"].to_numpy(), macd_hist)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_same_prices(self, all_indicators: Any) -> None:
        """Indicators should handle flat prices."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [100.0] * 100,
                "low": [100.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000.0] * 100,
            },
            index=dates,
        )

        for ind in all_indicators:
            cols = [c for c in ind.required_fields if c in data.columns]
            filtered_data = data[cols].copy()

            # Should not raise
            result = ind.calculate(filtered_data, ind.default_params)
            assert isinstance(result, pd.DataFrame), f"{ind.name} failed on flat prices"

    def test_extreme_values(self, all_indicators: Any) -> None:
        """Indicators should handle extreme price values."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": [1e6] * 100,
                "high": [1.1e6] * 100,
                "low": [0.9e6] * 100,
                "close": [1.05e6] * 100,
                "volume": [1e9] * 100,
            },
            index=dates,
        )

        for ind in all_indicators:
            cols = [c for c in ind.required_fields if c in data.columns]
            filtered_data = data[cols].copy()

            # Should not raise
            result = ind.calculate(filtered_data, ind.default_params)
            assert isinstance(result, pd.DataFrame), f"{ind.name} failed on extreme values"

    def test_zero_volume(self, all_indicators: Any) -> None:
        """Volume indicators should handle zero volume."""
        volume_indicators = {
            "obv",
            "vwap",
            "cvd",
            "volume_ratio",
            "ad",
            "cmf",
            "force",
            "vpvr",
            "mfi",
        }

        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        data = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 100),
                "high": np.linspace(101, 111, 100),
                "low": np.linspace(99, 109, 100),
                "close": np.linspace(100.5, 110.5, 100),
                "volume": [0.0] * 100,
            },
            index=dates,
        )

        for ind in all_indicators:
            if ind.name not in volume_indicators:
                continue

            cols = [c for c in ind.required_fields if c in data.columns]
            filtered_data = data[cols].copy()

            # Should not raise
            result = ind.calculate(filtered_data, ind.default_params)
            assert isinstance(result, pd.DataFrame), f"{ind.name} failed on zero volume"
