"""
Unit tests for RSI Indicator.

Tests:
- Normal calculation
- Edge cases: empty data, short data, integer dtype
- State extraction
- Zone classification
"""

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.indicators.momentum.rsi import RSIIndicator
from src.domain.signals.models import SignalCategory


class TestRSIIndicator:
    """Tests for RSI indicator."""

    @pytest.fixture
    def rsi(self) -> RSIIndicator:
        """Create RSI indicator instance."""
        return RSIIndicator()

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        return pd.DataFrame(
            {
                "open": close - np.random.rand(50) * 0.5,
                "high": close + np.random.rand(50) * 0.5,
                "low": close - np.random.rand(50) * 0.5,
                "close": close,
                "volume": np.random.randint(1000, 10000, 50),
            },
            index=dates,
        )

    def test_indicator_properties(self, rsi: RSIIndicator) -> None:
        """Test indicator metadata properties."""
        assert rsi.name == "rsi"
        assert rsi.category == SignalCategory.MOMENTUM
        assert rsi.required_fields == ["close"]
        assert rsi.warmup_periods == 15

    def test_default_params(self, rsi: RSIIndicator) -> None:
        """Test default parameters."""
        params = rsi.default_params
        assert params["period"] == 14
        assert params["overbought"] == 70
        assert params["oversold"] == 30

    def test_calculate_normal(self, rsi: RSIIndicator, sample_data: pd.DataFrame) -> None:
        """Test normal RSI calculation."""
        result = rsi.calculate(sample_data, {})

        assert "rsi" in result.columns
        assert len(result) == len(sample_data)
        # First 14 periods should be NaN (warmup)
        assert result["rsi"].iloc[:14].isna().all()
        # Rest should be valid RSI values (0-100)
        valid_rsi = result["rsi"].iloc[14:].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_calculate_empty_data(self, rsi: RSIIndicator) -> None:
        """Test RSI with empty DataFrame."""
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = rsi.calculate(empty_data, {})

        assert "rsi" in result.columns
        assert len(result) == 0

    def test_calculate_short_data(self, rsi: RSIIndicator) -> None:
        """Test RSI with data shorter than period."""
        short_data = pd.DataFrame(
            {
                "close": [100.0, 101.0, 99.0, 100.5, 102.0],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1h"),
        )
        result = rsi.calculate(short_data, {"period": 14})

        assert "rsi" in result.columns
        assert len(result) == 5
        # All should be NaN since data < period
        assert result["rsi"].isna().all()

    def test_calculate_integer_dtype(self, rsi: RSIIndicator) -> None:
        """Test RSI with integer close prices (should not crash)."""
        int_data = pd.DataFrame(
            {
                "close": np.arange(100, 150, dtype=np.int64),
            },
            index=pd.date_range("2024-01-01", periods=50, freq="1h"),
        )
        result = rsi.calculate(int_data, {})

        assert "rsi" in result.columns
        # Should convert to float and calculate
        valid_rsi = result["rsi"].dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.dtype == np.float64

    def test_calculate_exact_period_length(self, rsi: RSIIndicator) -> None:
        """Test RSI with data length exactly equal to period."""
        exact_data = pd.DataFrame(
            {
                "close": np.random.randn(14).cumsum() + 100,
            },
            index=pd.date_range("2024-01-01", periods=14, freq="1h"),
        )
        result = rsi.calculate(exact_data, {"period": 14})

        assert "rsi" in result.columns
        # All should be NaN (need period+1 for first valid RSI)
        assert result["rsi"].isna().all()

    def test_calculate_one_more_than_period(self, rsi: RSIIndicator) -> None:
        """Test RSI with data length = period + 1."""
        data = pd.DataFrame(
            {
                "close": np.random.randn(15).cumsum() + 100,
            },
            index=pd.date_range("2024-01-01", periods=15, freq="1h"),
        )
        result = rsi.calculate(data, {"period": 14})

        assert "rsi" in result.columns
        # Last value should be valid
        assert pd.notna(result["rsi"].iloc[-1])

    def test_get_state_overbought(self, rsi: RSIIndicator) -> None:
        """Test state extraction for overbought condition."""
        current = pd.Series({"rsi": 75.0})
        previous = pd.Series({"rsi": 70.0})

        state = rsi.get_state(current, previous)

        assert state["value"] == 75.0
        assert state["zone"] == "overbought"

    def test_get_state_oversold(self, rsi: RSIIndicator) -> None:
        """Test state extraction for oversold condition."""
        current = pd.Series({"rsi": 25.0})
        previous = pd.Series({"rsi": 30.0})

        state = rsi.get_state(current, previous)

        assert state["value"] == 25.0
        assert state["zone"] == "oversold"

    def test_get_state_neutral(self, rsi: RSIIndicator) -> None:
        """Test state extraction for neutral condition."""
        current = pd.Series({"rsi": 50.0})
        previous = pd.Series({"rsi": 45.0})

        state = rsi.get_state(current, previous)

        assert state["value"] == 50.0
        assert state["zone"] == "neutral"

    def test_get_state_nan_handling(self, rsi: RSIIndicator) -> None:
        """Test state extraction handles NaN values."""
        current = pd.Series({"rsi": np.nan})
        previous = None

        state = rsi.get_state(current, previous)

        # NaN should default to neutral zone
        assert state["zone"] == "neutral"
        assert state["value"] == 50  # Default value

    def test_get_state_boundary_values(self, rsi: RSIIndicator) -> None:
        """Test state at exact boundary values."""
        # Exactly at overbought threshold
        state = rsi.get_state(pd.Series({"rsi": 70.0}), None)
        assert state["zone"] == "overbought"

        # Exactly at oversold threshold
        state = rsi.get_state(pd.Series({"rsi": 30.0}), None)
        assert state["zone"] == "oversold"

        # Just below overbought
        state = rsi.get_state(pd.Series({"rsi": 69.99}), None)
        assert state["zone"] == "neutral"

        # Just above oversold
        state = rsi.get_state(pd.Series({"rsi": 30.01}), None)
        assert state["zone"] == "neutral"

    def test_calculate_custom_params(self, rsi: RSIIndicator, sample_data: pd.DataFrame) -> None:
        """Test RSI with custom parameters."""
        result = rsi.calculate(sample_data, {"period": 7})

        # Should have fewer NaN values with shorter period
        nan_count = result["rsi"].isna().sum()
        assert nan_count == 7  # First 7 values should be NaN

    def test_all_gains_rsi(self, rsi: RSIIndicator) -> None:
        """Test RSI when price only goes up (should approach 100)."""
        rising_data = pd.DataFrame(
            {
                "close": np.arange(100, 150, dtype=float),
            },
            index=pd.date_range("2024-01-01", periods=50, freq="1h"),
        )
        result = rsi.calculate(rising_data, {})

        valid_rsi = result["rsi"].dropna()
        # RSI should be very high with all gains
        assert valid_rsi.iloc[-1] > 90

    def test_all_losses_rsi(self, rsi: RSIIndicator) -> None:
        """Test RSI when price only goes down (should approach 0)."""
        falling_data = pd.DataFrame(
            {
                "close": np.arange(150, 100, -1, dtype=float),
            },
            index=pd.date_range("2024-01-01", periods=50, freq="1h"),
        )
        result = rsi.calculate(falling_data, {})

        valid_rsi = result["rsi"].dropna()
        # RSI should be very low with all losses
        assert valid_rsi.iloc[-1] < 10

    def test_get_state_custom_thresholds(self, rsi: RSIIndicator) -> None:
        """Test state extraction with custom overbought/oversold thresholds."""
        # Test 1: 75 with default (70/30) vs custom (80/20)
        current = pd.Series({"rsi": 75.0})

        # With default thresholds (70/30), 75 is overbought
        state_default = rsi.get_state(current, None)
        assert state_default["zone"] == "overbought"

        # With custom thresholds (80/20), 75 is neutral (75 < 80)
        state_custom = rsi.get_state(current, None, params={"overbought": 80, "oversold": 20})
        assert state_custom["zone"] == "neutral"
        assert state_custom["value"] == 75.0

        # Test 2: 25 with default (30) is oversold, but with custom (20) is neutral
        current_25 = pd.Series({"rsi": 25.0})

        # Default: 25 <= 30, so oversold
        state_default_25 = rsi.get_state(current_25, None)
        assert state_default_25["zone"] == "oversold"

        # Custom: 25 > 20, so neutral
        state_custom_25 = rsi.get_state(current_25, None, params={"oversold": 20})
        assert state_custom_25["zone"] == "neutral"

        # Test 3: 15 should be oversold with both default (30) and custom (20)
        current_15 = pd.Series({"rsi": 15.0})

        state_default_15 = rsi.get_state(current_15, None)
        assert state_default_15["zone"] == "oversold"

        state_custom_15 = rsi.get_state(current_15, None, params={"oversold": 20})
        assert state_custom_15["zone"] == "oversold"
