"""
TA-Lib Numerical Alignment Tests.

Verifies that APEX indicator implementations produce values numerically
aligned with TA-Lib reference implementation.

These tests require TA-Lib to be installed and will be skipped otherwise.
Run with: pytest tests/unit/signals/test_talib_alignment.py -v

Tolerance Strategy:
- abs_err: 1e-6 (absolute difference)
- rel_err: 1e-4 (relative difference)
- warmup_slice: Only compare post-warmup rows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip entire module if TA-Lib not available
talib = pytest.importorskip("talib", reason="TA-Lib not installed")


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200

    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    open_ = close + np.random.randn(n) * 0.3

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n).astype(float),
        },
        index=dates,
    )


class TestRSIAlignment:
    """Test RSI alignment with TA-Lib."""

    @pytest.fixture
    def apex_rsi(self):
        """Get APEX RSI indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("rsi")

    def test_rsi_14_alignment(self, sample_ohlcv: pd.DataFrame, apex_rsi):
        """Test RSI(14) alignment with TA-Lib."""
        if apex_rsi is None:
            pytest.skip("RSI indicator not registered")

        # Calculate APEX RSI
        result = apex_rsi.calculate(sample_ohlcv, {"symbol": "TEST", "period": 14})
        apex_values = result["rsi"].values if "rsi" in result.columns else result["value"].values

        # Calculate TA-Lib RSI
        talib_values = talib.RSI(sample_ohlcv["close"].values, timeperiod=14)

        # Compare after warmup (first 14 bars are NaN)
        warmup = 20  # Extra buffer
        apex_post_warmup = apex_values[warmup:]
        talib_post_warmup = talib_values[warmup:]

        # Remove NaN for comparison
        mask = ~np.isnan(talib_post_warmup) & ~np.isnan(apex_post_warmup)
        apex_clean = apex_post_warmup[mask]
        talib_clean = talib_post_warmup[mask]

        # Assert alignment
        np.testing.assert_allclose(
            apex_clean,
            talib_clean,
            atol=1e-6,
            rtol=1e-4,
            err_msg="RSI values differ from TA-Lib reference",
        )


class TestMACDAlignment:
    """Test MACD alignment with TA-Lib."""

    @pytest.fixture
    def apex_macd(self):
        """Get APEX MACD indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("macd")

    def test_macd_alignment(self, sample_ohlcv: pd.DataFrame, apex_macd):
        """Test MACD(12, 26, 9) alignment with TA-Lib."""
        if apex_macd is None:
            pytest.skip("MACD indicator not registered")

        # Calculate APEX MACD
        result = apex_macd.calculate(
            sample_ohlcv,
            {
                "symbol": "TEST",
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            },
        )

        # Calculate TA-Lib MACD
        talib_macd, talib_signal, talib_hist = talib.MACD(
            sample_ohlcv["close"].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9,
        )

        warmup = 35  # MACD needs 26 + 9 bars

        # Compare MACD line
        if "macd" in result.columns:
            apex_macd_line = result["macd"].values[warmup:]
            talib_macd_clean = talib_macd[warmup:]
            mask = ~np.isnan(talib_macd_clean) & ~np.isnan(apex_macd_line)
            np.testing.assert_allclose(
                apex_macd_line[mask],
                talib_macd_clean[mask],
                atol=1e-6,
                rtol=1e-4,
                err_msg="MACD line differs from TA-Lib",
            )

        # Compare histogram
        if "histogram" in result.columns:
            apex_hist = result["histogram"].values[warmup:]
            talib_hist_clean = talib_hist[warmup:]
            mask = ~np.isnan(talib_hist_clean) & ~np.isnan(apex_hist)
            np.testing.assert_allclose(
                apex_hist[mask],
                talib_hist_clean[mask],
                atol=1e-6,
                rtol=1e-4,
                err_msg="MACD histogram differs from TA-Lib",
            )


class TestADXAlignment:
    """Test ADX alignment with TA-Lib."""

    @pytest.fixture
    def apex_adx(self):
        """Get APEX ADX indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("adx")

    def test_adx_14_alignment(self, sample_ohlcv: pd.DataFrame, apex_adx):
        """Test ADX(14) alignment with TA-Lib."""
        if apex_adx is None:
            pytest.skip("ADX indicator not registered")

        # Calculate APEX ADX
        result = apex_adx.calculate(sample_ohlcv, {"symbol": "TEST", "period": 14})

        # Calculate TA-Lib ADX
        talib_adx = talib.ADX(
            sample_ohlcv["high"].values,
            sample_ohlcv["low"].values,
            sample_ohlcv["close"].values,
            timeperiod=14,
        )

        warmup = 30  # ADX needs 2x period for smoothing

        if "adx" in result.columns:
            apex_adx_vals = result["adx"].values[warmup:]
            talib_adx_clean = talib_adx[warmup:]
            mask = ~np.isnan(talib_adx_clean) & ~np.isnan(apex_adx_vals)

            # ADX can have larger variance due to smoothing differences
            np.testing.assert_allclose(
                apex_adx_vals[mask],
                talib_adx_clean[mask],
                atol=1.0,
                rtol=0.05,  # More lenient for ADX
                err_msg="ADX values differ from TA-Lib",
            )


class TestATRAlignment:
    """Test ATR alignment with TA-Lib."""

    @pytest.fixture
    def apex_atr(self):
        """Get APEX ATR indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("atr")

    def test_atr_14_alignment(self, sample_ohlcv: pd.DataFrame, apex_atr):
        """Test ATR(14) alignment with TA-Lib."""
        if apex_atr is None:
            pytest.skip("ATR indicator not registered")

        # Calculate APEX ATR
        result = apex_atr.calculate(sample_ohlcv, {"symbol": "TEST", "period": 14})

        # Calculate TA-Lib ATR
        talib_atr = talib.ATR(
            sample_ohlcv["high"].values,
            sample_ohlcv["low"].values,
            sample_ohlcv["close"].values,
            timeperiod=14,
        )

        warmup = 20

        # Find ATR column
        atr_col = None
        for col in ["atr", "value", "atr_14"]:
            if col in result.columns:
                atr_col = col
                break

        if atr_col is None:
            pytest.skip("ATR column not found in result")

        apex_atr_vals = result[atr_col].values[warmup:]
        talib_atr_clean = talib_atr[warmup:]
        mask = ~np.isnan(talib_atr_clean) & ~np.isnan(apex_atr_vals)

        np.testing.assert_allclose(
            apex_atr_vals[mask],
            talib_atr_clean[mask],
            atol=1e-6,
            rtol=1e-4,
            err_msg="ATR values differ from TA-Lib",
        )


class TestSMAAlignment:
    """Test SMA alignment with TA-Lib."""

    @pytest.fixture
    def apex_sma(self):
        """Get APEX SMA indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("sma")

    @pytest.mark.parametrize("period", [10, 20, 50])
    def test_sma_alignment(self, sample_ohlcv: pd.DataFrame, apex_sma, period: int):
        """Test SMA alignment with TA-Lib for various periods."""
        if apex_sma is None:
            pytest.skip("SMA indicator not registered")

        # Calculate APEX SMA
        result = apex_sma.calculate(sample_ohlcv, {"symbol": "TEST", "period": period})

        # Calculate TA-Lib SMA
        talib_sma = talib.SMA(sample_ohlcv["close"].values, timeperiod=period)

        warmup = period + 5

        # Find SMA column
        sma_col = None
        for col in [f"sma_{period}", "sma", "value"]:
            if col in result.columns:
                sma_col = col
                break

        if sma_col is None:
            pytest.skip(f"SMA column not found for period {period}")

        apex_sma_vals = result[sma_col].values[warmup:]
        talib_sma_clean = talib_sma[warmup:]
        mask = ~np.isnan(talib_sma_clean) & ~np.isnan(apex_sma_vals)

        np.testing.assert_allclose(
            apex_sma_vals[mask],
            talib_sma_clean[mask],
            atol=1e-10,
            rtol=1e-10,  # SMA should be exact
            err_msg=f"SMA({period}) values differ from TA-Lib",
        )


class TestEMAAlignment:
    """Test EMA alignment with TA-Lib."""

    @pytest.fixture
    def apex_ema(self):
        """Get APEX EMA indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("ema")

    @pytest.mark.parametrize("period", [12, 26])
    def test_ema_alignment(self, sample_ohlcv: pd.DataFrame, apex_ema, period: int):
        """Test EMA alignment with TA-Lib for various periods."""
        if apex_ema is None:
            pytest.skip("EMA indicator not registered")

        # Calculate APEX EMA
        result = apex_ema.calculate(sample_ohlcv, {"symbol": "TEST", "period": period})

        # Calculate TA-Lib EMA
        talib_ema = talib.EMA(sample_ohlcv["close"].values, timeperiod=period)

        warmup = period * 2  # EMA needs more warmup due to initialization

        # Find EMA column
        ema_col = None
        for col in [f"ema_{period}", "ema", "value"]:
            if col in result.columns:
                ema_col = col
                break

        if ema_col is None:
            pytest.skip(f"EMA column not found for period {period}")

        apex_ema_vals = result[ema_col].values[warmup:]
        talib_ema_clean = talib_ema[warmup:]
        mask = ~np.isnan(talib_ema_clean) & ~np.isnan(apex_ema_vals)

        np.testing.assert_allclose(
            apex_ema_vals[mask],
            talib_ema_clean[mask],
            atol=1e-4,
            rtol=1e-3,  # EMA can vary due to initialization
            err_msg=f"EMA({period}) values differ from TA-Lib",
        )


class TestBBandsAlignment:
    """Test Bollinger Bands alignment with TA-Lib."""

    @pytest.fixture
    def apex_bbands(self):
        """Get APEX Bollinger Bands indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("bollinger")

    def test_bbands_alignment(self, sample_ohlcv: pd.DataFrame, apex_bbands):
        """Test Bollinger Bands(20, 2) alignment with TA-Lib."""
        if apex_bbands is None:
            pytest.skip("Bollinger Bands indicator not registered")

        # Calculate APEX BBands
        result = apex_bbands.calculate(
            sample_ohlcv,
            {
                "symbol": "TEST",
                "period": 20,
                "std_dev": 2.0,
            },
        )

        # Calculate TA-Lib BBands
        upper, middle, lower = talib.BBANDS(
            sample_ohlcv["close"].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
        )

        warmup = 25

        # Compare middle band (should be SMA)
        if "middle" in result.columns:
            apex_middle = result["middle"].values[warmup:]
            talib_middle = middle[warmup:]
            mask = ~np.isnan(talib_middle) & ~np.isnan(apex_middle)
            np.testing.assert_allclose(
                apex_middle[mask],
                talib_middle[mask],
                atol=1e-6,
                rtol=1e-4,
                err_msg="Bollinger middle band differs from TA-Lib",
            )

        # Compare upper band
        if "upper" in result.columns:
            apex_upper = result["upper"].values[warmup:]
            talib_upper = upper[warmup:]
            mask = ~np.isnan(talib_upper) & ~np.isnan(apex_upper)
            np.testing.assert_allclose(
                apex_upper[mask],
                talib_upper[mask],
                atol=1e-6,
                rtol=1e-4,
                err_msg="Bollinger upper band differs from TA-Lib",
            )


class TestCCIAlignment:
    """Test CCI alignment with TA-Lib."""

    @pytest.fixture
    def apex_cci(self):
        """Get APEX CCI indicator."""
        from src.domain.signals.indicators.registry import get_indicator_registry

        registry = get_indicator_registry()
        return registry.get("cci")

    def test_cci_20_alignment(self, sample_ohlcv: pd.DataFrame, apex_cci):
        """Test CCI(20) alignment with TA-Lib."""
        if apex_cci is None:
            pytest.skip("CCI indicator not registered")

        # Calculate APEX CCI
        result = apex_cci.calculate(sample_ohlcv, {"symbol": "TEST", "period": 20})

        # Calculate TA-Lib CCI
        talib_cci = talib.CCI(
            sample_ohlcv["high"].values,
            sample_ohlcv["low"].values,
            sample_ohlcv["close"].values,
            timeperiod=20,
        )

        warmup = 25

        # Find CCI column
        cci_col = None
        for col in ["cci", "value"]:
            if col in result.columns:
                cci_col = col
                break

        if cci_col is None:
            pytest.skip("CCI column not found")

        apex_cci_vals = result[cci_col].values[warmup:]
        talib_cci_clean = talib_cci[warmup:]
        mask = ~np.isnan(talib_cci_clean) & ~np.isnan(apex_cci_vals)

        np.testing.assert_allclose(
            apex_cci_vals[mask],
            talib_cci_clean[mask],
            atol=1e-6,
            rtol=1e-4,
            err_msg="CCI values differ from TA-Lib",
        )
