"""
Tests for summary_builder.py — PackageManifest, budget enforcement,
_resolve_latest_close, _compute_daily_change, _extract_ticker_data_quality,
_build_ticker_summary, _build_market_overview.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.reporting.exceptions import SizeBudgetExceeded
from src.infrastructure.reporting.package.summary_builder import (
    PackageManifest,
    SummaryBuilder,
)

# =============================================================================
# PackageManifest
# =============================================================================


class TestPackageManifest:
    def test_to_dict(self) -> None:
        m = PackageManifest(
            version="1.0",
            created_at="2024-01-01T00:00:00",
            symbols=("AAPL", "SPY"),
            timeframes=("1d",),
            total_data_files=10,
            summary_size_kb=45.2,
            theme="dark",
        )
        d = m.to_dict()
        assert d["version"] == "1.0"
        assert d["symbols"] == ["AAPL", "SPY"]
        assert d["timeframes"] == ["1d"]
        assert d["total_data_files"] == 10
        assert d["theme"] == "dark"

    def test_is_frozen(self) -> None:
        m = PackageManifest(
            version="1.0",
            created_at="x",
            symbols=(),
            timeframes=(),
            total_data_files=0,
            summary_size_kb=0,
            theme="dark",
        )
        with pytest.raises(AttributeError):
            m.version = "2.0"  # type: ignore[misc]

    def test_roundtrip_json(self) -> None:
        m = PackageManifest(
            version="1.0",
            created_at="2024-01-01",
            symbols=("AAPL",),
            timeframes=("1d",),
            total_data_files=5,
            summary_size_kb=10.0,
            theme="dark",
        )
        serialized = json.dumps(m.to_dict())
        loaded = json.loads(serialized)
        assert loaded["version"] == "1.0"


# =============================================================================
# Budget enforcement
# =============================================================================


class TestBudgetEnforcement:
    def test_within_budget_no_error(self) -> None:
        builder = SummaryBuilder(enforce_budget=True)
        small_data = {"key": "value"}
        builder.check_budget("test", small_data, budget_kb=10)  # Should not raise

    def test_exceeds_budget_warning_only(self) -> None:
        """Without enforce_budget, exceeding logs warning but no exception."""
        builder = SummaryBuilder(enforce_budget=False)
        big_data = {"key": "x" * 20_000}
        # Should not raise
        builder.check_budget("test", big_data, budget_kb=1)

    def test_exceeds_budget_raises(self) -> None:
        builder = SummaryBuilder(enforce_budget=True)
        big_data = {"key": "x" * 20_000}
        with pytest.raises(SizeBudgetExceeded) as exc_info:
            builder.check_budget("test", big_data, budget_kb=1)
        assert exc_info.value.section == "test"
        assert exc_info.value.actual_kb > 1

    def test_top_contributors(self) -> None:
        builder = SummaryBuilder()
        data = {"big": "x" * 10_000, "small": "y", "medium": "z" * 100}
        contributors = builder._find_top_contributors(data)
        assert len(contributors) == 3
        # Sorted by size descending
        assert contributors[0].key == "big"
        assert contributors[0].pct_of_section > 50


# =============================================================================
# _compute_daily_change
# =============================================================================


def _make_daily_df(
    closes: list, start: str = "2024-01-08", periods: int | None = None
) -> pd.DataFrame:
    """Helper: build a 1d DataFrame with DatetimeIndex."""
    n = periods or len(closes)
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"close": closes, "volume": [1_000_000] * n}, index=dates)


def _make_hourly_df(closes: list, start: str, periods: int | None = None) -> pd.DataFrame:
    """Helper: build a 1h DataFrame with DatetimeIndex."""
    n = periods or len(closes)
    dates = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({"close": closes, "volume": [500_000] * n}, index=dates)


class TestResolveLatestClose:
    """Tests for _resolve_latest_close — cross-timeframe close resolution."""

    def test_only_1d_data(self) -> None:
        daily = _make_daily_df([100.0, 102.0, 105.0])
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        close, ts, tf = builder._resolve_latest_close("AAPL", data)
        assert close == 105.0
        assert tf == "1d"

    def test_1h_more_recent_than_1d(self) -> None:
        daily = _make_daily_df([100.0, 102.0], start="2024-01-08")
        hourly = _make_hourly_df([103.0, 104.0, 105.5], start="2024-01-10 09:00")
        data: Dict[Tuple[str, str], pd.DataFrame] = {
            ("AAPL", "1d"): daily,
            ("AAPL", "1h"): hourly,
        }
        builder = SummaryBuilder()
        close, ts, tf = builder._resolve_latest_close("AAPL", data)
        assert close == 105.5
        assert tf == "1h"

    def test_4h_present_and_more_recent(self) -> None:
        daily = _make_daily_df([100.0, 102.0], start="2024-01-08")
        four_h = pd.DataFrame(
            {"close": [106.0], "volume": [300_000]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-12 12:00")]),
        )
        data: Dict[Tuple[str, str], pd.DataFrame] = {
            ("AAPL", "1d"): daily,
            ("AAPL", "4h"): four_h,
        }
        builder = SummaryBuilder()
        close, ts, tf = builder._resolve_latest_close("AAPL", data)
        assert close == 106.0
        assert tf == "4h"

    def test_all_nan_sentinel_returns_none(self) -> None:
        bad_daily = pd.DataFrame(
            {"close": [np.nan, -1.0], "volume": [0, 0]},
            index=pd.date_range("2024-01-08", periods=2, freq="D"),
        )
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): bad_daily}
        builder = SummaryBuilder()
        close, ts, tf = builder._resolve_latest_close("AAPL", data)
        assert close is None
        assert ts is None
        assert tf is None

    def test_ignores_other_symbols(self) -> None:
        daily_aapl = _make_daily_df([100.0, 102.0])
        daily_spy = _make_daily_df([400.0, 410.0], start="2024-02-01")
        data: Dict[Tuple[str, str], pd.DataFrame] = {
            ("AAPL", "1d"): daily_aapl,
            ("SPY", "1d"): daily_spy,
        }
        builder = SummaryBuilder()
        close, ts, tf = builder._resolve_latest_close("AAPL", data)
        assert close == 102.0  # Not SPY's 410.0


class TestComputeDailyChange:
    """Tests for _compute_daily_change — trading-date aware change computation."""

    def test_intraday_scenario(self) -> None:
        """1h bar from today, 1d hasn't updated yet → prev = last 1d close."""
        daily = _make_daily_df([100.0, 102.0], start="2024-01-08")
        # Current close from today (Jan 10), 1d last bar is Jan 9
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        current_close = 105.0
        current_ts = datetime(2024, 1, 10, 14, 0)  # Today, after market hours
        change = builder._compute_daily_change("AAPL", current_close, current_ts, data)
        # prev = 102.0 (last 1d close), change = (105-102)/102 * 100
        assert change is not None
        assert change == pytest.approx(2.94, abs=0.01)

    def test_eod_scenario(self) -> None:
        """1d includes today → prev = second-to-last 1d close (yesterday)."""
        daily = _make_daily_df([100.0, 102.0, 105.0], start="2024-01-08")
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        current_close = 105.0
        current_ts = datetime(2024, 1, 10, 0, 0)  # Same date as last 1d bar
        change = builder._compute_daily_change("AAPL", current_close, current_ts, data)
        # prev = 102.0 (iloc[-2]), change = (105-102)/102 * 100
        assert change is not None
        assert change == pytest.approx(2.94, abs=0.01)

    def test_negative_change(self) -> None:
        daily = _make_daily_df([100.0, 102.0], start="2024-01-08")
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        change = builder._compute_daily_change("AAPL", 98.0, datetime(2024, 1, 10, 14, 0), data)
        assert change is not None
        assert change < 0

    def test_none_close(self) -> None:
        builder = SummaryBuilder()
        assert builder._compute_daily_change("AAPL", None, None, {}) is None

    def test_missing_1d_data(self) -> None:
        """No 1d DataFrame → returns None."""
        data: Dict[Tuple[str, str], pd.DataFrame] = {
            ("AAPL", "1h"): _make_hourly_df([100.0, 101.0], start="2024-01-08 09:00")
        }
        builder = SummaryBuilder()
        change = builder._compute_daily_change("AAPL", 101.0, datetime(2024, 1, 8, 10, 0), data)
        assert change is None

    def test_nan_prev_close_returns_none(self) -> None:
        daily = pd.DataFrame(
            {"close": [np.nan, 105.0], "volume": [0, 0]},
            index=pd.date_range("2024-01-08", periods=2, freq="D"),
        )
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        # EOD scenario: prev = iloc[-2] = NaN
        change = builder._compute_daily_change("AAPL", 105.0, datetime(2024, 1, 9, 0, 0), data)
        assert change is None

    def test_single_row_1d(self) -> None:
        daily = _make_daily_df([100.0], start="2024-01-08", periods=1)
        data: Dict[Tuple[str, str], pd.DataFrame] = {("AAPL", "1d"): daily}
        builder = SummaryBuilder()
        change = builder._compute_daily_change("AAPL", 105.0, datetime(2024, 1, 9, 14, 0), data)
        assert change is None


# =============================================================================
# _extract_ticker_data_quality
# =============================================================================


class TestExtractTickerDataQuality:
    def test_healthy_data(self) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame({"close": [100.0 + i for i in range(50)]}, index=dates)
        builder = SummaryBuilder()
        quality = builder._extract_ticker_data_quality("AAPL", df, None)
        assert quality["usable_bars"] == 50
        assert quality["sentinel_count"] == 0
        assert quality["regime_trustworthy"] is True
        assert quality["reasons"] == []

    def test_sentinel_values(self) -> None:
        df = pd.DataFrame({"close": [-1.0, -1.0, 100.0, 101.0, 102.0]})
        builder = SummaryBuilder()
        quality = builder._extract_ticker_data_quality("AAPL", df, None)
        assert quality["sentinel_count"] == 2
        assert quality["regime_trustworthy"] is False
        assert "SENTINEL_VALUES" in quality["reasons"]

    def test_nan_values(self) -> None:
        df = pd.DataFrame({"close": [100.0, np.nan, 102.0]})
        builder = SummaryBuilder()
        quality = builder._extract_ticker_data_quality("AAPL", df, None)
        assert "NAN_VALUES" in quality["reasons"]

    def test_none_df(self) -> None:
        builder = SummaryBuilder()
        quality = builder._extract_ticker_data_quality("AAPL", None, None)
        assert quality["usable_bars"] == 0
        assert quality["regime_trustworthy"] is True

    def test_regime_quality_invalid_close(self) -> None:
        """Regime output marking close as invalid sets regime_trustworthy=False."""
        df = pd.DataFrame({"close": [100.0]})
        regime = MagicMock()
        regime.quality.component_validity = {"close": False}
        regime.quality.component_issues = {}
        builder = SummaryBuilder()
        quality = builder._extract_ticker_data_quality("AAPL", df, regime)
        assert quality["regime_trustworthy"] is False
        assert "INVALID_CLOSE" in quality["reasons"]


# =============================================================================
# _build_market_overview
# =============================================================================


class TestBuildMarketOverview:
    def test_extracts_benchmarks(self, mock_regime_output: Any) -> None:
        regime_outputs = {
            "SPY": mock_regime_output(final_regime="R0"),
            "QQQ": mock_regime_output(final_regime="R1"),
        }
        builder = SummaryBuilder()
        overview = builder._build_market_overview(regime_outputs)
        assert "SPY" in overview["benchmarks"]
        assert "QQQ" in overview["benchmarks"]
        assert overview["benchmarks"]["SPY"]["regime"] == "R0"

    def test_missing_benchmarks(self) -> None:
        builder = SummaryBuilder()
        overview = builder._build_market_overview({})
        assert overview["benchmarks"] == {}

    def test_non_benchmark_symbols_excluded(self, mock_regime_output: Any) -> None:
        regime_outputs = {
            "AAPL": mock_regime_output(),  # Not a benchmark
            "SPY": mock_regime_output(),
        }
        builder = SummaryBuilder()
        overview = builder._build_market_overview(regime_outputs)
        assert "AAPL" not in overview["benchmarks"]
        assert "SPY" in overview["benchmarks"]
