"""
Tests for summary_builder.py — PackageManifest, budget enforcement,
_compute_daily_change, _extract_ticker_data_quality, _build_ticker_summary,
_build_market_overview.
"""

from __future__ import annotations

import json
from typing import Any
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


class TestComputeDailyChange:
    def test_positive_change(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100.0, 102.0, 105.0]}, index=dates)
        builder = SummaryBuilder()
        change = builder._compute_daily_change(df, "AAPL")
        assert change is not None
        assert change == pytest.approx(2.94, abs=0.01)

    def test_negative_change(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"close": [100.0, 95.0, 90.0]}, index=dates)
        builder = SummaryBuilder()
        change = builder._compute_daily_change(df, "AAPL")
        assert change is not None
        assert change < 0

    def test_none_df(self) -> None:
        builder = SummaryBuilder()
        assert builder._compute_daily_change(None, "AAPL") is None  # type: ignore[arg-type]

    def test_empty_df(self) -> None:
        builder = SummaryBuilder()
        df = pd.DataFrame()
        assert builder._compute_daily_change(df, "AAPL") is None

    def test_single_row(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        builder = SummaryBuilder()
        assert builder._compute_daily_change(df, "AAPL") is None

    def test_nan_close(self) -> None:
        df = pd.DataFrame({"close": [100.0, np.nan, 105.0]})
        builder = SummaryBuilder()
        # Last two rows: NaN and 105 -> NaN prev_close
        assert builder._compute_daily_change(df, "AAPL") is None

    def test_zero_prev_close(self) -> None:
        df = pd.DataFrame({"close": [0.0, 100.0]})
        builder = SummaryBuilder()
        assert builder._compute_daily_change(df, "AAPL") is None

    def test_no_close_column(self) -> None:
        df = pd.DataFrame({"open": [100.0, 101.0]})
        builder = SummaryBuilder()
        assert builder._compute_daily_change(df, "AAPL") is None


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
