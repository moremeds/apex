"""Tests for report data loading optimization (Parquet-first, no coverage check).

Tests the REAL SignalPipelineProcessor._generate_html_report() method,
mocking only external dependencies (historical_manager, PackageBuilder,
indicator registry) to verify:
- Timeframe-aware lookback via get_max_history_days()
- Parquet-first read via get_bars() (no coverage check)
- Fallback to ensure_data() only when get_bars() returns empty
- Lookback capped at 900 days regardless of source capacity
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from src.domain.events.domain_events import BarData
from src.domain.signals.pipeline.config import SignalPipelineConfig
from src.domain.signals.pipeline.processor import SignalPipelineProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(n: int, start: datetime, timeframe: str = "1h") -> List[BarData]:
    """Create a list of n BarData instances for testing."""
    tf_deltas = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }
    delta = tf_deltas.get(timeframe, timedelta(hours=1))
    bars: List[BarData] = []
    for i in range(n):
        ts = start + delta * i
        bars.append(
            BarData(
                symbol="TEST",
                timeframe=timeframe,
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1_000_000,
                bar_start=ts,
                timestamp=ts + delta,
            )
        )
    return bars


def _make_historical_manager(
    max_history_days: Dict[str, int],
    get_bars_results: Optional[Dict[Tuple[str, str], List[BarData]]] = None,
    ensure_data_results: Optional[Dict[Tuple[str, str], List[BarData]]] = None,
) -> MagicMock:
    """Build a mock HistoricalDataManager (no coverage check needed)."""
    get_bars_results = get_bars_results or {}
    ensure_data_results = ensure_data_results or {}

    manager = MagicMock()
    manager.get_max_history_days = MagicMock(side_effect=lambda tf: max_history_days.get(tf, 365))
    manager.get_bars = MagicMock(
        side_effect=lambda sym, tf, start, end: get_bars_results.get((sym, tf), [])
    )
    manager.ensure_data = AsyncMock(
        side_effect=lambda sym, tf, start, end: ensure_data_results.get((sym, tf), [])
    )
    return manager


def _make_processor(symbols: List[str], timeframes: List[str]) -> SignalPipelineProcessor:
    """Create a minimally-initialized processor for report generation tests."""
    config = SignalPipelineConfig(
        symbols=symbols,
        timeframes=timeframes,
        html_output="/tmp/test_report",
    )
    proc = SignalPipelineProcessor(config)
    # Mock service with indicator engine
    proc._service = MagicMock()
    proc._service._indicator_engine = MagicMock()
    proc._service._indicator_engine._indicators = []
    return proc


# Patches that stub out everything after the data-loading loop
_PATCHES = [
    patch(
        "src.domain.signals.pipeline.processor.SignalPipelineProcessor._compute_indicators_on_df",
        lambda self, df, indicators: df,
    ),
    patch("src.infrastructure.reporting.PackageBuilder"),
    patch(
        "src.domain.signals.indicators.registry.get_indicator_registry",
        return_value=MagicMock(get_all=MagicMock(return_value=[])),
    ),
]


def _run(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests: Lookback calculation
# ---------------------------------------------------------------------------


class TestLookbackViaProcessor:
    """Verify timeframe-aware lookback through the real processor method."""

    def test_lookback_uses_max_history_days_for_1h(self) -> None:
        """1h: get_max_history_days returns 730, lookback = 730 (< 900 cap)."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        manager = _make_historical_manager(
            max_history_days={"1h": 730},
            ensure_data_results={
                ("AAPL", "1h"): _make_bars(10, end - timedelta(days=5)),
            },
        )
        proc = _make_processor(["AAPL"], ["1h"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.get_max_history_days.assert_called_with("1h")
        # get_bars called first (returns empty), then ensure_data called
        call_args = manager.ensure_data.call_args
        actual_start = call_args.kwargs.get("start") or call_args[0][2]
        expected_start = end - timedelta(days=730)
        assert actual_start == expected_start

    def test_lookback_capped_at_900_for_1d(self) -> None:
        """1d: get_max_history_days returns 3650, capped at 900."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        manager = _make_historical_manager(
            max_history_days={"1d": 3650},
            ensure_data_results={
                ("AAPL", "1d"): _make_bars(10, end - timedelta(days=30), "1d"),
            },
        )
        proc = _make_processor(["AAPL"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        call_args = manager.ensure_data.call_args
        actual_start = call_args.kwargs.get("start") or call_args[0][2]
        expected_start = end - timedelta(days=900)
        assert actual_start == expected_start


# ---------------------------------------------------------------------------
# Tests: Parquet-first behavior (no coverage check)
# ---------------------------------------------------------------------------


class TestParquetFirstViaProcessor:
    """Verify get_bars() is always called first, with ensure_data() fallback."""

    def test_get_bars_returns_data_skips_ensure_data(self) -> None:
        """When get_bars returns fresh data, ensure_data is NOT called."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        # Bars ending within 5 days of end (fresh — passes freshness guard)
        cached_bars = _make_bars(50, end - timedelta(days=3))
        manager = _make_historical_manager(
            max_history_days={"1h": 730},
            get_bars_results={("AAPL", "1h"): cached_bars},
        )
        proc = _make_processor(["AAPL"], ["1h"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.get_bars.assert_called_once()
        manager.ensure_data.assert_not_called()

    def test_get_bars_empty_falls_back_to_ensure_data(self) -> None:
        """When get_bars returns [], ensure_data is called as fallback."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        downloaded_bars = _make_bars(100, end - timedelta(days=60))
        manager = _make_historical_manager(
            max_history_days={"1h": 730},
            get_bars_results={("AAPL", "1h"): []},
            ensure_data_results={("AAPL", "1h"): downloaded_bars},
        )
        proc = _make_processor(["AAPL"], ["1h"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.get_bars.assert_called_once()
        manager.ensure_data.assert_called_once()

    def test_mixed_symbols_parquet_and_fallback(self) -> None:
        """AAPL has cached bars (Parquet hit), MSFT doesn't (ensure_data fallback)."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        manager = _make_historical_manager(
            max_history_days={"1d": 3650},
            get_bars_results={
                ("AAPL", "1d"): _make_bars(40, end - timedelta(days=20), "1d"),
                ("MSFT", "1d"): [],  # Empty — will fall back
            },
            ensure_data_results={
                ("MSFT", "1d"): _make_bars(80, end - timedelta(days=50), "1d"),
            },
        )
        proc = _make_processor(["AAPL", "MSFT"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        # get_bars called for both symbols
        assert manager.get_bars.call_count == 2
        # ensure_data only called for MSFT (empty Parquet)
        manager.ensure_data.assert_called_once()
        ensure_sym = manager.ensure_data.call_args[0][0]
        assert ensure_sym == "MSFT"

    def test_ensure_data_exception_does_not_crash(self) -> None:
        """If ensure_data raises, the symbol is skipped without crashing."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        manager = _make_historical_manager(
            max_history_days={"1h": 730},
            get_bars_results={("AAPL", "1h"): []},
        )
        manager.ensure_data = AsyncMock(side_effect=RuntimeError("download failed"))
        proc = _make_processor(["AAPL"], ["1h"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                # Should not raise
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.ensure_data.assert_called_once()

    def test_no_coverage_check_called(self) -> None:
        """Verify has_complete_coverage is NEVER called (removed from code path)."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        manager = _make_historical_manager(
            max_history_days={"1d": 365},
            get_bars_results={("AAPL", "1d"): _make_bars(50, end - timedelta(days=30), "1d")},
        )
        # Add has_complete_coverage to track if it's called
        manager.has_complete_coverage = MagicMock(return_value=True)
        proc = _make_processor(["AAPL"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.has_complete_coverage.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Freshness guard (stale Parquet detection)
# ---------------------------------------------------------------------------


class TestFreshnessGuard:
    """Verify stale Parquet data triggers re-download via ensure_data."""

    def test_stale_data_triggers_redownload(self) -> None:
        """Bars >5 days behind end are stale — ensure_data is called as fallback."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        # Bars from 30 days ago — last bar is ~20 days old (stale)
        stale_bars = _make_bars(10, end - timedelta(days=30), "1d")
        fresh_bars = _make_bars(50, end - timedelta(days=5), "1d")
        manager = _make_historical_manager(
            max_history_days={"1d": 365},
            get_bars_results={("AAPL", "1d"): stale_bars},
            ensure_data_results={("AAPL", "1d"): fresh_bars},
        )
        proc = _make_processor(["AAPL"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        # get_bars returned stale data -> freshness guard triggered ensure_data
        manager.get_bars.assert_called_once()
        manager.ensure_data.assert_called_once()

    def test_fresh_data_skips_redownload(self) -> None:
        """Bars within 5 days of end are fresh — ensure_data NOT called."""
        end = datetime(2026, 2, 22, 16, 0, 0, tzinfo=timezone.utc)
        # Last bar is 2 days before end — within freshness threshold
        fresh_bars = _make_bars(50, end - timedelta(days=5), "1d")
        manager = _make_historical_manager(
            max_history_days={"1d": 365},
            get_bars_results={("AAPL", "1d"): fresh_bars},
        )
        proc = _make_processor(["AAPL"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.get_bars.assert_called_once()
        manager.ensure_data.assert_not_called()

    def test_weekend_gap_not_treated_as_stale(self) -> None:
        """3-day weekend gap (Fri close → Mon) should NOT trigger re-download."""
        # Monday morning — last bar is Friday's close (3 days ago)
        end = datetime(2026, 2, 23, 16, 0, 0, tzinfo=timezone.utc)  # Monday
        friday_bars = _make_bars(50, end - timedelta(days=8), "1d")
        # Last bar is ~3 days before end (Friday close to Monday)
        manager = _make_historical_manager(
            max_history_days={"1d": 365},
            get_bars_results={("AAPL", "1d"): friday_bars},
        )
        proc = _make_processor(["AAPL"], ["1d"])

        with patch.object(proc, "_get_intraday_end", return_value=pd.Timestamp(end)):
            for p in _PATCHES:
                p.start()
            try:
                _run(proc._generate_html_report(manager, "/tmp/test"))
            finally:
                for p in _PATCHES:
                    p.stop()

        manager.get_bars.assert_called_once()
        manager.ensure_data.assert_not_called()
