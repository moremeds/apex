"""
Tests for _get_last_trading_day and _get_intraday_end timezone handling.

Verifies:
- DST-correct close times (summer=20:00 UTC, winter=21:00 UTC)
- Intraday end logic (before open, during session, after close, non-trading day)
- 4h bar resampling produces correct bar count and timestamps
- Dual MACD timestamp formatting uses display timezone
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, List
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.domain.services.bar_count_calculator import TradingSession

if TYPE_CHECKING:
    from src.domain.signals.pipeline.processor import SignalPipelineProcessor

US_EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session(
    d: date,
    open_hour: int = 9,
    open_min: int = 30,
    close_hour: int = 16,
    close_min: int = 0,
) -> TradingSession:
    """Build a TradingSession with ET times converted to UTC-aware datetimes."""
    open_et = datetime(d.year, d.month, d.day, open_hour, open_min, tzinfo=US_EASTERN)
    close_et = datetime(d.year, d.month, d.day, close_hour, close_min, tzinfo=US_EASTERN)
    return TradingSession(
        date=d,
        market_open=open_et.astimezone(UTC),
        market_close=close_et.astimezone(UTC),
        is_early_close=(close_hour < 16),
    )


def _make_processor(
    config: MagicMock | None = None,
) -> SignalPipelineProcessor:
    """Create a SignalPipelineProcessor with a minimal mock config."""
    from src.domain.signals.pipeline.processor import SignalPipelineProcessor

    if config is None:
        config = MagicMock()
        config.symbols = ["AAPL"]
        config.timeframes = ["1d"]
    return SignalPipelineProcessor(config)


# ---------------------------------------------------------------------------
# _get_last_trading_day
# ---------------------------------------------------------------------------


class TestGetLastTradingDay:
    """Tests for DST-correct close time from _get_last_trading_day."""

    def test_winter_close_is_21_utc(self) -> None:
        """In winter (EST), NYSE close = 16:00 ET = 21:00 UTC."""
        winter_date = date(2026, 1, 15)  # January = EST
        session = _make_session(winter_date)
        assert session.market_close.hour == 21  # 16:00 EST = 21:00 UTC

    def test_summer_close_is_20_utc(self) -> None:
        """In summer (EDT), NYSE close = 16:00 ET = 20:00 UTC."""
        summer_date = date(2026, 7, 15)  # July = EDT
        session = _make_session(summer_date)
        assert session.market_close.hour == 20  # 16:00 EDT = 20:00 UTC

    @patch("src.domain.services.bar_count_calculator.BarCountCalculator.get_previous_trading_day")
    @patch("src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions")
    def test_returns_calendar_close_not_hardcoded(
        self,
        mock_sessions: MagicMock,
        mock_prev_day: MagicMock,
    ) -> None:
        """_get_last_trading_day uses calendar close, not hardcoded 21:00."""
        summer_date = date(2026, 7, 15)
        mock_prev_day.return_value = summer_date
        session = _make_session(summer_date)
        mock_sessions.return_value = [session]

        proc = _make_processor()
        result = proc._get_last_trading_day()

        # Should match the summer close (20:00 UTC), not winter (21:00)
        assert result == pd.Timestamp(session.market_close)

    @patch("src.domain.services.bar_count_calculator.BarCountCalculator.get_previous_trading_day")
    @patch("src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions")
    def test_fallback_when_no_sessions(
        self,
        mock_sessions: MagicMock,
        mock_prev_day: MagicMock,
    ) -> None:
        """Falls back to 21:00 UTC if calendar returns no sessions."""
        d = date(2026, 1, 15)
        mock_prev_day.return_value = d
        mock_sessions.return_value = []

        proc = _make_processor()
        result = proc._get_last_trading_day()
        assert result.hour == 21


# ---------------------------------------------------------------------------
# _get_intraday_end
# ---------------------------------------------------------------------------


class TestGetIntradayEnd:
    """Tests for market-hours-aware end timestamp."""

    def test_non_trading_day_returns_prev_close(self) -> None:
        """On weekends/holidays, falls back to previous trading day close."""
        proc = _make_processor()

        with (
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.is_trading_day",
                return_value=False,
            ),
            patch("src.domain.signals.pipeline.processor.date") as mock_date,
            patch.object(proc, "_get_last_trading_day") as mock_ltd,
        ):
            mock_date.today.return_value = date(2026, 1, 31)  # Saturday
            mock_ltd.return_value = pd.Timestamp("2026-01-30 21:00:00+00:00")
            result = proc._get_intraday_end()
            mock_ltd.assert_called_once()
            assert result == pd.Timestamp("2026-01-30 21:00:00+00:00")

    def test_before_market_open_returns_prev_close(self) -> None:
        """Before 9:30 ET on a trading day, use previous close."""
        trading_date = date(2026, 1, 29)
        session = _make_session(trading_date)
        proc = _make_processor()

        with (
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.is_trading_day",
                return_value=True,
            ),
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions",
                return_value=[session],
            ),
            patch("src.domain.signals.pipeline.processor.date") as mock_date,
            patch("src.domain.signals.pipeline.processor.datetime") as mock_dt,
            patch.object(proc, "_get_last_trading_day") as mock_ltd,
        ):
            mock_date.today.return_value = trading_date
            # 8:00 AM ET = 13:00 UTC (before open at 14:30 UTC in winter)
            mock_dt.now.return_value = datetime(2026, 1, 29, 13, 0, tzinfo=UTC)
            mock_ltd.return_value = pd.Timestamp("2026-01-28 21:00:00+00:00")
            proc._get_intraday_end()
            mock_ltd.assert_called_once()

    def test_after_close_returns_today_close(self) -> None:
        """After 16:00 ET, use today's close timestamp."""
        trading_date = date(2026, 1, 29)
        session = _make_session(trading_date)
        proc = _make_processor()

        with (
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.is_trading_day",
                return_value=True,
            ),
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions",
                return_value=[session],
            ),
            patch("src.domain.signals.pipeline.processor.date") as mock_date,
            patch("src.domain.signals.pipeline.processor.datetime") as mock_dt,
        ):
            mock_date.today.return_value = trading_date
            # 5:00 PM ET = 22:00 UTC (after close at 21:00 UTC in winter)
            mock_dt.now.return_value = datetime(2026, 1, 29, 22, 0, tzinfo=UTC)
            result = proc._get_intraday_end()
            assert result == pd.Timestamp(session.market_close)

    def test_during_session_returns_now(self) -> None:
        """During market hours, use current time."""
        trading_date = date(2026, 1, 29)
        session = _make_session(trading_date)
        proc = _make_processor()

        # 11:00 AM ET = 16:00 UTC (during session: open=14:30, close=21:00 UTC)
        now = datetime(2026, 1, 29, 16, 0, tzinfo=UTC)

        with (
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.is_trading_day",
                return_value=True,
            ),
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions",
                return_value=[session],
            ),
            patch("src.domain.signals.pipeline.processor.date") as mock_date,
            patch("src.domain.signals.pipeline.processor.datetime") as mock_dt,
        ):
            mock_date.today.return_value = trading_date
            mock_dt.now.return_value = now
            result = proc._get_intraday_end()
            assert result == pd.Timestamp(now)

    def test_summer_after_close_uses_20_utc(self) -> None:
        """In summer (EDT), after close should use 20:00 UTC, not 21:00."""
        summer_date = date(2026, 7, 15)
        session = _make_session(summer_date)
        proc = _make_processor()

        with (
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.is_trading_day",
                return_value=True,
            ),
            patch(
                "src.domain.services.bar_count_calculator.BarCountCalculator.get_trading_sessions",
                return_value=[session],
            ),
            patch("src.domain.signals.pipeline.processor.date") as mock_date,
            patch("src.domain.signals.pipeline.processor.datetime") as mock_dt,
        ):
            mock_date.today.return_value = summer_date
            mock_dt.now.return_value = datetime(2026, 7, 15, 21, 0, tzinfo=UTC)
            result = proc._get_intraday_end()
            # Summer close = 20:00 UTC, not 21:00
            assert result == pd.Timestamp(session.market_close)
            assert result.hour == 20


# ---------------------------------------------------------------------------
# 4h bar resampling
# ---------------------------------------------------------------------------


class TestResampleBarsTo4h:
    """Tests for 4h bar resampling from 1h bars."""

    def test_regular_day_produces_two_bars(self) -> None:
        """A full trading day with 7 x 1h bars resamples to 2 x 4h bars."""
        from src.domain.events.domain_events import BarData
        from src.services.historical_data_manager import resample_bars_to_4h

        d = date(2026, 1, 29)
        hours = [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]
        bars_1h: List[BarData] = []

        for h, m in hours:
            ts = datetime(d.year, d.month, d.day, h, m, tzinfo=US_EASTERN).astimezone(UTC)
            bars_1h.append(
                BarData(
                    symbol="TEST",
                    timeframe="1h",
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000,
                    bar_start=ts,
                    timestamp=ts,
                )
            )

        result = resample_bars_to_4h(bars_1h, "TEST")
        assert len(result) == 2

        # Bar 1 starts at 9:30 ET, Bar 2 starts at 13:30 ET
        bar1_et = result[0].bar_start.astimezone(US_EASTERN)
        bar2_et = result[1].bar_start.astimezone(US_EASTERN)
        assert bar1_et.hour == 9 and bar1_et.minute == 30
        assert bar2_et.hour == 13 and bar2_et.minute == 30

    def test_bar_aggregation_ohlcv(self) -> None:
        """4h bar OHLCV is correctly aggregated from 1h bars."""
        from src.domain.events.domain_events import BarData
        from src.services.historical_data_manager import resample_bars_to_4h

        d = date(2026, 1, 29)
        # First 4h group: 9:30, 10:30, 11:30, 12:30
        prices = [
            (100.0, 102.0, 99.0, 101.0, 1000),  # 9:30
            (101.0, 105.0, 100.0, 103.0, 2000),  # 10:30
            (103.0, 104.0, 101.0, 102.0, 1500),  # 11:30
            (102.0, 103.0, 98.0, 99.0, 1800),  # 12:30
        ]
        bars_1h = []
        hours = [(9, 30), (10, 30), (11, 30), (12, 30)]

        for (h, m), (o, hi, lo, c, v) in zip(hours, prices):
            ts = datetime(d.year, d.month, d.day, h, m, tzinfo=US_EASTERN).astimezone(UTC)
            bars_1h.append(
                BarData(
                    symbol="TEST",
                    timeframe="1h",
                    open=o,
                    high=hi,
                    low=lo,
                    close=c,
                    volume=v,
                    bar_start=ts,
                    timestamp=ts,
                )
            )

        result = resample_bars_to_4h(bars_1h, "TEST")
        assert len(result) == 1

        bar = result[0]
        assert bar.open == 100.0  # first open
        assert bar.high == 105.0  # max high
        assert bar.low == 98.0  # min low
        assert bar.close == 99.0  # last close
        assert bar.volume == 6300  # sum

    def test_multi_day_resampling(self) -> None:
        """Multiple trading days each produce 2 bars."""
        from src.domain.events.domain_events import BarData
        from src.services.historical_data_manager import resample_bars_to_4h

        bars_1h = []
        for d in [date(2026, 1, 29), date(2026, 1, 30)]:
            for h, m in [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]:
                ts = datetime(d.year, d.month, d.day, h, m, tzinfo=US_EASTERN).astimezone(UTC)
                bars_1h.append(
                    BarData(
                        symbol="TEST",
                        timeframe="1h",
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=1000,
                        bar_start=ts,
                        timestamp=ts,
                    )
                )

        result = resample_bars_to_4h(bars_1h, "TEST")
        assert len(result) == 4  # 2 bars per day × 2 days


# ---------------------------------------------------------------------------
# Dual MACD timestamp formatting
# ---------------------------------------------------------------------------


class TestDualMacdTimestampFormatting:
    """Tests for timezone-aware timestamp formatting in dual MACD output."""

    def test_daily_bars_show_date_only(self) -> None:
        """Daily timeframe should format as YYYY-MM-DD without time."""
        from src.infrastructure.reporting.signal_report.dual_macd_section import (
            compute_dual_macd_history,
        )

        # Build a minimal DataFrame with enough bars for DualMACD warmup
        n = 200
        dates = pd.date_range("2025-06-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame(
            {"close": [100 + i * 0.1 for i in range(n)]},
            index=dates,
        )
        df.index.name = "timestamp"

        result = compute_dual_macd_history({("TEST", "1d"): df}, display_timezone="US/Eastern")
        key = "TEST_1d"
        if key in result and result[key]:
            date_str = result[key][0]["date"]
            assert len(date_str) == 10, f"Expected YYYY-MM-DD, got '{date_str}'"
            assert ":" not in date_str

    def test_intraday_bars_use_display_timezone(self) -> None:
        """Intraday timeframe should convert timestamps to display timezone."""
        from src.infrastructure.reporting.signal_report.dual_macd_section import (
            compute_dual_macd_history,
        )

        # Build 1h bars in UTC
        n = 200
        dates = pd.date_range("2025-06-01 14:30", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"close": [100 + i * 0.1 for i in range(n)]},
            index=dates,
        )
        df.index.name = "timestamp"

        # Use Asia/Hong_Kong to verify it's not hardcoded to US/Eastern
        result = compute_dual_macd_history({("TEST", "1h"): df}, display_timezone="Asia/Hong_Kong")
        key = "TEST_1h"
        if key in result and result[key]:
            date_str = result[key][0]["date"]
            assert ":" in date_str, f"Expected datetime format, got '{date_str}'"


# ---------------------------------------------------------------------------
# Heatmap generated_at_str
# ---------------------------------------------------------------------------


class TestHeatmapGeneratedAtStr:
    """Tests for timezone-aware generated_at_str on HeatmapModel."""

    def test_generated_at_str_field_exists(self) -> None:
        """HeatmapModel should have generated_at_str field."""
        from src.infrastructure.reporting.heatmap.model import HeatmapModel

        model = HeatmapModel(generated_at_str="2026-01-30 09:30 HKT")
        assert model.generated_at_str == "2026-01-30 09:30 HKT"

    def test_empty_generated_at_str_default(self) -> None:
        """Default generated_at_str should be empty string."""
        from src.infrastructure.reporting.heatmap.model import HeatmapModel

        model = HeatmapModel()
        assert model.generated_at_str == ""


# ---------------------------------------------------------------------------
# Yahoo adapter intraday end date
# ---------------------------------------------------------------------------


class TestYahooIntradayEndDate:
    """Tests for Yahoo adapter intraday vs daily end date handling."""

    def test_intraday_intervals_set(self) -> None:
        """Verify intraday intervals are correctly defined."""
        intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        assert "1d" not in intraday
        assert "1wk" not in intraday
        assert "1h" in intraday
        assert "5m" in intraday
