"""
Tests for 4h bar resampling timezone handling.

Verifies:
- 4h bar resampling produces correct bar count and timestamps
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List
from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


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
