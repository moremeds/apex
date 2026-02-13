"""Unit tests for PEAD signal generator (VectorBT integration)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.domain.strategy.signals.pead import PeadSignalGenerator, _find_entry_bar


def _make_ohlcv(start: str = "2024-01-02", periods: int = 60, base: float = 100.0) -> pd.DataFrame:
    """Build synthetic daily OHLCV DataFrame."""
    dates = pd.bdate_range(start=start, periods=periods)
    rng = np.random.default_rng(42)
    close = base + np.cumsum(rng.normal(0, 0.5, periods))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, periods),
            "high": close + rng.uniform(0, 2, periods),
            "low": close - rng.uniform(0, 2, periods),
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, periods),
        },
        index=dates,
    )


def _make_earnings(
    report_date: str = "2024-01-10",
    sue: float = 3.5,
    gap: float = 0.04,
    vol: float = 2.8,
    quality: float = 72.0,
) -> dict:
    return {
        "report_date": report_date,
        "sue_score": sue,
        "gap_pct": gap,
        "volume_ratio": vol,
        "revenue_beat": True,
        "quality_score": quality,
    }


class TestPeadSignalGenerator:
    def test_entry_on_correct_bar(self) -> None:
        """Entry appears after report_date + delay bars."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(report_date="2024-01-10")]

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        # Should have exactly 1 entry
        assert entries.sum() == 1

        # Entry bar must be after 2024-01-10
        entry_date = entries[entries].index[0].date()
        assert entry_date > date(2024, 1, 10)

    def test_no_lookahead(self) -> None:
        """No entry signal before or on report_date."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(report_date="2024-01-10")]

        gen = PeadSignalGenerator()
        entries, _ = gen.generate(data, {"earnings_data": earnings})

        # All entries must be after report_date
        for ts in entries[entries].index:
            assert ts.date() > date(2024, 1, 10)

    def test_exit_at_target(self) -> None:
        """Position exits when close reaches profit target."""
        # Create data that rises steadily to hit 6% target
        dates = pd.bdate_range("2024-01-02", periods=30)
        close = np.concatenate(
            [
                np.full(8, 100.0),  # Flat before report
                np.full(2, 100.0),  # Delay bars
                np.linspace(100, 108, 20),  # Rising to +8%
            ]
        )
        data = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        earnings = [_make_earnings(report_date="2024-01-10")]

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() >= 1
        assert exits.sum() >= 1  # Should exit at target

    def test_exit_at_stop(self) -> None:
        """Position exits when close hits stop loss."""
        dates = pd.bdate_range("2024-01-02", periods=30)
        close = np.concatenate(
            [
                np.full(8, 100.0),
                np.full(2, 100.0),
                np.linspace(100, 93, 20),  # Dropping to -7%
            ]
        )
        data = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        earnings = [_make_earnings(report_date="2024-01-10")]

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() >= 1
        assert exits.sum() >= 1  # Should exit at stop

    def test_exit_at_timeout(self) -> None:
        """Position exits after max_hold_bars if neither target nor stop hit."""
        dates = pd.bdate_range("2024-01-02", periods=50)
        # Flat price — neither target nor stop triggers
        close = np.full(50, 100.0)
        data = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        earnings = [_make_earnings(report_date="2024-01-10")]

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings, "max_hold_bars": 10})

        assert entries.sum() == 1
        assert exits.sum() == 1  # Timeout exit

    def test_sue_filter(self) -> None:
        """Earnings below min_sue are filtered out."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(sue=1.0)]  # Below default min_sue=2.0

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() == 0  # Filtered out

    def test_gap_filter(self) -> None:
        """Earnings below min_gap_pct are filtered out."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(gap=0.01)]  # Below default min_gap_pct=0.02

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() == 0

    def test_volume_filter(self) -> None:
        """Earnings below min_volume_ratio are filtered out."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(vol=1.5)]  # Below default min_volume_ratio=2.0

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() == 0

    def test_no_earnings_data(self) -> None:
        """No entries when earnings_data is empty or missing."""
        data = _make_ohlcv()

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {})

        assert entries.sum() == 0
        assert exits.sum() == 0

    def test_quality_sizing(self) -> None:
        """Quality score drives position sizing when use_quality_sizing=True."""
        data = _make_ohlcv(start="2024-01-02", periods=30)
        earnings = [_make_earnings(quality=80.0)]

        gen = PeadSignalGenerator()
        gen.generate(data, {"earnings_data": earnings, "use_quality_sizing": True})

        assert hasattr(gen, "entry_sizes")
        # Quality 80 → 0.80 sizing (clamped to [0.1, 1.0])
        entry_idx = gen.entry_sizes[gen.entry_sizes < 1.0].index
        assert len(entry_idx) > 0

    def test_multiple_earnings(self) -> None:
        """Multiple earnings events produce multiple entries (if not overlapping)."""
        data = _make_ohlcv(start="2024-01-02", periods=120)
        earnings = [
            _make_earnings(report_date="2024-01-15"),
            _make_earnings(report_date="2024-04-15"),
        ]

        gen = PeadSignalGenerator()
        entries, exits = gen.generate(data, {"earnings_data": earnings, "max_hold_bars": 10})

        # Both should produce entries (sufficiently spaced)
        assert entries.sum() == 2

    def test_report_after_data_range_ignored(self) -> None:
        """Earnings after the data range produce no entries."""
        data = _make_ohlcv(start="2024-01-02", periods=20)
        earnings = [_make_earnings(report_date="2025-06-01")]

        gen = PeadSignalGenerator()
        entries, _ = gen.generate(data, {"earnings_data": earnings})

        assert entries.sum() == 0

    def test_warmup_bars_zero(self) -> None:
        """PEAD needs no indicator warmup."""
        gen = PeadSignalGenerator()
        assert gen.warmup_bars == 0


class TestFindEntryBar:
    def test_basic_delay(self) -> None:
        bar_dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        # report on Jan 2, delay=2 → first bar after report is Jan 3 (idx 1), + 1 delay = idx 2
        idx = _find_entry_bar(bar_dates, date(2024, 1, 2), delay=2)
        assert idx == 2

    def test_report_before_data(self) -> None:
        bar_dates = [date(2024, 1, 10), date(2024, 1, 11)]
        idx = _find_entry_bar(bar_dates, date(2024, 1, 5), delay=2)
        assert idx == 1  # First bar after report + 1 delay

    def test_report_after_data(self) -> None:
        bar_dates = [date(2024, 1, 2), date(2024, 1, 3)]
        idx = _find_entry_bar(bar_dates, date(2024, 6, 1), delay=2)
        assert idx is None

    def test_delay_exceeds_remaining(self) -> None:
        bar_dates = [date(2024, 1, 2), date(2024, 1, 3)]
        # report on Jan 2, delay=5 → not enough bars
        idx = _find_entry_bar(bar_dates, date(2024, 1, 2), delay=5)
        assert idx is None
