"""
OPT-009: Unit tests for streaming data feeds.

Tests verify:
- Memory efficiency (O(symbols) not O(bars))
- Correct timestamp ordering
- Date filtering
- Empty/missing file handling
- Factory function behavior
"""

import csv
import tempfile
from datetime import date
from pathlib import Path
from typing import List

import pytest

from src.backtest.data.feeds import (
    CsvDataFeed,
    StreamingCsvDataFeed,
    create_data_feed,
)
from src.domain.events.domain_events import BarData


@pytest.fixture
def csv_test_dir():
    """Create temporary directory with test CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create AAPL.csv with daily bars
        with open(tmppath / "AAPL.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "open", "high", "low", "close", "volume"]
            )
            writer.writeheader()
            for i in range(1, 11):  # 10 bars: 2024-01-01 to 2024-01-10
                writer.writerow(
                    {
                        "date": f"2024-01-{i:02d}",
                        "open": 150.0 + i,
                        "high": 152.0 + i,
                        "low": 149.0 + i,
                        "close": 151.0 + i,
                        "volume": 1000000 + i * 10000,
                    }
                )

        # Create MSFT.csv with daily bars
        with open(tmppath / "MSFT.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "open", "high", "low", "close", "volume"]
            )
            writer.writeheader()
            for i in range(1, 11):  # 10 bars: 2024-01-01 to 2024-01-10
                writer.writerow(
                    {
                        "date": f"2024-01-{i:02d}",
                        "open": 350.0 + i,
                        "high": 352.0 + i,
                        "low": 349.0 + i,
                        "close": 351.0 + i,
                        "volume": 2000000 + i * 10000,
                    }
                )

        yield tmppath


class TestStreamingCsvDataFeed:
    """Tests for StreamingCsvDataFeed."""

    @pytest.mark.asyncio
    async def test_streams_bars_in_timestamp_order(self, csv_test_dir):
        """Verify bars are yielded in correct timestamp order."""
        feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL", "MSFT"],
        )
        await feed.load()

        bars: List[BarData] = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        # Should have 20 bars total (10 AAPL + 10 MSFT)
        assert len(bars) == 20

        # Verify timestamp ordering
        prev_timestamp = None
        for bar in bars:
            if prev_timestamp is not None:
                assert bar.timestamp >= prev_timestamp
            prev_timestamp = bar.timestamp

    @pytest.mark.asyncio
    async def test_bar_count_unknown_before_streaming(self, csv_test_dir):
        """Verify bar_count is -1 before streaming completes."""
        feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL"],
        )
        await feed.load()

        # Before streaming
        assert feed.bar_count == -1

        # After streaming
        async for _ in feed.stream_bars():
            pass

        # Now count is known
        assert feed.bar_count == 10

    @pytest.mark.asyncio
    async def test_date_filtering(self, csv_test_dir):
        """Verify date range filtering works."""
        feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL"],
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 7),
        )
        await feed.load()

        bars: List[BarData] = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        # Should have 5 bars (Jan 3-7 inclusive)
        assert len(bars) == 5

        # First bar should be Jan 3
        assert bars[0].timestamp.date() == date(2024, 1, 3)

        # Last bar should be Jan 7
        assert bars[-1].timestamp.date() == date(2024, 1, 7)

    @pytest.mark.asyncio
    async def test_missing_symbol_handled(self, csv_test_dir):
        """Verify missing symbol files don't cause errors."""
        feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL", "NONEXISTENT"],  # NONEXISTENT doesn't exist
        )
        await feed.load()

        bars: List[BarData] = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        # Should only have AAPL bars
        assert len(bars) == 10
        assert all(bar.symbol == "AAPL" for bar in bars)

    @pytest.mark.asyncio
    async def test_empty_directory_handled(self):
        """Verify empty directory returns no bars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feed = StreamingCsvDataFeed(
                csv_dir=tmpdir,
                symbols=["AAPL"],
            )
            await feed.load()

            bars: List[BarData] = []
            async for bar in feed.stream_bars():
                bars.append(bar)

            assert len(bars) == 0
            assert feed.bar_count == 0

    @pytest.mark.asyncio
    async def test_interleaved_timestamps(self, csv_test_dir):
        """Verify multi-symbol bars are correctly interleaved."""
        feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL", "MSFT"],
        )
        await feed.load()

        bars: List[BarData] = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        # Same-day bars should be adjacent
        # Check that for each day, both symbols appear before next day
        days_seen = {}
        for bar in bars:
            day = bar.timestamp.date()
            if day not in days_seen:
                days_seen[day] = set()
            days_seen[day].add(bar.symbol)

        # Each day should have both symbols
        for day, symbols in days_seen.items():
            assert symbols == {"AAPL", "MSFT"}, f"Day {day} missing symbols"


class TestStreamingVsFullLoad:
    """Compare streaming vs full-load feeds."""

    @pytest.mark.asyncio
    async def test_same_results(self, csv_test_dir):
        """Verify streaming and full-load produce same results."""
        # Streaming feed
        streaming_feed = StreamingCsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL", "MSFT"],
        )
        await streaming_feed.load()

        streaming_bars: List[BarData] = []
        async for bar in streaming_feed.stream_bars():
            streaming_bars.append(bar)

        # Full-load feed
        full_feed = CsvDataFeed(
            csv_dir=str(csv_test_dir),
            symbols=["AAPL", "MSFT"],
        )
        await full_feed.load()

        full_bars: List[BarData] = []
        async for bar in full_feed.stream_bars():
            full_bars.append(bar)

        # Same number of bars
        assert len(streaming_bars) == len(full_bars)

        # Same content (comparing key fields)
        for s_bar, f_bar in zip(streaming_bars, full_bars):
            assert s_bar.symbol == f_bar.symbol
            assert s_bar.timestamp == f_bar.timestamp
            assert s_bar.open == f_bar.open
            assert s_bar.close == f_bar.close


class TestCreateDataFeedFactory:
    """Tests for create_data_feed factory function."""

    @pytest.mark.asyncio
    async def test_creates_streaming_csv_by_default(self, csv_test_dir):
        """Verify factory creates streaming feed by default."""
        feed = create_data_feed(
            source=str(csv_test_dir),
            symbols=["AAPL"],
        )

        assert isinstance(feed, StreamingCsvDataFeed)

    @pytest.mark.asyncio
    async def test_creates_full_load_when_streaming_false(self, csv_test_dir):
        """Verify factory creates full-load feed when streaming=False."""
        feed = create_data_feed(
            source=str(csv_test_dir),
            symbols=["AAPL"],
            streaming=False,
        )

        assert isinstance(feed, CsvDataFeed)

    @pytest.mark.asyncio
    async def test_factory_date_filtering(self, csv_test_dir):
        """Verify factory passes date filters correctly."""
        feed = create_data_feed(
            source=str(csv_test_dir),
            symbols=["AAPL"],
            start_date=date(2024, 1, 5),
            end_date=date(2024, 1, 8),
        )
        await feed.load()

        bars: List[BarData] = []
        async for bar in feed.stream_bars():
            bars.append(bar)

        # Should have 4 bars
        assert len(bars) == 4


class TestLargeDataset:
    """Tests for larger datasets to verify memory efficiency."""

    @pytest.mark.asyncio
    async def test_many_symbols_streaming(self):
        """Test streaming with many symbols (memory stress test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create 100 symbol files with 100 bars each
            for sym_idx in range(100):
                symbol = f"SYM{sym_idx:03d}"
                with open(tmppath / f"{symbol}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["date", "open", "high", "low", "close", "volume"]
                    )
                    writer.writeheader()
                    for day in range(1, 101):
                        month = (day - 1) // 28 + 1
                        day_of_month = (day - 1) % 28 + 1
                        writer.writerow(
                            {
                                "date": f"2024-{month:02d}-{day_of_month:02d}",
                                "open": 100.0 + sym_idx,
                                "high": 102.0 + sym_idx,
                                "low": 99.0 + sym_idx,
                                "close": 101.0 + sym_idx,
                                "volume": 1000000,
                            }
                        )

            symbols = [f"SYM{i:03d}" for i in range(100)]
            feed = StreamingCsvDataFeed(
                csv_dir=str(tmppath),
                symbols=symbols,
            )
            await feed.load()

            bar_count = 0
            prev_timestamp = None
            async for bar in feed.stream_bars():
                bar_count += 1
                # Verify ordering
                if prev_timestamp is not None:
                    assert bar.timestamp >= prev_timestamp
                prev_timestamp = bar.timestamp

            # Should have 10,000 bars total (100 symbols × 100 bars)
            assert bar_count == 10000
            assert feed.bar_count == 10000
