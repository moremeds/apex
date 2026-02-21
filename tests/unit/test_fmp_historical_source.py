"""Tests for FMP Historical Source Adapter wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.adapters.fmp.historical_source_adapter import (
    FMPHistoricalSourceAdapter,
)


@pytest.fixture
def adapter() -> FMPHistoricalSourceAdapter:
    """Create adapter with mocked inner FMPHistoricalAdapter."""
    with patch(
        "src.infrastructure.adapters.fmp.historical_source_adapter.FMPHistoricalAdapter"
    ) as mock_cls:
        mock_inner = MagicMock()
        mock_cls.return_value = mock_inner
        a = FMPHistoricalSourceAdapter()
        a._inner = mock_inner
        return a


class TestSourceProperties:
    def test_source_name(self, adapter: FMPHistoricalSourceAdapter) -> None:
        assert adapter.source_name == "fmp"

    def test_supports_timeframe_valid(self, adapter: FMPHistoricalSourceAdapter) -> None:
        assert adapter.supports_timeframe("1h") is True
        assert adapter.supports_timeframe("1d") is True
        assert adapter.supports_timeframe("4h") is True

    def test_supports_timeframe_invalid(self, adapter: FMPHistoricalSourceAdapter) -> None:
        assert adapter.supports_timeframe("2h") is False
        assert adapter.supports_timeframe("3d") is False

    def test_max_history_days_warmup_safe(self, adapter: FMPHistoricalSourceAdapter) -> None:
        # These must match Yahoo (warmup-safe), NOT FMP per-request caps
        assert adapter.get_max_history_days("1h") == 730
        assert adapter.get_max_history_days("4h") == 730
        assert adapter.get_max_history_days("1d") == 18250

    def test_get_supported_timeframes(self, adapter: FMPHistoricalSourceAdapter) -> None:
        tfs = adapter.get_supported_timeframes()
        assert "1h" in tfs
        assert "1d" in tfs
        assert "4h" in tfs


class TestFetchBars:
    @pytest.mark.asyncio
    async def test_fetch_bars_converts_dataframe(self, adapter: FMPHistoricalSourceAdapter) -> None:
        """Mock inner adapter returns a DataFrame, verify BarData list output."""
        dates = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [105.0] * 5,
                "low": [95.0] * 5,
                "close": [102.0] * 5,
                "volume": [1000] * 5,
            },
            index=dates,
        )
        adapter._inner.fetch_bars.return_value = df

        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 5)
        bars = await adapter.fetch_bars("AAPL", "1d", start, end)

        assert len(bars) == 5
        assert bars[0].symbol == "AAPL"
        assert bars[0].timeframe == "1d"
        assert bars[0].source == "fmp"
        assert bars[0].close == 102.0

    @pytest.mark.asyncio
    async def test_fetch_bars_chunks_large_range(self, adapter: FMPHistoricalSourceAdapter) -> None:
        """Request 1h over 200 days should chunk into ~85-day windows."""
        adapter._inner.fetch_bars.return_value = pd.DataFrame()

        start = datetime(2025, 1, 1)
        end = datetime(2025, 7, 20)  # ~200 days
        await adapter.fetch_bars("AAPL", "1h", start, end)

        # 200 days / 85 per chunk = 3 chunks (ceil)
        assert adapter._inner.fetch_bars.call_count == 3

    @pytest.mark.asyncio
    async def test_fetch_bars_empty_dataframe(self, adapter: FMPHistoricalSourceAdapter) -> None:
        adapter._inner.fetch_bars.return_value = pd.DataFrame()

        bars = await adapter.fetch_bars("AAPL", "1d", datetime(2025, 1, 1), datetime(2025, 1, 5))
        assert bars == []

    @pytest.mark.asyncio
    async def test_fetch_bars_deduplicates_across_chunks(
        self, adapter: FMPHistoricalSourceAdapter
    ) -> None:
        """Two chunks with overlapping bars should produce unique output."""
        # Chunk 1 returns bars for Jan 1-3, chunk 2 returns bars for Jan 3-5
        # Jan 3 appears in both chunks
        chunk1_dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        chunk2_dates = pd.date_range("2025-01-03", periods=3, freq="D", tz="UTC")

        def make_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "open": [100.0] * len(dates),
                    "high": [105.0] * len(dates),
                    "low": [95.0] * len(dates),
                    "close": [102.0] * len(dates),
                    "volume": [1000] * len(dates),
                },
                index=dates,
            )

        # Use side_effect to return different data per chunk call
        adapter._inner.fetch_bars.side_effect = [make_df(chunk1_dates), make_df(chunk2_dates)]

        # Force 2-day chunks so we get 2 chunks over a 3-day range
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 5)  # Will produce 2 chunks with chunk_days=2

        # Call _compute_chunks + fetch directly with small chunk_days
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 2)
        assert len(chunks) >= 2  # Must produce multiple chunks

        # Now test through fetch_bars by patching _FMP_CHUNK_DAYS
        adapter._inner.fetch_bars.side_effect = [make_df(chunk1_dates), make_df(chunk2_dates)]
        with patch(
            "src.infrastructure.adapters.fmp.historical_source_adapter._FMP_CHUNK_DAYS",
            {"1h": 2, "4h": 170, "1d": 2},
        ):
            bars = await adapter.fetch_bars("AAPL", "1d", start, end)

        # Jan 1, 2, 3, 4, 5 = 5 unique dates (Jan 3 deduplicated)
        assert len(bars) == 5


class TestChunkComputation:
    def test_single_chunk(self) -> None:
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 10)
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 30)
        assert len(chunks) == 1
        assert chunks[0] == (start, end)

    def test_multiple_chunks(self) -> None:
        start = datetime(2025, 1, 1)
        end = datetime(2025, 7, 1)  # ~181 days
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 85)
        assert len(chunks) == 3
        # Each chunk should be at most 85 days
        for cs, ce in chunks:
            assert (ce - cs).days <= 85

    def test_start_equals_end(self) -> None:
        """Single-day range must produce exactly one chunk."""
        dt = datetime(2025, 3, 15)
        chunks = FMPHistoricalSourceAdapter._compute_chunks(dt, dt, 85)
        assert len(chunks) == 1
        assert chunks[0] == (dt, dt)

    def test_trailing_day_not_dropped(self) -> None:
        """N*chunk + 1 days must not drop the trailing day.

        86 days with 85-day chunks: chunk 1 covers days 0-84,
        chunk 2 must cover day 85.
        """
        start = datetime(2025, 1, 1)
        end = start + timedelta(days=86)  # 87 days total
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 85)

        assert len(chunks) == 2
        # Last chunk must reach end exactly
        assert chunks[-1][1] == end
        # First chunk must not exceed 85 days
        assert (chunks[0][1] - chunks[0][0]).days == 85

    def test_exact_chunk_boundary(self) -> None:
        """Exactly chunk_days range should be one chunk, not two."""
        start = datetime(2025, 1, 1)
        end = start + timedelta(days=85)
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 85)
        assert len(chunks) == 1
        assert chunks[0] == (start, end)

    def test_chunks_cover_full_range(self) -> None:
        """All chunks together must cover [start, end] with no gaps."""
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)  # 364 days
        chunks = FMPHistoricalSourceAdapter._compute_chunks(start, end, 85)

        # First chunk starts at start
        assert chunks[0][0] == start
        # Last chunk ends at end
        assert chunks[-1][1] == end
        # Each chunk starts exactly 1 day after the previous ends (no gaps)
        for i in range(1, len(chunks)):
            assert chunks[i][0] == chunks[i - 1][1] + timedelta(days=1)
