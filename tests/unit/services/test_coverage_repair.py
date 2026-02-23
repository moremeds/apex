"""Tests for coverage auto-repair from existing Parquet data."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.interfaces.historical_source import DateRange
from src.services.historical_data_manager import HistoricalDataManager


class TestCoverageAutoRepair:
    """Test the coverage auto-repair logic in HistoricalDataManager.ensure_data()."""

    def _make_manager(self) -> HistoricalDataManager:
        """Create a mock HistoricalDataManager with the ensure_data repair logic."""
        with patch.object(HistoricalDataManager, "_init_sources"):
            manager = HistoricalDataManager.__new__(HistoricalDataManager)
            manager._base_dir = MagicMock()
            manager._source_priority = ["fmp", "yahoo"]
            manager._sources = {}
            manager._coverage_store = MagicMock()
            manager._bar_store = MagicMock()
            return manager

    @pytest.mark.asyncio
    async def test_repair_when_no_coverage_records(self) -> None:
        """Empty DuckDB + Parquet with data -> coverage created, gaps re-checked."""
        manager = self._make_manager()

        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 30)
        parquet_start = datetime(2025, 1, 2)
        parquet_end = datetime(2025, 6, 29)

        # First find_gaps returns a gap
        gap = DateRange(start=start, end=end)
        # After repair, second find_gaps returns empty (fully covered)
        manager._coverage_store.find_gaps.side_effect = [[gap], []]
        manager._coverage_store.get_coverage.return_value = []  # No records
        manager._bar_store.get_date_range.return_value = (parquet_start, parquet_end)
        manager._bar_store.get_bar_count.return_value = 125
        manager._bar_store.read_bars.return_value = []

        await manager.ensure_data("AAPL", "1d", start, end)

        # Verify coverage was repaired
        manager._coverage_store.update_coverage.assert_called_once_with(
            symbol="AAPL",
            timeframe="1d",
            source="local",
            start=parquet_start,
            end=parquet_end,
            bar_count=125,
        )
        # Verify gaps re-checked after repair
        assert manager._coverage_store.find_gaps.call_count == 2

    @pytest.mark.asyncio
    async def test_no_repair_when_partial_coverage(self) -> None:
        """DuckDB has records with gap -> NO repair attempted, gap preserved for download."""
        manager = self._make_manager()

        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 30)

        gap = DateRange(start=datetime(2025, 3, 1), end=end)
        manager._coverage_store.find_gaps.return_value = [gap]
        # Has existing coverage records (partial)
        manager._coverage_store.get_coverage.return_value = [
            MagicMock(start=start, end=datetime(2025, 2, 28))
        ]
        manager._bar_store.read_bars.return_value = []

        # Mock _download_range to avoid actual network calls
        manager._download_range = AsyncMock()

        await manager.ensure_data("AAPL", "1d", start, end)

        # update_coverage should NOT be called for repair
        manager._coverage_store.update_coverage.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_repair_when_no_parquet(self) -> None:
        """Empty DuckDB + no Parquet -> gap passed to download."""
        manager = self._make_manager()

        start = datetime(2025, 1, 1)
        end = datetime(2025, 6, 30)

        gap = DateRange(start=start, end=end)
        manager._coverage_store.find_gaps.return_value = [gap]
        manager._coverage_store.get_coverage.return_value = []
        manager._bar_store.get_date_range.return_value = None  # No parquet
        manager._bar_store.read_bars.return_value = []

        manager._download_range = AsyncMock()

        await manager.ensure_data("AAPL", "1d", start, end)

        # No repair should happen
        manager._coverage_store.update_coverage.assert_not_called()
        # Download should be attempted
        manager._download_range.assert_called_once()
