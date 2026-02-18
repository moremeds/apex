"""Tests for FMPIndexConstituentsAdapter: fetch_us_stocks, get_combined_universe fallback."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.infrastructure.adapters.fmp.index_constituents import FMPIndexConstituentsAdapter


@pytest.fixture()
def adapter() -> FMPIndexConstituentsAdapter:
    """Create adapter with a dummy API key (all network calls are mocked)."""
    return FMPIndexConstituentsAdapter(api_key="test-key")


class TestFetchUsStocks:
    def test_sorts_by_market_cap_descending(self, adapter: FMPIndexConstituentsAdapter) -> None:
        mock_data = [
            {"symbol": "SMALL", "marketCap": 1_000_000_000},
            {"symbol": "BIG", "marketCap": 100_000_000_000},
            {"symbol": "MID", "marketCap": 10_000_000_000},
        ]
        with patch.object(adapter, "_fmp_get", return_value=mock_data):
            result = adapter.fetch_us_stocks(max_symbols=10)
        assert result == ["BIG", "MID", "SMALL"]

    def test_caps_at_max_symbols(self, adapter: FMPIndexConstituentsAdapter) -> None:
        mock_data = [{"symbol": f"S{i}", "marketCap": i * 1e9} for i in range(10)]
        with patch.object(adapter, "_fmp_get", return_value=mock_data):
            result = adapter.fetch_us_stocks(max_symbols=3)
        assert len(result) == 3

    def test_handles_none_market_cap(self, adapter: FMPIndexConstituentsAdapter) -> None:
        """None marketCap should not crash — treated as 0."""
        mock_data = [
            {"symbol": "GOOD", "marketCap": 5_000_000_000},
            {"symbol": "NULL_CAP", "marketCap": None},
            {"symbol": "MISSING_CAP"},  # No marketCap key at all
        ]
        with patch.object(adapter, "_fmp_get", return_value=mock_data):
            result = adapter.fetch_us_stocks(max_symbols=10)
        assert "GOOD" in result
        assert "NULL_CAP" in result
        assert "MISSING_CAP" in result
        # GOOD should be first (highest cap)
        assert result[0] == "GOOD"

    def test_handles_string_market_cap(self, adapter: FMPIndexConstituentsAdapter) -> None:
        """Non-numeric marketCap (e.g. string) should be treated as 0."""
        mock_data = [
            {"symbol": "GOOD", "marketCap": 5_000_000_000},
            {"symbol": "BAD", "marketCap": "not a number"},
        ]
        with patch.object(adapter, "_fmp_get", return_value=mock_data):
            result = adapter.fetch_us_stocks(max_symbols=10)
        assert result == ["GOOD", "BAD"]

    def test_skips_invalid_symbols(self, adapter: FMPIndexConstituentsAdapter) -> None:
        mock_data = [
            {"symbol": "OK", "marketCap": 1e9},
            {"symbol": "", "marketCap": 2e9},  # Empty symbol
            {"symbol": 123, "marketCap": 3e9},  # Non-string symbol
            {"marketCap": 4e9},  # Missing symbol key
        ]
        with patch.object(adapter, "_fmp_get", return_value=mock_data):
            result = adapter.fetch_us_stocks(max_symbols=10)
        assert result == ["OK"]

    def test_empty_response(self, adapter: FMPIndexConstituentsAdapter) -> None:
        with patch.object(adapter, "_fmp_get", return_value=[]):
            result = adapter.fetch_us_stocks()
        assert result == []

    def test_non_list_response(self, adapter: FMPIndexConstituentsAdapter) -> None:
        with patch.object(adapter, "_fmp_get", return_value={"error": "forbidden"}):
            result = adapter.fetch_us_stocks()
        assert result == []


class TestGetCombinedUniverseFallback:
    def test_fallback_when_constituents_empty(self, adapter: FMPIndexConstituentsAdapter) -> None:
        """When sp500 + nasdaq both return 0 (402), should use company-screener."""
        with (
            patch.object(adapter, "fetch_sp500", return_value=[]),
            patch.object(adapter, "fetch_nasdaq", return_value=[]),
            patch.object(adapter, "fetch_russell_proxy", return_value=["RSL1"]),
            patch.object(adapter, "fetch_us_stocks", return_value=["FB1", "FB2"]) as mock_fallback,
        ):
            result = adapter.get_combined_universe(
                indices=["sp500", "nasdaq"],
                russell_proxy=True,
                fallback_max_symbols=500,
            )
        mock_fallback.assert_called_once_with(cap_min=500_000_000, max_symbols=500)
        assert "FB1" in result
        assert "FB2" in result
        assert "RSL1" in result

    def test_no_fallback_when_constituents_present(
        self, adapter: FMPIndexConstituentsAdapter
    ) -> None:
        """When constituent endpoints work, should NOT call fallback."""
        with (
            patch.object(adapter, "fetch_sp500", return_value=["AAPL"]),
            patch.object(adapter, "fetch_nasdaq", return_value=[]),
            patch.object(adapter, "fetch_russell_proxy", return_value=[]),
            patch.object(adapter, "fetch_us_stocks") as mock_fallback,
        ):
            result = adapter.get_combined_universe(indices=["sp500", "nasdaq"])
        mock_fallback.assert_not_called()
        assert "AAPL" in result
