"""Unit tests for EarningsService cache round-trip."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.services.earnings_service import EarningsCache, EarningsService


class TestEarningsCache:
    def test_cache_round_trip(self, tmp_path: Path) -> None:
        """Save + load preserves data."""
        cache_path = tmp_path / "earnings.json"
        service = EarningsService(cache_path=cache_path)

        cache = EarningsCache(
            version="1.0",
            updated_at=datetime(2025, 1, 27, 10, 0),
            source="fmp",
            earnings={"AAPL": [{"symbol": "AAPL", "report_date": "2025-01-27", "actual_eps": 2.5}]},
            skipped_count=3,
        )
        service._save_cache(cache)

        # Force reload
        service.clear_cache()
        loaded = service._load_cache()

        assert loaded.version == "1.0"
        assert loaded.source == "fmp"
        assert "AAPL" in loaded.earnings
        assert loaded.earnings["AAPL"][0]["actual_eps"] == 2.5
        assert loaded.skipped_count == 3

    def test_empty_cache(self, tmp_path: Path) -> None:
        """Missing file returns empty cache."""
        cache_path = tmp_path / "nonexistent.json"
        service = EarningsService(cache_path=cache_path)
        cache = service._load_cache()

        assert cache.earnings == {}
        assert cache.updated_at is None

    def test_cache_metadata(self, tmp_path: Path) -> None:
        """Metadata reflects cache state."""
        cache_path = tmp_path / "earnings.json"
        cache_path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "updated_at": "2025-01-27T10:00:00",
                    "source": "fmp",
                    "earnings": {"AAPL": [{}], "NVDA": [{}]},
                    "skipped_count": 5,
                    "errors": {"TSLA": "FMP tier limit"},
                }
            )
        )

        service = EarningsService(cache_path=cache_path)
        meta = service.get_cache_metadata()

        assert meta["symbol_count"] == 2
        assert meta["skipped_count"] == 5
        assert meta["error_count"] == 1
