"""Tests for MomentumDataService new methods: get_data_as_of_date, get_upcoming_earnings."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.services.momentum_data_service import MomentumDataService


@pytest.fixture()
def svc(tmp_path: Path) -> MomentumDataService:
    """Create service with temp dirs."""
    return MomentumDataService(
        universe_cache_path=tmp_path / "universe.json",
        historical_base_dir=tmp_path / "historical",
    )


def _write_parquet(hist_dir: Path, symbol: str, last_date: str) -> None:
    """Write a minimal Parquet file with given last date."""
    sym_dir = hist_dir / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(end=last_date, periods=10, freq="B")
    df = pd.DataFrame(
        {"close": range(10), "volume": range(10)},
        index=dates,
    )
    df.index.name = "timestamp"
    df.to_parquet(sym_dir / "1d.parquet")


class TestGetDataAsOfDate:
    def test_returns_max_date(self, svc: MomentumDataService, tmp_path: Path) -> None:
        hist = tmp_path / "historical"
        _write_parquet(hist, "AAPL", "2026-02-14")
        _write_parquet(hist, "MSFT", "2026-02-17")
        _write_parquet(hist, "GOOG", "2026-02-10")

        result = svc.get_data_as_of_date(["AAPL", "GOOG", "MSFT"])
        assert result == date(2026, 2, 17)

    def test_deterministic_sample(self, svc: MomentumDataService, tmp_path: Path) -> None:
        """Same input always yields same output (no random sampling)."""
        hist = tmp_path / "historical"
        _write_parquet(hist, "AAPL", "2026-02-14")
        _write_parquet(hist, "MSFT", "2026-02-17")

        r1 = svc.get_data_as_of_date(["MSFT", "AAPL"])
        r2 = svc.get_data_as_of_date(["MSFT", "AAPL"])
        assert r1 == r2

    def test_empty_symbols(self, svc: MomentumDataService) -> None:
        assert svc.get_data_as_of_date([]) is None

    def test_no_parquet_files(self, svc: MomentumDataService) -> None:
        assert svc.get_data_as_of_date(["AAPL", "MSFT"]) is None


class TestGetUpcomingEarnings:
    def test_happy_path(self, svc: MomentumDataService) -> None:
        today = date.today()
        tomorrow = today + timedelta(days=1)
        mock_calendar = [
            {"symbol": "AAPL", "date": tomorrow.isoformat()},
            {"symbol": "TSLA", "date": tomorrow.isoformat()},
            {"symbol": "OTHER", "date": tomorrow.isoformat()},
        ]

        with patch(
            "src.services.momentum_data_service.MomentumDataService.get_upcoming_earnings",
            wraps=svc.get_upcoming_earnings,
        ):
            with patch(
                "src.infrastructure.adapters.earnings.fmp_earnings.FMPEarningsAdapter"
            ) as mock_cls:
                instance = MagicMock()
                instance.fetch_earnings_calendar.return_value = mock_calendar
                mock_cls.return_value = instance

                result = svc.get_upcoming_earnings(["AAPL", "TSLA"], lookahead_days=7)

        assert "AAPL" in result
        assert "TSLA" in result
        assert "OTHER" not in result  # Not in provided symbols list

    def test_fail_open_on_error(self, svc: MomentumDataService) -> None:
        """Should return empty dict on any exception, not crash."""
        with patch(
            "src.infrastructure.adapters.earnings.fmp_earnings.FMPEarningsAdapter",
            side_effect=ValueError("no API key"),
        ):
            result = svc.get_upcoming_earnings(["AAPL"])
        assert result == {}
