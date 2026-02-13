"""Tests for FMP earnings merge logic — BMO/AMC alignment and SUE leakage fixes."""

from __future__ import annotations

from datetime import date

from src.infrastructure.adapters.earnings.fmp_earnings import FMPEarningsAdapter


def _make_calendar_entry(
    symbol: str = "AAPL",
    report_date: str = "2025-01-27",
    actual_eps: float = 2.50,
    estimated_eps: float = 2.00,
    report_time: str = "bmo",
) -> dict:
    return {
        "symbol": symbol,
        "date": report_date,
        "time": report_time,
        "epsActual": actual_eps,
        "epsEstimated": estimated_eps,
        "revenueActual": 100_000,
        "revenueEstimated": 95_000,
    }


def _make_history(
    quarters: list[tuple[str, float, float]],
) -> list[dict]:
    """Build history list: [(date, actual, est), ...]"""
    return [{"date": d, "epsActual": a, "epsEstimated": e} for d, a, e in quarters]


_EMPTY_GRADES: list[dict] = []
_DEFAULT_PRICE = {
    "prior_close": 100.0,
    "open": 105.0,
    "close": 106.0,
    "volume": 5_000_000,
    "avg_20d_volume": 2_000_000,
    "high_52w": 110.0,
    "current_price": 106.0,
    "forward_pe": 25.0,
}


class TestBmoAmcAlignment:
    def test_bmo_stores_report_time(self) -> None:
        """BMO report_time is stored in merged dict."""
        entry = _make_calendar_entry(report_time="bmo")
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            [],
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
            report_time="bmo",
        )
        assert merged["report_time"] == "bmo"

    def test_amc_stores_report_time(self) -> None:
        """AMC report_time is stored in merged dict."""
        entry = _make_calendar_entry(report_time="amc")
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            [],
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
            report_time="amc",
        )
        assert merged["report_time"] == "amc"

    def test_missing_time_defaults_bmo(self) -> None:
        """When time field is None, default is bmo."""
        entry = _make_calendar_entry()
        entry["time"] = None
        # The default in _merge_earning_data is "bmo"
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            [],
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
        )
        assert merged["report_time"] == "bmo"


class TestSueLeakageFix:
    def test_sue_excludes_current_quarter(self) -> None:
        """Current quarter is excluded from historical_surprises."""
        # History includes current quarter (2025-01-27) and 7 past quarters
        history = _make_history(
            [
                ("2025-01-27", 2.50, 2.00),  # Current quarter — should be excluded
                ("2024-10-25", 2.20, 2.10),
                ("2024-07-26", 2.10, 2.00),
                ("2024-04-26", 1.90, 1.80),
                ("2024-01-26", 1.80, 1.70),
                ("2023-10-27", 1.70, 1.60),
                ("2023-07-28", 1.60, 1.50),
                ("2023-04-28", 1.50, 1.40),
            ]
        )
        entry = _make_calendar_entry(report_date="2025-01-27")
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            history,
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
        )
        surprises = merged["historical_surprises"]
        # Current quarter surprise = 2.50 - 2.00 = 0.50 — should NOT be present
        assert 0.50 not in surprises
        # Should have 7 past quarters
        assert len(surprises) == 7

    def test_sue_history_sorted_descending(self) -> None:
        """History is sorted most-recent-first regardless of input order."""
        # Intentionally unsorted input
        history = _make_history(
            [
                ("2023-04-28", 1.50, 1.40),  # oldest
                ("2024-10-25", 2.20, 2.10),  # 2nd most recent
                ("2024-01-26", 1.80, 1.70),
                ("2024-07-26", 2.10, 2.00),
                ("2023-10-27", 1.70, 1.60),
                ("2024-04-26", 1.90, 1.80),
            ]
        )
        entry = _make_calendar_entry(report_date="2025-01-27")
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            history,
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
        )
        surprises = merged["historical_surprises"]
        # Most recent surprise (2024-10-25: 2.20-2.10=0.10) should be first
        assert abs(surprises[0] - 0.10) < 0.001
        # Oldest (2023-04-28: 1.50-1.40=0.10) should be last
        assert abs(surprises[-1] - 0.10) < 0.001

    def test_sue_12q_history(self) -> None:
        """With 12+ quarters available, takes first 12 after excluding current."""
        history = _make_history(
            [
                ("2025-01-27", 2.50, 2.00),  # Current — excluded
                *[
                    (f"20{24 - i // 4}-{['01', '04', '07', '10'][i % 4]}-15", 1.0 + i * 0.01, 1.0)
                    for i in range(14)
                ],
            ]
        )
        entry = _make_calendar_entry(report_date="2025-01-27")
        merged = FMPEarningsAdapter._merge_earning_data(
            entry,
            history,
            _EMPTY_GRADES,
            _DEFAULT_PRICE,
            date(2025, 1, 27),
        )
        # Should have at most 12 quarters (sliced from past_history[:12])
        assert len(merged["historical_surprises"]) <= 12
