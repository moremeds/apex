"""Tests for momentum runner helper functions."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.screeners.momentum.models import MomentumScreenResult


class TestWriteWatchlistJson:
    def test_errors_serialized_as_list(self, tmp_path: Path) -> None:
        """dict[str, str] errors should become a list of 'key: value' strings."""
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={"TSLA": "insufficient data", "GME": "delisted"},
        )

        out_path = _write_watchlist_json(result, tmp_path)
        data = json.loads(out_path.read_text())

        errors = data["errors"]
        assert isinstance(errors, list)
        assert len(errors) == 2
        assert "TSLA: insufficient data" in errors
        assert "GME: delisted" in errors

    def test_data_as_of_included_when_provided(self, tmp_path: Path) -> None:
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={},
        )

        out_path = _write_watchlist_json(result, tmp_path, data_as_of=date(2026, 2, 17))
        data = json.loads(out_path.read_text())

        assert data["data_as_of"] == "2026-02-17"

    def test_data_as_of_absent_when_none(self, tmp_path: Path) -> None:
        from src.runners.momentum_runner import _write_watchlist_json

        result = MomentumScreenResult(
            candidates=[],
            universe_size=100,
            passed_filters=50,
            regime="R0",
            generated_at=datetime(2026, 2, 18, 12, 0, 0),
            errors={},
        )

        out_path = _write_watchlist_json(result, tmp_path, data_as_of=None)
        data = json.loads(out_path.read_text())

        assert "data_as_of" not in data


class TestGenerateRebalanceDates:
    def test_produces_fridays(self) -> None:
        from src.runners.momentum_runner import _generate_rebalance_dates

        dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 1, 31))
        for d in dates:
            assert d.weekday() == 4, f"{d} is not a Friday"

    def test_empty_when_range_too_small(self) -> None:
        from src.runners.momentum_runner import _generate_rebalance_dates

        # Mon to Wed — no Friday in range
        dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 1, 3))
        assert dates == []

    def test_weekly_spacing(self) -> None:
        from src.runners.momentum_runner import _generate_rebalance_dates

        dates = _generate_rebalance_dates(date(2024, 1, 1), date(2024, 3, 31))
        for i in range(1, len(dates)):
            delta = (dates[i] - dates[i - 1]).days
            assert delta == 7, f"Gap {dates[i-1]} -> {dates[i]} = {delta} days"


class TestComputeMaxDrawdown:
    def test_no_drawdown_in_uptrend(self) -> None:
        from src.runners.momentum_runner import _compute_max_drawdown

        dd = _compute_max_drawdown([0.01, 0.02, 0.03])
        assert dd == 0.0

    def test_full_loss(self) -> None:
        from src.runners.momentum_runner import _compute_max_drawdown

        dd = _compute_max_drawdown([-0.5, -0.5])
        # cumulative: [0.5, 0.25], peak: [0.5, 0.5], drawdown: [0, -0.5]
        assert dd <= -0.5

    def test_empty_returns(self) -> None:
        from src.runners.momentum_runner import _compute_max_drawdown

        assert _compute_max_drawdown([]) == 0.0


class TestRunWalkForward:
    """Tests for _run_walk_forward including PIT market caps and searchsorted."""

    @staticmethod
    def _make_close_series(
        prices: list[float], start: date = date(2024, 1, 2)
    ) -> "pd.Series[float]":
        """Create a DatetimeIndex Series of daily closes (business days)."""
        idx = pd.bdate_range(start=start, periods=len(prices))
        return pd.Series(prices, index=idx, dtype=float)

    @staticmethod
    def _make_mock_screener(pick_symbols: list[str]) -> MagicMock:
        """Create a screener mock that always returns the given symbols."""
        from src.domain.screeners.momentum.models import (
            MomentumCandidate,
            MomentumSignal,
        )
        from src.domain.screeners.pead.models import LiquidityTier

        candidates = []
        for i, sym in enumerate(pick_symbols):
            signal = MomentumSignal(
                symbol=sym,
                momentum_12_1=0.1,
                fip=0.6,
                momentum_percentile=0.8,
                fip_percentile=0.7,
                composite_rank=0.75,
                last_close=100.0,
                market_cap=1e9,
                avg_daily_dollar_volume=50e6,
                liquidity_tier=LiquidityTier.LARGE_CAP,
                estimated_slippage_bps=10,
                lookback_days=252,
            )
            candidates.append(
                MomentumCandidate(
                    signal=signal,
                    rank=i + 1,
                    quality_label="STRONG",
                    position_size_factor=1.0,
                    regime="R0",
                )
            )

        mock_result = MagicMock()
        mock_result.candidates = candidates
        screener = MagicMock()
        screener.screen.return_value = mock_result
        return screener

    def test_pit_market_caps_scaled_by_price(self) -> None:
        """Market caps passed to screener should reflect rebalance-date prices."""
        from src.runners.momentum_runner import _run_walk_forward

        # AAPL: starts at $100, ends at $200 → at rebal, cap should be ~half
        prices = [100.0 + i for i in range(60)]  # 100..159
        series = self._make_close_series(prices)

        screener = self._make_mock_screener(["AAPL"])
        rebal_dates = [series.index[10].date()]  # early date where price ~110

        _run_walk_forward(
            screener=screener,
            all_close_series={"AAPL": series},
            all_volume_series={"AAPL": series * 1000},
            market_caps={"AAPL": 2e9},  # current cap at price=159
            rebalance_dates=rebal_dates,
            top_n=5,
            hold_days=5,
        )

        # Check what market_caps were passed to screener
        call_kwargs = screener.screen.call_args
        pit_caps = call_kwargs.kwargs.get("market_caps") or call_kwargs[1].get("market_caps")
        aapl_pit_cap = pit_caps["AAPL"]

        # At index 10, price = 110; latest price = 159
        # pit_cap should be 2e9 * (110/159) ≈ 1.384e9
        expected = 2e9 * (110.0 / 159.0)
        assert (
            abs(aapl_pit_cap - expected) < 1e6
        ), f"PIT cap {aapl_pit_cap:.0f} != expected {expected:.0f}"

    def test_rebal_on_holiday_uses_preceding_bar(self) -> None:
        """Rebalance date falling on a non-trading day should use the last available bar."""
        from src.runners.momentum_runner import _run_walk_forward

        # Create series with a gap (simulate weekend/holiday)
        idx = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-15",
                "2024-01-16",
                "2024-01-17",
                "2024-01-18",
                "2024-01-19",
                "2024-01-22",
                "2024-01-23",
                "2024-01-24",
            ]
        )
        prices = [100.0 + i * 2 for i in range(len(idx))]
        series = pd.Series(prices, index=idx, dtype=float)

        screener = self._make_mock_screener(["TEST"])

        # Rebalance on Saturday Jan 6 — should use Jan 3 (preceding bar)
        _run_walk_forward(
            screener=screener,
            all_close_series={"TEST": series},
            all_volume_series={"TEST": series * 1000},
            market_caps={"TEST": 1e9},
            rebalance_dates=[date(2024, 1, 5)],  # Friday Jan 5 — not in index
            top_n=5,
            hold_days=3,
        )

        # Screener should still be called (preceding bar Jan 3 exists)
        assert screener.screen.called

    def test_returns_empty_when_no_data(self) -> None:
        from src.runners.momentum_runner import _run_walk_forward

        screener = self._make_mock_screener([])
        result = _run_walk_forward(
            screener=screener,
            all_close_series={},
            all_volume_series={},
            market_caps={},
            rebalance_dates=[date(2024, 1, 5)],
            top_n=5,
            hold_days=5,
        )
        assert result == []


class TestRenderBacktestHtml:
    """Tests for the backtest HTML template."""

    def test_renders_complete_html(self) -> None:
        from src.infrastructure.reporting.momentum.templates import render_backtest_html

        data = {
            "backtest": {
                "start": "2023-01-01",
                "end": "2025-12-31",
                "top_n": 20,
                "hold_days": 5,
                "cumulative_return": 0.45,
                "sharpe_approx": 1.23,
                "max_drawdown": -0.15,
                "avg_weekly_return": 0.004,
                "periods": [
                    {
                        "date": "2023-01-06",
                        "n_picks": 20,
                        "avg_return": 0.012,
                        "picks": ["AAPL", "MSFT"],
                    },
                    {"date": "2023-01-13", "n_picks": 18, "avg_return": -0.005, "picks": ["NVDA"]},
                ],
            },
            "ablation": {
                "top_n": 20,
                "hold_days": 5,
                "configs": [
                    {
                        "label": "M-only",
                        "cumulative_return": 0.30,
                        "sharpe_approx": 0.9,
                        "max_drawdown": -0.20,
                        "avg_weekly_return": 0.003,
                    },
                    {
                        "label": "M+FIP",
                        "cumulative_return": 0.38,
                        "sharpe_approx": 1.1,
                        "max_drawdown": -0.18,
                        "avg_weekly_return": 0.0035,
                    },
                    {
                        "label": "M+FIP+Filters",
                        "cumulative_return": 0.45,
                        "sharpe_approx": 1.23,
                        "max_drawdown": -0.15,
                        "avg_weekly_return": 0.004,
                    },
                ],
            },
        }

        html = render_backtest_html(data)

        # Structure checks
        assert "<!DOCTYPE html>" in html
        assert "Momentum Backtest" in html
        assert "2023-01-01" in html
        assert "2025-12-31" in html

        # Summary cards
        assert "+45.0%" in html  # cum return
        assert "1.23" in html  # sharpe

        # Ablation table
        assert "M-only" in html
        assert "M+FIP" in html
        assert "M+FIP+Filters" in html

        # Period rows
        assert "2023-01-06" in html
        assert "AAPL, MSFT" in html

    def test_handles_empty_data(self) -> None:
        from src.infrastructure.reporting.momentum.templates import render_backtest_html

        html = render_backtest_html({"backtest": {}, "ablation": {}})
        assert "<!DOCTYPE html>" in html
        assert "Momentum Backtest" in html

    def test_ablation_rows_color_coded(self) -> None:
        from src.infrastructure.reporting.momentum.templates import _build_ablation_rows

        rows = _build_ablation_rows(
            [
                {
                    "label": "Positive",
                    "cumulative_return": 0.5,
                    "sharpe_approx": 1.0,
                    "max_drawdown": -0.05,
                    "avg_weekly_return": 0.01,
                },
                {
                    "label": "Negative",
                    "cumulative_return": -0.2,
                    "sharpe_approx": -0.5,
                    "max_drawdown": -0.3,
                    "avg_weekly_return": -0.01,
                },
            ]
        )
        # Positive return → green
        assert "#3fb950" in rows
        # Negative return → red
        assert "#f85149" in rows

    def test_period_rows_return_colors(self) -> None:
        from src.infrastructure.reporting.momentum.templates import _build_period_rows

        rows = _build_period_rows(
            [
                {"date": "2024-01-05", "n_picks": 10, "avg_return": 0.02, "picks": ["A"]},
                {"date": "2024-01-12", "n_picks": 8, "avg_return": -0.01, "picks": ["B"]},
            ]
        )
        assert "#3fb950" in rows  # positive
        assert "#f85149" in rows  # negative
