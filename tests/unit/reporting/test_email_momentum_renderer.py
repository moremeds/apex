"""Tests for src.infrastructure.reporting.email_momentum_renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.reporting.email_momentum_renderer import (
    render_momentum_email_text,
)


@pytest.fixture()
def watchlist_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "momentum_watchlist.json"


@pytest.fixture()
def sample_candidates() -> list[dict]:
    return [
        {
            "rank": 1,
            "symbol": "AAPL",
            "momentum_12_1": 0.3215,
            "fip": 1.42,
            "momentum_percentile": 0.95,
            "fip_percentile": 0.88,
            "composite_rank": 0.92,
            "last_close": 198.50,
            "market_cap": 3100000000000,
            "avg_daily_dollar_volume": 12500000000,
            "liquidity_tier": "mega",
            "estimated_slippage_bps": 1,
            "lookback_days": 252,
            "quality_label": "STRONG",
            "position_size_factor": 1.0,
            "regime": "R0",
        },
        {
            "rank": 2,
            "symbol": "NVDA",
            "momentum_12_1": 0.2850,
            "fip": 0.95,
            "momentum_percentile": 0.90,
            "fip_percentile": 0.75,
            "composite_rank": 0.83,
            "last_close": 875.20,
            "market_cap": 2200000000000,
            "avg_daily_dollar_volume": 18000000000,
            "liquidity_tier": "mega",
            "estimated_slippage_bps": 2,
            "lookback_days": 252,
            "quality_label": "GOOD",
            "position_size_factor": 0.8,
            "regime": "R0",
        },
    ]


class TestRenderMomentumEmailText:
    def test_render_missing_file(self, tmp_path: Path) -> None:
        result = render_momentum_email_text(tmp_path / "nonexistent.json")
        assert result == "Momentum Screen: No data available."

    def test_render_zero_candidates(self, watchlist_path: Path) -> None:
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 500,
            "passed_filters": 120,
            "regime": "R1",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": [],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))

        assert "0 momentum candidates" in result
        assert "Choppy/Extended" in result
        assert "R1" in result
        assert "universe: 500" in result

    def test_render_candidates(self, watchlist_path: Path, sample_candidates: list[dict]) -> None:
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": sample_candidates,
            "universe_size": 500,
            "passed_filters": 120,
            "regime": "R0",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": [],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))

        # Verify header
        assert "Momentum Screen" in result
        assert "Healthy Uptrend" in result

        # Verify candidate fields
        assert "#1 AAPL [MEGA]" in result
        assert "#2 NVDA [MEGA]" in result
        assert "Mom 12-1: +32.1%" in result
        assert "FIP: 1.42" in result
        assert "Composite: 0.92" in result
        assert "Quality: STRONG" in result
        assert "Close: $198.50" in result
        assert "MktCap: $3.1T" in result
        assert "ADDV: $12,500,000,000" in result
        assert "Slippage: ~1bps" in result
        assert "Size: 100%" in result

    def test_regime_name_in_output(self, watchlist_path: Path) -> None:
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 100,
            "passed_filters": 30,
            "regime": "R2",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": [],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))
        assert "Risk-Off" in result

    def test_errors_displayed(self, watchlist_path: Path) -> None:
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 100,
            "passed_filters": 30,
            "regime": "R0",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": ["TSLA: insufficient data", "GME: delisted"],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))
        assert "Errors (2)" in result
        assert "TSLA: insufficient data" in result

    def test_data_as_of_displayed(self, watchlist_path: Path) -> None:
        """data_as_of field should appear in email output when present."""
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 100,
            "passed_filters": 30,
            "regime": "R0",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "data_as_of": "2026-02-17",
            "errors": [],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))
        assert "Data as of: 2026-02-17" in result

    def test_data_as_of_missing(self, watchlist_path: Path) -> None:
        """No data_as_of should not break rendering."""
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 100,
            "passed_filters": 30,
            "regime": "R0",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": [],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))
        assert "Data as of" not in result

    def test_errors_as_list_works(self, watchlist_path: Path) -> None:
        """Errors serialized as list (from dict) should render correctly."""
        watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "candidates": [],
            "universe_size": 100,
            "passed_filters": 30,
            "regime": "R0",
            "generated_at": "2026-02-18T21:30:00+00:00",
            "errors": ["TSLA: momentum computation failed", "GME: FIP computation failed"],
        }
        watchlist_path.write_text(json.dumps(data))

        result = render_momentum_email_text(str(watchlist_path))
        assert "Errors (2)" in result
        assert "TSLA: momentum computation failed" in result
        assert "GME: FIP computation failed" in result
