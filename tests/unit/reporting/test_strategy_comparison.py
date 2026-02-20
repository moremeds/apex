"""
Tests for strategy_comparison — _classify_health, _param_badge,
StrategyComparisonBuilder add_strategy/build, StrategyMetrics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.infrastructure.reporting.strategy_comparison.builder import (
    StrategyComparisonBuilder,
    StrategyMetrics,
)
from src.infrastructure.reporting.strategy_comparison.templates import (
    _classify_health,
    _param_badge,
)

# =============================================================================
# _classify_health
# =============================================================================


class TestClassifyHealth:
    def test_baseline(self) -> None:
        """buy_and_hold always returns BASELINE."""
        css, label = _classify_health("buy_and_hold", {})
        assert label == "BASELINE"
        assert "baseline" in css

    def test_healthy(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.55,
            "total_return": 0.10,
            "max_drawdown": -0.15,
            "sharpe": 1.2,
        }
        css, label = _classify_health("trend_pulse", m)
        assert label == "HEALTHY"
        assert "green" in css

    def test_broken_zero_trades(self) -> None:
        m = {"trade_count": 0, "win_rate": 0, "total_return": 0, "max_drawdown": 0, "sharpe": 0}
        css, label = _classify_health("bad_strategy", m)
        assert label == "BROKEN"
        assert "red" in css

    def test_broken_low_win_rate(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.10,  # < 0.15 threshold
            "total_return": 0.05,
            "max_drawdown": -0.10,
            "sharpe": 0.5,
        }
        _, label = _classify_health("strat", m)
        assert label == "BROKEN"

    def test_broken_severe_drawdown(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.55,
            "total_return": 0.10,
            "max_drawdown": -0.55,  # < -0.50 threshold
            "sharpe": 1.0,
        }
        _, label = _classify_health("strat", m)
        assert label == "BROKEN"

    def test_broken_large_loss(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.55,
            "total_return": -0.06,  # < -0.05 threshold
            "max_drawdown": -0.20,
            "sharpe": -0.5,
        }
        _, label = _classify_health("strat", m)
        assert label == "BROKEN"

    def test_needs_work_low_sharpe(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.35,
            "total_return": 0.01,
            "max_drawdown": -0.20,
            "sharpe": 0.05,  # < 0.1 threshold
        }
        _, label = _classify_health("strat", m)
        assert label == "NEEDS WORK"

    def test_needs_work_negative_return(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.35,
            "total_return": -0.01,  # < 0 threshold for NEEDS WORK
            "max_drawdown": -0.20,
            "sharpe": 0.5,
        }
        _, label = _classify_health("strat", m)
        assert label == "NEEDS WORK"

    def test_needs_work_low_win_rate(self) -> None:
        m = {
            "trade_count": 50,
            "win_rate": 0.25,  # < 0.30 for NEEDS WORK
            "total_return": 0.05,
            "max_drawdown": -0.20,
            "sharpe": 0.5,
        }
        _, label = _classify_health("strat", m)
        assert label == "NEEDS WORK"


# =============================================================================
# _param_badge
# =============================================================================


class TestParamBadge:
    def test_baseline_returns_dash(self) -> None:
        display, css = _param_badge({}, is_baseline=True)
        assert display == "–"
        assert css == ""

    def test_zero_total_returns_dash(self) -> None:
        display, _ = _param_badge({"effective_params": 5, "total_params": 0}, False)
        assert display == "–"

    def test_under_budget(self) -> None:
        display, css = _param_badge({"effective_params": 4, "total_params": 10}, False)
        assert "4/8" in display
        assert css == "positive"

    def test_at_budget(self) -> None:
        display, css = _param_badge({"effective_params": 7, "total_params": 10}, False)
        assert "7/8" in display
        assert "warning" in css

    def test_over_budget(self) -> None:
        display, css = _param_badge({"effective_params": 10, "total_params": 15}, False)
        assert "10/8" in display
        assert css == "negative"


# =============================================================================
# StrategyMetrics
# =============================================================================


class TestStrategyMetrics:
    def test_to_dict_rounding(self) -> None:
        m = StrategyMetrics(
            name="test",
            sharpe=1.23456,
            sortino=2.34567,
            total_return=0.123456,
            max_drawdown=-0.234567,
        )
        d = m.to_dict()
        assert d["sharpe"] == 1.235
        assert d["sortino"] == 2.346
        assert d["total_return"] == 0.1235
        assert d["max_drawdown"] == -0.2346

    def test_tier_b_fields_optional(self) -> None:
        m = StrategyMetrics(name="test")
        d = m.to_dict()
        assert "tier_b_sharpe" not in d
        assert "tier_b_return" not in d

    def test_tier_b_fields_present(self) -> None:
        m = StrategyMetrics(name="test", tier_b_sharpe=0.8, tier_b_return=0.05)
        d = m.to_dict()
        assert d["tier_b_sharpe"] == 0.8
        assert d["tier_b_return"] == 0.05


# =============================================================================
# StrategyComparisonBuilder
# =============================================================================


class TestStrategyComparisonBuilder:
    def test_add_strategy(self) -> None:
        builder = StrategyComparisonBuilder()
        m = StrategyMetrics(name="trend_pulse", sharpe=1.5)
        builder.add_strategy("trend_pulse", m)
        assert "trend_pulse" in builder._strategies

    def test_build_empty_returns_path(self) -> None:
        builder = StrategyComparisonBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.html")
            result = builder.build(path)
            assert result == path

    def test_build_creates_html(self) -> None:
        builder = StrategyComparisonBuilder(title="Test Dashboard")
        m = StrategyMetrics(
            name="trend_pulse",
            sharpe=1.5,
            total_return=0.15,
            max_drawdown=-0.10,
            win_rate=0.55,
            trade_count=50,
            profit_factor=1.8,
        )
        builder.add_strategy("trend_pulse", m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dashboard.html")
            builder.build(path)
            content = Path(path).read_text(encoding="utf-8")
            assert "trend_pulse" in content
            assert "<html" in content.lower()

    def test_set_symbols(self) -> None:
        builder = StrategyComparisonBuilder()
        builder.set_symbols(["AAPL", "SPY"])
        assert builder._symbols == ["AAPL", "SPY"]

    def test_build_creates_parent_dirs(self) -> None:
        builder = StrategyComparisonBuilder()
        m = StrategyMetrics(
            name="test",
            sharpe=1.0,
            total_return=0.1,
            max_drawdown=-0.1,
            win_rate=0.5,
            trade_count=10,
            profit_factor=1.5,
        )
        builder.add_strategy("test", m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "deep" / "report.html")
            builder.build(path)
            assert Path(path).exists()

    def test_to_json_data_empty(self) -> None:
        """to_json_data returns empty dict when no strategies added."""
        builder = StrategyComparisonBuilder()
        assert builder.to_json_data() == {}

    def test_to_json_data_returns_full_bundle(self) -> None:
        """to_json_data returns full comparison bundle with all expected keys."""
        builder = StrategyComparisonBuilder(
            title="Test Dashboard",
            universe_name="test_universe",
            period="2024-01-01 to 2024-12-31",
        )
        m = StrategyMetrics(
            name="trend_pulse",
            sharpe=1.5,
            total_return=0.15,
            max_drawdown=-0.10,
            win_rate=0.55,
            trade_count=50,
        )
        builder.add_strategy("trend_pulse", m)
        builder.set_symbols(["AAPL", "SPY", "NVDA"])
        builder.set_sector_map({"Technology": ["AAPL", "NVDA"], "Finance": ["SPY"]})

        data = builder.to_json_data()

        # Required top-level keys
        assert "title" in data
        assert "generated_at" in data
        assert "universe_name" in data
        assert "period" in data
        assert "strategy_count" in data
        assert "symbols" in data
        assert "strategies" in data
        assert "sector_map" in data

        # Values
        assert data["title"] == "Test Dashboard"
        assert data["universe_name"] == "test_universe"
        assert data["strategy_count"] == 1
        assert data["symbols"] == ["AAPL", "SPY", "NVDA"]
        assert "trend_pulse" in data["strategies"]
        assert data["strategies"]["trend_pulse"]["sharpe"] == 1.5

        # sector_map shape: Dict[str, List[str]]
        assert isinstance(data["sector_map"], dict)
        assert isinstance(data["sector_map"]["Technology"], list)
        assert "AAPL" in data["sector_map"]["Technology"]

    def test_to_json_data_matches_build_data(self) -> None:
        """to_json_data returns same structure as build() would use."""
        builder = StrategyComparisonBuilder()
        m = StrategyMetrics(
            name="buy_and_hold",
            sharpe=0.8,
            total_return=0.12,
            max_drawdown=-0.15,
        )
        builder.add_strategy("buy_and_hold", m)
        builder.set_symbols(["SPY"])

        data = builder.to_json_data()
        assert isinstance(data["strategies"]["buy_and_hold"], dict)
        assert data["strategies"]["buy_and_hold"]["name"] == "buy_and_hold"
        assert data["strategies"]["buy_and_hold"]["sharpe"] == 0.8
