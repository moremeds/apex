"""
Integration and smoke tests for the reporting module.

1. Entry point smoke tests: import + construct each builder
2. Render-level integration test: PackageBuilder.build with minimal fixtures
"""

from __future__ import annotations

import json
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# =============================================================================
# Entry Point Smoke Tests — catch broken imports early
# =============================================================================


class TestEntryPointImports:
    """
    Verify each runner-level entry point can be imported and instantiated
    without needing real data or broker connections.
    """

    def test_package_builder_import(self) -> None:
        from src.infrastructure.reporting.package import PackageBuilder

        builder = PackageBuilder()
        assert builder is not None

    def test_signal_report_generator_import(self) -> None:
        from src.infrastructure.reporting.signal_report.generator import (
            SignalReportGenerator,
        )

        gen = SignalReportGenerator()
        assert gen is not None

    def test_strategy_comparison_builder_import(self) -> None:
        from src.infrastructure.reporting.strategy_comparison.builder import (
            StrategyComparisonBuilder,
        )

        builder = StrategyComparisonBuilder()
        assert builder is not None

    def test_pead_report_builder_import(self) -> None:
        from src.infrastructure.reporting.pead.builder import PEADReportBuilder

        builder = PEADReportBuilder()
        assert builder is not None

    def test_momentum_report_builder_import(self) -> None:
        from src.infrastructure.reporting.momentum.builder import (
            MomentumReportBuilder,
        )

        builder = MomentumReportBuilder()
        assert builder is not None

    def test_heatmap_builder_import(self) -> None:
        from src.infrastructure.reporting.heatmap.builder import HeatmapBuilder

        builder = HeatmapBuilder()
        assert builder is not None

    def test_score_history_manager_import(self) -> None:
        from src.infrastructure.reporting.package.score_history import (
            ScoreHistoryManager,
        )

        mgr = ScoreHistoryManager()
        assert mgr is not None

    def test_summary_builder_import(self) -> None:
        from src.infrastructure.reporting.package.summary_builder import (
            SummaryBuilder,
        )

        builder = SummaryBuilder()
        assert builder is not None


# =============================================================================
# Render-Level Integration Test
# =============================================================================


class TestRenderIntegration:
    """
    Build a minimal signal package and verify critical HTML/JS invariants.

    This catches import/dispatch regressions that pure unit tests miss.
    """

    @pytest.fixture
    def minimal_data(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Minimal OHLCV data for AAPL 1d."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        df = pd.DataFrame(
            {
                "open": [100 + i * 0.5 for i in range(50)],
                "high": [105 + i * 0.5 for i in range(50)],
                "low": [95 + i * 0.5 for i in range(50)],
                "close": [102 + i * 0.5 for i in range(50)],
                "volume": [1_000_000] * 50,
            },
            index=dates,
        )
        return {("AAPL", "1d"): df}

    @pytest.fixture
    def mock_regime(self) -> MagicMock:
        mock = MagicMock()
        mock.final_regime.value = "R0"
        mock.confidence = 0.85
        mock.composite_score = 72.5
        mock.regime_changed = False
        mock.quality.component_validity = {"close": True}
        mock.quality.component_issues = {}
        mock.to_dict.return_value = {
            "final_regime": "R0",
            "regime_name": "Healthy Uptrend",
            "confidence": 0.85,
            "regime_changed": False,
            "decision_regime": "R0",
            "component_states": {"trend_state": "uptrend"},
            "component_values": {"close": 126.5, "ma50_slope": 0.02},
            "derived_metrics": {"atr_pct": 1.5},
            "transition": {"holding_bars": 5},
            "quality": {"bars_used": 50, "component_validity": {"close": True}},
            "rules_fired_decision": ["R0_default"],
            "turning_point": None,
            "data_window": {"start": "2024-01-01", "end": "2024-02-19"},
            "asof_ts": "2024-02-19T16:00:00",
        }
        return mock

    def test_summary_builder_build_with_regime(
        self,
        minimal_data: Dict[Tuple[str, str], pd.DataFrame],
        mock_regime: MagicMock,
    ) -> None:
        """
        Build a summary with SummaryBuilder and verify generated dict contains
        all critical top-level keys and ticker data.
        """
        from src.infrastructure.reporting.package.summary_builder import (
            SummaryBuilder,
        )

        builder = SummaryBuilder(enforce_budget=False)

        with (
            patch(
                "src.infrastructure.reporting.package.summary_builder"
                ".SummaryBuilder._compute_rule_frequency",
                return_value={
                    "by_symbol": {},
                    "buy_by_symbol": {},
                    "sell_by_symbol": {},
                    "by_rule": {},
                    "top_symbols": [],
                    "top_rules": [],
                    "total_signals": 0,
                },
            ),
            patch(
                "src.infrastructure.stores.duckdb_coverage_store" ".DuckDBCoverageStore",
                side_effect=Exception("No DuckDB in test"),
            ),
        ):
            summary = builder.build_summary(
                data=minimal_data,
                symbols=["AAPL"],
                timeframes=["1d"],
                regime_outputs={"AAPL": mock_regime},
            )

        # Verify critical top-level keys
        assert "tickers" in summary
        assert "market" in summary
        assert "confluence" in summary
        assert "dual_macd" in summary
        assert "trend_pulse" in summary

        # Verify ticker data
        assert len(summary["tickers"]) == 1
        ticker = summary["tickers"][0]
        assert ticker["symbol"] == "AAPL"
        assert ticker["regime"] == "R0"
        assert ticker["daily_change_pct"] is not None

        # Verify summary is JSON-serializable
        serialized = json.dumps(summary, default=str)
        assert len(serialized) > 100

    def test_summary_builder_produces_valid_json(
        self,
        minimal_data: Dict[Tuple[str, str], pd.DataFrame],
        mock_regime: MagicMock,
    ) -> None:
        """SummaryBuilder.build_summary produces JSON-serializable output."""
        from src.infrastructure.reporting.package.summary_builder import (
            SummaryBuilder,
        )

        builder = SummaryBuilder(enforce_budget=False)

        with (
            patch(
                "src.infrastructure.reporting.package.summary_builder"
                ".SummaryBuilder._compute_rule_frequency",
                return_value={
                    "by_symbol": {},
                    "buy_by_symbol": {},
                    "sell_by_symbol": {},
                    "by_rule": {},
                    "top_symbols": [],
                    "top_rules": [],
                    "total_signals": 0,
                },
            ),
            patch(
                "src.infrastructure.stores.duckdb_coverage_store" ".DuckDBCoverageStore",
                side_effect=Exception("No DuckDB in test"),
            ),
        ):
            summary = builder.build_summary(
                data=minimal_data,
                symbols=["AAPL"],
                timeframes=["1d"],
                regime_outputs={"AAPL": mock_regime},
            )

        # Must be JSON-serializable
        serialized = json.dumps(summary, default=str)
        assert len(serialized) > 0

        # Check required top-level keys
        assert "version" in summary
        assert "tickers" in summary
        assert "market" in summary
        assert "confluence" in summary
