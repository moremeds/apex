"""Tests for gate_policy_clustering module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.backtest.analysis.dual_macd.behavioral_models import BehavioralMetrics
from src.backtest.analysis.dual_macd.behavioral_report import SymbolResult
from src.backtest.analysis.dual_macd.gate_policy_clustering import (
    _extract_features,
    generate_cluster_policies,
)


def _make_result(
    symbol: str,
    baseline: int = 100,
    allowed: int = 80,
    blocked_loss_ratio: float = 0.7,
    blocked_avg_pnl: float = -0.02,
    size_down_count: int = 0,
    bypass_count: int = 0,
    size_down_avg_pnl: float = 0.0,
    bypass_avg_pnl: float = 0.0,
) -> SymbolResult:
    m = BehavioralMetrics(
        blocked_trade_loss_ratio=blocked_loss_ratio,
        blocked_trade_avg_pnl=blocked_avg_pnl,
        allowed_trade_count=allowed,
        baseline_trade_count=baseline,
        size_down_count=size_down_count,
        bypass_count=bypass_count,
        size_down_avg_pnl=size_down_avg_pnl,
        bypass_avg_pnl=bypass_avg_pnl,
    )
    return SymbolResult(symbol=symbol, metrics=m, decisions=[])


class TestFeatureExtraction:
    def test_basic_features(self) -> None:
        r = _make_result("AAPL", baseline=100, allowed=80, blocked_loss_ratio=0.7)
        f = _extract_features(r)
        assert f.shape == (7,)
        assert f[0] == pytest.approx(0.2)  # blocked_ratio = 20/100
        assert f[1] == pytest.approx(0.7)  # blocked_loss_ratio
        assert f[5] == pytest.approx(0.8)  # allowed_trade_ratio
        assert f[6] == 0.0  # passes constraints

    def test_fail_flag(self) -> None:
        r = _make_result("BAD", baseline=100, allowed=50, blocked_loss_ratio=0.3)
        f = _extract_features(r)
        assert f[6] == 1.0  # fails both constraints

    def test_no_blocks_passes(self) -> None:
        r = _make_result("GOOD", baseline=100, allowed=100, blocked_loss_ratio=0.0)
        f = _extract_features(r)
        assert f[0] == 0.0  # no blocks
        assert f[6] == 0.0  # passes


class TestClusterGeneration:
    def test_generates_yaml(self, tmp_path: Path) -> None:
        results = [
            _make_result(
                "AAPL", baseline=100, allowed=70, blocked_loss_ratio=0.9, blocked_avg_pnl=-0.05
            ),
            _make_result(
                "MSFT", baseline=100, allowed=75, blocked_loss_ratio=0.8, blocked_avg_pnl=-0.03
            ),
            _make_result(
                "TSLA", baseline=100, allowed=95, blocked_loss_ratio=0.1, blocked_avg_pnl=0.02
            ),
            _make_result(
                "AMZN", baseline=100, allowed=85, blocked_loss_ratio=0.5, blocked_avg_pnl=-0.01
            ),
        ]

        out = tmp_path / "clusters.yaml"
        generate_cluster_policies(
            results=results,
            source_params={"slope_lookback": 2, "hist_norm_window": 126},
            output_path=out,
        )

        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert data["status"] == "candidate"
        assert "clusters" in data
        # All 4 symbols accounted for
        all_syms = []
        for cluster in data["clusters"].values():
            all_syms.extend(cluster["symbols"])
        assert sorted(all_syms) == ["AAPL", "AMZN", "MSFT", "TSLA"]

    def test_macro_proxy_override(self, tmp_path: Path) -> None:
        results = [
            _make_result("SPY", baseline=100, allowed=90, blocked_loss_ratio=0.8),
            _make_result("QQQ", baseline=100, allowed=90, blocked_loss_ratio=0.8),
            _make_result("AAPL", baseline=100, allowed=70, blocked_loss_ratio=0.9),
            _make_result("TSLA", baseline=100, allowed=95, blocked_loss_ratio=0.1),
        ]

        out = tmp_path / "clusters.yaml"
        generate_cluster_policies(
            results=results,
            source_params={"slope_lookback": 2, "hist_norm_window": 126},
            output_path=out,
        )

        data = yaml.safe_load(out.read_text())
        macro = data["clusters"].get("MACRO_PROXY", {})
        assert "SPY" in macro.get("symbols", [])
        assert "QQQ" in macro.get("symbols", [])
        assert macro.get("size_factor") == 0.7

    def test_roundtrip_yaml(self, tmp_path: Path) -> None:
        results = [
            _make_result("A", baseline=100, allowed=70, blocked_loss_ratio=0.9),
            _make_result("B", baseline=100, allowed=95, blocked_loss_ratio=0.1),
            _make_result("C", baseline=100, allowed=80, blocked_loss_ratio=0.6),
            _make_result("D", baseline=100, allowed=85, blocked_loss_ratio=0.5),
        ]

        out = tmp_path / "clusters.yaml"
        generate_cluster_policies(
            results=results,
            source_params={"slope_lookback": 3, "hist_norm_window": 252},
            output_path=out,
        )

        # Verify round-trip: load and re-parse
        data = yaml.safe_load(out.read_text())
        assert data["status"] == "candidate"
        assert isinstance(data["source_params"], dict)
        assert data["source_params"]["slope_lookback"] == 3

    def test_too_few_symbols(self, tmp_path: Path) -> None:
        results = [_make_result("ONLY")]
        out = tmp_path / "clusters.yaml"
        generate_cluster_policies(
            results=results,
            source_params={},
            output_path=out,
        )
        # Should not crash; file may or may not exist
