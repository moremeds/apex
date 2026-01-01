"""Tests for strategy parity harness."""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest.core import RunSpec, TimeWindow, RunResult, RunMetrics, RunStatus
from src.backtest.execution.parity import (
    StrategyParityHarness,
    ParityConfig,
    ParityResult,
    DriftType,
    DriftDetail,
)
from src.backtest.execution.engines import EngineType


@pytest.fixture
def run_spec():
    """Create sample run specification."""
    window = TimeWindow(
        window_id="fold_1",
        fold_index=0,
        train_start=date(2023, 1, 1),
        train_end=date(2023, 2, 28),
        test_start=date(2023, 3, 1),
        test_end=date(2023, 4, 10),
        is_train=False,
        is_oos=True,
    )
    return RunSpec(
        trial_id="trial_test123",
        symbol="AAPL",
        window=window,
        profile_version="v1",
        data_version="test",
        params={"fast_period": 10, "slow_period": 30},
        experiment_id="exp_test",
    )


def create_mock_engine(engine_type: EngineType, result: RunResult):
    """Create a mock engine that returns a fixed result."""
    engine = MagicMock()
    engine.engine_type = engine_type
    engine.run.return_value = result
    return engine


def create_run_result(
    run_spec: RunSpec,
    sharpe: float = 1.5,
    total_return: float = 0.15,
    max_drawdown: float = 0.10,
    win_rate: float = 0.55,
    total_trades: int = 50,
    status: RunStatus = RunStatus.SUCCESS,
) -> RunResult:
    """Create a RunResult with specified metrics."""
    return RunResult(
        run_id=run_spec.run_id,
        trial_id=run_spec.trial_id,
        experiment_id=run_spec.experiment_id or "",
        symbol=run_spec.symbol,
        window_id=run_spec.window.window_id,
        status=status,
        metrics=RunMetrics(
            sharpe=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
        ),
        started_at=datetime.now(),
        completed_at=datetime.now(),
    )


class TestParityConfig:
    """Test parity configuration."""

    def test_default_tolerances(self):
        """Default tolerances should be sensible."""
        config = ParityConfig()
        assert config.sharpe_tolerance == 0.05
        assert config.return_tolerance == 0.01
        assert config.max_dd_tolerance == 0.02

    def test_custom_tolerances(self):
        """Custom tolerances should be settable."""
        config = ParityConfig(
            sharpe_tolerance=0.10,
            return_tolerance=0.02,
        )
        assert config.sharpe_tolerance == 0.10
        assert config.return_tolerance == 0.02


class TestParityResult:
    """Test parity result structure."""

    def test_parity_ok_summary(self, run_spec):
        """Parity OK should have clear summary."""
        result = ParityResult(
            spec=run_spec,
            reference_engine=EngineType.APEX,
            test_engine=EngineType.VECTORBT,
            is_parity=True,
            drift_detected=[],
        )
        assert "Parity OK" in result.summary

    def test_parity_failed_summary(self, run_spec):
        """Failed parity should list drifts."""
        drift = DriftDetail(
            drift_type=DriftType.PNL_MISMATCH,
            field="total_return",
            reference_value=0.15,
            test_value=0.10,
            difference=0.05,
            tolerance=0.01,
            message="Return differs",
        )
        result = ParityResult(
            spec=run_spec,
            reference_engine=EngineType.APEX,
            test_engine=EngineType.VECTORBT,
            is_parity=False,
            drift_detected=[drift],
        )
        assert "critical" in result.summary.lower()

    def test_critical_drifts_filter(self, run_spec):
        """Critical drifts should be filterable."""
        critical = DriftDetail(
            drift_type=DriftType.PNL_MISMATCH,
            field="total_return",
            reference_value=0.15,
            test_value=0.10,
            difference=0.05,
            tolerance=0.01,
            message="Return differs",
        )
        warning = DriftDetail(
            drift_type=DriftType.METRIC_MISMATCH,
            field="sharpe",
            reference_value=1.5,
            test_value=1.4,
            difference=0.07,
            tolerance=0.05,
            message="Sharpe differs",
        )
        result = ParityResult(
            spec=run_spec,
            reference_engine=EngineType.APEX,
            test_engine=EngineType.VECTORBT,
            is_parity=False,
            drift_detected=[critical, warning],
        )
        assert len(result.critical_drifts) == 1
        assert result.critical_drifts[0].field == "total_return"

    def test_to_dict(self, run_spec):
        """Result should convert to dict."""
        result = ParityResult(
            spec=run_spec,
            reference_engine=EngineType.APEX,
            test_engine=EngineType.VECTORBT,
            is_parity=True,
            drift_detected=[],
        )
        d = result.to_dict()
        assert d["spec_id"] == run_spec.run_id
        assert d["is_parity"] is True


class TestDriftDetail:
    """Test drift detail structure."""

    def test_pnl_drift_is_critical(self):
        """PNL mismatch should be critical."""
        drift = DriftDetail(
            drift_type=DriftType.PNL_MISMATCH,
            field="total_return",
            reference_value=0.15,
            test_value=0.10,
            difference=0.05,
            tolerance=0.01,
            message="Return differs",
        )
        assert drift.is_critical is True

    def test_metric_drift_not_critical(self):
        """Metric mismatch should not be critical."""
        drift = DriftDetail(
            drift_type=DriftType.METRIC_MISMATCH,
            field="sharpe",
            reference_value=1.5,
            test_value=1.4,
            difference=0.07,
            tolerance=0.05,
            message="Sharpe differs",
        )
        assert drift.is_critical is False


class TestStrategyParityHarness:
    """Test parity harness comparison."""

    def test_compare_identical_results(self, run_spec):
        """Identical results should pass parity."""
        result = create_run_result(run_spec)

        ref_engine = create_mock_engine(EngineType.APEX, result)
        test_engine = create_mock_engine(EngineType.VECTORBT, result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        assert parity.is_parity is True
        assert len(parity.drift_detected) == 0

    def test_compare_detects_return_drift(self, run_spec):
        """Should detect return drift beyond tolerance."""
        ref_result = create_run_result(run_spec, total_return=0.15)
        test_result = create_run_result(run_spec, total_return=0.05)  # 10% diff

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        assert parity.is_parity is False
        assert any(d.field == "total_return" for d in parity.drift_detected)

    def test_compare_detects_trade_count_drift(self, run_spec):
        """Should detect trade count drift."""
        ref_result = create_run_result(run_spec, total_trades=50)
        test_result = create_run_result(run_spec, total_trades=40)

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        assert any(d.field == "total_trades" for d in parity.drift_detected)

    def test_compare_tolerates_small_drift(self, run_spec):
        """Small drifts within tolerance should pass."""
        ref_result = create_run_result(run_spec, total_return=0.150)
        test_result = create_run_result(run_spec, total_return=0.155)  # 0.5% diff

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        # 0.5% is within 1% tolerance
        pnl_drifts = [d for d in parity.drift_detected if d.field == "total_return"]
        assert len(pnl_drifts) == 0

    def test_compare_handles_status_mismatch(self, run_spec):
        """Should detect status mismatch."""
        ref_result = create_run_result(run_spec, status=RunStatus.SUCCESS)
        test_result = create_run_result(run_spec, status=RunStatus.FAIL_DATA)

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        assert parity.is_parity is False
        assert any(d.field == "status" for d in parity.drift_detected)

    def test_compare_tracks_timing(self, run_spec):
        """Should track comparison timing."""
        result = create_run_result(run_spec)

        ref_engine = create_mock_engine(EngineType.APEX, result)
        test_engine = create_mock_engine(EngineType.VECTORBT, result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity = harness.compare(run_spec)

        assert parity.comparison_time >= 0
        assert parity.reference_time >= 0
        assert parity.test_time >= 0


class TestParityBatch:
    """Test batch parity comparison."""

    def test_compare_batch_multiple_specs(self, run_spec):
        """Batch should compare multiple specs."""
        result = create_run_result(run_spec)

        ref_engine = create_mock_engine(EngineType.APEX, result)
        test_engine = create_mock_engine(EngineType.VECTORBT, result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        results = harness.compare_batch([run_spec, run_spec, run_spec])

        assert len(results) == 3
        assert all(r.is_parity for r in results)

    def test_compare_batch_empty(self):
        """Empty batch should return empty list."""
        ref_engine = MagicMock()
        test_engine = MagicMock()

        harness = StrategyParityHarness(ref_engine, test_engine)
        results = harness.compare_batch([])

        assert results == []


class TestParityReport:
    """Test parity report generation."""

    def test_generate_report_all_passed(self, run_spec):
        """Report should show all passed."""
        result = create_run_result(run_spec)

        ref_engine = create_mock_engine(EngineType.APEX, result)
        test_engine = create_mock_engine(EngineType.VECTORBT, result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity_results = harness.compare_batch([run_spec])
        report = harness.generate_report(parity_results)

        assert "Passed: 1" in report
        assert "Failed: 0" in report

    def test_generate_report_with_failures(self, run_spec):
        """Report should show failures."""
        ref_result = create_run_result(run_spec, total_return=0.20)
        test_result = create_run_result(run_spec, total_return=0.05)

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        harness = StrategyParityHarness(ref_engine, test_engine)
        parity_results = harness.compare_batch([run_spec])
        report = harness.generate_report(parity_results)

        assert "FAILURES:" in report
        assert "total_return" in report


class TestCustomTolerances:
    """Test custom tolerance configuration."""

    def test_strict_tolerances(self, run_spec):
        """Strict tolerances should catch smaller drifts."""
        ref_result = create_run_result(run_spec, sharpe=1.50)
        test_result = create_run_result(run_spec, sharpe=1.47)  # 2% diff

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        strict_config = ParityConfig(sharpe_tolerance=0.01)  # 1% tolerance
        harness = StrategyParityHarness(ref_engine, test_engine, strict_config)
        parity = harness.compare(run_spec)

        assert any(d.field == "sharpe" for d in parity.drift_detected)

    def test_relaxed_tolerances(self, run_spec):
        """Relaxed tolerances should allow larger drifts."""
        ref_result = create_run_result(run_spec, sharpe=1.50)
        test_result = create_run_result(run_spec, sharpe=1.35)  # 10% diff

        ref_engine = create_mock_engine(EngineType.APEX, ref_result)
        test_engine = create_mock_engine(EngineType.VECTORBT, test_result)

        relaxed_config = ParityConfig(sharpe_tolerance=0.15)  # 15% tolerance
        harness = StrategyParityHarness(ref_engine, test_engine, relaxed_config)
        parity = harness.compare(run_spec)

        sharpe_drifts = [d for d in parity.drift_detected if d.field == "sharpe"]
        assert len(sharpe_drifts) == 0
