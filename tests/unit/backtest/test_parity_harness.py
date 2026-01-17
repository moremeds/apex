"""Tests for strategy parity harness."""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest.core import RunMetrics, RunResult, RunSpec, RunStatus, TimeWindow
from src.backtest.execution.engines import EngineType
from src.backtest.execution.parity import (
    DriftDetail,
    DriftType,
    ParityConfig,
    ParityResult,
    SignalCapture,
    SignalParityResult,
    StrategyParityHarness,
    compare_signal_parity,
)


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


# =============================================================================
# Signal Parity Tests (Phase 5)
# =============================================================================


@pytest.fixture
def sample_index():
    """Create sample datetime index for signal testing."""
    return pd.date_range("2023-01-01", periods=100, freq="D")


class TestSignalParityResult:
    """Test SignalParityResult dataclass."""

    def test_passed_summary(self):
        """Passed result should have clear summary."""
        result = SignalParityResult(
            passed=True,
            warmup_bars=20,
            total_bars=100,
            compared_bars=80,
            entry_matches=80,
            exit_matches=80,
        )
        assert "PARITY OK" in result.summary()
        assert "80 bars compared" in result.summary()
        assert "20 warmup" in result.summary()

    def test_failed_summary(self):
        """Failed result should show accuracy stats."""
        result = SignalParityResult(
            passed=False,
            warmup_bars=20,
            total_bars=100,
            compared_bars=80,
            entry_matches=70,
            entry_mismatches=10,
            exit_matches=75,
            exit_mismatches=5,
            first_entry_mismatch_idx=datetime(2023, 1, 25),
            first_exit_mismatch_idx=datetime(2023, 2, 1),
        )
        summary = result.summary()
        assert "PARITY FAILED" in summary
        assert "entries" in summary.lower()

    def test_entry_accuracy_calculation(self):
        """Entry accuracy should be matches / total."""
        result = SignalParityResult(
            passed=True,
            warmup_bars=10,
            total_bars=100,
            compared_bars=90,
            entry_matches=85,
            entry_mismatches=5,
        )
        assert result.entry_accuracy == 85 / 90

    def test_exit_accuracy_calculation(self):
        """Exit accuracy should be matches / total."""
        result = SignalParityResult(
            passed=True,
            warmup_bars=10,
            total_bars=100,
            compared_bars=90,
            exit_matches=90,
            exit_mismatches=0,
        )
        assert result.exit_accuracy == 1.0

    def test_accuracy_when_no_signals(self):
        """Accuracy should be 1.0 when no signals to compare."""
        result = SignalParityResult(
            passed=True,
            warmup_bars=100,
            total_bars=100,
            compared_bars=0,
            entry_matches=0,
            entry_mismatches=0,
        )
        assert result.entry_accuracy == 1.0
        assert result.exit_accuracy == 1.0


class TestCompareSignalParity:
    """Test compare_signal_parity function."""

    def test_identical_signals_pass(self, sample_index):
        """Identical signals should pass parity."""
        entries = pd.Series([i % 20 == 0 for i in range(100)], index=sample_index)
        exits = pd.Series([i % 20 == 10 for i in range(100)], index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=entries,
            vectorbt_exits=exits,
            captured_entries=entries.copy(),
            captured_exits=exits.copy(),
            warmup_bars=20,
        )

        assert result.passed is True
        assert result.entry_mismatches == 0
        assert result.exit_mismatches == 0

    def test_different_entries_fail(self, sample_index):
        """Different entry signals should fail parity."""
        vbt_entries = pd.Series([i % 10 == 0 for i in range(100)], index=sample_index)
        cap_entries = pd.Series([i % 10 == 1 for i in range(100)], index=sample_index)
        exits = pd.Series(False, index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=vbt_entries,
            vectorbt_exits=exits,
            captured_entries=cap_entries,
            captured_exits=exits.copy(),
            warmup_bars=10,
        )

        assert result.passed is False
        assert result.entry_mismatches > 0
        assert "Entry mismatch" in result.mismatches[0]

    def test_different_exits_fail(self, sample_index):
        """Different exit signals should fail parity."""
        entries = pd.Series([i == 5 for i in range(100)], index=sample_index)
        vbt_exits = pd.Series([i == 50 for i in range(100)], index=sample_index)
        cap_exits = pd.Series([i == 60 for i in range(100)], index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=entries,
            vectorbt_exits=vbt_exits,
            captured_entries=entries.copy(),
            captured_exits=cap_exits,
            warmup_bars=0,
        )

        assert result.passed is False
        assert result.exit_mismatches > 0

    def test_warmup_skipped(self, sample_index):
        """Warmup period should be skipped in comparison."""
        # Different during warmup, same after
        vbt_entries = pd.Series([i < 20 for i in range(100)], index=sample_index)
        cap_entries = pd.Series([i >= 20 and i < 40 for i in range(100)], index=sample_index)
        exits = pd.Series(False, index=sample_index)

        # After warmup of 40, both are False
        result = compare_signal_parity(
            vectorbt_entries=vbt_entries,
            vectorbt_exits=exits,
            captured_entries=cap_entries,
            captured_exits=exits.copy(),
            warmup_bars=40,
        )

        # After bar 40, vbt_entries is False (i >= 20 condition not met)
        # cap_entries is also False (i >= 20 and i < 40 not met for i >= 40)
        assert result.passed is True
        assert result.compared_bars == 60

    def test_nan_treated_as_false(self, sample_index):
        """NaN values should be treated as False."""
        entries = pd.Series([True, False, False] + [False] * 97, index=sample_index)
        entries.iloc[1] = float("nan")  # NaN in vectorbt

        cap_entries = pd.Series([True, False, False] + [False] * 97, index=sample_index)

        exits = pd.Series(False, index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=entries,
            vectorbt_exits=exits,
            captured_entries=cap_entries,
            captured_exits=exits.copy(),
            warmup_bars=0,
        )

        assert result.passed is True

    def test_first_mismatch_recorded(self, sample_index):
        """First mismatch index should be recorded as datetime."""
        vbt_entries = pd.Series([i == 25 for i in range(100)], index=sample_index)
        cap_entries = pd.Series([i == 30 for i in range(100)], index=sample_index)
        exits = pd.Series(False, index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=vbt_entries,
            vectorbt_exits=exits,
            captured_entries=cap_entries,
            captured_exits=exits.copy(),
            warmup_bars=10,
        )

        assert result.first_entry_mismatch_idx is not None
        assert isinstance(result.first_entry_mismatch_idx, datetime)

    def test_index_mismatch_fails(self):
        """Mismatched indices should fail with clear message."""
        index1 = pd.date_range("2023-01-01", periods=50, freq="D")
        index2 = pd.date_range("2023-02-01", periods=50, freq="D")

        result = compare_signal_parity(
            vectorbt_entries=pd.Series(False, index=index1),
            vectorbt_exits=pd.Series(False, index=index1),
            captured_entries=pd.Series(False, index=index2),
            captured_exits=pd.Series(False, index=index2),
            warmup_bars=0,
        )

        assert result.passed is False
        assert "Index mismatch" in result.mismatches[0]

    def test_warmup_exceeds_bars(self, sample_index):
        """Warmup exceeding total bars should pass with 0 compared."""
        entries = pd.Series(True, index=sample_index)
        exits = pd.Series(False, index=sample_index)

        result = compare_signal_parity(
            vectorbt_entries=entries,
            vectorbt_exits=exits,
            captured_entries=entries.copy(),
            captured_exits=exits.copy(),
            warmup_bars=150,  # More than 100 bars
        )

        assert result.passed is True
        assert result.compared_bars == 0


class TestSignalCapture:
    """Test SignalCapture order recording."""

    def test_record_buy_creates_entry(self, sample_index):
        """BUY order while flat should create entry signal."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "BUY", 100)

        entries, exits = capture.get_signals()
        assert entries.iloc[10] is True or entries.iloc[10] == True
        assert exits.sum() == 0

    def test_record_sell_creates_exit(self, sample_index):
        """SELL order while long should create exit signal."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        # First buy to establish position
        capture.record_order(sample_index[10], "AAPL", "BUY", 100)
        # Then sell to exit
        capture.record_order(sample_index[20], "AAPL", "SELL", 100)

        entries, exits = capture.get_signals()
        assert entries.iloc[10] is True or entries.iloc[10] == True
        assert exits.iloc[20] is True or exits.iloc[20] == True

    def test_sell_while_flat_no_exit(self, sample_index):
        """SELL order while flat should not create exit signal."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "SELL", 100)

        entries, exits = capture.get_signals()
        assert entries.sum() == 0
        assert exits.sum() == 0

    def test_buy_while_long_no_entry(self, sample_index):
        """BUY order while already long should not create entry signal."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "BUY", 100)
        capture.record_order(sample_index[15], "AAPL", "BUY", 50)  # Adding to position

        entries, exits = capture.get_signals()
        assert entries.sum() == 1  # Only first entry

    def test_shadow_position_tracking(self, sample_index):
        """Shadow position should track multiple orders correctly."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        # Entry
        capture.record_order(sample_index[10], "AAPL", "BUY", 100)
        # Partial exit (still long)
        capture.record_order(sample_index[20], "AAPL", "SELL", 50)
        # Full exit
        capture.record_order(sample_index[25], "AAPL", "SELL", 50)
        # Re-entry
        capture.record_order(sample_index[30], "AAPL", "BUY", 100)

        entries, exits = capture.get_signals()

        # Two entries: initial and re-entry
        assert entries.iloc[10] is True or entries.iloc[10] == True
        assert entries.iloc[30] is True or entries.iloc[30] == True
        assert entries.sum() == 2

        # Two exits: partial and full (both count as exits from long)
        assert exits.iloc[20] is True or exits.iloc[20] == True
        assert exits.iloc[25] is True or exits.iloc[25] == True
        assert exits.sum() == 2

    def test_symbol_filtering(self, sample_index):
        """Orders for different symbols should be filtered."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "BUY", 100)
        capture.record_order(sample_index[15], "MSFT", "BUY", 100)  # Different symbol

        entries, exits = capture.get_signals()
        assert entries.sum() == 1  # Only AAPL order counted

    def test_unmatched_timestamps_tracked(self):
        """Timestamps not in index should be tracked."""
        index = pd.date_range("2023-01-01", periods=10, freq="D")
        capture = SignalCapture(index, symbol="AAPL")

        # Timestamp outside index
        out_of_range = datetime(2023, 2, 15)
        capture.record_order(out_of_range, "AAPL", "BUY", 100)

        entries, exits = capture.get_signals()
        assert entries.sum() == 0  # Not matched
        assert capture.unmatched_count == 1

    def test_reset_clears_state(self, sample_index):
        """Reset should clear all captured signals and position."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "BUY", 100)
        capture.reset()

        entries, exits = capture.get_signals()
        assert entries.sum() == 0
        assert exits.sum() == 0

    def test_case_insensitive_side(self, sample_index):
        """Order side should be case-insensitive."""
        capture = SignalCapture(sample_index, symbol="AAPL")

        capture.record_order(sample_index[10], "AAPL", "buy", 100)
        capture.record_order(sample_index[20], "AAPL", "SELL", 100)

        entries, exits = capture.get_signals()
        assert entries.sum() == 1
        assert exits.sum() == 1
