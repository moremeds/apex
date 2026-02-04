"""
Strategy parity harness for comparing backtest engines.

The single biggest source of production bugs is subtle drift between
different execution paths (VectorBT vs Apex, live vs backtest). This
harness detects discrepancies before they cause real losses.

Key comparisons:
- Signal timing: Do both engines generate the same entry/exit signals?
- Trade execution: Same fills at same prices?
- P&L calculation: Final results within tolerance?
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...core import RunResult, RunSpec
from ..engines import BacktestEngine

from .models import DriftDetail, DriftType, ParityConfig, ParityResult


class StrategyParityHarness:
    """
    Harness for comparing strategy results between two backtest engines.

    Detects drift in:
    - Signal generation (entry/exit timing)
    - Trade execution (fills, prices)
    - P&L calculation (returns, drawdowns)
    - Computed metrics (Sharpe, win rate)

    Example:
        harness = StrategyParityHarness(
            reference_engine=apex_engine,
            test_engine=vectorbt_engine,
        )

        result = harness.compare(run_spec)
        assert result.is_parity, f"Parity failed: {result.summary}"

        # Batch comparison
        results = harness.compare_batch(specs)
        failures = [r for r in results if not r.is_parity]

    Data Source Note:
        VectorBTEngine uses the provided `data` and `secondary_data` parameters.
        ApexEngine currently ignores these parameters and always loads from the
        historical store (HistoricalStoreDataFeed). This can cause false parity
        drifts if the preloaded VectorBT data differs from the stored data.

        For true parity testing with identical data, ensure both engines use
        the same underlying data source, or implement a DataFrame-backed feed
        for ApexEngine.
    """

    def __init__(
        self,
        reference_engine: BacktestEngine,
        test_engine: BacktestEngine,
        config: Optional[ParityConfig] = None,
    ):
        self.reference_engine = reference_engine
        self.test_engine = test_engine
        self.config = config or ParityConfig()

    def compare(
        self,
        spec: RunSpec,
        data: Optional[pd.DataFrame] = None,
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> ParityResult:
        """
        Compare a single run between two engines.

        Args:
            spec: Run specification to test
            data: Optional pre-loaded primary timeframe data (shared by both engines)
            secondary_data: Optional secondary timeframe data for MTF strategies.
                Keys are timeframe strings (e.g., "1h", "4h").
                Values are DataFrames with OHLCV columns.

        Returns:
            ParityResult with detailed comparison
        """
        start_time = datetime.now()

        # Run reference engine (with MTF data if provided)
        ref_start = datetime.now()
        ref_result = self.reference_engine.run(spec, data, secondary_data)
        ref_time = (datetime.now() - ref_start).total_seconds()

        # Run test engine (with MTF data if provided)
        test_start = datetime.now()
        test_result = self.test_engine.run(spec, data, secondary_data)
        test_time = (datetime.now() - test_start).total_seconds()

        # Compare results
        drifts = self._compare_results(ref_result, test_result)

        # Determine parity
        is_parity = len(drifts) == 0 or (
            not self.config.fail_on_warnings and all(not d.is_critical for d in drifts)
        )

        total_time = (datetime.now() - start_time).total_seconds()

        return ParityResult(
            spec=spec,
            reference_engine=self.reference_engine.engine_type,
            test_engine=self.test_engine.engine_type,
            reference_result=ref_result,
            test_result=test_result,
            is_parity=is_parity,
            drift_detected=drifts,
            comparison_time=total_time,
            reference_time=ref_time,
            test_time=test_time,
        )

    def compare_batch(
        self,
        specs: List[RunSpec],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        secondary_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    ) -> List[ParityResult]:
        """
        Compare multiple runs between engines.

        Args:
            specs: List of run specifications
            data: Optional dict of symbol -> DataFrame (primary timeframe)
            secondary_data: Optional dict of symbol -> {timeframe: DataFrame}
                for multi-timeframe strategies

        Returns:
            List of ParityResults
        """
        results = []
        for spec in specs:
            symbol_data = data.get(spec.symbol) if data else None
            symbol_secondary = secondary_data.get(spec.symbol) if secondary_data else None
            results.append(self.compare(spec, symbol_data, symbol_secondary))
        return results

    def _compare_results(
        self,
        ref: RunResult,
        test: RunResult,
    ) -> List[DriftDetail]:
        """Compare two RunResults and return detected drifts."""
        drifts = []

        # Check status
        if ref.status != test.status:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.STATUS_MISMATCH,
                    field="status",
                    reference_value=ref.status.value,
                    test_value=test.status.value,
                    difference=1.0,
                    tolerance=0.0,
                    message=f"Status mismatch: {ref.status.value} vs {test.status.value}",
                )
            )
            # If statuses differ, skip metric comparison
            return drifts

        # Skip metric comparison if either failed
        if not ref.is_success or not test.is_success:
            return drifts

        # Compare metrics
        ref_m = ref.metrics
        test_m = test.metrics

        # Trade count
        trade_diff = abs(ref_m.total_trades - test_m.total_trades)
        if trade_diff > self.config.trade_count_tolerance:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.TRADE_COUNT,
                    field="total_trades",
                    reference_value=ref_m.total_trades,
                    test_value=test_m.total_trades,
                    difference=trade_diff,
                    tolerance=self.config.trade_count_tolerance,
                    message=f"Trade count differs by {trade_diff}",
                )
            )

        # Total return (critical)
        return_diff = abs(ref_m.total_return - test_m.total_return)
        if return_diff > self.config.return_tolerance:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.PNL_MISMATCH,
                    field="total_return",
                    reference_value=ref_m.total_return,
                    test_value=test_m.total_return,
                    difference=return_diff,
                    tolerance=self.config.return_tolerance,
                    message=f"Return differs by {return_diff:.4f}",
                )
            )

        # Max drawdown
        dd_diff = abs(ref_m.max_drawdown - test_m.max_drawdown)
        if dd_diff > self.config.max_dd_tolerance:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.METRIC_MISMATCH,
                    field="max_drawdown",
                    reference_value=ref_m.max_drawdown,
                    test_value=test_m.max_drawdown,
                    difference=dd_diff,
                    tolerance=self.config.max_dd_tolerance,
                    message=f"Max drawdown differs by {dd_diff:.4f}",
                )
            )

        # Sharpe ratio (relative comparison)
        if ref_m.sharpe != 0:
            sharpe_rel_diff = abs(ref_m.sharpe - test_m.sharpe) / abs(ref_m.sharpe)
            if sharpe_rel_diff > self.config.sharpe_tolerance:
                drifts.append(
                    DriftDetail(
                        drift_type=DriftType.METRIC_MISMATCH,
                        field="sharpe",
                        reference_value=ref_m.sharpe,
                        test_value=test_m.sharpe,
                        difference=sharpe_rel_diff,
                        tolerance=self.config.sharpe_tolerance,
                        message=f"Sharpe ratio differs by {sharpe_rel_diff:.1%}",
                    )
                )

        # Win rate
        wr_diff = abs(ref_m.win_rate - test_m.win_rate)
        if wr_diff > self.config.win_rate_tolerance:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.METRIC_MISMATCH,
                    field="win_rate",
                    reference_value=ref_m.win_rate,
                    test_value=test_m.win_rate,
                    difference=wr_diff,
                    tolerance=self.config.win_rate_tolerance,
                    message=f"Win rate differs by {wr_diff:.1%}",
                )
            )

        return drifts

    def compare_signals(
        self,
        ref_signals: pd.DataFrame,
        test_signals: pd.DataFrame,
    ) -> List[DriftDetail]:
        """
        Compare signal DataFrames for timing drift.

        Args:
            ref_signals: DataFrame with 'date', 'signal' columns
            test_signals: DataFrame with 'date', 'signal' columns

        Returns:
            List of signal timing drifts
        """
        drifts = []

        # Align signals by date
        ref_entries = set(ref_signals[ref_signals["signal"] == "entry"]["date"])
        test_entries = set(test_signals[test_signals["signal"] == "entry"]["date"])

        # Check for missing entries
        missing_in_test = ref_entries - test_entries
        extra_in_test = test_entries - ref_entries

        if missing_in_test:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.SIGNAL_TIMING,
                    field="entry_signals",
                    reference_value=len(ref_entries),
                    test_value=len(test_entries),
                    difference=len(missing_in_test),
                    tolerance=0,
                    message=f"Missing {len(missing_in_test)} entry signals in test",
                )
            )

        if extra_in_test:
            drifts.append(
                DriftDetail(
                    drift_type=DriftType.SIGNAL_TIMING,
                    field="entry_signals",
                    reference_value=len(ref_entries),
                    test_value=len(test_entries),
                    difference=len(extra_in_test),
                    tolerance=0,
                    message=f"Extra {len(extra_in_test)} entry signals in test",
                )
            )

        return drifts

    def generate_report(self, results: List[ParityResult]) -> str:
        """Generate a text report of parity comparison results."""
        lines = [
            "=" * 60,
            "PARITY COMPARISON REPORT",
            "=" * 60,
            f"Total comparisons: {len(results)}",
            f"Passed: {sum(1 for r in results if r.is_parity)}",
            f"Failed: {sum(1 for r in results if not r.is_parity)}",
            "",
        ]

        # Failed comparisons
        failures = [r for r in results if not r.is_parity]
        if failures:
            lines.append("FAILURES:")
            lines.append("-" * 40)
            for r in failures:
                lines.append(f"\n  {r.spec.symbol} ({r.spec.window.window_id})")
                lines.append(f"    {r.summary}")
                for d in r.drift_detected:
                    critical = " [CRITICAL]" if d.is_critical else ""
                    lines.append(f"    - {d.field}: {d.message}{critical}")

        # Timing stats
        if results:
            ref_times = [r.reference_time for r in results]
            test_times = [r.test_time for r in results]
            lines.extend(
                [
                    "",
                    "TIMING:",
                    f"  Reference engine avg: {np.mean(ref_times):.3f}s",
                    f"  Test engine avg: {np.mean(test_times):.3f}s",
                    (
                        f"  Speedup: {np.mean(ref_times) / np.mean(test_times):.1f}x"
                        if np.mean(test_times) > 0
                        else ""
                    ),
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)
