"""
Strategy parity harness for comparing backtest engines.

The single biggest source of production bugs is subtle drift between
different execution paths (VectorBT vs Apex, live vs backtest). This
harness detects discrepancies before they cause real losses.

Key comparisons:
- Signal timing: Do both engines generate the same entry/exit signals?
- Trade execution: Same fills at same prices?
- P&L calculation: Final results within tolerance?

Signal Parity (Phase 5):
- Exact (entries, exits) match between VectorBT SignalGenerator and event-driven Strategy
- Warmup rule: Skip first `warmup_bars` bars before comparison
- Both paths must use same TA-Lib indicator wrappers

Usage:
    harness = StrategyParityHarness(
        reference_engine=ApexEngine(),
        test_engine=VectorBTEngine(),
    )
    result = harness.compare(run_spec)

    if not result.is_parity:
        print(f"Parity failed: {result.summary}")

Signal Parity Usage:
    result = compare_signal_parity(
        vectorbt_entries, vectorbt_exits,
        captured_entries, captured_exits,
        warmup_bars=50,
    )
    assert result.passed, result.mismatches
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core import RunResult, RunSpec
from .engines import BacktestEngine, EngineType


class DriftType(str, Enum):
    """Types of parity drift detected."""

    NONE = "none"
    STATUS_MISMATCH = "status_mismatch"  # Different execution status
    SIGNAL_TIMING = "signal_timing"  # Different entry/exit times
    TRADE_COUNT = "trade_count"  # Different number of trades
    PRICE_EXECUTION = "price_execution"  # Different fill prices
    PNL_MISMATCH = "pnl_mismatch"  # P&L beyond tolerance
    METRIC_MISMATCH = "metric_mismatch"  # Computed metrics differ


@dataclass
class DriftDetail:
    """Details about a specific drift detection."""

    drift_type: DriftType
    field: str
    reference_value: Any
    test_value: Any
    difference: float
    tolerance: float
    message: str

    @property
    def is_critical(self) -> bool:
        """Whether this drift is critical (status, P&L, or execution related)."""
        return self.drift_type in (
            DriftType.STATUS_MISMATCH,
            DriftType.PNL_MISMATCH,
            DriftType.PRICE_EXECUTION,
        )


@dataclass
class ParityResult:
    """
    Result of parity comparison between two engines.

    Contains detailed drift analysis and diagnostic information.
    """

    spec: RunSpec
    reference_engine: EngineType
    test_engine: EngineType

    # Results from each engine
    reference_result: Optional[RunResult] = None
    test_result: Optional[RunResult] = None

    # Parity status
    is_parity: bool = False
    drift_detected: List[DriftDetail] = field(default_factory=list)

    # Timing
    comparison_time: float = 0.0  # seconds
    reference_time: float = 0.0
    test_time: float = 0.0

    @property
    def summary(self) -> str:
        """Human-readable summary of parity result."""
        if self.is_parity:
            return "Parity OK - engines produce equivalent results"

        critical = [d for d in self.drift_detected if d.is_critical]
        warnings = [d for d in self.drift_detected if not d.is_critical]

        parts = []
        if critical:
            parts.append(f"{len(critical)} critical drifts: {[d.field for d in critical]}")
        if warnings:
            parts.append(f"{len(warnings)} warnings: {[d.field for d in warnings]}")

        return " | ".join(parts) or "Unknown drift"

    @property
    def critical_drifts(self) -> List[DriftDetail]:
        """Get only critical drifts."""
        return [d for d in self.drift_detected if d.is_critical]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "spec_id": self.spec.run_id,
            "symbol": self.spec.symbol,
            "reference_engine": self.reference_engine.value,
            "test_engine": self.test_engine.value,
            "is_parity": self.is_parity,
            "drift_count": len(self.drift_detected),
            "critical_count": len(self.critical_drifts),
            "comparison_time_seconds": self.comparison_time,
            "summary": self.summary,
            "drifts": [
                {
                    "type": d.drift_type.value,
                    "field": d.field,
                    "reference": d.reference_value,
                    "test": d.test_value,
                    "difference": d.difference,
                    "tolerance": d.tolerance,
                }
                for d in self.drift_detected
            ],
        }


@dataclass
class ParityConfig:
    """Configuration for parity comparison tolerances."""

    # Metric tolerances (as fractions)
    sharpe_tolerance: float = 0.05  # 5% relative difference
    return_tolerance: float = 0.01  # 1% absolute difference
    max_dd_tolerance: float = 0.02  # 2% absolute difference
    win_rate_tolerance: float = 0.05  # 5% absolute difference

    # Trade tolerances
    trade_count_tolerance: int = 2  # Allow this many trade differences
    price_tolerance_pct: float = 0.001  # 0.1% price difference

    # Signal tolerances
    signal_timing_days: int = 1  # Allow signals within N days

    # Whether to fail on any drift or only critical
    fail_on_warnings: bool = False


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


# =============================================================================
# Signal Parity Testing (Phase 5)
# =============================================================================


@dataclass
class SignalParityResult:
    """Result of signal-level parity comparison."""

    passed: bool
    warmup_bars: int
    total_bars: int
    compared_bars: int  # total_bars - warmup_bars

    # Match statistics
    entry_matches: int = 0
    entry_mismatches: int = 0
    exit_matches: int = 0
    exit_mismatches: int = 0

    # First mismatch details (timestamps, not indices)
    first_entry_mismatch_idx: Optional[datetime] = None
    first_exit_mismatch_idx: Optional[datetime] = None

    # Detailed mismatches (for debugging)
    mismatches: List[str] = field(default_factory=list)

    @property
    def entry_accuracy(self) -> float:
        """Percentage of matching entry signals."""
        total = self.entry_matches + self.entry_mismatches
        return self.entry_matches / total if total > 0 else 1.0

    @property
    def exit_accuracy(self) -> float:
        """Percentage of matching exit signals."""
        total = self.exit_matches + self.exit_mismatches
        return self.exit_matches / total if total > 0 else 1.0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.passed:
            return (
                f"PARITY OK: {self.compared_bars} bars compared "
                f"(skipped {self.warmup_bars} warmup)"
            )
        return (
            f"PARITY FAILED: entries={self.entry_accuracy:.1%}, "
            f"exits={self.exit_accuracy:.1%} | "
            f"First mismatch at entry={self.first_entry_mismatch_idx}, "
            f"exit={self.first_exit_mismatch_idx}"
        )


def compare_signal_parity(
    vectorbt_entries: pd.Series,
    vectorbt_exits: pd.Series,
    captured_entries: pd.Series,
    captured_exits: pd.Series,
    warmup_bars: int,
) -> SignalParityResult:
    """
    Compare signals from VectorBT SignalGenerator vs event-driven Strategy.

    Args:
        vectorbt_entries: Boolean series from SignalGenerator.generate()
        vectorbt_exits: Boolean series from SignalGenerator.generate()
        captured_entries: Boolean series captured from event-driven strategy
        captured_exits: Boolean series captured from event-driven strategy
        warmup_bars: Number of initial bars to skip (indicator warmup)

    Returns:
        SignalParityResult with match/mismatch statistics

    Note:
        - All series must have the same index
        - Comparison is exact match (True == True, False == False)
        - NaN values are treated as False
    """
    # Ensure all 4 series have the same index
    indices_match = (
        vectorbt_entries.index.equals(captured_entries.index)
        and vectorbt_exits.index.equals(captured_exits.index)
        and vectorbt_entries.index.equals(vectorbt_exits.index)
    )
    if not indices_match:
        return SignalParityResult(
            passed=False,
            warmup_bars=warmup_bars,
            total_bars=len(vectorbt_entries),
            compared_bars=0,
            mismatches=["Index mismatch between signal series - all must have same index"],
        )

    total_bars = len(vectorbt_entries)
    compared_bars = total_bars - warmup_bars

    if compared_bars <= 0:
        return SignalParityResult(
            passed=True,
            warmup_bars=warmup_bars,
            total_bars=total_bars,
            compared_bars=0,
            mismatches=["No bars to compare after warmup"],
        )

    # Skip warmup bars
    vbt_entries = vectorbt_entries.iloc[warmup_bars:].fillna(False).astype(bool)
    vbt_exits = vectorbt_exits.iloc[warmup_bars:].fillna(False).astype(bool)
    cap_entries = captured_entries.iloc[warmup_bars:].fillna(False).astype(bool)
    cap_exits = captured_exits.iloc[warmup_bars:].fillna(False).astype(bool)

    # Compare entries
    entry_match = vbt_entries == cap_entries
    entry_matches = entry_match.sum()
    entry_mismatches = (~entry_match).sum()

    # Compare exits
    exit_match = vbt_exits == cap_exits
    exit_matches = exit_match.sum()
    exit_mismatches = (~exit_match).sum()

    # Find first mismatches
    first_entry_mismatch = None
    first_exit_mismatch = None
    mismatches = []

    if entry_mismatches > 0:
        mismatch_idx = entry_match[~entry_match].index
        first_entry_mismatch = mismatch_idx[0] if len(mismatch_idx) > 0 else None
        # Report first 5 mismatches
        for idx in list(mismatch_idx)[:5]:
            mismatches.append(
                f"Entry mismatch at {idx}: "
                f"vectorbt={vbt_entries.loc[idx]}, captured={cap_entries.loc[idx]}"
            )

    if exit_mismatches > 0:
        mismatch_idx = exit_match[~exit_match].index
        first_exit_mismatch = mismatch_idx[0] if len(mismatch_idx) > 0 else None
        for idx in list(mismatch_idx)[:5]:
            mismatches.append(
                f"Exit mismatch at {idx}: "
                f"vectorbt={vbt_exits.loc[idx]}, captured={cap_exits.loc[idx]}"
            )

    passed = bool(entry_mismatches == 0 and exit_mismatches == 0)

    return SignalParityResult(
        passed=passed,
        warmup_bars=warmup_bars,
        total_bars=total_bars,
        compared_bars=compared_bars,
        entry_matches=int(entry_matches),
        entry_mismatches=int(entry_mismatches),
        exit_matches=int(exit_matches),
        exit_mismatches=int(exit_mismatches),
        first_entry_mismatch_idx=first_entry_mismatch,
        first_exit_mismatch_idx=first_exit_mismatch,
        mismatches=mismatches,
    )


def compare_directional_signal_parity(
    vectorbt_long_entries: pd.Series,
    vectorbt_long_exits: pd.Series,
    vectorbt_short_entries: pd.Series,
    vectorbt_short_exits: pd.Series,
    captured_long_entries: pd.Series,
    captured_long_exits: pd.Series,
    captured_short_entries: pd.Series,
    captured_short_exits: pd.Series,
    warmup_bars: int,
) -> SignalParityResult:
    """
    Compare directional signals from VectorBT SignalGenerator vs event-driven Strategy.

    Used for strategies that generate both long AND short signals (e.g., MTF RSI Trend).

    Args:
        vectorbt_long_entries: Long entry signals from SignalGenerator.generate_directional()
        vectorbt_long_exits: Long exit signals
        vectorbt_short_entries: Short entry signals
        vectorbt_short_exits: Short exit signals
        captured_long_entries: Long entries captured from event-driven strategy
        captured_long_exits: Long exits captured
        captured_short_entries: Short entries captured
        captured_short_exits: Short exits captured
        warmup_bars: Number of initial bars to skip

    Returns:
        SignalParityResult aggregating all 4 signal comparisons
    """
    # Compare long signals
    long_result = compare_signal_parity(
        vectorbt_long_entries,
        vectorbt_long_exits,
        captured_long_entries,
        captured_long_exits,
        warmup_bars,
    )

    # Compare short signals
    short_result = compare_signal_parity(
        vectorbt_short_entries,
        vectorbt_short_exits,
        captured_short_entries,
        captured_short_exits,
        warmup_bars,
    )

    # Aggregate results
    passed = long_result.passed and short_result.passed

    # Combine mismatches with side labels
    mismatches = []
    for m in long_result.mismatches:
        mismatches.append(f"[LONG] {m}")
    for m in short_result.mismatches:
        mismatches.append(f"[SHORT] {m}")

    return SignalParityResult(
        passed=passed,
        warmup_bars=warmup_bars,
        total_bars=long_result.total_bars,
        compared_bars=long_result.compared_bars,
        entry_matches=long_result.entry_matches + short_result.entry_matches,
        entry_mismatches=long_result.entry_mismatches + short_result.entry_mismatches,
        exit_matches=long_result.exit_matches + short_result.exit_matches,
        exit_mismatches=long_result.exit_mismatches + short_result.exit_mismatches,
        first_entry_mismatch_idx=(
            long_result.first_entry_mismatch_idx or short_result.first_entry_mismatch_idx
        ),
        first_exit_mismatch_idx=(
            long_result.first_exit_mismatch_idx or short_result.first_exit_mismatch_idx
        ),
        mismatches=mismatches,
    )


class SignalCapture:
    """
    Captures entry/exit signals from event-driven strategy execution.

    Attach to a strategy during backtest to record when orders are placed,
    then convert to boolean signal series for parity comparison.

    Uses shadow position tracking to correctly classify entry/exit across
    multiple orders in the same bar.

    Usage:
        capture = SignalCapture(data.index, symbol="AAPL")

        # During backtest, call on each order
        capture.record_order(timestamp, symbol, side, quantity)

        # After backtest, get signals
        entries, exits = capture.get_signals()

    Note:
        - For parity testing, timestamps must exactly match data.index
        - Long-only strategies only (short positions treated as flat)
    """

    def __init__(self, index: pd.DatetimeIndex, symbol: str = ""):
        """
        Initialize signal capture.

        Args:
            index: DatetimeIndex to align signals to (from OHLCV data)
            symbol: Symbol being tracked (for multi-symbol strategies)
        """
        self.index = index
        self.symbol = symbol
        self._entries: List[datetime] = []
        self._exits: List[datetime] = []
        self._shadow_position: float = 0.0  # Track position state per order
        self._unmatched_timestamps: List[datetime] = []  # For debugging

    def record_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
    ) -> None:
        """
        Record an order request as entry/exit signal.

        Uses shadow position to track state across multiple orders in same bar.

        Args:
            timestamp: Bar timestamp when order was placed
            symbol: Symbol for the order
            side: "BUY" or "SELL"
            quantity: Order quantity (for shadow position update)
        """
        if self.symbol and symbol != self.symbol:
            return

        # Map order to entry/exit based on shadow position state
        if side.upper() == "BUY":
            if self._shadow_position == 0:
                # BUY while flat → entry
                self._entries.append(timestamp)
            self._shadow_position += quantity

        elif side.upper() == "SELL":
            if self._shadow_position > 0:
                # SELL while long → exit
                self._exits.append(timestamp)
            self._shadow_position = max(0, self._shadow_position - quantity)

    def get_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Convert captured orders to boolean signal series.

        Returns:
            (entries, exits): Boolean series aligned to self.index

        Note:
            Timestamps must exactly match index. Unmatched timestamps are logged.
        """
        entries = pd.Series(False, index=self.index)
        exits = pd.Series(False, index=self.index)
        self._unmatched_timestamps.clear()

        # Mark entry timestamps (exact match only)
        for ts in self._entries:
            if ts in entries.index:
                entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark exit timestamps (exact match only)
        for ts in self._exits:
            if ts in exits.index:
                exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        return entries, exits

    @property
    def unmatched_count(self) -> int:
        """Number of timestamps that didn't match index."""
        return len(self._unmatched_timestamps)

    def reset(self) -> None:
        """Clear captured signals and reset shadow position."""
        self._entries.clear()
        self._exits.clear()
        self._shadow_position = 0.0
        self._unmatched_timestamps.clear()


class DirectionalSignalCapture:
    """
    Captures both long AND short entry/exit signals from event-driven strategy.

    Extends SignalCapture to track directional positions and generate
    4 signal series for parity comparison with DirectionalSignalGenerator.

    Position tracking:
    - Positive shadow_position = long
    - Negative shadow_position = short
    - Zero = flat

    Signal mapping:
    - BUY while flat → long_entry
    - BUY while short → short_exit (covering)
    - SELL while flat → short_entry
    - SELL while long → long_exit

    Usage:
        capture = DirectionalSignalCapture(data.index, symbol="AAPL")

        # During backtest, call on each order
        capture.record_order(timestamp, symbol, side, quantity)

        # After backtest, get all 4 signal series
        long_entries, long_exits, short_entries, short_exits = capture.get_directional_signals()
    """

    def __init__(self, index: pd.DatetimeIndex, symbol: str = ""):
        """
        Initialize directional signal capture.

        Args:
            index: DatetimeIndex to align signals to (from OHLCV data)
            symbol: Symbol being tracked (for multi-symbol strategies)
        """
        self.index = index
        self.symbol = symbol
        self._long_entries: List[datetime] = []
        self._long_exits: List[datetime] = []
        self._short_entries: List[datetime] = []
        self._short_exits: List[datetime] = []
        self._shadow_position: float = 0.0
        self._unmatched_timestamps: List[datetime] = []

    def record_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
    ) -> None:
        """
        Record an order request as directional entry/exit signal.

        Handles reversal orders that cross through zero position:
        - BUY 200 when short 100 → short_exit AND long_entry
        - SELL 200 when long 100 → long_exit AND short_entry

        Args:
            timestamp: Bar timestamp when order was placed
            symbol: Symbol for the order
            side: "BUY" or "SELL"
            quantity: Order quantity
        """
        if self.symbol and symbol != self.symbol:
            return

        if side.upper() == "BUY":
            if self._shadow_position == 0:
                # BUY while flat → long entry
                self._long_entries.append(timestamp)
            elif self._shadow_position < 0:
                # BUY while short
                self._short_exits.append(timestamp)
                # Check for reversal: BUY quantity exceeds short position
                if quantity > abs(self._shadow_position):
                    # Reversal: also entering long
                    self._long_entries.append(timestamp)
            # else: adding to existing long, no signal
            self._shadow_position += quantity

        elif side.upper() == "SELL":
            if self._shadow_position == 0:
                # SELL while flat → short entry
                self._short_entries.append(timestamp)
            elif self._shadow_position > 0:
                # SELL while long
                self._long_exits.append(timestamp)
                # Check for reversal: SELL quantity exceeds long position
                if quantity > self._shadow_position:
                    # Reversal: also entering short
                    self._short_entries.append(timestamp)
            # else: adding to existing short, no signal
            self._shadow_position -= quantity

    def get_directional_signals(
        self,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Convert captured orders to 4 boolean signal series.

        Returns:
            (long_entries, long_exits, short_entries, short_exits):
            All are boolean Series aligned to self.index
        """
        long_entries = pd.Series(False, index=self.index)
        long_exits = pd.Series(False, index=self.index)
        short_entries = pd.Series(False, index=self.index)
        short_exits = pd.Series(False, index=self.index)
        self._unmatched_timestamps.clear()

        # Mark long entries
        for ts in self._long_entries:
            if ts in long_entries.index:
                long_entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark long exits
        for ts in self._long_exits:
            if ts in long_exits.index:
                long_exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark short entries
        for ts in self._short_entries:
            if ts in short_entries.index:
                short_entries.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        # Mark short exits
        for ts in self._short_exits:
            if ts in short_exits.index:
                short_exits.loc[ts] = True
            else:
                self._unmatched_timestamps.append(ts)

        return long_entries, long_exits, short_entries, short_exits

    def get_signals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Get combined entry/exit signals (backward compatibility).

        Returns all entries (long or short) as entries, all exits as exits.
        """
        long_e, long_x, short_e, short_x = self.get_directional_signals()
        entries = long_e | short_e
        exits = long_x | short_x
        return entries, exits

    @property
    def unmatched_count(self) -> int:
        """Number of timestamps that didn't match index."""
        return len(self._unmatched_timestamps)

    def reset(self) -> None:
        """Clear captured signals and reset shadow position."""
        self._long_entries.clear()
        self._long_exits.clear()
        self._short_entries.clear()
        self._short_exits.clear()
        self._shadow_position = 0.0
        self._unmatched_timestamps.clear()
