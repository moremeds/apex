"""
Data models for parity testing.

Contains:
- DriftType enum for categorizing parity drifts
- DriftDetail for drift details
- ParityResult for comparison results
- ParityConfig for tolerance settings
- SignalParityResult for signal-level comparison
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...core import RunResult, RunSpec
    from ..engines import EngineType


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

    spec: "RunSpec"
    reference_engine: "EngineType"
    test_engine: "EngineType"

    # Results from each engine
    reference_result: Optional["RunResult"] = None
    test_result: Optional["RunResult"] = None

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
