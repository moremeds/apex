"""
Signal Service v2 - Core Schema Definitions.

PR-01 Deliverable: Frozen dataclasses for schema stability.

Schema Version: signal_v2@1.0

Key Design Principles:
1. All schema classes are frozen (immutable) to prevent accidental mutation
2. Schema version validation for serialization/deserialization
3. Explicit documentation of field semantics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# SCHEMA VERSION
# =============================================================================

SCHEMA_VERSION: Tuple[int, int] = (1, 0)
SCHEMA_VERSION_STR: str = f"signal_v2@{SCHEMA_VERSION[0]}.{SCHEMA_VERSION[1]}"


class SchemaVersionError(Exception):
    """Raised when schema version is incompatible."""

    def __init__(self, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(f"Schema version mismatch: expected {expected}, got {actual}")


def validate_schema_version(
    version_str: str, expected_prefix: str = "signal_v2@"
) -> Tuple[int, int]:
    """
    Validate and parse a schema version string.

    Args:
        version_str: Version string like "signal_v2@1.0"
        expected_prefix: Expected prefix for the schema

    Returns:
        Tuple of (major, minor) version numbers

    Raises:
        SchemaVersionError: If version format is invalid or incompatible
    """
    if not version_str or not version_str.startswith(expected_prefix):
        raise SchemaVersionError(SCHEMA_VERSION_STR, version_str or "None")

    try:
        version_part = version_str.split("@")[1]
        parts = version_part.split(".")
        major, minor = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (IndexError, ValueError) as e:
        raise SchemaVersionError(SCHEMA_VERSION_STR, version_str) from e

    # Check major version compatibility (must match exactly)
    if major != SCHEMA_VERSION[0]:
        raise SchemaVersionError(SCHEMA_VERSION_STR, version_str)

    return (major, minor)


# =============================================================================
# BAR VALIDATION REPORT (PR-01 Core Deliverable)
# =============================================================================


class BarReductionReason(Enum):
    """Reasons for bar count reduction during validation."""

    WEEKEND_HOLIDAY = "weekend_holiday"  # Non-trading days removed
    NAN_GAP = "nan_gap"  # NaN values or data gaps trimmed
    WARMUP = "warmup"  # Warmup period for indicators (e.g., SMA200 needs 200 bars)
    MARKET_HOURS = "market_hours"  # Outside market hours (for intraday)
    DATA_QUALITY = "data_quality"  # Failed quality checks (OHLC consistency, etc.)
    MISSING_DATA = "missing_data"  # Data source returned fewer bars than requested


@dataclass(frozen=True)
class BarReduction:
    """A single bar count reduction with explanation."""

    reason: BarReductionReason
    bars_removed: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reason": self.reason.value,
            "bars_removed": self.bars_removed,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarReduction":
        """Deserialize from dictionary."""
        return cls(
            reason=BarReductionReason(data["reason"]),
            bars_removed=data["bars_removed"],
            description=data["description"],
        )


@dataclass(frozen=True)
class BarValidationReport:
    """
    Comprehensive bar count validation report.

    Solves the "350 vs 252" mystery by providing full transparency on
    how many bars were requested, loaded, trimmed, and validated.

    Usage:
        $ python -m src.runners.signal_runner --symbols AAPL --validate-bars

        BarValidationReport:
          requested_bars: 550  (calendar days)
          loaded_bars: 389     (actual data fetched)
          usable_bars: 340     (after NaN/gap trim)
          validated_bars: 140  (after SMA(200) warmup)
          reasons:
            - "Weekends/holidays removed: 161 bars"
            - "NaN gaps trimmed: 49 bars"
            - "Warmup for SMA(200): 200 bars"

    Invariants:
        requested_bars >= loaded_bars >= usable_bars >= validated_bars
    """

    # Schema version for serialization compatibility
    schema_version: str = SCHEMA_VERSION_STR

    # Symbol and timeframe being validated
    symbol: str = ""
    timeframe: str = "1d"

    # Timestamp of validation
    validated_at: Optional[datetime] = None

    # === THE 4 COUNTS ===

    # Calendar days/periods requested (e.g., 550 for ~1.5 years of daily data)
    requested_bars: int = 0

    # Actual bars fetched from data source (after weekend/holiday removal)
    loaded_bars: int = 0

    # Bars after NaN/gap trimming (usable for indicator calculation)
    usable_bars: int = 0

    # Bars after warmup period for longest indicator (e.g., SMA(200))
    # This is the effective number of bars for signal generation
    validated_bars: int = 0

    # === REDUCTION EXPLANATIONS ===

    # List of reductions explaining each count transition
    reductions: Tuple[BarReduction, ...] = field(default_factory=tuple)

    # === DATA SOURCE INFO ===

    # Data source used (e.g., "ib", "yahoo", "cache")
    data_source: str = ""

    # Date range of loaded data
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # === QUALITY FLAGS ===

    # Whether we have enough bars for the longest warmup requirement
    warmup_satisfied: bool = False

    # Longest warmup period required (e.g., 252 for regime detector)
    warmup_required: int = 252

    # Indicator requiring the longest warmup
    warmup_indicator: str = ""

    def __post_init__(self) -> None:
        """Validate invariants."""
        # Note: Can't raise in frozen dataclass __post_init__ easily,
        # but we document the invariants for callers
        pass

    @property
    def is_valid(self) -> bool:
        """Check if the report satisfies all invariants."""
        return (
            self.requested_bars >= self.loaded_bars >= self.usable_bars >= self.validated_bars >= 0
            and self.warmup_satisfied
        )

    @property
    def coverage_pct(self) -> float:
        """Percentage of requested bars that are validated."""
        if self.requested_bars == 0:
            return 0.0
        return (self.validated_bars / self.requested_bars) * 100

    @property
    def reasons(self) -> List[str]:
        """Human-readable list of reduction reasons."""
        return [f"{r.description}: {r.bars_removed} bars" for r in self.reductions]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "requested_bars": self.requested_bars,
            "loaded_bars": self.loaded_bars,
            "usable_bars": self.usable_bars,
            "validated_bars": self.validated_bars,
            "reductions": [r.to_dict() for r in self.reductions],
            "data_source": self.data_source,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "warmup_satisfied": self.warmup_satisfied,
            "warmup_required": self.warmup_required,
            "warmup_indicator": self.warmup_indicator,
            "is_valid": self.is_valid,
            "coverage_pct": round(self.coverage_pct, 1),
            "reasons": self.reasons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BarValidationReport":
        """Deserialize from dictionary with schema validation."""
        # Validate schema version
        validate_schema_version(data.get("schema_version", ""))

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION_STR),
            symbol=data.get("symbol", ""),
            timeframe=data.get("timeframe", "1d"),
            validated_at=(
                datetime.fromisoformat(data["validated_at"]) if data.get("validated_at") else None
            ),
            requested_bars=data.get("requested_bars", 0),
            loaded_bars=data.get("loaded_bars", 0),
            usable_bars=data.get("usable_bars", 0),
            validated_bars=data.get("validated_bars", 0),
            reductions=tuple(BarReduction.from_dict(r) for r in data.get("reductions", [])),
            data_source=data.get("data_source", ""),
            start_date=(
                datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None
            ),
            end_date=(datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None),
            warmup_satisfied=data.get("warmup_satisfied", False),
            warmup_required=data.get("warmup_required", 252),
            warmup_indicator=data.get("warmup_indicator", ""),
        )

    def format_report(self) -> str:
        """Format as human-readable report string."""
        lines = [
            f"BarValidationReport for {self.symbol} ({self.timeframe}):",
            f"  requested_bars:  {self.requested_bars:>6}  (calendar days requested)",
            f"  loaded_bars:     {self.loaded_bars:>6}  (actual data fetched)",
            f"  usable_bars:     {self.usable_bars:>6}  (after NaN/gap trim)",
            f"  validated_bars:  {self.validated_bars:>6}  (after {self.warmup_indicator or 'warmup'})",
            "",
            f"  coverage: {self.coverage_pct:.1f}% | warmup_satisfied: {self.warmup_satisfied}",
            "",
            "  reasons:",
        ]
        for reason in self.reasons:
            lines.append(f"    - {reason}")

        return "\n".join(lines)


# =============================================================================
# BAR VALIDATION BUILDER (Mutable builder for constructing frozen report)
# =============================================================================


class BarValidationBuilder:
    """
    Builder for constructing BarValidationReport incrementally.

    Since BarValidationReport is frozen, use this builder to collect
    data during bar loading and validation, then call build() to
    create the immutable report.

    Usage:
        builder = BarValidationBuilder(symbol="AAPL", timeframe="1d")
        builder.set_requested_bars(550)
        builder.set_loaded_bars(389)
        builder.add_reduction(BarReductionReason.WEEKEND_HOLIDAY, 161, "Weekends/holidays removed")
        builder.set_usable_bars(340)
        builder.add_reduction(BarReductionReason.NAN_GAP, 49, "NaN gaps trimmed")
        builder.set_validated_bars(140, warmup_required=200, warmup_indicator="SMA(200)")
        builder.add_reduction(BarReductionReason.WARMUP, 200, "Warmup for SMA(200)")

        report = builder.build()
    """

    def __init__(self, symbol: str, timeframe: str = "1d") -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.requested_bars = 0
        self.loaded_bars = 0
        self.usable_bars = 0
        self.validated_bars = 0
        self.reductions: List[BarReduction] = []
        self.data_source = ""
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.warmup_required = 252
        self.warmup_indicator = ""

    def set_requested_bars(self, count: int) -> "BarValidationBuilder":
        """Set the number of calendar days/periods requested."""
        self.requested_bars = count
        return self

    def set_loaded_bars(
        self,
        count: int,
        source: str = "",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> "BarValidationBuilder":
        """Set the number of bars actually loaded from data source."""
        self.loaded_bars = count
        if source:
            self.data_source = source
        if start:
            self.start_date = start
        if end:
            self.end_date = end
        return self

    def set_usable_bars(self, count: int) -> "BarValidationBuilder":
        """Set the number of usable bars after NaN/gap trimming."""
        self.usable_bars = count
        return self

    def set_validated_bars(
        self,
        count: int,
        warmup_required: int = 252,
        warmup_indicator: str = "",
    ) -> "BarValidationBuilder":
        """Set the number of validated bars after warmup."""
        self.validated_bars = count
        self.warmup_required = warmup_required
        self.warmup_indicator = warmup_indicator
        return self

    def add_reduction(
        self,
        reason: BarReductionReason,
        bars_removed: int,
        description: str,
    ) -> "BarValidationBuilder":
        """Add a reduction explanation."""
        self.reductions.append(
            BarReduction(
                reason=reason,
                bars_removed=bars_removed,
                description=description,
            )
        )
        return self

    def build(self) -> BarValidationReport:
        """Build the frozen BarValidationReport."""
        warmup_satisfied = self.validated_bars >= 0 and self.usable_bars >= self.warmup_required

        return BarValidationReport(
            schema_version=SCHEMA_VERSION_STR,
            symbol=self.symbol,
            timeframe=self.timeframe,
            validated_at=datetime.now(tz=None),  # Use naive UTC datetime
            requested_bars=self.requested_bars,
            loaded_bars=self.loaded_bars,
            usable_bars=self.usable_bars,
            validated_bars=self.validated_bars,
            reductions=tuple(self.reductions),
            data_source=self.data_source,
            start_date=self.start_date,
            end_date=self.end_date,
            warmup_satisfied=warmup_satisfied,
            warmup_required=self.warmup_required,
            warmup_indicator=self.warmup_indicator,
        )


# =============================================================================
# SUMMARY SCHEMA (PR-03 Preview)
# =============================================================================


@dataclass(frozen=True)
class SymbolSummary:
    """
    Condensed summary for a single symbol (~1.5 KB budget).

    Used in summary.json for the package format. Contains only
    essential fields needed for the symbol selector and overview.
    """

    symbol: str
    regime: str  # "R0", "R1", "R2", "R3"
    regime_name: str
    confidence: int
    regime_changed: bool
    prev_regime: Optional[str]

    # Component states (compact)
    trend_state: str
    vol_state: str
    chop_state: str
    ext_state: str

    # Key metrics (subset)
    close: float
    atr_pct_63: float
    chop_pct: float

    # Turning point (if available)
    top_prob: Optional[float] = None
    bottom_prob: Optional[float] = None

    # Top signal (most relevant)
    top_signal_rule: Optional[str] = None
    top_signal_direction: Optional[str] = None
    top_signal_strength: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "confidence": self.confidence,
            "regime_changed": self.regime_changed,
            "prev_regime": self.prev_regime,
            "component_states": {
                "trend": self.trend_state,
                "vol": self.vol_state,
                "chop": self.chop_state,
                "ext": self.ext_state,
            },
            "key_metrics": {
                "close": round(self.close, 2),
                "atr_pct_63": round(self.atr_pct_63, 1),
                "chop_pct": round(self.chop_pct, 1),
            },
            "turning_point": (
                {"top_prob": self.top_prob, "bottom_prob": self.bottom_prob}
                if self.top_prob is not None
                else None
            ),
            "top_signal": (
                {
                    "rule": self.top_signal_rule,
                    "direction": self.top_signal_direction,
                    "strength": self.top_signal_strength,
                }
                if self.top_signal_rule
                else None
            ),
        }


# =============================================================================
# SIZE BUDGET CONSTANTS (PR-03)
# =============================================================================


class SizeBudget:
    """
    Size budget constants for summary.json sections.

    Enforced at construction time in SummaryBuilder to prevent
    field pruning pain later.
    """

    MAX_TOTAL_KB: int = 200
    MARKET_BUDGET_KB: int = 8
    SECTORS_BUDGET_KB: int = 20
    TICKERS_BUDGET_KB: int = 100  # ~1.5 KB per symbol for 65 symbols
    HIGHLIGHTS_BUDGET_KB: int = 40
    METADATA_BUDGET_KB: int = 32

    @classmethod
    def validate(cls, section: str, size_bytes: int) -> bool:
        """Check if a section is within budget."""
        budget_map = {
            "market": cls.MARKET_BUDGET_KB,
            "sectors": cls.SECTORS_BUDGET_KB,
            "tickers": cls.TICKERS_BUDGET_KB,
            "highlights": cls.HIGHLIGHTS_BUDGET_KB,
            "metadata": cls.METADATA_BUDGET_KB,
        }
        budget_kb = budget_map.get(section.lower(), 0)
        return size_bytes <= budget_kb * 1024
