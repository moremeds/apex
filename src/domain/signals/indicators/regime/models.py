"""
Regime Detection Models and Enums.

Defines the core types for the 3-level hierarchical regime detection system:
- MarketRegime: R0 (Healthy), R1 (Choppy), R2 (Risk-Off), R3 (Rebound)
- Component states: TrendState, VolState, ChopState, ExtState, IVState
- RegimeState: Tracks regime with hysteresis for stable transitions
- RegimeOutput: Complete output with explainability primitives

Schema Version: regime_output@1.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from .rule_trace import RuleTrace


# Market benchmark symbols for regime hierarchy
MARKET_BENCHMARKS = {"QQQ", "SPY", "IWM", "DIA"}


class MarketRegime(Enum):
    """
    Market regime classification with trading implications.

    Priority Order (highest to lowest for classification):
    1. R2 (Risk-Off) - Veto power, always checked first
    2. R3 (Rebound)  - Only if NOT in active downtrend + structural confirm
    3. R1 (Choppy)   - Only if NOT in strong trend acceleration
    4. R0 (Healthy)  - Default when conditions are favorable
    """

    R0_HEALTHY_UPTREND = "R0"
    R1_CHOPPY_EXTENDED = "R1"
    R2_RISK_OFF = "R2"
    R3_REBOUND_WINDOW = "R3"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        names = {
            "R0": "Healthy Uptrend",
            "R1": "Choppy/Extended",
            "R2": "Risk-Off",
            "R3": "Rebound Window",
        }
        return names.get(self.value, self.value)

    @property
    def severity(self) -> int:
        """Severity level for comparison (higher = more restrictive)."""
        severity_map = {
            "R0": 0,
            "R1": 1,
            "R3": 2,  # R3 is between R1 and R2 in severity
            "R2": 3,
        }
        return severity_map.get(self.value, 0)


class TrendState(Enum):
    """
    Trend state based on MA relationships and slope.

    Classification:
    - UP: close > MA200 AND MA50_slope > 0 AND close > MA50
    - DOWN: close < MA200 AND MA50_slope < 0
    - NEUTRAL: otherwise
    """

    UP = "trend_up"
    DOWN = "trend_down"
    NEUTRAL = "neutral"


class VolState(Enum):
    """
    Volatility state based on dual-window ATR percentiles.

    Classification (realized volatility):
    - HIGH: ATR_pct_63 > 80 OR ATR_pct_252 > 85
    - LOW: ATR_pct_63 < 20 AND ATR_pct_252 < 25
    - NORMAL: otherwise
    """

    HIGH = "vol_high"
    NORMAL = "vol_normal"
    LOW = "vol_low"


class IVState(Enum):
    """
    Implied volatility state based on VIX/VXN percentile.

    Only applicable at MARKET level (QQQ/SPY).
    - HIGH: VIX_pct_63 > 75 (risk warning for short put)
    - ELEVATED: VIX_pct_63 in [50, 75] (caution)
    - NORMAL: VIX_pct_63 in [25, 50] (average volatility)
    - LOW: VIX_pct_63 < 25 (quiet market, favorable for premium selling)
    - NA: IV data unavailable
    """

    HIGH = "iv_high"
    ELEVATED = "iv_elevated"
    NORMAL = "iv_normal"
    LOW = "iv_low"
    NA = "na"


class ChopState(Enum):
    """
    Choppiness state based on CHOP index percentile and MA20 crosses.

    Classification:
    - CHOPPY: CHOP_pct_252 > 70 OR MA20_crosses >= 4 (in last 10 bars)
    - TRENDING: CHOP_pct_252 < 30 AND MA20_crosses <= 1
    - NEUTRAL: otherwise

    Note: 61.8/38.2 levels are for display only, not classification.
    """

    CHOPPY = "choppy"
    TRENDING = "trending"
    NEUTRAL = "neutral"


class ExtState(Enum):
    """
    Extension state (distance from mean).

    Classification (ext = (close - MA20) / ATR20):
    - OVERBOUGHT: ext > 2.0
    - OVERSOLD: ext < -2.0
    - SLIGHTLY_HIGH: ext > 1.5
    - SLIGHTLY_LOW: ext < -1.5
    - NEUTRAL: otherwise
    """

    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    SLIGHTLY_HIGH = "slightly_high"
    SLIGHTLY_LOW = "slightly_low"
    NEUTRAL = "neutral"


class FallbackReason(Enum):
    """Reason for fallback to default regime."""

    NONE = "none"
    WARMUP = "warmup"
    NAN = "nan"
    MISSING_DATA = "missing_data"
    EXCEPTION = "exception"


# Hysteresis thresholds for regime transitions
ENTRY_HYSTERESIS = {
    MarketRegime.R2_RISK_OFF: 0,  # Immediate - no delay for risk-off
    MarketRegime.R3_REBOUND_WINDOW: 2,  # Require 2 bars to confirm bottom
    MarketRegime.R1_CHOPPY_EXTENDED: 3,  # Require 3 bars to avoid noise
    MarketRegime.R0_HEALTHY_UPTREND: 2,  # Require 2 bars to confirm recovery
}

EXIT_HYSTERESIS = {
    MarketRegime.R2_RISK_OFF: 5,  # Stay cautious for 5 bars after R2
    MarketRegime.R3_REBOUND_WINDOW: 3,  # Don't exit R3 too quickly
    MarketRegime.R1_CHOPPY_EXTENDED: 2,
    MarketRegime.R0_HEALTHY_UPTREND: 1,  # Can exit R0 quickly if conditions worsen
}


@dataclass
class RegimeState:
    """
    Tracks regime with proper hysteresis for stable transitions.

    The pending_regime/pending_count pattern ensures:
    1. R2 triggers immediately (entry_hysteresis = 0)
    2. Other regimes require confirmation bars
    3. Exit from current regime respects exit_hysteresis
    """

    current_regime: MarketRegime = MarketRegime.R1_CHOPPY_EXTENDED
    pending_regime: Optional[MarketRegime] = None
    pending_count: int = 0
    bars_in_current: int = 0
    last_regime_change: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_regime": self.current_regime.value,
            "pending_regime": self.pending_regime.value if self.pending_regime else None,
            "pending_count": self.pending_count,
            "bars_in_current": self.bars_in_current,
            "last_regime_change": (
                self.last_regime_change.isoformat() if self.last_regime_change else None
            ),
        }


@dataclass
class ComponentStates:
    """
    Aggregated component states for regime classification.

    Contains all the individual component classifications that
    feed into the regime decision tree.
    """

    trend_state: TrendState = TrendState.NEUTRAL
    vol_state: VolState = VolState.NORMAL
    chop_state: ChopState = ChopState.NEUTRAL
    ext_state: ExtState = ExtState.NEUTRAL
    iv_state: IVState = IVState.NA

    def to_dict(self) -> Dict[str, str]:
        """Serialize to dictionary."""
        return {
            "trend_state": self.trend_state.value,
            "vol_state": self.vol_state.value,
            "chop_state": self.chop_state.value,
            "ext_state": self.ext_state.value,
            "iv_state": self.iv_state.value,
        }


@dataclass
class ComponentValues:
    """
    Raw numeric values from component calculations.

    Used for debugging, reporting, and threshold tuning.
    This is the legacy structure - prefer DerivedMetrics for new code.
    """

    # Price and MAs
    close: float = 0.0
    ma20: float = 0.0
    ma50: float = 0.0
    ma200: float = 0.0
    ma50_slope: float = 0.0

    # Volatility
    atr20: float = 0.0
    atr_pct: float = 0.0  # ATR as % of close
    atr_pct_63: float = 50.0  # ATR percentile (3-month)
    atr_pct_252: float = 50.0  # ATR percentile (1-year)

    # Implied volatility (market level only)
    iv_value: Optional[float] = None  # VIX/VXN raw value
    iv_pct_63: Optional[float] = None  # IV percentile

    # Choppiness
    chop: float = 50.0
    chop_pct_252: float = 50.0
    ma20_crosses: int = 0

    # Extension
    ext: float = 0.0

    # Structural (for R3 confirmation)
    last_5_bar_high: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "close": self.close,
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "ma50_slope": self.ma50_slope,
            "atr20": self.atr20,
            "atr_pct": self.atr_pct,
            "atr_pct_63": self.atr_pct_63,
            "atr_pct_252": self.atr_pct_252,
            "iv_value": self.iv_value,
            "iv_pct_63": self.iv_pct_63,
            "chop": self.chop,
            "chop_pct_252": self.chop_pct_252,
            "ma20_crosses": self.ma20_crosses,
            "ext": self.ext,
            "last_5_bar_high": self.last_5_bar_high,
        }


# ============================================================================
# NEW EXPLAINABILITY DATACLASSES (PR1)
# ============================================================================


@dataclass
class DataWindow:
    """
    Time window used for regime classification.

    Documents exactly which data range was used, enabling auditability.
    """

    start_ts: datetime = field(default_factory=datetime.now)
    end_ts: datetime = field(default_factory=datetime.now)
    bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "bars": self.bars,
        }


@dataclass
class BarSnapshot:
    """
    Single bar snapshot for history display.

    Captures a point-in-time view of key metrics and component states
    for the "Last N Bars Context" table in reports.
    """

    ts: datetime
    close: float
    key_metrics: Dict[str, float] = field(
        default_factory=dict
    )  # {"atr_pctile_short": 82, "chop_pctile": 78}
    component_states: Dict[str, str] = field(
        default_factory=dict
    )  # {"vol": "HIGH", "trend": "UP", ...}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ts": self.ts.isoformat() if self.ts else None,
            "close": self.close,
            "key_metrics": self.key_metrics,
            "component_states": self.component_states,
        }


@dataclass
class InputsUsed:
    """
    Raw input values for auditability.

    Contains the actual OHLCV values and historical snapshots used
    in regime classification. This enables full reproducibility.
    """

    close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    history: List[BarSnapshot] = field(default_factory=list)  # Last N bars with timestamps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "history": [h.to_dict() for h in self.history],
        }


@dataclass
class DerivedMetrics:
    """
    Calculated intermediate values with explicit naming.

    This is the enhanced version of ComponentValues with explicit
    percentile reference window documentation to avoid confusion.
    """

    # Moving averages
    ma20: float = 0.0
    ma50: float = 0.0
    ma200: float = 0.0
    ma50_slope: float = 0.0

    # Volatility (explicit reference windows)
    atr_value: float = 0.0  # Raw ATR20 value
    atr_pctile_short_window: float = 0.0  # Percentile vs last 63 bars
    atr_pctile_long_window: float = 0.0  # Percentile vs last 252 bars
    atr_reference_windows: Tuple[int, int] = (63, 252)  # Document the windows

    # Choppiness (explicit reference)
    chop_value: float = 0.0  # Raw CHOP index
    chop_pctile: float = 0.0  # Percentile vs last 252 bars
    chop_reference_window: int = 252
    ma20_crosses: int = 0

    # Extension
    ext_atr_units: float = 0.0  # (close - MA20) / ATR

    # IV (market level only)
    iv_value: Optional[float] = None
    iv_pctile: Optional[float] = None  # Percentile vs last 63 bars
    iv_reference_window: int = 63

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "ma50_slope": self.ma50_slope,
            "atr_value": self.atr_value,
            "atr_pctile_short_window": self.atr_pctile_short_window,
            "atr_pctile_long_window": self.atr_pctile_long_window,
            "atr_reference_windows": list(self.atr_reference_windows),
            "chop_value": self.chop_value,
            "chop_pctile": self.chop_pctile,
            "chop_reference_window": self.chop_reference_window,
            "ma20_crosses": self.ma20_crosses,
            "ext_atr_units": self.ext_atr_units,
            "iv_value": self.iv_value,
            "iv_pctile": self.iv_pctile,
            "iv_reference_window": self.iv_reference_window,
        }

    @classmethod
    def from_component_values(cls, cv: ComponentValues) -> "DerivedMetrics":
        """Create from legacy ComponentValues for backward compatibility."""
        return cls(
            ma20=cv.ma20,
            ma50=cv.ma50,
            ma200=cv.ma200,
            ma50_slope=cv.ma50_slope,
            atr_value=cv.atr20,
            atr_pctile_short_window=cv.atr_pct_63,
            atr_pctile_long_window=cv.atr_pct_252,
            chop_value=cv.chop,
            chop_pctile=cv.chop_pct_252,
            ma20_crosses=cv.ma20_crosses,
            ext_atr_units=cv.ext,
            iv_value=cv.iv_value,
            iv_pctile=cv.iv_pct_63,
        )


@dataclass
class DataQuality:
    """
    Data quality and fallback tracking.

    Provides explicit component validity (no inference needed in HTML)
    and tracks any data quality issues that affected classification.
    """

    warmup_ok: bool = False
    warmup_bars_needed: int = 252
    warmup_bars_available: int = 0

    nan_counts: Dict[str, int] = field(
        default_factory=dict
    )  # metric_name -> nan count
    missing_columns: List[str] = field(default_factory=list)

    fallback_reason: FallbackReason = FallbackReason.NONE
    fallback_active: bool = False
    exception_msg: Optional[str] = None  # Sanitized, only when fallback_reason == EXCEPTION

    # Explicit component validity (computed in detector, not inferred in HTML)
    component_validity: Dict[str, bool] = field(
        default_factory=dict
    )  # {"trend": True, "vol": True, ...}
    component_issues: Dict[str, str] = field(
        default_factory=dict
    )  # {"iv": "not available for symbol"}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "warmup_ok": self.warmup_ok,
            "warmup_bars_needed": self.warmup_bars_needed,
            "warmup_bars_available": self.warmup_bars_available,
            "nan_counts": self.nan_counts,
            "missing_columns": self.missing_columns,
            "fallback_reason": self.fallback_reason.value,
            "fallback_active": self.fallback_active,
            "exception_msg": self.exception_msg,
            "component_validity": self.component_validity,
            "component_issues": self.component_issues,
        }


@dataclass
class RegimeTransitionState:
    """
    Hysteresis state machine tracking.

    Separates the state machine tracking from the regime output
    for clearer explainability.
    """

    pending_regime: Optional[MarketRegime] = None
    pending_count: int = 0
    entry_threshold: int = 0  # Bars needed to enter pending_regime
    exit_threshold: int = 0  # Bars needed to exit current regime
    bars_in_current: int = 0
    last_transition_ts: Optional[datetime] = None
    transition_reason: Optional[str] = None  # "CHOP_pct crossed above 70"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pending_regime": (
                self.pending_regime.value if self.pending_regime else None
            ),
            "pending_count": self.pending_count,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "bars_in_current": self.bars_in_current,
            "last_transition_ts": (
                self.last_transition_ts.isoformat() if self.last_transition_ts else None
            ),
            "transition_reason": self.transition_reason,
        }


@dataclass
class RegimeOutput:
    """
    Complete regime detection output for a single bar.

    Combines regime classification, confidence, component states,
    transition information, and explainability primitives.

    Schema version: regime_output@1.0
    """

    # === SCHEMA & IDENTITY ===
    schema_version: str = "regime_output@1.0"
    symbol: str = ""
    asof_ts: Optional[datetime] = None  # Bar timestamp
    bar_interval: str = "1d"  # "1d", "4h", "1h", "30m"
    data_window: DataWindow = field(default_factory=DataWindow)

    # === REGIME CLASSIFICATION (Separated) ===
    decision_regime: MarketRegime = MarketRegime.R1_CHOPPY_EXTENDED  # Raw tree output
    final_regime: MarketRegime = MarketRegime.R1_CHOPPY_EXTENDED  # After hysteresis
    regime_name: str = "Choppy/Extended"
    confidence: int = 50

    # === COMPONENT STATES & VALUES ===
    component_states: ComponentStates = field(default_factory=ComponentStates)
    component_values: ComponentValues = field(default_factory=ComponentValues)

    # === EXPLAINABILITY ===
    inputs_used: InputsUsed = field(default_factory=InputsUsed)
    derived_metrics: DerivedMetrics = field(default_factory=DerivedMetrics)
    rules_fired_decision: List["RuleTrace"] = field(default_factory=list)  # Decision tree
    rules_fired_hysteresis: List["RuleTrace"] = field(
        default_factory=list
    )  # Hysteresis
    quality: DataQuality = field(default_factory=DataQuality)

    # === TRANSITION STATE ===
    transition: RegimeTransitionState = field(default_factory=RegimeTransitionState)
    regime_changed: bool = False
    previous_regime: Optional[MarketRegime] = None

    # === LEGACY FIELDS (backward compatibility) ===
    # These mirror final_regime/bars_in_regime for existing code
    @property
    def regime(self) -> MarketRegime:
        """Legacy alias for final_regime."""
        return self.final_regime

    @property
    def bars_in_regime(self) -> int:
        """Legacy alias for transition.bars_in_current."""
        return self.transition.bars_in_current

    @property
    def timestamp(self) -> Optional[datetime]:
        """Legacy alias for asof_ts."""
        return self.asof_ts

    @property
    def is_market_level(self) -> bool:
        """Check if symbol is a market benchmark."""
        return self.symbol.upper() in MARKET_BENCHMARKS

    def to_dict(self, precision: int = 4) -> Dict[str, Any]:
        """
        Serialize to dict with stable key ordering and rounding.

        Used for snapshots and JSON export.

        Args:
            precision: Decimal places for float rounding (default 4)

        Returns:
            Dictionary with stable ordering suitable for JSON serialization
        """

        def round_floats(obj: Any, prec: int) -> Any:
            if isinstance(obj, float):
                return round(obj, prec)
            elif isinstance(obj, dict):
                return {k: round_floats(v, prec) for k, v in sorted(obj.items())}
            elif isinstance(obj, list):
                return [round_floats(v, prec) for v in obj]
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, "item"):  # numpy scalar types (float64, int64, etc.)
                return round_floats(obj.item(), prec)
            else:
                return obj

        # Build dict manually for stable ordering
        result = {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "asof_ts": self.asof_ts.isoformat() if self.asof_ts else None,
            "bar_interval": self.bar_interval,
            "data_window": self.data_window.to_dict(),
            "decision_regime": self.decision_regime.value,
            "final_regime": self.final_regime.value,
            "regime_name": self.regime_name,
            "confidence": self.confidence,
            "component_states": self.component_states.to_dict(),
            "component_values": self.component_values.to_dict(),
            "inputs_used": self.inputs_used.to_dict(),
            "derived_metrics": self.derived_metrics.to_dict(),
            "rules_fired_decision": [r.to_dict() for r in self.rules_fired_decision],
            "rules_fired_hysteresis": [r.to_dict() for r in self.rules_fired_hysteresis],
            "quality": self.quality.to_dict(),
            "transition": self.transition.to_dict(),
            "regime_changed": self.regime_changed,
            "previous_regime": (
                self.previous_regime.value if self.previous_regime else None
            ),
        }

        return cast(Dict[str, Any], round_floats(result, precision))

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Serialize to legacy dictionary format for backward compatibility.

        Use to_dict() for new code with explainability fields.
        """
        return {
            "regime": self.final_regime.value,
            "regime_name": self.regime_name,
            "confidence": self.confidence,
            "component_states": self.component_states.to_dict(),
            "components": self.component_values.to_dict(),
            "transition": {
                "regime_changed": self.regime_changed,
                "previous_regime": (
                    self.previous_regime.value if self.previous_regime else None
                ),
                "bars_in_regime": self.transition.bars_in_current,
            },
            "timestamp": self.asof_ts.isoformat() if self.asof_ts else None,
            "symbol": self.symbol,
            "is_market_level": self.is_market_level,
        }
