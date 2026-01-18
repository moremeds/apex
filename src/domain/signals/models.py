"""
Trading Signal Domain Models.

Defines core domain models for the trading signal engine:
- TradingSignal: Generated signal with context and metadata
- SignalRule: Rule definition for signal generation
- Divergence: Price/indicator divergence detection result
- Enums: Categories, directions, priorities, condition types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ...utils.timezone import now_utc


class SignalCategory(Enum):
    """Category of the trading signal based on indicator type."""

    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"
    REGIME = "regime"


class SignalDirection(Enum):
    """Direction of the trading signal."""

    BUY = "buy"
    SELL = "sell"
    ALERT = "alert"  # Informational, no directional bias


class SignalPriority(Enum):
    """Priority level of the signal for display and filtering."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SignalStatus(Enum):
    """Lifecycle status of a trading signal."""

    ACTIVE = "active"  # Signal is current and actionable
    INVALIDATED = (
        "invalidated"  # Superseded by newer signal for same (symbol, indicator, timeframe)
    )


class ConditionType(Enum):
    """Type of condition that triggers a signal rule."""

    THRESHOLD_CROSS_UP = "threshold_cross_up"
    THRESHOLD_CROSS_DOWN = "threshold_cross_down"
    STATE_CHANGE = "state_change"
    CROSS_UP = "cross_up"
    CROSS_DOWN = "cross_down"
    RANGE_ENTRY = "range_entry"
    RANGE_EXIT = "range_exit"
    CUSTOM = "custom"


class DivergenceType(Enum):
    """Type of divergence between price and indicator."""

    BULLISH = "bullish"  # Price lower low, indicator higher low
    BEARISH = "bearish"  # Price higher high, indicator lower high
    HIDDEN_BULLISH = "hidden_bullish"  # Price higher low, indicator lower low
    HIDDEN_BEARISH = "hidden_bearish"  # Price lower high, indicator higher high


@dataclass
class TradingSignal:
    """
    Generated trading signal with full context.

    Represents a detected trading condition based on technical indicators
    and rule evaluation. Used for display in TUI and logging.
    """

    # Unique identifier: "{category}:{indicator}:{symbol}:{timeframe}"
    signal_id: str

    # Core identification
    symbol: str
    category: SignalCategory
    indicator: str  # e.g., "rsi", "macd", "supertrend"
    direction: SignalDirection
    strength: int  # 0-100 signal strength
    priority: SignalPriority
    timeframe: str  # e.g., "1m", "5m", "1h", "1d"

    # Trigger context
    trigger_rule: str  # Rule name that triggered this signal
    current_value: float  # Current indicator value
    threshold: Optional[float] = None  # Threshold if applicable
    previous_value: Optional[float] = None  # For state change detection

    # Metadata
    timestamp: datetime = field(default_factory=now_utc)
    cooldown_until: Optional[datetime] = None
    message: str = ""  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Signal lifecycle (invalidation tracking)
    status: SignalStatus = SignalStatus.ACTIVE
    invalidated_by: Optional[str] = None  # signal_id of superseding signal
    invalidated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/API."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "category": self.category.value,
            "indicator": self.indicator,
            "direction": self.direction.value,
            "strength": self.strength,
            "priority": self.priority.value,
            "timeframe": self.timeframe,
            "trigger_rule": self.trigger_rule,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "previous_value": self.previous_value,
            "timestamp": self.timestamp.isoformat(),
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "message": self.message,
            "metadata": self.metadata,
            "status": self.status.value,
            "invalidated_by": self.invalidated_by,
            "invalidated_at": self.invalidated_at.isoformat() if self.invalidated_at else None,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        direction_symbol = {"buy": "▲", "sell": "▼", "alert": "●"}.get(self.direction.value, "?")
        return (
            f"{direction_symbol} [{self.priority.value.upper()}] "
            f"{self.symbol} {self.indicator} ({self.timeframe}) - {self.message}"
        )

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"TradingSignal(id={self.signal_id!r}, "
            f"direction={self.direction.value}, "
            f"strength={self.strength})"
        )


@dataclass
class SignalRule:
    """
    Rule definition for generating trading signals.

    Defines when and how to generate a TradingSignal based on
    indicator values and state transitions.
    """

    name: str  # Unique rule identifier
    indicator: str  # Indicator name (e.g., "rsi", "macd")
    category: SignalCategory
    direction: SignalDirection
    strength: int  # 0-100 base signal strength
    priority: SignalPriority

    # Condition specification
    condition_type: ConditionType
    condition_config: Dict[str, Any]  # Type-specific parameters

    # Filtering
    timeframes: Tuple[str, ...] = ("1h", "4h", "1d")
    cooldown_seconds: int = 3600  # 1 hour default
    min_volume: Optional[float] = None
    enabled: bool = True

    # Message template with placeholders: {symbol}, {indicator}, {value}, {threshold}
    message_template: str = ""

    def format_message(
        self,
        symbol: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Format the message template with actual values."""
        if not self.message_template:
            return f"{symbol} {self.indicator} signal triggered"

        return self.message_template.format(
            symbol=symbol,
            indicator=self.indicator,
            value=f"{value:.2f}" if value is not None else "N/A",
            threshold=f"{threshold:.2f}" if threshold is not None else "N/A",
        )

    def check_condition(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """
        Check if the rule condition is met.

        Uses pluggable evaluators from conditions module for each ConditionType.

        Args:
            prev_state: Previous indicator state (None if first evaluation)
            curr_state: Current indicator state

        Returns:
            True if condition is met, False otherwise
        """
        # Handle CUSTOM separately (requires handler in config)
        if self.condition_type == ConditionType.CUSTOM:
            handler = self.condition_config.get("handler")
            if handler is None:
                raise NotImplementedError(
                    f"CUSTOM condition type requires a callable in condition_config['handler']. "
                    f"Rule '{self.name}' is missing the handler."
                )
            return bool(handler(self.condition_config, prev_state, curr_state))

        # Use pluggable evaluator
        from .conditions import EVALUATORS

        evaluator = EVALUATORS.get(self.condition_type)
        if evaluator is None:
            return False

        return evaluator.evaluate(self.condition_config, prev_state, curr_state)


@dataclass
class Divergence:
    """
    Detected divergence between price and indicator.

    Represents a divergence pattern where price and indicator
    move in opposite directions, potentially signaling reversal.
    """

    type: DivergenceType
    indicator: str
    symbol: str
    timeframe: str

    # Price pivot points
    price_point1: Tuple[datetime, float]  # First pivot (older)
    price_point2: Tuple[datetime, float]  # Second pivot (newer)

    # Indicator pivot points
    indicator_point1: Tuple[datetime, float]
    indicator_point2: Tuple[datetime, float]

    # Quality metrics
    strength: int  # 0-100 based on angle difference
    bars_apart: int  # Distance between pivots in bars

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type.value,
            "indicator": self.indicator,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "price_point1": {
                "time": self.price_point1[0].isoformat(),
                "value": self.price_point1[1],
            },
            "price_point2": {
                "time": self.price_point2[0].isoformat(),
                "value": self.price_point2[1],
            },
            "indicator_point1": {
                "time": self.indicator_point1[0].isoformat(),
                "value": self.indicator_point1[1],
            },
            "indicator_point2": {
                "time": self.indicator_point2[0].isoformat(),
                "value": self.indicator_point2[1],
            },
            "strength": self.strength,
            "bars_apart": self.bars_apart,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"{self.type.value.upper()} divergence: "
            f"{self.symbol} {self.indicator} ({self.timeframe}) "
            f"strength={self.strength}"
        )


@dataclass
class IndicatorTrace:
    """
    Trace of indicator calculation for observability (Phase 3).

    Links signals to raw indicator values and regime decisions,
    enabling full auditability of the signal generation pipeline.

    Each IndicatorTrace captures a single indicator's state at a given bar,
    including the raw values, derived state, and which rules triggered.
    """

    indicator_name: str  # e.g., "rsi", "macd", "atr", "regime"
    timeframe: str  # e.g., "1m", "5m", "1h", "1d"
    bar_ts: datetime  # Timestamp of the bar
    symbol: str = ""  # Symbol this trace belongs to

    # Raw indicator values: {"rsi": 28.5, "macd": -1.2, "macd_signal": -1.0}
    raw: Dict[str, float] = field(default_factory=dict)

    # Derived state: {"zone": "oversold", "direction": "down", "strength": "strong"}
    state: Dict[str, Any] = field(default_factory=dict)

    # Rules that triggered on this bar: ["rsi_oversold_entry", "rsi_bullish_divergence"]
    rules_triggered_now: List[str] = field(default_factory=list)

    # Lookback period used in calculation (for context)
    lookback: int = 0

    # Optional previous raw values for delta calculation
    prev_raw: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "indicator_name": self.indicator_name,
            "timeframe": self.timeframe,
            "bar_ts": self.bar_ts.isoformat(),
            "symbol": self.symbol,
            "raw": self.raw,
            "state": self.state,
            "rules_triggered_now": self.rules_triggered_now,
            "lookback": self.lookback,
            "prev_raw": self.prev_raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndicatorTrace":
        """Deserialize from dictionary."""
        return cls(
            indicator_name=data.get("indicator_name", ""),
            timeframe=data.get("timeframe", ""),
            bar_ts=(
                datetime.fromisoformat(data["bar_ts"]) if data.get("bar_ts") else datetime.utcnow()
            ),
            symbol=data.get("symbol", ""),
            raw=data.get("raw", {}),
            state=data.get("state", {}),
            rules_triggered_now=data.get("rules_triggered_now", []),
            lookback=data.get("lookback", 0),
            prev_raw=data.get("prev_raw"),
        )

    def get_delta(self, key: str) -> Optional[float]:
        """Calculate delta from previous raw value for a key."""
        if self.prev_raw is None:
            return None
        if key not in self.raw or key not in self.prev_raw:
            return None
        return self.raw[key] - self.prev_raw[key]

    def __str__(self) -> str:
        """Human-readable representation."""
        rules_str = ", ".join(self.rules_triggered_now) if self.rules_triggered_now else "none"
        return (
            f"IndicatorTrace({self.indicator_name}@{self.timeframe}, "
            f"raw={self.raw}, state={self.state}, rules=[{rules_str}])"
        )


@dataclass
class ConfluenceScore:
    """
    Multi-indicator confluence scoring result.

    Aggregates signals from multiple indicators to determine
    overall market sentiment alignment.
    """

    symbol: str
    timeframe: str
    bullish_count: int
    bearish_count: int
    neutral_count: int
    alignment_score: int  # -100 (all bearish) to +100 (all bullish)
    diverging_pairs: List[Tuple[str, str, str]]  # (ind1, ind2, reason)
    strongest_signal: Optional[str]  # Dominant indicator direction
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "alignment_score": self.alignment_score,
            "diverging_pairs": [
                {"ind1": p[0], "ind2": p[1], "reason": p[2]} for p in self.diverging_pairs
            ],
            "strongest_signal": self.strongest_signal,
            "timestamp": self.timestamp.isoformat(),
        }
