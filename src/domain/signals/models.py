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


class SignalCategory(Enum):
    """Category of the trading signal based on indicator type."""

    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"


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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cooldown_until: Optional[datetime] = None
    message: str = ""  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        direction_symbol = {"buy": "▲", "sell": "▼", "alert": "●"}.get(
            self.direction.value, "?"
        )
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

        Args:
            prev_state: Previous indicator state (None if first evaluation)
            curr_state: Current indicator state

        Returns:
            True if condition is met, False otherwise
        """
        if self.condition_type == ConditionType.THRESHOLD_CROSS_UP:
            return self._check_threshold_cross_up(prev_state, curr_state)
        elif self.condition_type == ConditionType.THRESHOLD_CROSS_DOWN:
            return self._check_threshold_cross_down(prev_state, curr_state)
        elif self.condition_type == ConditionType.STATE_CHANGE:
            return self._check_state_change(prev_state, curr_state)
        elif self.condition_type == ConditionType.CROSS_UP:
            return self._check_cross_up(prev_state, curr_state)
        elif self.condition_type == ConditionType.CROSS_DOWN:
            return self._check_cross_down(prev_state, curr_state)
        elif self.condition_type == ConditionType.RANGE_ENTRY:
            return self._check_range_entry(prev_state, curr_state)
        elif self.condition_type == ConditionType.RANGE_EXIT:
            return self._check_range_exit(prev_state, curr_state)
        elif self.condition_type == ConditionType.CUSTOM:
            raise NotImplementedError(
                f"CUSTOM condition type requires a custom handler. "
                f"Rule '{self.name}' must provide a callable in condition_config['handler']."
            )
        return False

    def _check_threshold_cross_up(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Value crosses above threshold."""
        if prev_state is None:
            return False
        field = self.condition_config.get("field", "value")
        threshold = self.condition_config.get("threshold", 0)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)
        return prev_val < threshold <= curr_val

    def _check_threshold_cross_down(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Value crosses below threshold."""
        if prev_state is None:
            return False
        field = self.condition_config.get("field", "value")
        threshold = self.condition_config.get("threshold", 0)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)
        return prev_val > threshold >= curr_val

    def _check_state_change(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """State transitions from one value to another."""
        if prev_state is None:
            return False
        field = self.condition_config.get("field", "zone")
        from_states = self.condition_config.get("from", [])
        to_states = self.condition_config.get("to", [])
        prev_val = prev_state.get(field)
        curr_val = curr_state.get(field)
        return prev_val in from_states and curr_val in to_states

    def _check_cross_up(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Line A crosses above Line B."""
        if prev_state is None:
            return False
        line_a = self.condition_config.get("line_a", "fast")
        line_b = self.condition_config.get("line_b", "slow")
        prev_a = prev_state.get(line_a, 0)
        prev_b = prev_state.get(line_b, 0)
        curr_a = curr_state.get(line_a, 0)
        curr_b = curr_state.get(line_b, 0)
        return prev_a <= prev_b and curr_a > curr_b

    def _check_cross_down(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Line A crosses below Line B."""
        if prev_state is None:
            return False
        line_a = self.condition_config.get("line_a", "fast")
        line_b = self.condition_config.get("line_b", "slow")
        prev_a = prev_state.get(line_a, 0)
        prev_b = prev_state.get(line_b, 0)
        curr_a = curr_state.get(line_a, 0)
        curr_b = curr_state.get(line_b, 0)
        return prev_a >= prev_b and curr_a < curr_b

    def _check_range_entry(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Value enters a range."""
        if prev_state is None:
            return False
        field = self.condition_config.get("field", "value")
        lower = self.condition_config.get("lower", 0)
        upper = self.condition_config.get("upper", 100)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)
        was_outside = prev_val < lower or prev_val > upper
        is_inside = lower <= curr_val <= upper
        return was_outside and is_inside

    def _check_range_exit(
        self,
        prev_state: Optional[Dict[str, Any]],
        curr_state: Dict[str, Any],
    ) -> bool:
        """Value exits a range."""
        if prev_state is None:
            return False
        field = self.condition_config.get("field", "value")
        lower = self.condition_config.get("lower", 0)
        upper = self.condition_config.get("upper", 100)
        prev_val = prev_state.get(field, 0)
        curr_val = curr_state.get(field, 0)
        was_inside = lower <= prev_val <= upper
        is_outside = curr_val < lower or curr_val > upper
        return was_inside and is_outside


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
