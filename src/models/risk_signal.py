"""
Risk Signal Model - Structured risk alerts with actionable information.

Represents a detected risk condition with context, severity, and suggested actions.
Used by the Risk Signal Engine to provide structured output to the dashboard.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class SignalLevel(Enum):
    """Level at which the signal was detected."""
    PORTFOLIO = "PORTFOLIO"  # Portfolio-wide risk (e.g., total delta breach)
    POSITION = "POSITION"    # Single position risk (e.g., stop loss hit)
    STRATEGY = "STRATEGY"    # Multi-leg strategy risk (e.g., diagonal delta flip)


class SignalSeverity(Enum):
    """Severity of the risk signal."""
    INFO = "INFO"            # Informational - monitor
    WARNING = "WARNING"      # Warning - take action soon
    CRITICAL = "CRITICAL"    # Critical - immediate action required


class SuggestedAction(Enum):
    """Suggested action to address the risk."""
    MONITOR = "MONITOR"                # Continue monitoring
    REDUCE = "REDUCE"                  # Reduce position size
    CLOSE = "CLOSE"                    # Close position entirely
    ROLL = "ROLL"                      # Roll to different expiry/strike
    HEDGE = "HEDGE"                    # Add hedge position
    HALT_NEW_TRADES = "HALT_NEW_TRADES"  # Stop opening new positions


@dataclass
class RiskSignal:
    """
    Structured risk signal with actionable information.

    Represents a detected risk condition with full context for decision-making.
    Supports debouncing, cooldown, and severity escalation.
    """

    # Unique identifier for deduplication (format: "level:symbol:rule")
    signal_id: str

    # When the signal was generated
    timestamp: datetime

    # Classification
    level: SignalLevel
    severity: SignalSeverity

    # Context - what triggered this signal
    symbol: Optional[str] = None
    strategy_type: Optional[str] = None  # e.g., "DIAGONAL", "CREDIT_SPREAD"

    # Rule details - what was breached
    trigger_rule: str = ""  # e.g., "Delta_Limit_Breach", "Stop_Loss_Hit"
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    breach_pct: Optional[float] = None  # Percentage over threshold

    # Action guidance
    suggested_action: SuggestedAction = SuggestedAction.MONITOR
    action_details: str = ""  # Human-readable explanation

    # Metadata
    layer: int = 1  # 1-4 pyramid layer (1=Hard Limits, 2=Greeks, 3=Regime, 4=Events)
    cooldown_until: Optional[datetime] = None  # When cooldown expires
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for logging/API.

        Returns:
            Dictionary representation of the signal
        """
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "strategy_type": self.strategy_type,
            "trigger_rule": self.trigger_rule,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "breach_pct": self.breach_pct,
            "suggested_action": self.suggested_action.value,
            "action_details": self.action_details,
            "layer": self.layer,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_breach(cls, breach, layer: int = 1) -> RiskSignal:
        """
        Convert legacy LimitBreach to RiskSignal (for backward compatibility).

        Args:
            breach: LimitBreach object from RuleEngine
            layer: Risk pyramid layer (default 1 for hard limits)

        Returns:
            RiskSignal object
        """
        from ..domain.services.rule_engine import BreachSeverity

        # Map breach severity to signal severity
        severity = (
            SignalSeverity.CRITICAL
            if breach.severity == BreachSeverity.HARD
            else SignalSeverity.WARNING
        )

        # Map to suggested action
        if breach.severity == BreachSeverity.HARD:
            suggested_action = SuggestedAction.HALT_NEW_TRADES
        else:
            suggested_action = SuggestedAction.MONITOR

        # Generate signal ID
        signal_id = f"PORTFOLIO:{breach.limit_name}:{breach.severity.value}"

        return cls(
            signal_id=signal_id,
            timestamp=datetime.now(),
            level=SignalLevel.PORTFOLIO,
            severity=severity,
            trigger_rule=breach.limit_name,
            current_value=breach.current_value,
            threshold=breach.limit_value,
            breach_pct=breach.breach_pct(),
            suggested_action=suggested_action,
            action_details=f"Limit breached by {breach.breach_pct():.1f}%",
            layer=layer,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [
            f"[{self.severity.value}]",
            f"Level={self.level.value}",
        ]

        if self.symbol:
            parts.append(f"Symbol={self.symbol}")

        parts.append(f"Rule={self.trigger_rule}")

        if self.current_value is not None and self.threshold is not None:
            parts.append(f"Value={self.current_value:.2f} (Threshold={self.threshold:.2f})")

        if self.suggested_action != SuggestedAction.MONITOR:
            parts.append(f"Action={self.suggested_action.value}")

        return " | ".join(parts)

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"RiskSignal(id={self.signal_id!r}, "
            f"severity={self.severity.value}, "
            f"level={self.level.value}, "
            f"rule={self.trigger_rule!r})"
        )
