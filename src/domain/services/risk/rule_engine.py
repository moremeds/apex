"""
Rule Engine - Soft/hard breach detection.

Evaluates risk limits against current snapshot and classifies breaches
as SOFT (warning) or HARD (critical).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from src.models.risk_snapshot import RiskSnapshot
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class BreachSeverity(Enum):
    """Breach severity levels."""

    SOFT = "SOFT"  # Warning threshold (e.g., 80% of limit)
    HARD = "HARD"  # Critical breach (limit exceeded)


@dataclass
class LimitBreach:
    """Represents a risk limit breach."""

    limit_name: str
    severity: BreachSeverity
    current_value: float
    limit_value: float
    underlying: str = ""  # For per-underlying limits

    def breach_pct(self) -> float:
        """Calculate breach percentage."""
        if self.limit_value == 0:
            return 0.0
        return (self.current_value / self.limit_value) * 100

    def description(self) -> str:
        """Human-readable breach description."""
        pct = self.breach_pct()
        if self.underlying:
            return f"{self.limit_name} [{self.underlying}]: {self.current_value:,.0f} / {self.limit_value:,.0f} ({pct:.1f}%)"
        return (
            f"{self.limit_name}: {self.current_value:,.0f} / {self.limit_value:,.0f} ({pct:.1f}%)"
        )


class RuleEngine:
    """
    Risk limit rule engine.

    Evaluates portfolio metrics against configured limits and
    classifies breaches as SOFT or HARD.
    """

    def __init__(self, risk_limits: Dict[str, Any], soft_threshold: float = 0.80):
        """
        Initialize rule engine.

        Args:
            risk_limits: Risk limits configuration dict.
            soft_threshold: Soft breach threshold (default: 80% of limit).
        """
        self.risk_limits = risk_limits
        self.soft_threshold = soft_threshold

    def evaluate(self, snapshot: RiskSnapshot) -> List[LimitBreach]:
        """
        Evaluate snapshot against all risk limits.

        Args:
            snapshot: Current risk snapshot.

        Returns:
            List of LimitBreach objects (empty if no breaches).
        """
        breaches: List[LimitBreach] = []

        # Max total gross notional
        max_notional = self.risk_limits.get("max_total_gross_notional")
        if max_notional:
            breaches.extend(
                self._check_limit(
                    "max_total_gross_notional",
                    snapshot.total_gross_notional,
                    max_notional,
                )
            )

        # Per-underlying notional limits
        per_underlying = self.risk_limits.get("max_per_underlying_notional", {})
        default_limit = per_underlying.get("default")
        for underlying, notional in snapshot.notional_by_underlying.items():
            limit = per_underlying.get(underlying, default_limit)
            if limit:
                breaches.extend(
                    self._check_limit(
                        "max_per_underlying_notional",
                        abs(notional),
                        limit,
                        underlying=underlying,
                    )
                )

        # Portfolio delta range
        delta_range = self.risk_limits.get("portfolio_delta_range")
        if delta_range:
            breaches.extend(
                self._check_range("portfolio_delta", snapshot.portfolio_delta, delta_range)
            )

        # Portfolio vega range
        vega_range = self.risk_limits.get("portfolio_vega_range")
        if vega_range:
            breaches.extend(
                self._check_range("portfolio_vega", snapshot.portfolio_vega, vega_range)
            )

        # Portfolio theta range
        theta_range = self.risk_limits.get("portfolio_theta_range")
        if theta_range:
            breaches.extend(
                self._check_range("portfolio_theta", snapshot.portfolio_theta, theta_range)
            )

        # Margin utilization
        max_margin = self.risk_limits.get("max_margin_utilization")
        if max_margin:
            breaches.extend(
                self._check_limit(
                    "max_margin_utilization",
                    snapshot.margin_utilization,
                    max_margin,
                )
            )

        # Concentration percentage
        max_concentration = self.risk_limits.get("max_concentration_pct")
        if max_concentration:
            breaches.extend(
                self._check_limit(
                    "max_concentration_pct",
                    snapshot.concentration_pct,
                    max_concentration,
                )
            )

        return breaches

    def _check_limit(
        self, name: str, value: float, limit: float, underlying: str = ""
    ) -> List[LimitBreach]:
        """Check single limit (value <= limit)."""
        breaches = []
        soft_limit = limit * self.soft_threshold

        if value >= limit:
            breaches.append(
                LimitBreach(
                    limit_name=name,
                    severity=BreachSeverity.HARD,
                    current_value=value,
                    limit_value=limit,
                    underlying=underlying,
                )
            )
        elif value >= soft_limit:
            breaches.append(
                LimitBreach(
                    limit_name=name,
                    severity=BreachSeverity.SOFT,
                    current_value=value,
                    limit_value=limit,
                    underlying=underlying,
                )
            )

        return breaches

    def _check_range(self, name: str, value: float, range_limits: List[float]) -> List[LimitBreach]:
        """
        Check range limit (min <= value <= max) with soft threshold support.

        Args:
            name: Name of the limit being checked.
            value: Current value to check.
            range_limits: List of [min_limit, max_limit].

        Returns:
            List of LimitBreach objects (empty if no breach).
        """
        breaches = []

        # Validate range_limits has exactly 2 elements
        if len(range_limits) != 2:
            logger.error(
                f"Invalid range_limits for {name}: expected [min, max], got {range_limits}"
            )
            return []

        min_limit, max_limit = range_limits

        # Calculate soft thresholds (percentage of the way towards the limit)
        range_size = max_limit - min_limit
        if range_size > 0:
            soft_buffer = range_size * (1 - self.soft_threshold) / 2
            soft_min = min_limit + soft_buffer
            soft_max = max_limit - soft_buffer
        else:
            # If range is 0 or negative, don't use soft thresholds
            soft_min = min_limit
            soft_max = max_limit

        # Check for breaches
        if value < min_limit:
            breaches.append(
                LimitBreach(
                    limit_name=f"{name}_min",
                    severity=BreachSeverity.HARD,
                    current_value=value,
                    limit_value=min_limit,
                )
            )
        elif value < soft_min:
            breaches.append(
                LimitBreach(
                    limit_name=f"{name}_min",
                    severity=BreachSeverity.SOFT,
                    current_value=value,
                    limit_value=min_limit,
                )
            )
        elif value > max_limit:
            breaches.append(
                LimitBreach(
                    limit_name=f"{name}_max",
                    severity=BreachSeverity.HARD,
                    current_value=value,
                    limit_value=max_limit,
                )
            )
        elif value > soft_max:
            breaches.append(
                LimitBreach(
                    limit_name=f"{name}_max",
                    severity=BreachSeverity.SOFT,
                    current_value=value,
                    limit_value=max_limit,
                )
            )

        return breaches
