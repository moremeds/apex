"""Threshold helper for standardized alert severity checking."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ThresholdDirection(Enum):
    """Direction for threshold comparison."""
    ABOVE = "above"  # Alert when value >= threshold (e.g., VIX)
    BELOW = "below"  # Alert when value <= threshold (e.g., stop loss)


@dataclass
class Threshold:
    """
    Standardized threshold for warning/critical alerts.

    Reduces duplicate threshold comparison logic across detectors.
    Supports both "above threshold" (e.g., VIX) and "below threshold" (e.g., stop loss).

    Usage:
        vix_threshold = Threshold(warning=20, critical=30, direction=ThresholdDirection.ABOVE)
        severity = vix_threshold.check(current_vix)  # Returns WARNING, CRITICAL, or None

        stop_loss_threshold = Threshold(warning=-0.40, critical=-0.60, direction=ThresholdDirection.BELOW)
        severity = stop_loss_threshold.check(pnl_pct)  # Returns severity based on loss
    """
    warning: float
    critical: float
    direction: ThresholdDirection = ThresholdDirection.ABOVE

    def check(self, value: float) -> Optional[str]:
        """
        Check value against thresholds.

        Args:
            value: Current value to check.

        Returns:
            "CRITICAL", "WARNING", or None if no threshold breached.
        """
        if self.direction == ThresholdDirection.ABOVE:
            if value >= self.critical:
                return "CRITICAL"
            elif value >= self.warning:
                return "WARNING"
        else:  # BELOW
            if value <= self.critical:
                return "CRITICAL"
            elif value <= self.warning:
                return "WARNING"
        return None

    def breach_pct(self, value: float) -> float:
        """
        Calculate how far past the threshold the value is (as percentage).

        For ABOVE: (value - warning) / (critical - warning)
        For BELOW: (warning - value) / (warning - critical)

        Returns:
            Percentage where 0 = at warning, 1 = at critical, >1 = past critical
        """
        if self.direction == ThresholdDirection.ABOVE:
            if value < self.warning:
                return 0.0
            range_size = self.critical - self.warning
            if range_size == 0:
                return 1.0 if value >= self.warning else 0.0
            return (value - self.warning) / range_size
        else:  # BELOW
            if value > self.warning:
                return 0.0
            range_size = self.warning - self.critical
            if range_size == 0:
                return 1.0 if value <= self.warning else 0.0
            return (self.warning - value) / range_size


@dataclass
class RangeThreshold:
    """
    Threshold for values that must stay within a range.

    Used for portfolio Greeks (e.g., delta should be within [-50000, 50000]).
    """
    min_value: float
    max_value: float
    soft_margin_pct: float = 0.80  # Warning at 80% of limit

    def check(self, value: float) -> Optional[str]:
        """
        Check if value is within acceptable range.

        Args:
            value: Current value to check.

        Returns:
            "CRITICAL" if outside range, "WARNING" if in soft margin, None if OK.
        """
        if value < self.min_value or value > self.max_value:
            return "CRITICAL"

        # Check soft margin (warning zone)
        range_size = self.max_value - self.min_value
        soft_min = self.min_value + (range_size * (1 - self.soft_margin_pct) / 2)
        soft_max = self.max_value - (range_size * (1 - self.soft_margin_pct) / 2)

        if value < soft_min or value > soft_max:
            return "WARNING"

        return None

    def breach_pct(self, value: float) -> float:
        """
        Calculate how close to the limit the value is.

        Returns:
            0 = at center, 1 = at hard limit, >1 = past limit
        """
        center = (self.min_value + self.max_value) / 2
        half_range = (self.max_value - self.min_value) / 2

        if half_range == 0:
            return 0.0

        distance_from_center = abs(value - center)
        return distance_from_center / half_range
