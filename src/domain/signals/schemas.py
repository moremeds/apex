"""Bar data-quality types shared by the signal data validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

# =============================================================================
# DATA QUALITY TYPES (PR-A: Data Quality Gates)
# =============================================================================


class InvalidValueReason(Enum):
    """Reasons for invalid bar values."""

    NONE = "none"  # Value is valid
    SENTINEL_NEGATIVE = "sentinel_negative"  # -1.0 sentinel value
    ZERO_VALUE = "zero_value"  # 0.0 value (invalid for close)
    NAN_VALUE = "nan_value"  # NaN value
    NEGATIVE_VALUE = "negative_value"  # Negative value (invalid for OHLCV)


@dataclass(frozen=True)
class BarQualityResult:
    """
    Result of bar data quality validation.

    PR-A Deliverable: Single entry point for data quality checks.
    Used by DataQualityValidator to report on bar validity.

    Attributes:
        valid: Whether the bar data is valid for downstream processing
        close: The validated close price (0.0 if invalid)
        timestamp: The bar timestamp
        invalid_reason: Reason for invalidity (NONE if valid)
        sentinel_counts: Count of sentinel values (-1.0) by column
        nan_counts: Count of NaN values by column
        dropped_bar_count: Number of bars dropped during cleaning
        usable_bar_count: Number of bars usable after cleaning
    """

    valid: bool
    close: float
    timestamp: Optional[datetime]
    invalid_reason: InvalidValueReason = InvalidValueReason.NONE
    sentinel_counts: Dict[str, int] = field(default_factory=dict)
    nan_counts: Dict[str, int] = field(default_factory=dict)
    dropped_bar_count: int = 0
    usable_bar_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "valid": self.valid,
            "close": self.close,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "invalid_reason": self.invalid_reason.value,
            "sentinel_counts": self.sentinel_counts,
            "nan_counts": self.nan_counts,
            "dropped_bar_count": self.dropped_bar_count,
            "usable_bar_count": self.usable_bar_count,
        }

    @classmethod
    def valid_result(
        cls,
        close: float,
        timestamp: Optional[datetime],
        usable_bar_count: int,
    ) -> "BarQualityResult":
        """Factory for valid result."""
        return cls(
            valid=True,
            close=close,
            timestamp=timestamp,
            invalid_reason=InvalidValueReason.NONE,
            usable_bar_count=usable_bar_count,
        )

    @classmethod
    def invalid_result(
        cls,
        reason: InvalidValueReason,
        timestamp: Optional[datetime] = None,
        sentinel_counts: Optional[Dict[str, int]] = None,
        nan_counts: Optional[Dict[str, int]] = None,
    ) -> "BarQualityResult":
        """Factory for invalid result."""
        return cls(
            valid=False,
            close=0.0,
            timestamp=timestamp,
            invalid_reason=reason,
            sentinel_counts=sentinel_counts or {},
            nan_counts=nan_counts or {},
        )
