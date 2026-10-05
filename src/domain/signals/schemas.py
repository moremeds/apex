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
from typing import Any, Dict, Optional, Tuple

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
