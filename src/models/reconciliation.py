"""Reconciliation issue model for position discrepancies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .position import Position


class IssueType(Enum):
    """Type of reconciliation issue."""

    MISSING = "MISSING"  # Position in one source but not another
    DRIFT = "DRIFT"  # Quantity mismatch between sources
    STALE = "STALE"  # Position not updated for threshold period


@dataclass
class ReconciliationIssue:
    """Represents a position reconciliation discrepancy."""

    issue_type: IssueType
    symbol: str
    underlying: str

    # Source positions (None if missing from that source)
    ib_position: Optional[Position] = None
    manual_position: Optional[Position] = None
    cached_position: Optional[Position] = None

    # Discrepancy details
    quantity_delta: Optional[int] = None
    staleness_seconds: Optional[float] = None

    # Metadata
    detected_at: datetime = None
    severity: str = "WARNING"  # WARNING, CRITICAL

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now()

    def description(self) -> str:
        """Human-readable description of the issue."""
        if self.issue_type == IssueType.MISSING:
            sources = []
            if self.ib_position:
                sources.append("IB")
            if self.manual_position:
                sources.append("MANUAL")
            if self.cached_position:
                sources.append("CACHED")
            return f"{self.symbol} present in {sources} but missing from other source(s)"

        elif self.issue_type == IssueType.DRIFT:
            return f"{self.symbol} quantity mismatch: delta={self.quantity_delta}"

        elif self.issue_type == IssueType.STALE:
            return f"{self.symbol} not updated for {self.staleness_seconds:.0f}s"

        return f"Unknown issue for {self.symbol}"
