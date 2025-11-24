"""
Position Reconciler - Detect MISSING, DRIFT, STALE positions.

Compares positions from multiple sources (IBKR vs manual YAML vs cached)
and identifies discrepancies.
"""

from __future__ import annotations
from typing import Dict, List, Set
from datetime import datetime
from ...models.position import Position, PositionSource
from ...models.reconciliation import ReconciliationIssue, IssueType


class Reconciler:
    """
    Position reconciliation service.

    Detects:
    - MISSING: Position in one source but not another
    - DRIFT: Quantity mismatch between sources
    - STALE: Position not updated for threshold period
    """

    def __init__(self, stale_threshold_seconds: int = 300):
        """
        Initialize reconciler.

        Args:
            stale_threshold_seconds: Threshold for marking positions stale (default: 5 min).
        """
        self.stale_threshold_seconds = stale_threshold_seconds

    def reconcile(
        self,
        ib_positions: List[Position],
        manual_positions: List[Position],
        cached_positions: List[Position],
    ) -> List[ReconciliationIssue]:
        """
        Reconcile positions from multiple sources.

        Args:
            ib_positions: Positions from Interactive Brokers.
            manual_positions: Positions from manual YAML file.
            cached_positions: Positions from previous snapshot.

        Returns:
            List of ReconciliationIssue objects.
        """
        issues: List[ReconciliationIssue] = []

        # Build position maps by key
        ib_map = {p.key(): p for p in ib_positions}
        manual_map = {p.key(): p for p in manual_positions}
        cached_map = {p.key(): p for p in cached_positions}

        # All unique keys across sources
        all_keys: Set[tuple] = set(ib_map.keys()) | set(manual_map.keys()) | set(cached_map.keys())

        for key in all_keys:
            ib_pos = ib_map.get(key)
            manual_pos = manual_map.get(key)
            cached_pos = cached_map.get(key)

            # Detect MISSING
            sources_present = sum([ib_pos is not None, manual_pos is not None])
            if sources_present == 1:
                issue = ReconciliationIssue(
                    issue_type=IssueType.MISSING,
                    symbol=key[0],
                    underlying=key[1],
                    ib_position=ib_pos,
                    manual_position=manual_pos,
                    cached_position=cached_pos,
                    severity="WARNING",
                )
                issues.append(issue)

            # Detect DRIFT (quantity mismatch)
            if ib_pos and manual_pos:
                if ib_pos.quantity != manual_pos.quantity:
                    issue = ReconciliationIssue(
                        issue_type=IssueType.DRIFT,
                        symbol=key[0],
                        underlying=key[1],
                        ib_position=ib_pos,
                        manual_position=manual_pos,
                        cached_position=cached_pos,
                        quantity_delta=ib_pos.quantity - manual_pos.quantity,
                        severity="CRITICAL",
                    )
                    issues.append(issue)

            # Detect STALE
            for pos in [ib_pos, manual_pos]:
                if pos and self._is_stale(pos):
                    issue = ReconciliationIssue(
                        issue_type=IssueType.STALE,
                        symbol=key[0],
                        underlying=key[1],
                        ib_position=ib_pos if pos == ib_pos else None,
                        manual_position=manual_pos if pos == manual_pos else None,
                        staleness_seconds=(datetime.now() - pos.last_updated).total_seconds(),
                        severity="WARNING",
                    )
                    issues.append(issue)
                    break  # Only report once per position

        return issues

    def _is_stale(self, position: Position) -> bool:
        """Check if position exceeds staleness threshold."""
        age_seconds = (datetime.now() - position.last_updated).total_seconds()
        return age_seconds > self.stale_threshold_seconds
