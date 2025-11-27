"""
Position Reconciler - Detect MISSING, DRIFT, STALE positions.

Compares positions from multiple sources (IBKR vs manual YAML vs cached)
and identifies discrepancies.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional
from datetime import datetime
import logging
from ...models.position import Position, PositionSource, AssetType
from ...models.reconciliation import ReconciliationIssue, IssueType

logger = logging.getLogger(__name__)


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

    def remove_expired_options(
        self,
        positions: List[Position],
        ref_date=None,
    ) -> List[Position]:
        """
        Drop expired option positions so they do not linger in downstream views.

        Args:
            positions: Positions to evaluate.
            ref_date: Optional reference date for expiry calculations (defaults to today).

        Returns:
            Filtered list with expired options removed.
        """
        filtered: List[Position] = []
        for pos in positions:
            if pos.asset_type != AssetType.OPTION:
                filtered.append(pos)
                continue

            dte = pos.days_to_expiry(ref_date=ref_date)
            if dte is None or dte >= 0:
                filtered.append(pos)
            else:
                logger.info(f"Dropping expired option position {pos.symbol} (DTE={dte})")

        return filtered

    def merge_positions(
        self,
        ib_positions: List[Position],
        manual_positions: List[Position],
        futu_positions: Optional[List[Position]] = None,
    ) -> List[Position]:
        """
        Merge positions across sources while preserving business rules.

        For positions with the same key:
        - Aggregates quantities across accounts/sources
        - Manual avg_price takes precedence (user-specified cost basis)
        - Broker sources (IB, FUTU) take precedence over manual for metadata

        Args:
            ib_positions: Positions from Interactive Brokers.
            manual_positions: Positions from manual YAML file.
            futu_positions: Positions from Futu OpenD (optional).

        Returns:
            Merged list of positions with aggregated quantities.
        """
        merged: Dict[tuple, Position] = {}

        # Build manual map for avg_price lookups
        manual_map = {p.key(): p for p in manual_positions}

        # Build list of all broker positions (IB first, then Futu)
        broker_positions = ib_positions.copy()
        if futu_positions:
            broker_positions.extend(futu_positions)

        # Process broker positions (primary source for quantity/metadata)
        for position in broker_positions:
            key = position.key()

            if key not in merged:
                # Check if manual has avg_price override for this position
                manual_pos = manual_map.get(key)
                if manual_pos is not None:
                    # Use manual avg_price (user-specified cost basis)
                    position.avg_price = manual_pos.avg_price
                    # Also preserve manual strategy_tag if set
                    if manual_pos.strategy_tag:
                        position.strategy_tag = manual_pos.strategy_tag

                merged[key] = position
            else:
                # Aggregate quantities (same position from multiple accounts)
                existing = merged[key]
                total_quantity = existing.quantity + position.quantity
                existing.quantity = total_quantity
                # Keep the existing avg_price (already set from manual or first broker position)

        # Process manual positions that don't exist in any broker
        for position in manual_positions:
            key = position.key()

            if key not in merged:
                # Position only in manual - use as-is
                merged[key] = position

        return list(merged.values())

    def merge_all_positions(self, positions_by_source: Dict[str, List[Position]]) -> List[Position]:
        """
        Merge positions from multiple sources using a flexible dict input.

        This method provides a more flexible interface for multi-broker scenarios.

        Args:
            positions_by_source: Dict mapping source name to positions.
                                 Keys can be "ib", "futu", "manual", etc.

        Returns:
            Merged list of positions.
        """
        ib_positions = positions_by_source.get("ib", [])
        futu_positions = positions_by_source.get("futu", [])
        manual_positions = positions_by_source.get("manual", [])

        return self.merge_positions(ib_positions, manual_positions, futu_positions)

    def _is_stale(self, position: Position) -> bool:
        """Check if position exceeds staleness threshold."""
        age_seconds = (datetime.now() - position.last_updated).total_seconds()
        return age_seconds > self.stale_threshold_seconds
