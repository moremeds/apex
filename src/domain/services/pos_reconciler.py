"""
Position Reconciler - Detect MISSING, DRIFT, STALE positions.

Compares positions from multiple sources (IBKR vs manual YAML vs cached)
and identifies discrepancies.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, TYPE_CHECKING
from datetime import datetime
from dataclasses import replace
import logging
from ...models.position import Position, PositionSource, AssetType
from ...models.reconciliation import ReconciliationIssue, IssueType

if TYPE_CHECKING:
    from ...models.account import AccountInfo

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
        futu_positions: Optional[List[Position]] = None,
    ) -> List[ReconciliationIssue]:
        """
        Reconcile positions from multiple sources.

        Compares all broker positions (IB, Futu) against manual positions
        and cached state to detect discrepancies.

        Args:
            ib_positions: Positions from Interactive Brokers.
            manual_positions: Positions from manual YAML file.
            cached_positions: Positions from previous snapshot.
            futu_positions: Positions from Futu (optional).

        Returns:
            List of ReconciliationIssue objects.
        """
        issues: List[ReconciliationIssue] = []

        # Build position maps by key
        ib_map = {p.key(): p for p in ib_positions}
        manual_map = {p.key(): p for p in manual_positions}
        cached_map = {p.key(): p for p in cached_positions}
        futu_map = {p.key(): p for p in (futu_positions or [])}

        # All unique keys across sources
        all_keys: Set[tuple] = (
            set(ib_map.keys()) | set(manual_map.keys()) |
            set(cached_map.keys()) | set(futu_map.keys())
        )

        for key in all_keys:
            ib_pos = ib_map.get(key)
            manual_pos = manual_map.get(key)
            cached_pos = cached_map.get(key)
            futu_pos = futu_map.get(key)

            # Detect MISSING: position in only one broker source (IB xor Futu)
            # Note: manual-only positions are intentional overrides, not issues
            broker_sources = sum([ib_pos is not None, futu_pos is not None])
            if broker_sources == 1 and not manual_pos:
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

            # Detect DRIFT (quantity mismatch between IB and manual)
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

            # Detect DRIFT (quantity mismatch between Futu and manual)
            if futu_pos and manual_pos:
                if futu_pos.quantity != manual_pos.quantity:
                    issue = ReconciliationIssue(
                        issue_type=IssueType.DRIFT,
                        symbol=key[0],
                        underlying=key[1],
                        ib_position=futu_pos,  # Reuse ib_position field for Futu
                        manual_position=manual_pos,
                        cached_position=cached_pos,
                        quantity_delta=futu_pos.quantity - manual_pos.quantity,
                        severity="CRITICAL",
                    )
                    issues.append(issue)

            # Detect STALE
            for pos in [ib_pos, futu_pos, manual_pos]:
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

        # Track which positions exist in which sources (for multi-broker visibility)
        sources_by_key: Dict[tuple, set] = {}

        # Process broker positions (primary source for quantity/metadata)
        for position in broker_positions:
            key = position.key()

            # Track source
            if key not in sources_by_key:
                sources_by_key[key] = set()
            sources_by_key[key].add(position.source)

            if key not in merged:
                # Check if manual has avg_price override for this position
                manual_pos = manual_map.get(key)
                if manual_pos is not None:
                    # Clone position with manual overrides (avoid mutating original)
                    overrides = {"avg_price": manual_pos.avg_price}
                    if manual_pos.strategy_tag:
                        overrides["strategy_tag"] = manual_pos.strategy_tag
                    merged[key] = replace(position, **overrides)
                else:
                    # Clone to avoid side effects on original position
                    merged[key] = replace(position)
            else:
                # Aggregate quantities (same position from multiple accounts)
                existing = merged[key]
                # Clone with updated quantity to avoid mutating existing
                merged[key] = replace(existing, quantity=existing.quantity + position.quantity)

        # Process manual positions that don't exist in any broker
        for position in manual_positions:
            key = position.key()

            if key not in merged:
                # Position only in manual - use as-is
                merged[key] = position

        # Set all_sources field on each merged position (using replace to avoid mutation)
        for key in list(merged.keys()):
            pos = merged[key]
            if key in sources_by_key:
                merged[key] = replace(pos, all_sources=list(sources_by_key[key]))
            else:
                # Position from manual only
                merged[key] = replace(pos, all_sources=[pos.source])

        # Log merged position sources for debugging
        source_counts = {}
        multi_source_count = 0
        for pos in merged.values():
            src = pos.source.value if pos.source else "None"
            source_counts[src] = source_counts.get(src, 0) + 1
            if len(pos.all_sources) > 1:
                multi_source_count += 1
        logger.info(f"Merged position sources: {source_counts}, multi-source positions: {multi_source_count}")

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

    def aggregate_account_info(
        self,
        ib_account: Optional["AccountInfo"],
        futu_account: Optional["AccountInfo"],
    ) -> "AccountInfo":
        """
        Aggregate account info from multiple brokers.

        Single source of truth for account aggregation logic.

        Args:
            ib_account: AccountInfo from IB (may be None).
            futu_account: AccountInfo from Futu (may be None).

        Returns:
            Aggregated AccountInfo with combined values.
        """
        from ...models.account import AccountInfo

        # Start with zero values
        aggregated = AccountInfo(
            net_liquidation=0.0,
            total_cash=0.0,
            buying_power=0.0,
            margin_used=0.0,
            margin_available=0.0,
            maintenance_margin=0.0,
            init_margin_req=0.0,
            excess_liquidity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            account_id="AGGREGATED",
        )

        # Add IB account values
        if ib_account:
            aggregated.net_liquidation += ib_account.net_liquidation
            aggregated.total_cash += ib_account.total_cash
            aggregated.buying_power += ib_account.buying_power
            aggregated.margin_used += ib_account.margin_used
            aggregated.margin_available += ib_account.margin_available
            aggregated.maintenance_margin += ib_account.maintenance_margin
            aggregated.init_margin_req += ib_account.init_margin_req
            aggregated.excess_liquidity += ib_account.excess_liquidity
            aggregated.realized_pnl += ib_account.realized_pnl
            aggregated.unrealized_pnl += ib_account.unrealized_pnl

        # Add Futu account values
        if futu_account:
            aggregated.net_liquidation += futu_account.net_liquidation
            aggregated.total_cash += futu_account.total_cash
            aggregated.buying_power += futu_account.buying_power
            aggregated.margin_used += futu_account.margin_used
            aggregated.margin_available += futu_account.margin_available
            aggregated.maintenance_margin += futu_account.maintenance_margin
            aggregated.init_margin_req += futu_account.init_margin_req
            aggregated.excess_liquidity += futu_account.excess_liquidity
            aggregated.realized_pnl += futu_account.realized_pnl
            aggregated.unrealized_pnl += futu_account.unrealized_pnl

        return aggregated
