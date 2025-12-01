"""Repository for position snapshots and changes."""

from __future__ import annotations
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from ..duckdb_adapter import DuckDBAdapter
from src.models.position_risk import PositionRisk
from src.models.position import Position

logger = logging.getLogger(__name__)


class PositionRepository:
    """Repository for position-related persistence operations."""

    def __init__(self, db: DuckDBAdapter):
        self.db = db

    def save_snapshots(self, position_risks: List[PositionRisk], snapshot_time: datetime) -> int:
        """
        Save position snapshots in batch.

        Args:
            position_risks: List of PositionRisk objects
            snapshot_time: Timestamp for this snapshot

        Returns:
            Number of records inserted
        """
        if not position_risks:
            return 0

        records = []
        for pr in position_risks:
            pos = pr.position
            records.append((
                snapshot_time,
                pr.symbol,
                pos.underlying,
                pos.asset_type.value if pos.asset_type else None,
                pos.quantity,
                pos.avg_price,
                pr.mark_price,
                pr.unrealized_pnl,
                pr.daily_pnl,
                pr.delta,
                pr.gamma,
                pr.vega,
                pr.theta,
                pr.iv,
                pos.expiry,
                pos.strike,
                pos.right,  # "C" or "P" for options, None for stocks
                pos.source.value if pos.source else None,
                pr.has_market_data,
            ))

        self.db.executemany("""
            INSERT INTO position_snapshots (
                id, snapshot_time, symbol, underlying, asset_type, quantity,
                avg_price, mark_price, unrealized_pnl, daily_pnl,
                delta, gamma, vega, theta, iv,
                expiry, strike, option_type, source, has_market_data
            ) VALUES (
                nextval('position_snapshots_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, records)

        logger.debug(f"Saved {len(records)} position snapshots")
        return len(records)

    def save_change(
        self,
        change_type: str,
        symbol: str,
        underlying: str,
        quantity_before: Optional[float],
        quantity_after: Optional[float],
        avg_price_before: Optional[float] = None,
        avg_price_after: Optional[float] = None,
        source: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Save a position change record."""
        self.db.execute("""
            INSERT INTO position_changes (
                id, change_time, change_type, symbol, underlying,
                quantity_before, quantity_after, avg_price_before, avg_price_after,
                source, notes
            ) VALUES (
                nextval('position_changes_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            datetime.now(),
            change_type,
            symbol,
            underlying,
            quantity_before,
            quantity_after,
            avg_price_before,
            avg_price_after,
            source,
            notes,
        ))

    def save_changes_batch(self, changes: List[Dict[str, Any]]) -> int:
        """Save multiple position changes in batch."""
        if not changes:
            return 0

        records = []
        now = datetime.now()
        for c in changes:
            records.append((
                now,
                c["change_type"],
                c["symbol"],
                c["underlying"],
                c.get("quantity_before"),
                c.get("quantity_after"),
                c.get("avg_price_before"),
                c.get("avg_price_after"),
                c.get("source"),
                c.get("notes"),
            ))

        self.db.executemany("""
            INSERT INTO position_changes (
                id, change_time, change_type, symbol, underlying,
                quantity_before, quantity_after, avg_price_before, avg_price_after,
                source, notes
            ) VALUES (
                nextval('position_changes_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, records)

        return len(records)

    def get_latest_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot for a symbol."""
        return self.db.fetch_one("""
            SELECT * FROM position_snapshots
            WHERE symbol = ?
            ORDER BY snapshot_time DESC
            LIMIT 1
        """, (symbol,))

    def get_snapshots_range(
        self,
        start_time: datetime,
        end_time: datetime,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get position snapshots within a time range."""
        if symbol:
            query = """
                SELECT * FROM position_snapshots
                WHERE snapshot_time BETWEEN ? AND ? AND symbol = ?
                ORDER BY snapshot_time
            """
            rows = self.db.fetch_all(query, (start_time, end_time, symbol))
        else:
            query = """
                SELECT * FROM position_snapshots
                WHERE snapshot_time BETWEEN ? AND ?
                ORDER BY snapshot_time
            """
            rows = self.db.fetch_all(query, (start_time, end_time))

        return rows if rows else []

    def get_changes_today(self) -> List[Dict[str, Any]]:
        """Get all position changes for today."""
        rows = self.db.fetch_all("""
            SELECT * FROM position_changes
            WHERE change_time >= CURRENT_DATE
            ORDER BY change_time DESC
        """)
        return rows if rows else []

    def get_changes_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get position changes within a time range."""
        rows = self.db.fetch_all("""
            SELECT * FROM position_changes
            WHERE change_time BETWEEN ? AND ?
            ORDER BY change_time DESC
        """, (start_time, end_time))
        return rows if rows else []

    def get_changes_recent_days(self, days: int = 5, exclude_today: bool = True) -> List[Dict[str, Any]]:
        """
        Get position changes for the last N days.

        Args:
            days: Number of days to look back (default 5)
            exclude_today: If True, exclude today's changes (default True)

        Returns:
            List of position change records ordered by time descending
        """
        # DuckDB doesn't support parameterized intervals, so we calculate the date in Python
        from datetime import timedelta
        start_date = date.today() - timedelta(days=days)

        if exclude_today:
            # Get changes from N days ago up to (but not including) today
            rows = self.db.fetch_all("""
                SELECT * FROM position_changes
                WHERE change_time >= ?
                  AND change_time < CURRENT_DATE
                ORDER BY change_time DESC
            """, (start_date,))
        else:
            # Get changes from N days ago up to now
            rows = self.db.fetch_all("""
                SELECT * FROM position_changes
                WHERE change_time >= ?
                ORDER BY change_time DESC
            """, (start_date,))
        return rows if rows else []

    def cleanup_old_snapshots(self, days_to_keep: int = 90) -> int:
        """Delete position snapshots older than specified days."""
        result = self.db.execute("""
            DELETE FROM position_snapshots
            WHERE snapshot_time < CURRENT_DATE - INTERVAL ? DAY
        """, (days_to_keep,))
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Deleted {deleted} old position snapshots (>{days_to_keep} days)")
        return deleted
