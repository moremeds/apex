"""Repository for portfolio snapshots and daily P&L."""

from __future__ import annotations
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from ..duckdb_adapter import DuckDBAdapter
from src.models.risk_snapshot import RiskSnapshot

logger = logging.getLogger(__name__)


class PortfolioRepository:
    """Repository for portfolio-level persistence operations."""

    def __init__(self, db: DuckDBAdapter):
        self.db = db

    def save_snapshot(self, snapshot: RiskSnapshot) -> None:
        """Save a portfolio snapshot."""
        self.db.execute("""
            INSERT INTO portfolio_snapshots (
                id, snapshot_time, total_unrealized_pnl, total_daily_pnl,
                portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta,
                total_gross_notional, total_net_notional, margin_utilization,
                total_net_liquidation, total_positions, positions_with_missing_md,
                concentration_pct, max_underlying_symbol
            ) VALUES (
                nextval('portfolio_snapshots_id_seq'),
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            snapshot.timestamp,
            snapshot.total_unrealized_pnl,
            snapshot.total_daily_pnl,
            snapshot.portfolio_delta,
            snapshot.portfolio_gamma,
            snapshot.portfolio_vega,
            snapshot.portfolio_theta,
            snapshot.total_gross_notional,
            snapshot.total_net_notional,
            snapshot.margin_utilization,
            getattr(snapshot, 'total_net_liquidation', None),
            snapshot.total_positions,
            getattr(snapshot, 'positions_with_missing_md', 0),
            snapshot.concentration_pct,
            snapshot.max_underlying_symbol,
        ))

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent portfolio snapshot."""
        result = self.db.fetch_one("""
            SELECT * FROM portfolio_snapshots
            ORDER BY snapshot_time DESC
            LIMIT 1
        """)
        return dict(result) if result else None

    def get_snapshots_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get portfolio snapshots within a time range."""
        rows = self.db.fetch_all("""
            SELECT * FROM portfolio_snapshots
            WHERE snapshot_time BETWEEN ? AND ?
            ORDER BY snapshot_time
        """, (start_time, end_time))
        return [dict(row) for row in rows] if rows else []

    def get_snapshots_df(self, start_time: datetime, end_time: datetime):
        """Get portfolio snapshots as DataFrame for analytics."""
        return self.db.fetch_df("""
            SELECT * FROM portfolio_snapshots
            WHERE snapshot_time BETWEEN ? AND ?
            ORDER BY snapshot_time
        """, (start_time, end_time))

    # Daily P&L operations

    def save_daily_pnl(
        self,
        trade_date: date,
        unrealized_pnl: Optional[float] = None,
        daily_pnl: Optional[float] = None,
        is_open: bool = True,
        total_positions: Optional[int] = None,
    ) -> None:
        """Save or update daily P&L record."""
        existing = self.db.fetch_one(
            "SELECT id FROM daily_pnl WHERE trade_date = ?",
            (trade_date,)
        )

        if existing:
            if is_open:
                self.db.execute("""
                    UPDATE daily_pnl
                    SET unrealized_pnl_open = ?, total_positions = ?
                    WHERE trade_date = ?
                """, (unrealized_pnl, total_positions, trade_date))
            else:
                self.db.execute("""
                    UPDATE daily_pnl
                    SET unrealized_pnl_close = ?, daily_pnl = ?, total_positions = ?
                    WHERE trade_date = ?
                """, (unrealized_pnl, daily_pnl, total_positions, trade_date))
        else:
            if is_open:
                self.db.execute("""
                    INSERT INTO daily_pnl (id, trade_date, unrealized_pnl_open, total_positions)
                    VALUES (nextval('daily_pnl_id_seq'), ?, ?, ?)
                """, (trade_date, unrealized_pnl, total_positions))
            else:
                self.db.execute("""
                    INSERT INTO daily_pnl (id, trade_date, unrealized_pnl_close, daily_pnl, total_positions)
                    VALUES (nextval('daily_pnl_id_seq'), ?, ?, ?, ?)
                """, (trade_date, unrealized_pnl, daily_pnl, total_positions))

    def update_daily_drawdown(self, trade_date: date, current_pnl: float) -> None:
        """Update max drawdown and peak P&L for the day."""
        existing = self.db.fetch_one(
            "SELECT peak_pnl, max_drawdown FROM daily_pnl WHERE trade_date = ?",
            (trade_date,)
        )

        if existing:
            peak_pnl = existing[0] or current_pnl
            max_drawdown = existing[1] or 0

            if current_pnl > peak_pnl:
                peak_pnl = current_pnl

            drawdown = peak_pnl - current_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            self.db.execute("""
                UPDATE daily_pnl
                SET peak_pnl = ?, max_drawdown = ?
                WHERE trade_date = ?
            """, (peak_pnl, max_drawdown, trade_date))

    def increment_positions_opened(self, trade_date: date, count: int = 1) -> None:
        """Increment the count of positions opened today."""
        self.db.execute("""
            UPDATE daily_pnl
            SET positions_opened = COALESCE(positions_opened, 0) + ?
            WHERE trade_date = ?
        """, (count, trade_date))

    def increment_positions_closed(self, trade_date: date, count: int = 1) -> None:
        """Increment the count of positions closed today."""
        self.db.execute("""
            UPDATE daily_pnl
            SET positions_closed = COALESCE(positions_closed, 0) + ?
            WHERE trade_date = ?
        """, (count, trade_date))

    def get_daily_pnl(self, trade_date: date) -> Optional[Dict[str, Any]]:
        """Get daily P&L record for a specific date."""
        result = self.db.fetch_one(
            "SELECT * FROM daily_pnl WHERE trade_date = ?",
            (trade_date,)
        )
        return dict(result) if result else None

    def get_daily_pnl_range(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get daily P&L records within a date range."""
        rows = self.db.fetch_all("""
            SELECT * FROM daily_pnl
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, (start_date, end_date))
        return [dict(row) for row in rows] if rows else []

    def get_daily_pnl_df(self, start_date: date, end_date: date):
        """Get daily P&L as DataFrame for analytics."""
        return self.db.fetch_df("""
            SELECT * FROM daily_pnl
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, (start_date, end_date))

    def cleanup_old_snapshots(self, days_to_keep: int = 365) -> int:
        """Delete portfolio snapshots older than specified days."""
        result = self.db.execute("""
            DELETE FROM portfolio_snapshots
            WHERE snapshot_time < CURRENT_DATE - INTERVAL ? DAY
        """, (days_to_keep,))
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Deleted {deleted} old portfolio snapshots (>{days_to_keep} days)")
        return deleted
