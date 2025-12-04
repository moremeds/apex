"""Repository for order and trade history persistence."""

from __future__ import annotations
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
import logging
import json

from ..duckdb_adapter import DuckDBAdapter
from src.models.order import Order, Execution, Trade, OrderSource, OrderStatus, OrderSide, OrderType

logger = logging.getLogger(__name__)


class OrderRepository:
    """
    Repository for order and trade persistence operations.

    Handles upsert logic for orders (by order_id + source + account_id)
    and trades (by trade_id + source + account_id).
    """

    def __init__(self, db: DuckDBAdapter):
        self.db = db

    def upsert_orders(self, orders: List[Order]) -> Dict[str, int]:
        """
        Upsert orders into the database (insert or update).

        Uses composite key (source, order_id, account_id) for deduplication.

        Args:
            orders: List of Order objects to upsert

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        if not orders:
            return {"inserted": 0, "updated": 0}

        inserted = 0
        updated = 0

        for order in orders:
            # Check if order exists
            existing = self.db.fetch_one("""
                SELECT id FROM orders
                WHERE source = ? AND order_id = ? AND account_id = ?
            """, (order.source.value, order.order_id, order.account_id))

            if existing:
                # Update existing order
                self.db.execute("""
                    UPDATE orders SET
                        symbol = ?,
                        underlying = ?,
                        asset_type = ?,
                        side = ?,
                        order_type = ?,
                        quantity = ?,
                        limit_price = ?,
                        stop_price = ?,
                        status = ?,
                        filled_quantity = ?,
                        avg_fill_price = ?,
                        commission = ?,
                        created_time = ?,
                        submitted_time = ?,
                        filled_time = ?,
                        updated_time = ?,
                        expiry = ?,
                        strike = ?,
                        option_right = ?,
                        broker_order_id = ?,
                        exchange = ?,
                        time_in_force = ?,
                        notes = ?
                    WHERE source = ? AND order_id = ? AND account_id = ?
                """, (
                    order.symbol,
                    order.underlying,
                    order.asset_type,
                    order.side.value,
                    order.order_type.value,
                    order.quantity,
                    order.limit_price,
                    order.stop_price,
                    order.status.value,
                    order.filled_quantity,
                    order.avg_fill_price,
                    order.commission,
                    order.created_time,
                    order.submitted_time,
                    order.filled_time,
                    order.updated_time,
                    order.expiry,
                    order.strike,
                    order.right,
                    order.broker_order_id,
                    order.exchange,
                    order.time_in_force,
                    order.notes,
                    order.source.value,
                    order.order_id,
                    order.account_id,
                ))
                updated += 1
            else:
                # Insert new order
                self.db.execute("""
                    INSERT INTO orders (
                        id, source, order_id, account_id, symbol, underlying, asset_type,
                        side, order_type, quantity, limit_price, stop_price, status,
                        filled_quantity, avg_fill_price, commission,
                        created_time, submitted_time, filled_time, updated_time,
                        expiry, strike, option_right, broker_order_id, exchange,
                        time_in_force, notes
                    ) VALUES (
                        nextval('orders_id_seq'),
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    order.source.value,
                    order.order_id,
                    order.account_id,
                    order.symbol,
                    order.underlying,
                    order.asset_type,
                    order.side.value,
                    order.order_type.value,
                    order.quantity,
                    order.limit_price,
                    order.stop_price,
                    order.status.value,
                    order.filled_quantity,
                    order.avg_fill_price,
                    order.commission,
                    order.created_time,
                    order.submitted_time,
                    order.filled_time,
                    order.updated_time,
                    order.expiry,
                    order.strike,
                    order.right,
                    order.broker_order_id,
                    order.exchange,
                    order.time_in_force,
                    order.notes,
                ))
                inserted += 1

        logger.info(f"Upserted orders: {inserted} inserted, {updated} updated")
        return {"inserted": inserted, "updated": updated}

    def upsert_trades(self, trades: List[Trade]) -> Dict[str, int]:
        """
        Upsert trades into the database (insert or update).

        Uses composite key (source, trade_id, account_id) for deduplication.

        Args:
            trades: List of Trade objects to upsert

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        if not trades:
            return {"inserted": 0, "updated": 0}

        inserted = 0
        updated = 0

        for trade in trades:
            # Check if trade exists
            existing = self.db.fetch_one("""
                SELECT id FROM trades
                WHERE source = ? AND trade_id = ? AND account_id = ?
            """, (trade.source.value, trade.trade_id, trade.account_id))

            if existing:
                # Update existing trade (rare, but handle it)
                self.db.execute("""
                    UPDATE trades SET
                        order_id = ?,
                        symbol = ?,
                        underlying = ?,
                        asset_type = ?,
                        side = ?,
                        quantity = ?,
                        price = ?,
                        commission = ?,
                        trade_time = ?,
                        expiry = ?,
                        strike = ?,
                        option_right = ?,
                        exchange = ?,
                        liquidity = ?
                    WHERE source = ? AND trade_id = ? AND account_id = ?
                """, (
                    trade.order_id,
                    trade.symbol,
                    trade.underlying,
                    trade.asset_type,
                    trade.side.value,
                    trade.quantity,
                    trade.price,
                    trade.commission,
                    trade.trade_time,
                    trade.expiry,
                    trade.strike,
                    trade.right,
                    trade.exchange,
                    trade.liquidity,
                    trade.source.value,
                    trade.trade_id,
                    trade.account_id,
                ))
                updated += 1
            else:
                # Insert new trade
                self.db.execute("""
                    INSERT INTO trades (
                        id, source, trade_id, order_id, account_id, symbol, underlying,
                        asset_type, side, quantity, price, commission, trade_time,
                        expiry, strike, option_right, exchange, liquidity
                    ) VALUES (
                        nextval('trades_id_seq'),
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    trade.source.value,
                    trade.trade_id,
                    trade.order_id,
                    trade.account_id,
                    trade.symbol,
                    trade.underlying,
                    trade.asset_type,
                    trade.side.value,
                    trade.quantity,
                    trade.price,
                    trade.commission,
                    trade.trade_time,
                    trade.expiry,
                    trade.strike,
                    trade.right,
                    trade.exchange,
                    trade.liquidity,
                ))
                inserted += 1

        logger.info(f"Upserted trades: {inserted} inserted, {updated} updated")
        return {"inserted": inserted, "updated": updated}

    def get_orders(
        self,
        source: Optional[OrderSource] = None,
        account_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query orders with optional filters.

        Args:
            source: Filter by source (IB/FUTU)
            account_id: Filter by account
            status: Filter by order status
            symbol: Filter by symbol
            start_time: Filter orders created after this time
            end_time: Filter orders created before this time
            limit: Maximum records to return

        Returns:
            List of order records as dicts
        """
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)
        if account_id:
            conditions.append("account_id = ?")
            params.append(account_id)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("created_time >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("created_time <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = self.db.fetch_all(f"""
            SELECT * FROM orders
            WHERE {where_clause}
            ORDER BY created_time DESC
            LIMIT ?
        """, tuple(params))

        return rows if rows else []

    def get_trades(
        self,
        source: Optional[OrderSource] = None,
        account_id: Optional[str] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query trades with optional filters.

        Args:
            source: Filter by source (IB/FUTU)
            account_id: Filter by account
            symbol: Filter by symbol
            start_time: Filter trades after this time
            end_time: Filter trades before this time
            limit: Maximum records to return

        Returns:
            List of trade records as dicts
        """
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source.value)
        if account_id:
            conditions.append("account_id = ?")
            params.append(account_id)
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time:
            conditions.append("trade_time >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("trade_time <= ?")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = self.db.fetch_all(f"""
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY trade_time DESC
            LIMIT ?
        """, tuple(params))

        return rows if rows else []

    def get_order_by_id(
        self,
        source: OrderSource,
        order_id: str,
        account_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific order by its composite key."""
        return self.db.fetch_one("""
            SELECT * FROM orders
            WHERE source = ? AND order_id = ? AND account_id = ?
        """, (source.value, order_id, account_id))

    def get_trade_by_id(
        self,
        source: OrderSource,
        trade_id: str,
        account_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific trade by its composite key."""
        return self.db.fetch_one("""
            SELECT * FROM trades
            WHERE source = ? AND trade_id = ? AND account_id = ?
        """, (source.value, trade_id, account_id))

    def get_trades_by_order(
        self,
        source: OrderSource,
        order_id: str,
        account_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all trades for a specific order."""
        rows = self.db.fetch_all("""
            SELECT * FROM trades
            WHERE source = ? AND order_id = ? AND account_id = ?
            ORDER BY trade_time
        """, (source.value, order_id, account_id))
        return rows if rows else []

    def get_open_orders(self, source: Optional[OrderSource] = None) -> List[Dict[str, Any]]:
        """Get all open/pending orders."""
        open_statuses = (
            OrderStatus.PENDING.value,
            OrderStatus.SUBMITTED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        )

        if source:
            rows = self.db.fetch_all("""
                SELECT * FROM orders
                WHERE status IN (?, ?, ?) AND source = ?
                ORDER BY created_time DESC
            """, (*open_statuses, source.value))
        else:
            rows = self.db.fetch_all("""
                SELECT * FROM orders
                WHERE status IN (?, ?, ?)
                ORDER BY created_time DESC
            """, open_statuses)

        return rows if rows else []

    def get_daily_trade_summary(
        self,
        trade_date: Optional[date] = None,
        source: Optional[OrderSource] = None,
    ) -> Dict[str, Any]:
        """
        Get daily trade summary (for reconciliation).

        Args:
            trade_date: Date to summarize (default: today)
            source: Filter by source

        Returns:
            Dict with summary statistics
        """
        trade_date = trade_date or date.today()
        start_time = datetime.combine(trade_date, datetime.min.time())
        end_time = datetime.combine(trade_date, datetime.max.time())

        if source:
            result = self.db.fetch_one("""
                SELECT
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) as total_bought,
                    SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) as total_sold,
                    SUM(quantity * price) as total_notional,
                    SUM(commission) as total_commission,
                    COUNT(DISTINCT symbol) as unique_symbols
                FROM trades
                WHERE trade_time BETWEEN ? AND ? AND source = ?
            """, (start_time, end_time, source.value))
        else:
            result = self.db.fetch_one("""
                SELECT
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) as total_bought,
                    SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) as total_sold,
                    SUM(quantity * price) as total_notional,
                    SUM(commission) as total_commission,
                    COUNT(DISTINCT symbol) as unique_symbols
                FROM trades
                WHERE trade_time BETWEEN ? AND ?
            """, (start_time, end_time))

        return result if result else {
            "trade_count": 0,
            "total_bought": 0,
            "total_sold": 0,
            "total_notional": 0,
            "total_commission": 0,
            "unique_symbols": 0,
        }

    def cleanup_old_orders(self, days_to_keep: int = 365) -> int:
        """Delete orders older than specified days."""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        result = self.db.execute("""
            DELETE FROM orders
            WHERE created_time < ? AND status IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED')
        """, (cutoff,))
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Deleted {deleted} old orders (>{days_to_keep} days)")
        return deleted

    def cleanup_old_trades(self, days_to_keep: int = 365) -> int:
        """Delete trades older than specified days."""
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        result = self.db.execute("""
            DELETE FROM trades
            WHERE trade_time < ?
        """, (cutoff,))
        deleted = result.fetchone()[0] if result else 0
        logger.info(f"Deleted {deleted} old trades (>{days_to_keep} days)")
        return deleted
