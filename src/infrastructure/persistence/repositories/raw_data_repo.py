"""Repository for raw API payload persistence (audit trail)."""

from __future__ import annotations
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging
import json

from ..duckdb_adapter import DuckDBAdapter

logger = logging.getLogger(__name__)


class RawDataRepository:
    """
    Repository for raw API payload persistence.

    Stores original JSON payloads from broker APIs for:
    - Audit trail and compliance
    - Replay and debugging
    - Future re-processing with updated normalizers

    All operations are idempotent using ON CONFLICT DO UPDATE.
    """

    def __init__(self, db: DuckDBAdapter):
        self.db = db

    # =========================================================================
    # Futu Raw Data Operations
    # =========================================================================

    def persist_futu_order_raw(
        self,
        acc_id: int,
        order_id: str,
        payload: dict,
        create_time_raw: Optional[str] = None,
        update_time_raw: Optional[str] = None,
    ) -> bool:
        """
        Persist raw Futu order payload.

        Args:
            acc_id: Futu account ID
            order_id: Order ID from Futu
            payload: Full API response dict
            create_time_raw: Original create time string from API
            update_time_raw: Original update time string from API

        Returns:
            True if inserted, False if updated existing
        """
        # Check if exists
        existing = self.db.fetch_one(
            "SELECT 1 FROM orders_raw_futu WHERE acc_id = ? AND order_id = ?",
            (acc_id, order_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        if existing:
            self.db.execute("""
                UPDATE orders_raw_futu SET
                    payload = ?,
                    create_time_raw_str = ?,
                    update_time_raw_str = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE acc_id = ? AND order_id = ?
            """, (payload_json, create_time_raw, update_time_raw, acc_id, order_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO orders_raw_futu (
                    acc_id, order_id, payload, create_time_raw_str, update_time_raw_str
                ) VALUES (?, ?, ?, ?, ?)
            """, (acc_id, order_id, payload_json, create_time_raw, update_time_raw))
            return True

    def persist_futu_deal_raw(
        self,
        acc_id: int,
        deal_id: str,
        order_id: Optional[str],
        payload: dict,
        trade_time_raw: Optional[str] = None,
    ) -> bool:
        """
        Persist raw Futu deal (execution) payload.

        Args:
            acc_id: Futu account ID
            deal_id: Deal ID from Futu
            order_id: Parent order ID
            payload: Full API response dict
            trade_time_raw: Original trade time string from API

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one(
            "SELECT 1 FROM trades_raw_futu WHERE acc_id = ? AND deal_id = ?",
            (acc_id, deal_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        if existing:
            self.db.execute("""
                UPDATE trades_raw_futu SET
                    order_id = ?,
                    payload = ?,
                    trade_time_raw_str = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE acc_id = ? AND deal_id = ?
            """, (order_id, payload_json, trade_time_raw, acc_id, deal_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO trades_raw_futu (
                    acc_id, deal_id, order_id, payload, trade_time_raw_str
                ) VALUES (?, ?, ?, ?, ?)
            """, (acc_id, deal_id, order_id, payload_json, trade_time_raw))
            return True

    def persist_futu_fee_raw(
        self,
        acc_id: int,
        order_id: str,
        fee_amount: float,
        fee_details: Optional[dict],
        payload: dict,
    ) -> bool:
        """
        Persist raw Futu fee payload.

        Args:
            acc_id: Futu account ID
            order_id: Order ID
            fee_amount: Total fee amount
            fee_details: Breakdown by fee type
            payload: Full API response dict

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one(
            "SELECT 1 FROM fees_raw_futu WHERE acc_id = ? AND order_id = ?",
            (acc_id, order_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)
        fee_details_json = json.dumps(fee_details, ensure_ascii=False, default=str) if fee_details else None

        if existing:
            self.db.execute("""
                UPDATE fees_raw_futu SET
                    fee_amount = ?,
                    fee_details = ?,
                    payload = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE acc_id = ? AND order_id = ?
            """, (fee_amount, fee_details_json, payload_json, acc_id, order_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO fees_raw_futu (
                    acc_id, order_id, fee_amount, fee_details, payload
                ) VALUES (?, ?, ?, ?, ?)
            """, (acc_id, order_id, fee_amount, fee_details_json, payload_json))
            return True

    def batch_persist_futu_orders_raw(
        self,
        acc_id: int,
        orders: List[dict],
    ) -> Dict[str, int]:
        """
        Batch persist raw Futu orders.

        Args:
            acc_id: Futu account ID
            orders: List of order dicts from API

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        inserted = 0
        updated = 0

        for order in orders:
            order_id = str(order.get('order_id', ''))
            if not order_id:
                continue

            is_new = self.persist_futu_order_raw(
                acc_id=acc_id,
                order_id=order_id,
                payload=order,
                create_time_raw=order.get('create_time'),
                update_time_raw=order.get('updated_time') or order.get('update_time'),
            )
            if is_new:
                inserted += 1
            else:
                updated += 1

        logger.info(f"Batch persisted Futu orders: {inserted} new, {updated} updated")
        return {"inserted": inserted, "updated": updated}

    def batch_persist_futu_deals_raw(
        self,
        acc_id: int,
        deals: List[dict],
    ) -> Dict[str, int]:
        """
        Batch persist raw Futu deals (executions).

        Args:
            acc_id: Futu account ID
            deals: List of deal dicts from API

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        inserted = 0
        updated = 0

        for deal in deals:
            deal_id = str(deal.get('deal_id', ''))
            if not deal_id:
                continue

            is_new = self.persist_futu_deal_raw(
                acc_id=acc_id,
                deal_id=deal_id,
                order_id=str(deal.get('order_id', '')),
                payload=deal,
                trade_time_raw=deal.get('create_time'),
            )
            if is_new:
                inserted += 1
            else:
                updated += 1

        logger.info(f"Batch persisted Futu deals: {inserted} new, {updated} updated")
        return {"inserted": inserted, "updated": updated}

    def batch_persist_futu_fees_raw(
        self,
        acc_id: int,
        fees: List[dict],
    ) -> Dict[str, int]:
        """
        Batch persist raw Futu fee records.

        Args:
            acc_id: Futu account ID
            fees: List of fee dicts from API

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        inserted = 0
        updated = 0

        for fee in fees:
            order_id = str(fee.get('order_id', ''))
            if not order_id:
                continue

            is_new = self.persist_futu_fee_raw(
                acc_id=acc_id,
                order_id=order_id,
                fee_amount=float(fee.get('fee_amount', 0) or 0),
                fee_details=fee.get('fee_list'),
                payload=fee,
            )
            if is_new:
                inserted += 1
            else:
                updated += 1

        logger.info(f"Batch persisted Futu fees: {inserted} new, {updated} updated")
        return {"inserted": inserted, "updated": updated}

    # =========================================================================
    # IB Raw Data Operations
    # =========================================================================

    def persist_ib_order_raw(
        self,
        account: str,
        perm_id: int,
        payload: dict,
        client_order_id: Optional[str] = None,
        order_ref: Optional[str] = None,
        create_time_raw: Optional[str] = None,
        update_time_raw: Optional[str] = None,
        source: str = 'API',
    ) -> bool:
        """
        Persist raw IB order payload.

        Args:
            account: IB account ID
            perm_id: Permanent order ID from IB
            payload: Full API response dict
            client_order_id: Client-assigned order ID
            order_ref: Order reference string
            create_time_raw: Original create time string
            update_time_raw: Original update time string
            source: Data source ('API' or 'FLEX')

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one(
            "SELECT 1 FROM orders_raw_ib WHERE account = ? AND perm_id = ?",
            (account, perm_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        if existing:
            self.db.execute("""
                UPDATE orders_raw_ib SET
                    client_order_id = ?,
                    order_ref = ?,
                    payload = ?,
                    create_time_raw_str = ?,
                    update_time_raw_str = ?,
                    source = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE account = ? AND perm_id = ?
            """, (client_order_id, order_ref, payload_json, create_time_raw, update_time_raw, source, account, perm_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO orders_raw_ib (
                    account, perm_id, client_order_id, order_ref, payload,
                    create_time_raw_str, update_time_raw_str, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (account, perm_id, client_order_id, order_ref, payload_json, create_time_raw, update_time_raw, source))
            return True

    def persist_ib_execution_raw(
        self,
        account: str,
        exec_id: str,
        payload: dict,
        perm_id: Optional[int] = None,
        order_ref: Optional[str] = None,
        trade_time_raw: Optional[str] = None,
        source: str = 'API',
    ) -> bool:
        """
        Persist raw IB execution payload.

        Args:
            account: IB account ID
            exec_id: Execution ID from IB
            payload: Full API response dict
            perm_id: Permanent order ID
            order_ref: Order reference string
            trade_time_raw: Original trade time string
            source: Data source ('API' or 'FLEX')

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one(
            "SELECT 1 FROM trades_raw_ib WHERE account = ? AND exec_id = ?",
            (account, exec_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str)

        if existing:
            self.db.execute("""
                UPDATE trades_raw_ib SET
                    perm_id = ?,
                    order_ref = ?,
                    payload = ?,
                    trade_time_raw_str = ?,
                    source = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE account = ? AND exec_id = ?
            """, (perm_id, order_ref, payload_json, trade_time_raw, source, account, exec_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO trades_raw_ib (
                    account, exec_id, perm_id, order_ref, payload,
                    trade_time_raw_str, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (account, exec_id, perm_id, order_ref, payload_json, trade_time_raw, source))
            return True

    def persist_ib_commission_raw(
        self,
        account: str,
        exec_id: str,
        commission: float,
        currency: str,
        payload: Optional[dict] = None,
        realized_pnl: Optional[float] = None,
        source: str = 'API',
    ) -> bool:
        """
        Persist raw IB commission report.

        Args:
            account: IB account ID
            exec_id: Execution ID
            commission: Commission amount
            currency: Commission currency
            payload: Full API response dict
            realized_pnl: Realized P&L if available
            source: Data source ('API' or 'FLEX')

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one(
            "SELECT 1 FROM fees_raw_ib WHERE account = ? AND exec_id = ?",
            (account, exec_id)
        )

        payload_json = json.dumps(payload, ensure_ascii=False, default=str) if payload else None

        if existing:
            self.db.execute("""
                UPDATE fees_raw_ib SET
                    commission = ?,
                    currency = ?,
                    realized_pnl = ?,
                    payload = ?,
                    source = ?,
                    ingest_ts = CURRENT_TIMESTAMP
                WHERE account = ? AND exec_id = ?
            """, (commission, currency, realized_pnl, payload_json, source, account, exec_id))
            return False
        else:
            self.db.execute("""
                INSERT INTO fees_raw_ib (
                    account, exec_id, commission, currency, realized_pnl, payload, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (account, exec_id, commission, currency, realized_pnl, payload_json, source))
            return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    def get_futu_orders_raw(
        self,
        acc_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get raw Futu orders with optional date filter."""
        conditions = ["acc_id = ?"]
        params: List[Any] = [acc_id]

        if start_date:
            conditions.append("ingest_ts >= ?")
            params.append(datetime.combine(start_date, datetime.min.time()))
        if end_date:
            conditions.append("ingest_ts <= ?")
            params.append(datetime.combine(end_date, datetime.max.time()))

        params.append(limit)
        where_clause = " AND ".join(conditions)

        rows = self.db.fetch_all(f"""
            SELECT * FROM orders_raw_futu
            WHERE {where_clause}
            ORDER BY ingest_ts DESC
            LIMIT ?
        """, tuple(params))

        return rows if rows else []

    def get_ib_orders_raw(
        self,
        account: str,
        source: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get raw IB orders with optional filters."""
        conditions = ["account = ?"]
        params: List[Any] = [account]

        if source:
            conditions.append("source = ?")
            params.append(source)
        if start_date:
            conditions.append("ingest_ts >= ?")
            params.append(datetime.combine(start_date, datetime.min.time()))
        if end_date:
            conditions.append("ingest_ts <= ?")
            params.append(datetime.combine(end_date, datetime.max.time()))

        params.append(limit)
        where_clause = " AND ".join(conditions)

        rows = self.db.fetch_all(f"""
            SELECT * FROM orders_raw_ib
            WHERE {where_clause}
            ORDER BY ingest_ts DESC
            LIMIT ?
        """, tuple(params))

        return rows if rows else []

    def count_futu_orders_raw(self, acc_id: Optional[int] = None) -> int:
        """Count raw Futu orders."""
        if acc_id:
            result = self.db.fetch_one(
                "SELECT COUNT(*) as cnt FROM orders_raw_futu WHERE acc_id = ?",
                (acc_id,)
            )
        else:
            result = self.db.fetch_one("SELECT COUNT(*) as cnt FROM orders_raw_futu")
        return result["cnt"] if result else 0

    def count_futu_deals_raw(self, acc_id: Optional[int] = None) -> int:
        """Count raw Futu deals."""
        if acc_id:
            result = self.db.fetch_one(
                "SELECT COUNT(*) as cnt FROM trades_raw_futu WHERE acc_id = ?",
                (acc_id,)
            )
        else:
            result = self.db.fetch_one("SELECT COUNT(*) as cnt FROM trades_raw_futu")
        return result["cnt"] if result else 0

    def count_ib_orders_raw(self, account: Optional[str] = None) -> int:
        """Count raw IB orders."""
        if account:
            result = self.db.fetch_one(
                "SELECT COUNT(*) as cnt FROM orders_raw_ib WHERE account = ?",
                (account,)
            )
        else:
            result = self.db.fetch_one("SELECT COUNT(*) as cnt FROM orders_raw_ib")
        return result["cnt"] if result else 0

    def get_data_boundaries(self, broker: str) -> Dict[str, Any]:
        """
        Get earliest and latest timestamps for a broker's raw data.

        Args:
            broker: 'FUTU' or 'IB'

        Returns:
            Dict with min/max timestamps for orders and trades
        """
        if broker.upper() == 'FUTU':
            orders_result = self.db.fetch_one("""
                SELECT MIN(ingest_ts) as min_ts, MAX(ingest_ts) as max_ts
                FROM orders_raw_futu
            """)
            trades_result = self.db.fetch_one("""
                SELECT MIN(ingest_ts) as min_ts, MAX(ingest_ts) as max_ts
                FROM trades_raw_futu
            """)
        else:  # IB
            orders_result = self.db.fetch_one("""
                SELECT MIN(ingest_ts) as min_ts, MAX(ingest_ts) as max_ts
                FROM orders_raw_ib
            """)
            trades_result = self.db.fetch_one("""
                SELECT MIN(ingest_ts) as min_ts, MAX(ingest_ts) as max_ts
                FROM trades_raw_ib
            """)

        return {
            "orders_min": orders_result.get("min_ts") if orders_result else None,
            "orders_max": orders_result.get("max_ts") if orders_result else None,
            "trades_min": trades_result.get("min_ts") if trades_result else None,
            "trades_max": trades_result.get("max_ts") if trades_result else None,
        }

    # =========================================================================
    # Strategy Classification Operations
    # =========================================================================

    def upsert_strategy_mapping(
        self,
        broker: str,
        account_id: str,
        order_uid: str,
        strategy_id: str,
        strategy_type: str,
        strategy_name: Optional[str] = None,
        confidence: float = 0.0,
        leg_index: Optional[int] = None,
        legs: Optional[List[dict]] = None,
        classify_version: str = 'v1',
    ) -> bool:
        """
        Upsert strategy classification for an order.

        Args:
            broker: Broker source ('FUTU', 'IB')
            account_id: Account ID
            order_uid: Unified order ID
            strategy_id: Strategy group ID
            strategy_type: Strategy type name
            strategy_name: Human-readable strategy name
            confidence: Classification confidence (0-1)
            leg_index: Leg position in multi-leg strategy
            legs: List of all legs in this strategy
            classify_version: Classifier version

        Returns:
            True if inserted, False if updated existing
        """
        existing = self.db.fetch_one("""
            SELECT 1 FROM order_strategy_map
            WHERE broker = ? AND account_id = ? AND order_uid = ?
        """, (broker, account_id, order_uid))

        legs_json = json.dumps(legs, ensure_ascii=False, default=str) if legs else None

        if existing:
            self.db.execute("""
                UPDATE order_strategy_map SET
                    strategy_id = ?,
                    strategy_type = ?,
                    strategy_name = ?,
                    confidence = ?,
                    leg_index = ?,
                    legs = ?,
                    classify_version = ?,
                    updated_ts = CURRENT_TIMESTAMP
                WHERE broker = ? AND account_id = ? AND order_uid = ?
            """, (strategy_id, strategy_type, strategy_name, confidence, leg_index, legs_json, classify_version, broker, account_id, order_uid))
            return False
        else:
            self.db.execute("""
                INSERT INTO order_strategy_map (
                    broker, account_id, order_uid, strategy_id, strategy_type,
                    strategy_name, confidence, leg_index, legs, classify_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (broker, account_id, order_uid, strategy_id, strategy_type, strategy_name, confidence, leg_index, legs_json, classify_version))
            return True

    def get_unclassified_orders(
        self,
        broker: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get orders that haven't been strategy-classified yet.

        Returns orders from the normalized 'orders' table that are not in
        order_strategy_map.
        """
        if broker:
            rows = self.db.fetch_all("""
                SELECT o.* FROM orders o
                LEFT JOIN order_strategy_map osm
                    ON o.source = osm.broker
                    AND o.account_id = osm.account_id
                    AND CONCAT(o.source, '_', o.account_id, '_', o.order_id) = osm.order_uid
                WHERE osm.order_uid IS NULL
                    AND o.source = ?
                    AND o.status IN ('FILLED', 'PARTIALLY_FILLED')
                ORDER BY o.created_time DESC
                LIMIT ?
            """, (broker, limit))
        else:
            rows = self.db.fetch_all("""
                SELECT o.* FROM orders o
                LEFT JOIN order_strategy_map osm
                    ON o.source = osm.broker
                    AND o.account_id = osm.account_id
                    AND CONCAT(o.source, '_', o.account_id, '_', o.order_id) = osm.order_uid
                WHERE osm.order_uid IS NULL
                    AND o.status IN ('FILLED', 'PARTIALLY_FILLED')
                ORDER BY o.created_time DESC
                LIMIT ?
            """, (limit,))

        return rows if rows else []

    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy classification statistics."""
        result = self.db.fetch_all("""
            SELECT
                strategy_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM order_strategy_map
            GROUP BY strategy_type
            ORDER BY count DESC
        """)

        return {
            "by_type": result if result else [],
            "total": sum(r["count"] for r in result) if result else 0,
        }
