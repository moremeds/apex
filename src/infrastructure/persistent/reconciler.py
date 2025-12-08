"""
Reconciler for validating order-trade-fee consistency.

Detects anomalies like:
- Orders with filled_qty != sum of trade quantities
- Filled orders missing fee records
- Orphan trades (no matching order)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of reconciliation anomalies."""

    # Order-Trade mismatches
    FILL_QTY_MISMATCH = "fill_qty_mismatch"
    ORPHAN_TRADE = "orphan_trade"
    ORDER_WITHOUT_TRADES = "order_without_trades"

    # Fee issues
    MISSING_FEE = "missing_fee"
    ORPHAN_FEE = "orphan_fee"
    FEE_MISMATCH = "fee_mismatch"

    # Data quality
    MISSING_TIMESTAMP = "missing_timestamp"
    INVALID_PRICE = "invalid_price"
    DUPLICATE_RECORD = "duplicate_record"


@dataclass
class Anomaly:
    """Reconciliation anomaly record."""

    anomaly_type: AnomalyType
    broker: str
    account_id: str
    order_uid: Optional[str]
    trade_uid: Optional[str]
    fee_uid: Optional[str]
    description: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    severity: str = "WARNING"  # WARNING, ERROR, CRITICAL
    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "broker": self.broker,
            "account_id": self.account_id,
            "order_uid": self.order_uid,
            "trade_uid": self.trade_uid,
            "fee_uid": self.fee_uid,
            "description": self.description,
            "expected": self.expected_value,
            "actual": self.actual_value,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
        }


class Reconciler:
    """
    Validates consistency between orders, trades, and fees.

    Runs after normalization to ensure data integrity.
    """

    # Tolerance for floating point comparisons
    QTY_TOLERANCE = 0.001
    PRICE_TOLERANCE = 0.0001

    def __init__(self, store):
        """
        Initialize reconciler.

        Args:
            store: PostgresStore instance for database queries.
        """
        self.store = store

    async def run(
        self,
        broker: Optional[str] = None,
    ) -> List[Anomaly]:
        """
        Run full reconciliation checks.

        Args:
            broker: Optional broker filter (FUTU, IB).

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        logger.info(f"Starting reconciliation{f' for {broker}' if broker else ''}")

        # Run all checks
        anomalies.extend(await self._check_fill_quantities(broker))
        anomalies.extend(await self._check_missing_fees(broker))
        anomalies.extend(await self._check_orphan_trades(broker))

        logger.info(f"Reconciliation complete: {len(anomalies)} anomalies found")

        return anomalies

    async def _check_fill_quantities(
        self,
        broker: Optional[str] = None,
    ) -> List[Anomaly]:
        """
        Check that order filled_qty matches sum of trade quantities.

        Returns:
            List of fill quantity mismatch anomalies.
        """
        anomalies = []

        sql = """
        SELECT
            o.broker,
            o.account_id,
            o.order_uid,
            o.symbol,
            o.filled_qty AS order_filled,
            COALESCE(SUM(t.qty), 0) AS trade_sum,
            o.filled_qty - COALESCE(SUM(t.qty), 0) AS discrepancy
        FROM orders_norm o
        LEFT JOIN trades_norm t ON o.order_uid = t.order_uid
            AND o.broker = t.broker AND o.account_id = t.account_id
        WHERE o.status IN ('FILLED', 'PARTIALLY_FILLED')
        """

        if broker:
            sql += f" AND o.broker = '{broker}'"

        sql += """
        GROUP BY o.broker, o.account_id, o.order_uid, o.symbol, o.filled_qty
        HAVING ABS(o.filled_qty - COALESCE(SUM(t.qty), 0)) > $1
        """

        async with self.store.acquire() as conn:
            rows = await conn.fetch(sql, self.QTY_TOLERANCE)

            for row in rows:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.FILL_QTY_MISMATCH,
                    broker=row["broker"],
                    account_id=row["account_id"],
                    order_uid=row["order_uid"],
                    trade_uid=None,
                    fee_uid=None,
                    description=f"Order {row['order_uid']} ({row['symbol']}): filled_qty ({row['order_filled']}) != trade sum ({row['trade_sum']})",
                    expected_value=float(row["order_filled"]) if row["order_filled"] else 0,
                    actual_value=float(row["trade_sum"]) if row["trade_sum"] else 0,
                    severity="WARNING",
                ))

        logger.debug(f"Fill quantity check: {len(anomalies)} mismatches")
        return anomalies

    async def _check_missing_fees(
        self,
        broker: Optional[str] = None,
    ) -> List[Anomaly]:
        """
        Check that filled orders have associated fee records.

        Returns:
            List of missing fee anomalies.
        """
        anomalies = []

        # Futu: fees are at order level
        if broker is None or broker == "FUTU":
            sql_futu = """
            SELECT
                o.broker,
                o.account_id,
                o.order_uid,
                o.symbol
            FROM orders_norm o
            LEFT JOIN fees_norm f ON o.order_uid = f.order_uid
                AND o.broker = f.broker AND o.account_id = f.account_id
            WHERE o.broker = 'FUTU'
              AND o.status = 'FILLED'
              AND f.fee_uid IS NULL
            """

            async with self.store.acquire() as conn:
                rows = await conn.fetch(sql_futu)

                for row in rows:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.MISSING_FEE,
                        broker=row["broker"],
                        account_id=row["account_id"],
                        order_uid=row["order_uid"],
                        trade_uid=None,
                        fee_uid=None,
                        description=f"Futu order {row['order_uid']} ({row['symbol']}) is filled but has no fee record",
                        severity="WARNING",
                    ))

        # IB: fees are at trade level
        if broker is None or broker == "IB":
            sql_ib = """
            SELECT
                t.broker,
                t.account_id,
                t.trade_uid,
                t.order_uid,
                t.symbol
            FROM trades_norm t
            LEFT JOIN fees_norm f ON t.trade_uid = f.trade_uid
                AND t.broker = f.broker AND t.account_id = f.account_id
            WHERE t.broker = 'IB'
              AND f.fee_uid IS NULL
            """

            async with self.store.acquire() as conn:
                rows = await conn.fetch(sql_ib)

                for row in rows:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.MISSING_FEE,
                        broker=row["broker"],
                        account_id=row["account_id"],
                        order_uid=row["order_uid"],
                        trade_uid=row["trade_uid"],
                        fee_uid=None,
                        description=f"IB trade {row['trade_uid']} ({row['symbol']}) has no fee record",
                        severity="WARNING",
                    ))

        logger.debug(f"Missing fee check: {len(anomalies)} missing")
        return anomalies

    async def _check_orphan_trades(
        self,
        broker: Optional[str] = None,
    ) -> List[Anomaly]:
        """
        Check for trades without matching orders.

        Returns:
            List of orphan trade anomalies.
        """
        anomalies = []

        sql = """
        SELECT
            t.broker,
            t.account_id,
            t.trade_uid,
            t.order_uid,
            t.symbol,
            t.qty,
            t.price,
            t.trade_time_utc
        FROM trades_norm t
        LEFT JOIN orders_norm o ON t.order_uid = o.order_uid
            AND t.broker = o.broker AND t.account_id = o.account_id
        WHERE t.order_uid IS NOT NULL
          AND o.order_uid IS NULL
        """

        if broker:
            sql += f" AND t.broker = '{broker}'"

        async with self.store.acquire() as conn:
            rows = await conn.fetch(sql)

            for row in rows:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.ORPHAN_TRADE,
                    broker=row["broker"],
                    account_id=row["account_id"],
                    order_uid=row["order_uid"],
                    trade_uid=row["trade_uid"],
                    fee_uid=None,
                    description=f"Trade {row['trade_uid']} references order {row['order_uid']} which doesn't exist",
                    severity="WARNING",
                ))

        logger.debug(f"Orphan trade check: {len(anomalies)} orphans")
        return anomalies

    async def generate_report(
        self,
        anomalies: List[Anomaly],
    ) -> Dict[str, Any]:
        """
        Generate summary report from anomalies.

        Args:
            anomalies: List of detected anomalies.

        Returns:
            Report dictionary with summary and details.
        """
        if not anomalies:
            return {
                "status": "OK",
                "total_anomalies": 0,
                "by_type": {},
                "by_severity": {},
                "details": [],
            }

        # Group by type
        by_type = {}
        for a in anomalies:
            key = a.anomaly_type.value
            by_type[key] = by_type.get(key, 0) + 1

        # Group by severity
        by_severity = {}
        for a in anomalies:
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1

        # Determine overall status
        if "CRITICAL" in by_severity:
            status = "CRITICAL"
        elif "ERROR" in by_severity:
            status = "ERROR"
        else:
            status = "WARNING"

        return {
            "status": status,
            "total_anomalies": len(anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
            "details": [a.to_dict() for a in anomalies[:100]],  # Limit details
        }


async def run_reconciliation(store, broker: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run reconciliation and return report.

    Args:
        store: PostgresStore instance.
        broker: Optional broker filter.

    Returns:
        Reconciliation report dict.
    """
    reconciler = Reconciler(store)
    anomalies = await reconciler.run(broker)
    return await reconciler.generate_report(anomalies)
