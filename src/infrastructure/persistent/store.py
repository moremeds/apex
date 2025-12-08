"""
PostgreSQL store for the persistent layer.

Provides async connection pooling and upsert methods for all tables.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

import asyncpg

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from .schemas import get_schema_sql

logger = logging.getLogger(__name__)

# Exchange timezone mappings
EXCHANGE_TZ = {
    "US": ZoneInfo("America/New_York"),
    "HK": ZoneInfo("Asia/Hong_Kong"),
    "CN": ZoneInfo("Asia/Shanghai"),
    "SG": ZoneInfo("Asia/Singapore"),
    "JP": ZoneInfo("Asia/Tokyo"),
}
UTC = timezone.utc


def _detect_market_from_code(code: Optional[str]) -> str:
    """Detect market from Futu code prefix (e.g., 'US.AAPL' -> 'US', 'HK.00700' -> 'HK')."""
    if code and "." in code:
        prefix = code.split(".")[0].upper()
        if prefix in EXCHANGE_TZ:
            return prefix
    return "US"


def _parse_futu_timestamp(time_str: Optional[str], market: str = "US") -> Optional[datetime]:
    """
    Parse Futu timestamp and convert to UTC based on market timezone.

    Futu returns timestamps in local exchange time:
    - US market: US Eastern time
    - HK market: Hong Kong time
    - etc.

    Format: "YYYY-MM-DD HH:MM:SS"
    """
    if not time_str:
        return None
    try:
        tz = EXCHANGE_TZ.get(market, EXCHANGE_TZ["US"])
        naive_dt = datetime.strptime(str(time_str), "%Y-%m-%d %H:%M:%S")
        local_dt = naive_dt.replace(tzinfo=tz)
        return local_dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


class PostgresStore:
    """
    Async PostgreSQL store with connection pooling.

    Usage:
        store = PostgresStore(dsn="postgresql://user:pass@localhost:5432/risk_db")
        await store.connect()
        await store.upsert_orders_raw_futu(records)
        await store.close()
    """

    def __init__(
        self,
        dsn: str,
        pool_min: int = 2,
        pool_max: int = 10,
    ):
        """
        Initialize PostgresStore.

        Args:
            dsn: PostgreSQL connection string.
            pool_min: Minimum pool connections.
            pool_max: Maximum pool connections.
        """
        self.dsn = dsn
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool and initialize schema."""
        if self._pool is not None:
            return

        logger.info(f"Connecting to PostgreSQL: {self._sanitize_dsn(self.dsn)}")
        self._pool = await asyncpg.create_pool(
            self.dsn,
            min_size=self.pool_min,
            max_size=self.pool_max,
        )
        await self._init_schema()
        logger.info("PostgreSQL connection pool established")

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed")

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        schema_sql = get_schema_sql()
        async with self._pool.acquire() as conn:
            # Execute each statement separately to handle errors better
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
            for stmt in statements:
                try:
                    await conn.execute(stmt)
                except Exception as e:
                    # Log but continue - some statements may fail if already exists
                    logger.debug(f"Schema statement skipped: {e}")
        logger.info("PostgreSQL schema initialized")

    async def ensure_schema(self) -> None:
        """Verify schema exists and create if needed."""
        async with self._pool.acquire() as conn:
            # Log current database for verification
            db_name = await conn.fetchval("SELECT current_database()")
            logger.info(f"Connected to database: {db_name}")

            # Check if orders_raw_futu table exists as a proxy for schema existence
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'orders_raw_futu'
                )
            """)
            if not result:
                logger.info("Schema not found, creating tables...")
                await self._init_schema()
                # Verify tables were created
                await self._verify_schema(conn)
            else:
                logger.info("Schema already exists")
                # List existing tables for confirmation
                await self._verify_schema(conn)

    async def _verify_schema(self, conn) -> None:
        """Verify and log created tables."""
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND (
                table_name LIKE '%_raw_%'
                OR table_name LIKE 'apex_%'
                OR table_name LIKE 'positions_%'
                OR table_name LIKE '%_signals'
            )
            ORDER BY table_name
        """)
        table_names = [t['table_name'] for t in tables]
        if table_names:
            logger.info(f"Verified {len(table_names)} tables: {', '.join(table_names)}")
        else:
            logger.error("No tables found! Schema creation may have failed.")

    @asynccontextmanager
    async def acquire(self):
        """Context manager for acquiring a connection."""
        if self._pool is None:
            raise RuntimeError("Store not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            yield conn

    def _sanitize_dsn(self, dsn: str) -> str:
        """Sanitize DSN for logging (hide password)."""
        if "@" in dsn:
            parts = dsn.split("@")
            prefix = parts[0].rsplit(":", 1)[0]
            return f"{prefix}:****@{parts[1]}"
        return dsn

    # =========================================================================
    # RAW TABLE UPSERTS - FUTU
    # =========================================================================

    async def upsert_orders_raw_futu(self, records: List[Dict[str, Any]]) -> int:
        """
        Upsert raw Futu orders with full payload preservation.

        Args:
            records: List of order dicts from Futu API.

        Returns:
            Number of records upserted.
        """
        if not records:
            return 0

        sql = """
        INSERT INTO orders_raw_futu (
            acc_id, order_id, payload,
            code, stock_name, trd_side, order_type, order_status,
            qty, price, dealt_qty, dealt_avg_price,
            create_time_raw_str, update_time_raw_str, create_time_utc, update_time_utc,
            aux_price, trail_type, trail_value, trail_spread,
            time_in_force, fill_outside_rth, remark, last_err_msg
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        ON CONFLICT (acc_id, order_id)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            code = EXCLUDED.code,
            stock_name = EXCLUDED.stock_name,
            trd_side = EXCLUDED.trd_side,
            order_type = EXCLUDED.order_type,
            order_status = EXCLUDED.order_status,
            qty = EXCLUDED.qty,
            price = EXCLUDED.price,
            dealt_qty = EXCLUDED.dealt_qty,
            dealt_avg_price = EXCLUDED.dealt_avg_price,
            create_time_raw_str = EXCLUDED.create_time_raw_str,
            update_time_raw_str = EXCLUDED.update_time_raw_str,
            create_time_utc = EXCLUDED.create_time_utc,
            update_time_utc = EXCLUDED.update_time_utc,
            aux_price = EXCLUDED.aux_price,
            trail_type = EXCLUDED.trail_type,
            trail_value = EXCLUDED.trail_value,
            trail_spread = EXCLUDED.trail_spread,
            time_in_force = EXCLUDED.time_in_force,
            fill_outside_rth = EXCLUDED.fill_outside_rth,
            remark = EXCLUDED.remark,
            last_err_msg = EXCLUDED.last_err_msg,
            ingest_ts = NOW()
        """

        def safe_float(val) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_str(val) -> Optional[str]:
            if val is None:
                return None
            return str(val)

        # Pre-process records with market-aware timestamp conversion
        processed = []
        for r in records:
            code = r.get("code", "")
            market = _detect_market_from_code(code)
            processed.append((
                r.get("acc_id"),
                str(r.get("order_id", "")),
                json.dumps(r),
                safe_str(code),
                safe_str(r.get("stock_name")),
                safe_str(r.get("trd_side")),
                safe_str(r.get("order_type")),
                safe_str(r.get("order_status")),
                safe_float(r.get("qty")),
                safe_float(r.get("price")),
                safe_float(r.get("dealt_qty")),
                safe_float(r.get("dealt_avg_price")),
                r.get("create_time"),
                r.get("updated_time"),
                _parse_futu_timestamp(r.get("create_time"), market),
                _parse_futu_timestamp(r.get("updated_time"), market),
                safe_float(r.get("aux_price")),
                safe_str(r.get("trail_type")),
                safe_float(r.get("trail_value")),
                safe_float(r.get("trail_spread")),
                safe_str(r.get("time_in_force")),
                r.get("fill_outside_rth"),
                safe_str(r.get("remark")),
                safe_str(r.get("last_err_msg")),
            ))

        async with self.acquire() as conn:
            await conn.executemany(sql, processed)

        logger.debug(f"Upserted {len(records)} raw Futu orders")
        return len(records)

    async def upsert_trades_raw_futu(self, records: List[Dict[str, Any]]) -> int:
        """Upsert raw Futu trades (deals)."""
        if not records:
            return 0

        sql = """
        INSERT INTO trades_raw_futu (
            acc_id, deal_id, order_id, payload,
            code, stock_name, trd_side, qty, price, status,
            counter_broker_id, counter_broker_name,
            trade_time_raw_str, trade_time_utc
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (acc_id, deal_id)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            order_id = EXCLUDED.order_id,
            code = EXCLUDED.code,
            stock_name = EXCLUDED.stock_name,
            trd_side = EXCLUDED.trd_side,
            qty = EXCLUDED.qty,
            price = EXCLUDED.price,
            status = EXCLUDED.status,
            counter_broker_id = EXCLUDED.counter_broker_id,
            counter_broker_name = EXCLUDED.counter_broker_name,
            trade_time_raw_str = EXCLUDED.trade_time_raw_str,
            trade_time_utc = EXCLUDED.trade_time_utc,
            ingest_ts = NOW()
        """

        def safe_float(val) -> Optional[float]:
            if val is None:
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_str(val) -> Optional[str]:
            if val is None:
                return None
            return str(val)

        # Pre-process records with market-aware timestamp conversion
        processed = []
        for r in records:
            code = r.get("code", "")
            market = _detect_market_from_code(code)
            processed.append((
                r.get("acc_id"),
                str(r.get("deal_id", "")),
                str(r.get("order_id", "")) if r.get("order_id") else None,
                json.dumps(r),
                safe_str(code),
                safe_str(r.get("stock_name")),
                safe_str(r.get("trd_side")),
                safe_float(r.get("qty")),
                safe_float(r.get("price")),
                safe_str(r.get("status")),
                safe_str(r.get("counter_broker_id")),
                safe_str(r.get("counter_broker_name")),
                r.get("create_time"),
                _parse_futu_timestamp(r.get("create_time"), market),
            ))

        async with self.acquire() as conn:
            await conn.executemany(sql, processed)

        logger.debug(f"Upserted {len(records)} raw Futu trades")
        return len(records)

    async def upsert_fees_raw_futu(self, records: List[Dict[str, Any]]) -> int:
        """Upsert raw Futu fees."""
        if not records:
            return 0

        sql = """
        INSERT INTO fees_raw_futu (acc_id, order_id, fee_amount, fee_list, payload)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (acc_id, order_id)
        DO UPDATE SET
            fee_amount = EXCLUDED.fee_amount,
            fee_list = EXCLUDED.fee_list,
            payload = EXCLUDED.payload,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r.get("acc_id"),
                    str(r.get("order_id", "")),
                    float(r.get("fee_amount", 0) or 0),
                    json.dumps(r.get("fee_list")) if r.get("fee_list") else None,
                    json.dumps(r),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} raw Futu fees")
        return len(records)

    # =========================================================================
    # RAW TABLE UPSERTS - IB
    # =========================================================================

    async def upsert_orders_raw_ib(self, records: List[Dict[str, Any]]) -> int:
        """Upsert raw IB orders."""
        if not records:
            return 0

        sql = """
        INSERT INTO orders_raw_ib (
            account, perm_id, client_order_id, order_ref, order_id_composite, payload,
            create_time_raw_str, update_time_raw_str, source
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (account, order_id_composite)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            order_ref = EXCLUDED.order_ref,
            update_time_raw_str = EXCLUDED.update_time_raw_str,
            source = EXCLUDED.source,
            ingest_ts = NOW()
        """

        def make_composite_id(r: Dict) -> str:
            """Generate composite order ID from perm_id or client_order_id."""
            perm_id = r.get("perm_id")
            client_order_id = r.get("client_order_id")
            order_id = r.get("order_id")
            if perm_id:
                return f"perm_{perm_id}"
            if client_order_id:
                return f"client_{client_order_id}"
            if order_id:
                return str(order_id)
            return f"unknown_{hash(json.dumps(r, sort_keys=True, default=str)) % 10**10}"

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r.get("account"),
                    r.get("perm_id"),
                    r.get("client_order_id"),
                    r.get("order_ref"),
                    make_composite_id(r),
                    json.dumps(r),
                    r.get("create_time"),
                    r.get("update_time"),
                    r.get("source", "API"),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} raw IB orders")
        return len(records)

    async def upsert_trades_raw_ib(self, records: List[Dict[str, Any]]) -> int:
        """Upsert raw IB trades (executions)."""
        if not records:
            return 0

        sql = """
        INSERT INTO trades_raw_ib (account, exec_id, perm_id, order_ref, payload, trade_time_raw_str, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (account, exec_id)
        DO UPDATE SET
            payload = EXCLUDED.payload,
            perm_id = EXCLUDED.perm_id,
            order_ref = EXCLUDED.order_ref,
            trade_time_raw_str = EXCLUDED.trade_time_raw_str,
            source = EXCLUDED.source,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r.get("account"),
                    r.get("exec_id"),
                    r.get("perm_id"),
                    r.get("order_ref"),
                    json.dumps(r),
                    r.get("trade_time"),
                    r.get("source", "API"),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} raw IB trades")
        return len(records)

    async def upsert_fees_raw_ib(self, records: List[Dict[str, Any]]) -> int:
        """Upsert raw IB fees (commission reports)."""
        if not records:
            return 0

        sql = """
        INSERT INTO fees_raw_ib (account, exec_id, commission, currency, realized_pnl, payload, source)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (account, exec_id)
        DO UPDATE SET
            commission = EXCLUDED.commission,
            currency = EXCLUDED.currency,
            realized_pnl = EXCLUDED.realized_pnl,
            payload = EXCLUDED.payload,
            source = EXCLUDED.source,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r.get("account"),
                    r.get("exec_id"),
                    float(r.get("commission", 0) or 0),
                    r.get("currency", "USD"),
                    float(r.get("realized_pnl", 0) or 0) if r.get("realized_pnl") else None,
                    json.dumps(r) if r else None,
                    r.get("source", "API"),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} raw IB fees")
        return len(records)

    # =========================================================================
    # NORMALIZED TABLE UPSERTS
    # =========================================================================

    async def upsert_orders_norm(self, records: List[Dict[str, Any]]) -> int:
        """Upsert normalized orders."""
        if not records:
            return 0

        sql = """
        INSERT INTO apex_order (
            broker, account_id, order_uid, instrument_type, symbol, stock_name, underlying,
            exchange, strike, expiry, option_right, side, trd_side, qty, limit_price,
            order_type, order_type_raw, time_in_force,
            aux_price, trail_type, trail_value, trail_spread, fill_outside_rth,
            status, status_raw, filled_qty, avg_fill_price,
            last_err_msg, remark,
            create_time_utc, update_time_utc, total_fee, order_reconstructed, raw_ref
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34)
        ON CONFLICT (broker, account_id, order_uid)
        DO UPDATE SET
            status = EXCLUDED.status,
            status_raw = EXCLUDED.status_raw,
            filled_qty = EXCLUDED.filled_qty,
            avg_fill_price = EXCLUDED.avg_fill_price,
            update_time_utc = EXCLUDED.update_time_utc,
            total_fee = EXCLUDED.total_fee,
            last_err_msg = EXCLUDED.last_err_msg,
            remark = EXCLUDED.remark,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r["broker"],
                    r["account_id"],
                    r["order_uid"],
                    r.get("instrument_type"),
                    r["symbol"],
                    r.get("stock_name"),
                    r.get("underlying"),
                    r.get("exchange"),
                    r.get("strike"),
                    r.get("expiry"),
                    r.get("option_right"),
                    r["side"],
                    r.get("trd_side"),
                    r.get("qty"),
                    r.get("limit_price"),
                    r.get("order_type"),
                    r.get("order_type_raw"),
                    r.get("time_in_force"),
                    r.get("aux_price"),
                    r.get("trail_type"),
                    r.get("trail_value"),
                    r.get("trail_spread"),
                    r.get("fill_outside_rth"),
                    r.get("status"),
                    r.get("status_raw"),
                    r.get("filled_qty"),
                    r.get("avg_fill_price"),
                    r.get("last_err_msg"),
                    r.get("remark"),
                    r.get("create_time_utc"),
                    r.get("update_time_utc"),
                    r.get("total_fee", 0),
                    r.get("order_reconstructed", False),
                    json.dumps(r.get("raw_ref", {})),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} normalized orders")
        return len(records)

    async def update_order_fees_from_apex_fees(self) -> int:
        """
        Update total_fee in apex_order by aggregating from apex_fees.

        Returns:
            Number of orders updated.
        """
        sql = """
        UPDATE apex_order o
        SET total_fee = COALESCE(f.total, 0)
        FROM (
            SELECT broker, account_id, order_uid, SUM(amount) as total
            FROM apex_fees
            WHERE order_uid IS NOT NULL
            GROUP BY broker, account_id, order_uid
        ) f
        WHERE o.broker = f.broker
          AND o.account_id = f.account_id
          AND o.order_uid = f.order_uid
        """

        async with self.acquire() as conn:
            result = await conn.execute(sql)
            # Parse "UPDATE X" to get count
            count = int(result.split()[-1]) if result else 0

        logger.info(f"Updated total_fee for {count} orders")
        return count

    async def upsert_trades_norm(self, records: List[Dict[str, Any]]) -> int:
        """Upsert normalized trades."""
        if not records:
            return 0

        sql = """
        INSERT INTO apex_trades (
            broker, account_id, trade_uid, order_uid, instrument_type, symbol, stock_name,
            underlying, strike, expiry, option_right, side, trd_side, qty, price,
            exchange, status, counter_broker_id, counter_broker_name,
            position_effect, realized_pnl, trade_time_utc, update_time_utc, raw_ref
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
        ON CONFLICT (broker, account_id, trade_uid)
        DO UPDATE SET
            price = EXCLUDED.price,
            qty = EXCLUDED.qty,
            trd_side = EXCLUDED.trd_side,
            status = EXCLUDED.status,
            position_effect = EXCLUDED.position_effect,
            realized_pnl = EXCLUDED.realized_pnl,
            trade_time_utc = EXCLUDED.trade_time_utc,
            update_time_utc = EXCLUDED.update_time_utc,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r["broker"],
                    r["account_id"],
                    r["trade_uid"],
                    r.get("order_uid"),
                    r.get("instrument_type"),
                    r["symbol"],
                    r.get("stock_name"),
                    r.get("underlying"),
                    r.get("strike"),
                    r.get("expiry"),
                    r.get("option_right"),
                    r["side"],
                    r.get("trd_side"),
                    r["qty"],
                    r["price"],
                    r.get("exchange"),
                    r.get("status"),
                    r.get("counter_broker_id"),
                    r.get("counter_broker_name"),
                    r.get("position_effect"),
                    r.get("realized_pnl"),
                    r["trade_time_utc"],
                    r.get("update_time_utc"),
                    json.dumps(r.get("raw_ref", {})),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} normalized trades")
        return len(records)

    async def upsert_fees_norm(self, records: List[Dict[str, Any]]) -> int:
        """Upsert normalized fees."""
        if not records:
            return 0

        sql = """
        INSERT INTO apex_fees (
            broker, account_id, fee_uid, order_uid, trade_uid,
            fee_type, amount, currency, raw_ref
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (broker, account_id, fee_uid, fee_type)
        DO UPDATE SET
            amount = EXCLUDED.amount,
            currency = EXCLUDED.currency,
            ingest_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r["broker"],
                    r["account_id"],
                    r["fee_uid"],
                    r.get("order_uid"),
                    r.get("trade_uid"),
                    r.get("fee_type", "COMMISSION"),
                    r["amount"],
                    r.get("currency", "USD"),
                    json.dumps(r.get("raw_ref", {})),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} normalized fees")
        return len(records)

    # =========================================================================
    # STRATEGY MAP UPSERTS
    # =========================================================================

    async def upsert_strategy_mappings(self, records: List[Dict[str, Any]]) -> int:
        """Upsert strategy mappings."""
        if not records:
            return 0

        sql = """
        INSERT INTO apex_strategy_analysis (
            broker, account_id, order_uid, strategy_id, strategy_type,
            strategy_name, strategy_outcome, is_closed, trade_duration,
            involved_orders, confidence, legs, open_time, close_time, updated_time, classify_version
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (broker, account_id, order_uid)
        DO UPDATE SET
            strategy_id = EXCLUDED.strategy_id,
            strategy_type = EXCLUDED.strategy_type,
            strategy_name = EXCLUDED.strategy_name,
            strategy_outcome = EXCLUDED.strategy_outcome,
            is_closed = EXCLUDED.is_closed,
            trade_duration = EXCLUDED.trade_duration,
            involved_orders = EXCLUDED.involved_orders,
            confidence = EXCLUDED.confidence,
            legs = EXCLUDED.legs,
            close_time = EXCLUDED.close_time,
            updated_time = EXCLUDED.updated_time,
            classify_version = EXCLUDED.classify_version,
            insert_ts = NOW()
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    r["broker"],
                    r["account_id"],
                    r["order_uid"],
                    r["strategy_id"],
                    r["strategy_type"],
                    r.get("strategy_name"),
                    r.get("strategy_outcome"),
                    r.get("is_closed", False),
                    r.get("trade_duration"),
                    json.dumps(r.get("involved_orders", [])),
                    r.get("confidence"),
                    json.dumps(r.get("legs", [])),
                    r.get("open_time"),
                    r.get("close_time"),
                    r.get("updated_time"),
                    r.get("classify_version", "v1"),
                )
                for r in records
            ])

        logger.debug(f"Upserted {len(records)} strategy mappings")
        return len(records)

    # =========================================================================
    # POSITION SNAPSHOTS
    # =========================================================================

    async def save_position_snapshot(
        self,
        snapshot_id: str,
        account_id: str,
        positions: List[Dict[str, Any]],
        snapshot_time: Optional[datetime] = None,
    ) -> int:
        """
        Save a position snapshot.

        Args:
            snapshot_id: Unique ID for this snapshot.
            account_id: Account identifier.
            positions: List of position dicts.
            snapshot_time: Snapshot timestamp (default: now).

        Returns:
            Number of positions saved.
        """
        if not positions:
            return 0

        snapshot_time = snapshot_time or datetime.utcnow()

        sql = """
        INSERT INTO positions_snapshot (
            snapshot_id, account_id, snapshot_time_utc, symbol, underlying,
            instrument_type, qty, avg_cost, market_value, unrealized_pnl,
            delta, gamma, theta, vega, payload
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
        ON CONFLICT (snapshot_id, account_id, symbol)
        DO UPDATE SET
            qty = EXCLUDED.qty,
            market_value = EXCLUDED.market_value,
            unrealized_pnl = EXCLUDED.unrealized_pnl,
            delta = EXCLUDED.delta,
            gamma = EXCLUDED.gamma,
            theta = EXCLUDED.theta,
            vega = EXCLUDED.vega,
            payload = EXCLUDED.payload
        """

        async with self.acquire() as conn:
            await conn.executemany(sql, [
                (
                    snapshot_id,
                    account_id,
                    snapshot_time,
                    p["symbol"],
                    p.get("underlying"),
                    p.get("instrument_type"),
                    p.get("qty"),
                    p.get("avg_cost"),
                    p.get("market_value"),
                    p.get("unrealized_pnl"),
                    p.get("delta"),
                    p.get("gamma"),
                    p.get("theta"),
                    p.get("vega"),
                    json.dumps(p) if p else None,
                )
                for p in positions
            ])

        logger.debug(f"Saved snapshot {snapshot_id} with {len(positions)} positions")
        return len(positions)

    async def get_snapshot_at(
        self,
        account_id: str,
        at_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get the closest snapshot at or before the given time.

        Args:
            account_id: Account identifier.
            at_time: Target timestamp.

        Returns:
            List of position dicts from the snapshot.
        """
        sql = """
        WITH closest_snapshot AS (
            SELECT DISTINCT ON (account_id) snapshot_id, snapshot_time_utc
            FROM positions_snapshot
            WHERE account_id = $1 AND snapshot_time_utc <= $2
            ORDER BY account_id, snapshot_time_utc DESC
        )
        SELECT p.*
        FROM positions_snapshot p
        JOIN closest_snapshot c ON p.snapshot_id = c.snapshot_id AND p.account_id = c.account_id
        """

        async with self.acquire() as conn:
            rows = await conn.fetch(sql, account_id, at_time)
            return [dict(row) for row in rows]

    # =========================================================================
    # QUERY UTILITIES
    # =========================================================================

    async def get_data_boundaries(
        self,
        table: str,
        time_col: str,
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get earliest and latest timestamps for incremental load.

        Args:
            table: Table name.
            time_col: Timestamp column name.

        Returns:
            Tuple of (earliest, latest) timestamps.
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT MIN({time_col}), MAX({time_col}) FROM {table}
            """)
            return row[0], row[1]

    async def query_raw_payload(
        self,
        table: str,
        jsonb_path: str,
        value: str,
    ) -> List[Dict[str, Any]]:
        """
        Query raw tables by JSONB field.

        Args:
            table: Table name (e.g., 'orders_raw_futu').
            jsonb_path: JSONB field path (e.g., 'code').
            value: Value to match.

        Returns:
            List of matching records.
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT * FROM {table}
                WHERE payload->>'{jsonb_path}' = $1
            """, value)
            return [dict(row) for row in rows]

    async def get_unclassified_trades(
        self,
        broker: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get trades that haven't been classified yet.

        Args:
            broker: Optional broker filter.
            limit: Maximum records to return.

        Returns:
            List of unclassified trade dicts.
        """
        sql = """
        SELECT t.*
        FROM apex_trades t
        LEFT JOIN apex_strategy_analysis m ON t.order_uid = m.order_uid
            AND t.broker = m.broker AND t.account_id = m.account_id
        WHERE m.order_uid IS NULL
        """

        if broker:
            sql += f" AND t.broker = '{broker}'"

        sql += f" ORDER BY t.trade_time_utc DESC LIMIT {limit}"

        async with self.acquire() as conn:
            rows = await conn.fetch(sql)
            return [dict(row) for row in rows]

    async def truncate_normalized_tables(self) -> None:
        """Truncate all normalized tables (for full reload)."""
        tables = [
            "apex_order",
            "apex_trades",
            "apex_fees",
            "apex_strategy_analysis",
        ]

        async with self.acquire() as conn:
            for table in tables:
                await conn.execute(f"TRUNCATE TABLE {table}")
                logger.info(f"Truncated {table}")

    async def drop_all_tables(self) -> None:
        """Drop all tables (for schema reset). Use with caution!"""
        tables = [
            # Strategy and signals first (may have FKs)
            "apex_strategy_analysis",
            "risk_signals",
            "trading_signals",
            # Normalized tables
            "apex_fees",
            "apex_trades",
            "apex_order",
            # Snapshots
            "positions_snapshot",
            # Raw tables last
            "fees_raw_futu",
            "trades_raw_futu",
            "orders_raw_futu",
            "fees_raw_ib",
            "trades_raw_ib",
            "orders_raw_ib",
        ]

        async with self.acquire() as conn:
            for table in tables:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                logger.info(f"Dropped table {table}")

        logger.warning("All tables dropped. Schema will be recreated on next connect.")

    async def get_order_count(self, broker: Optional[str] = None) -> int:
        """Get count of normalized orders."""
        sql = "SELECT COUNT(*) FROM apex_order"
        if broker:
            sql += f" WHERE broker = '{broker}'"

        async with self.acquire() as conn:
            row = await conn.fetchrow(sql)
            return row[0]

    async def get_trade_count(self, broker: Optional[str] = None) -> int:
        """Get count of normalized trades."""
        sql = "SELECT COUNT(*) FROM apex_trades"
        if broker:
            sql += f" WHERE broker = '{broker}'"

        async with self.acquire() as conn:
            row = await conn.fetchrow(sql)
            return row[0]
