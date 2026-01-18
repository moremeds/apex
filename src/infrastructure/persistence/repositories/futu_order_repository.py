"""
Repository for Futu raw orders persistence.

Handles UPSERT operations for orders from Futu's history_order_list_query() API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from asyncpg import Record

from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories.base import BaseRepository
from src.utils.timezone import parse_futu_timestamp

logger = logging.getLogger(__name__)


# =============================================================================
# Futu Value Sanitization Helpers
# =============================================================================
# Futu SDK returns 'N/A', '', or None for missing values which break type casts.


def _sanitize_decimal(value: Any, default: Decimal | None = None) -> Decimal | None:
    """Convert Futu value to Decimal, handling N/A and empty strings."""
    if value is None or value == "" or value == "N/A" or str(value).upper() == "N/A":
        return default
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _sanitize_int(value: Any, default: int = 0) -> int:
    """Convert Futu value to int, handling N/A and empty strings."""
    if value is None or value == "" or value == "N/A" or str(value).upper() == "N/A":
        return default
    try:
        return int(float(value))  # Handle "123.0" strings
    except Exception:
        return default


def _sanitize_bool(value: Any, default: bool | None = None) -> bool | None:
    """Convert Futu value to bool, handling N/A and empty strings."""
    if value is None or value == "" or value == "N/A" or str(value).upper() == "N/A":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "y")
    return bool(value)


def _sanitize_str(value: Any, default: str | None = None) -> str | None:
    """Convert Futu value to string, handling N/A."""
    if value is None or value == "N/A" or str(value).upper() == "N/A":
        return default
    return str(value) if value != "" else default


@dataclass
class FutuRawOrder:
    """Futu raw order entity."""

    order_id: str
    account_id: str
    market: str
    code: str
    stock_name: Optional[str]
    trd_side: str
    order_type: str
    order_status: str
    qty: Decimal
    price: Optional[Decimal]
    currency: Optional[str]
    dealt_qty: Decimal
    dealt_avg_price: Optional[Decimal]
    time_in_force: Optional[str]
    fill_outside_rth: Optional[bool]
    session: Optional[str]
    aux_price: Optional[Decimal]
    trail_type: Optional[str]
    trail_value: Optional[Decimal]
    trail_spread: Optional[Decimal]
    remark: Optional[str]
    last_err_msg: Optional[str]
    create_time: datetime
    updated_time: datetime
    raw_data: Optional[Dict[str, Any]] = None
    loaded_at: Optional[datetime] = None
    id: Optional[int] = None


class FutuOrderRepository(BaseRepository[FutuRawOrder]):
    """
    Repository for Futu raw orders.

    Handles persistence of order data from Futu's history_order_list_query() API.
    Uses UPSERT pattern with (order_id, account_id) as the conflict key.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "futu_raw_orders"

    @property
    def conflict_columns(self) -> List[str]:
        return ["order_id", "account_id"]

    def _to_entity(self, record: Record) -> FutuRawOrder:
        """Convert database record to FutuRawOrder entity."""
        return FutuRawOrder(
            id=record["id"],
            order_id=record["order_id"],
            account_id=record["account_id"],
            market=record["market"],
            code=record["code"],
            stock_name=record["stock_name"],
            trd_side=record["trd_side"],
            order_type=record["order_type"],
            order_status=record["order_status"],
            qty=record["qty"],
            price=record["price"],
            currency=record["currency"],
            dealt_qty=record["dealt_qty"],
            dealt_avg_price=record["dealt_avg_price"],
            time_in_force=record["time_in_force"],
            fill_outside_rth=record["fill_outside_rth"],
            session=record["session"],
            aux_price=record["aux_price"],
            trail_type=record["trail_type"],
            trail_value=record["trail_value"],
            trail_spread=record["trail_spread"],
            remark=record["remark"],
            last_err_msg=record["last_err_msg"],
            create_time=record["create_time"],
            updated_time=record["updated_time"],
            raw_data=self._from_json(record["raw_data"]),
            loaded_at=record["loaded_at"],
        )

    def _to_row(self, entity: FutuRawOrder) -> Dict[str, Any]:
        """Convert FutuRawOrder entity to database row."""
        return {
            "order_id": entity.order_id,
            "account_id": entity.account_id,
            "market": entity.market,
            "code": entity.code,
            "stock_name": entity.stock_name,
            "trd_side": entity.trd_side,
            "order_type": entity.order_type,
            "order_status": entity.order_status,
            "qty": entity.qty,
            "price": entity.price,
            "currency": entity.currency,
            "dealt_qty": entity.dealt_qty,
            "dealt_avg_price": entity.dealt_avg_price,
            "time_in_force": entity.time_in_force,
            "fill_outside_rth": entity.fill_outside_rth,
            "session": entity.session,
            "aux_price": entity.aux_price,
            "trail_type": entity.trail_type,
            "trail_value": entity.trail_value,
            "trail_spread": entity.trail_spread,
            "remark": entity.remark,
            "last_err_msg": entity.last_err_msg,
            "create_time": entity.create_time,
            "updated_time": entity.updated_time,
            "raw_data": self._to_json(entity.raw_data),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def find_by_order_id(self, order_id: str, account_id: str) -> Optional[FutuRawOrder]:
        """
        Find order by order_id and account_id.

        Args:
            order_id: Futu order ID.
            account_id: Futu account ID.

        Returns:
            FutuRawOrder if found, None otherwise.
        """
        query = """
            SELECT * FROM futu_raw_orders
            WHERE order_id = $1 AND account_id = $2
        """
        record = await self._db.fetchrow(query, order_id, account_id)
        return self._to_entity(record) if record else None

    async def find_by_account(
        self,
        account_id: str,
        market: Optional[str] = None,
        limit: int = 1000,
    ) -> List[FutuRawOrder]:
        """
        Find orders by account.

        Args:
            account_id: Futu account ID.
            market: Optional market filter (US, HK, CN).
            limit: Maximum number of records.

        Returns:
            List of orders.
        """
        if market:
            query = """
                SELECT * FROM futu_raw_orders
                WHERE account_id = $1 AND market = $2
                ORDER BY create_time DESC
                LIMIT $3
            """
            records = await self._db.fetch(query, account_id, market, limit)
        else:
            query = """
                SELECT * FROM futu_raw_orders
                WHERE account_id = $1
                ORDER BY create_time DESC
                LIMIT $2
            """
            records = await self._db.fetch(query, account_id, limit)

        return [self._to_entity(r) for r in records]

    async def find_by_date_range(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
        market: Optional[str] = None,
    ) -> List[FutuRawOrder]:
        """
        Find orders within a date range.

        Args:
            account_id: Futu account ID.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).
            market: Optional market filter.

        Returns:
            List of orders within the date range.
        """
        if market:
            query = """
                SELECT * FROM futu_raw_orders
                WHERE account_id = $1
                  AND market = $2
                  AND create_time >= $3
                  AND create_time <= $4
                ORDER BY create_time DESC
            """
            records = await self._db.fetch(query, account_id, market, start_date, end_date)
        else:
            query = """
                SELECT * FROM futu_raw_orders
                WHERE account_id = $1
                  AND create_time >= $2
                  AND create_time <= $3
                ORDER BY create_time DESC
            """
            records = await self._db.fetch(query, account_id, start_date, end_date)

        return [self._to_entity(r) for r in records]

    async def find_by_code(
        self,
        account_id: str,
        code: str,
        limit: int = 100,
    ) -> List[FutuRawOrder]:
        """
        Find orders for a specific security.

        Args:
            account_id: Futu account ID.
            code: Futu security code (e.g., 'US.AAPL').
            limit: Maximum number of records.

        Returns:
            List of orders for the security.
        """
        query = """
            SELECT * FROM futu_raw_orders
            WHERE account_id = $1 AND code = $2
            ORDER BY create_time DESC
            LIMIT $3
        """
        records = await self._db.fetch(query, account_id, code, limit)
        return [self._to_entity(r) for r in records]

    async def find_filled_orders(
        self,
        account_id: str,
        start_date: Optional[datetime] = None,
        market: Optional[str] = None,
    ) -> List[FutuRawOrder]:
        """
        Find all filled orders (FILLED_ALL status).

        Args:
            account_id: Futu account ID.
            start_date: Optional start date filter.
            market: Optional market filter.

        Returns:
            List of filled orders.
        """
        conditions = ["account_id = $1", "order_status = 'FILLED_ALL'"]
        params: List[Any] = [account_id]
        param_idx = 2

        if start_date:
            conditions.append(f"create_time >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if market:
            conditions.append(f"market = ${param_idx}")
            params.append(market)

        query = f"""
            SELECT * FROM futu_raw_orders
            WHERE {' AND '.join(conditions)}
            ORDER BY create_time DESC
        """
        records = await self._db.fetch(query, *params)
        return [self._to_entity(r) for r in records]

    async def get_latest_order_time(
        self,
        account_id: str,
        market: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        Get the timestamp of the most recent order.

        Used for incremental sync to determine where to resume.

        Args:
            account_id: Futu account ID.
            market: Optional market filter.

        Returns:
            Timestamp of the most recent order, or None if no orders.
        """
        if market:
            query = """
                SELECT MAX(updated_time) FROM futu_raw_orders
                WHERE account_id = $1 AND market = $2
            """
            result = await self._db.fetchval(query, account_id, market)
        else:
            query = """
                SELECT MAX(updated_time) FROM futu_raw_orders
                WHERE account_id = $1
            """
            result = await self._db.fetchval(query, account_id)

        if result is None:
            return None
        if isinstance(result, datetime):
            return result
        return None

    # -------------------------------------------------------------------------
    # Conversion from Futu API
    # -------------------------------------------------------------------------

    @classmethod
    def from_futu_order(
        cls,
        order_data: Dict[str, Any],
        account_id: str,
        market: str,
    ) -> FutuRawOrder:
        """
        Convert Futu API order data to FutuRawOrder entity.

        Uses sanitization helpers to handle Futu's 'N/A' and '' values.

        Args:
            order_data: Raw order dict from Futu API.
            account_id: Futu account ID.
            market: Trading market (US, HK, CN).

        Returns:
            FutuRawOrder entity.
        """
        return FutuRawOrder(
            order_id=str(order_data.get("order_id", "")),
            account_id=account_id,
            market=market,
            code=_sanitize_str(order_data.get("code"), "") or "",
            stock_name=_sanitize_str(order_data.get("stock_name")),
            trd_side=_sanitize_str(order_data.get("trd_side"), "UNKNOWN") or "UNKNOWN",
            order_type=_sanitize_str(order_data.get("order_type"), "UNKNOWN") or "UNKNOWN",
            order_status=_sanitize_str(order_data.get("order_status"), "UNKNOWN") or "UNKNOWN",
            qty=_sanitize_decimal(order_data.get("qty"), Decimal("0")) or Decimal("0"),
            price=_sanitize_decimal(order_data.get("price")),
            currency=_sanitize_str(order_data.get("currency")),
            dealt_qty=_sanitize_decimal(order_data.get("dealt_qty"), Decimal("0")) or Decimal("0"),
            dealt_avg_price=_sanitize_decimal(order_data.get("dealt_avg_price")),
            time_in_force=_sanitize_str(order_data.get("time_in_force")),
            fill_outside_rth=_sanitize_bool(order_data.get("fill_outside_rth")),
            session=_sanitize_str(order_data.get("session")),
            aux_price=_sanitize_decimal(order_data.get("aux_price")),
            trail_type=_sanitize_str(order_data.get("trail_type")),
            trail_value=_sanitize_decimal(order_data.get("trail_value")),
            trail_spread=_sanitize_decimal(order_data.get("trail_spread")),
            remark=_sanitize_str(order_data.get("remark")),
            last_err_msg=_sanitize_str(order_data.get("last_err_msg")),
            create_time=parse_futu_timestamp(order_data.get("create_time"), market),
            updated_time=parse_futu_timestamp(order_data.get("updated_time"), market),
            raw_data=order_data,
        )
