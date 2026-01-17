"""
Repository for Futu raw deals (executions) persistence.

Handles UPSERT operations for deals from Futu's history_deal_list_query() API.
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


def _sanitize_int(value: Any, default: int | None = None) -> int | None:
    """Convert Futu value to int, handling N/A and empty strings."""
    if value is None or value == "" or value == "N/A" or str(value).upper() == "N/A":
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _sanitize_str(value: Any, default: str | None = None) -> str | None:
    """Convert Futu value to string, handling N/A."""
    if value is None or value == "N/A" or str(value).upper() == "N/A":
        return default
    return str(value) if value != "" else default


@dataclass
class FutuRawDeal:
    """Futu raw deal (execution) entity."""

    deal_id: str
    order_id: str
    account_id: str
    market: str
    code: str
    stock_name: Optional[str]
    trd_side: str
    qty: Decimal
    price: Decimal
    status: Optional[str]
    counter_broker_id: Optional[int]
    counter_broker_name: Optional[str]
    create_time: datetime
    raw_data: Optional[Dict[str, Any]] = None
    loaded_at: Optional[datetime] = None
    id: Optional[int] = None


class FutuDealRepository(BaseRepository[FutuRawDeal]):
    """
    Repository for Futu raw deals (executions).

    Handles persistence of deal data from Futu's history_deal_list_query() API.
    Uses UPSERT pattern with (deal_id, account_id) as the conflict key.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "futu_raw_deals"

    @property
    def conflict_columns(self) -> List[str]:
        return ["deal_id", "account_id"]

    def _to_entity(self, record: Record) -> FutuRawDeal:
        """Convert database record to FutuRawDeal entity."""
        return FutuRawDeal(
            id=record["id"],
            deal_id=record["deal_id"],
            order_id=record["order_id"],
            account_id=record["account_id"],
            market=record["market"],
            code=record["code"],
            stock_name=record["stock_name"],
            trd_side=record["trd_side"],
            qty=record["qty"],
            price=record["price"],
            status=record["status"],
            counter_broker_id=record["counter_broker_id"],
            counter_broker_name=record["counter_broker_name"],
            create_time=record["create_time"],
            raw_data=self._from_json(record["raw_data"]),
            loaded_at=record["loaded_at"],
        )

    def _to_row(self, entity: FutuRawDeal) -> Dict[str, Any]:
        """Convert FutuRawDeal entity to database row."""
        return {
            "deal_id": entity.deal_id,
            "order_id": entity.order_id,
            "account_id": entity.account_id,
            "market": entity.market,
            "code": entity.code,
            "stock_name": entity.stock_name,
            "trd_side": entity.trd_side,
            "qty": entity.qty,
            "price": entity.price,
            "status": entity.status,
            "counter_broker_id": entity.counter_broker_id,
            "counter_broker_name": entity.counter_broker_name,
            "create_time": entity.create_time,
            "raw_data": self._to_json(entity.raw_data),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def find_by_deal_id(self, deal_id: str, account_id: str) -> Optional[FutuRawDeal]:
        """
        Find deal by deal_id and account_id.

        Args:
            deal_id: Futu deal ID.
            account_id: Futu account ID.

        Returns:
            FutuRawDeal if found, None otherwise.
        """
        query = """
            SELECT * FROM futu_raw_deals
            WHERE deal_id = $1 AND account_id = $2
        """
        record = await self._db.fetchrow(query, deal_id, account_id)
        return self._to_entity(record) if record else None

    async def find_by_order_id(self, order_id: str, account_id: str) -> List[FutuRawDeal]:
        """
        Find all deals for a specific order.

        Args:
            order_id: Futu order ID.
            account_id: Futu account ID.

        Returns:
            List of deals for the order.
        """
        query = """
            SELECT * FROM futu_raw_deals
            WHERE order_id = $1 AND account_id = $2
            ORDER BY create_time
        """
        records = await self._db.fetch(query, order_id, account_id)
        return [self._to_entity(r) for r in records]

    async def find_by_account(
        self,
        account_id: str,
        market: Optional[str] = None,
        limit: int = 1000,
    ) -> List[FutuRawDeal]:
        """
        Find deals by account.

        Args:
            account_id: Futu account ID.
            market: Optional market filter (US, HK, CN).
            limit: Maximum number of records.

        Returns:
            List of deals.
        """
        if market:
            query = """
                SELECT * FROM futu_raw_deals
                WHERE account_id = $1 AND market = $2
                ORDER BY create_time DESC
                LIMIT $3
            """
            records = await self._db.fetch(query, account_id, market, limit)
        else:
            query = """
                SELECT * FROM futu_raw_deals
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
    ) -> List[FutuRawDeal]:
        """
        Find deals within a date range.

        Args:
            account_id: Futu account ID.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).
            market: Optional market filter.

        Returns:
            List of deals within the date range.
        """
        if market:
            query = """
                SELECT * FROM futu_raw_deals
                WHERE account_id = $1
                  AND market = $2
                  AND create_time >= $3
                  AND create_time <= $4
                ORDER BY create_time DESC
            """
            records = await self._db.fetch(query, account_id, market, start_date, end_date)
        else:
            query = """
                SELECT * FROM futu_raw_deals
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
    ) -> List[FutuRawDeal]:
        """
        Find deals for a specific security.

        Args:
            account_id: Futu account ID.
            code: Futu security code (e.g., 'US.AAPL').
            limit: Maximum number of records.

        Returns:
            List of deals for the security.
        """
        query = """
            SELECT * FROM futu_raw_deals
            WHERE account_id = $1 AND code = $2
            ORDER BY create_time DESC
            LIMIT $3
        """
        records = await self._db.fetch(query, account_id, code, limit)
        return [self._to_entity(r) for r in records]

    async def get_latest_deal_time(
        self,
        account_id: str,
        market: Optional[str] = None,
    ) -> Optional[datetime]:
        """
        Get the timestamp of the most recent deal.

        Used for incremental sync to determine where to resume.

        Args:
            account_id: Futu account ID.
            market: Optional market filter.

        Returns:
            Timestamp of the most recent deal, or None if no deals.
        """
        if market:
            query = """
                SELECT MAX(create_time) FROM futu_raw_deals
                WHERE account_id = $1 AND market = $2
            """
            result = await self._db.fetchval(query, account_id, market)
        else:
            query = """
                SELECT MAX(create_time) FROM futu_raw_deals
                WHERE account_id = $1
            """
            result = await self._db.fetchval(query, account_id)

        if result is None:
            return None
        if isinstance(result, datetime):
            return result
        return None

    async def get_total_volume_by_code(
        self,
        account_id: str,
        code: str,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Decimal]:
        """
        Calculate total buy/sell volume for a security.

        Args:
            account_id: Futu account ID.
            code: Futu security code.
            start_date: Optional start date filter.

        Returns:
            Dictionary with 'buy_qty', 'sell_qty', 'net_qty'.
        """
        conditions = ["account_id = $1", "code = $2"]
        params: List[Any] = [account_id, code]

        if start_date:
            conditions.append("create_time >= $3")
            params.append(start_date)

        query = f"""
            SELECT
                COALESCE(SUM(CASE WHEN trd_side = 'BUY' THEN qty ELSE 0 END), 0) as buy_qty,
                COALESCE(SUM(CASE WHEN trd_side = 'SELL' THEN qty ELSE 0 END), 0) as sell_qty
            FROM futu_raw_deals
            WHERE {' AND '.join(conditions)}
        """
        record = await self._db.fetchrow(query, *params)

        if record is None:
            return {"buy_qty": Decimal(0), "sell_qty": Decimal(0), "net_qty": Decimal(0)}

        buy_qty = record["buy_qty"] or Decimal(0)
        sell_qty = record["sell_qty"] or Decimal(0)

        return {
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "net_qty": buy_qty - sell_qty,
        }

    # -------------------------------------------------------------------------
    # Conversion from Futu API
    # -------------------------------------------------------------------------

    @classmethod
    def from_futu_deal(
        cls,
        deal_data: Dict[str, Any],
        account_id: str,
        market: str,
    ) -> FutuRawDeal:
        """
        Convert Futu API deal data to FutuRawDeal entity.

        Args:
            deal_data: Raw deal dict from Futu API.
            account_id: Futu account ID.
            market: Trading market (US, HK, CN).

        Returns:
            FutuRawDeal entity.
        """
        return FutuRawDeal(
            deal_id=str(deal_data.get("deal_id", "")),
            order_id=str(deal_data.get("order_id", "")),
            account_id=account_id,
            market=market,
            code=_sanitize_str(deal_data.get("code"), "") or "",
            stock_name=_sanitize_str(deal_data.get("stock_name")),
            trd_side=_sanitize_str(deal_data.get("trd_side"), "UNKNOWN") or "UNKNOWN",
            qty=_sanitize_decimal(deal_data.get("qty"), Decimal("0")) or Decimal("0"),
            price=_sanitize_decimal(deal_data.get("price"), Decimal("0")) or Decimal("0"),
            status=_sanitize_str(deal_data.get("status")),
            counter_broker_id=_sanitize_int(deal_data.get("counter_broker_id")),
            counter_broker_name=_sanitize_str(deal_data.get("counter_broker_name")),
            create_time=parse_futu_timestamp(deal_data.get("create_time"), market),
            raw_data=deal_data,
        )
