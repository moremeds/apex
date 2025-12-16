"""
Repository for IB raw executions persistence.

Handles UPSERT operations for executions from IB's reqExecutions() API.
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
from src.utils.timezone import parse_ib_timestamp

logger = logging.getLogger(__name__)


@dataclass
class IbRawExecution:
    """IB raw execution entity."""

    exec_id: str
    order_id: int
    perm_id: int
    account_id: str
    client_id: Optional[int]
    symbol: str
    sec_type: str
    exchange: Optional[str]
    currency: Optional[str]
    # Option-specific fields
    expiry: Optional[str]
    strike: Optional[Decimal]
    right: Optional[str]
    # Execution details
    side: str
    shares: Decimal
    price: Decimal
    cum_qty: Optional[Decimal]
    avg_price: Optional[Decimal]
    liquidation: int
    model_code: Optional[str]
    last_liquidity: Optional[int]
    exec_time: datetime
    raw_data: Optional[Dict[str, Any]] = None
    loaded_at: Optional[datetime] = None
    id: Optional[int] = None


class IbExecutionRepository(BaseRepository[IbRawExecution]):
    """
    Repository for IB raw executions.

    Handles persistence of execution data from IB's reqExecutions() API.
    Uses UPSERT pattern with (exec_id, account_id) as the conflict key.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "ib_raw_executions"

    @property
    def conflict_columns(self) -> List[str]:
        return ["exec_id", "account_id"]

    def _to_entity(self, record: Record) -> IbRawExecution:
        """Convert database record to IbRawExecution entity."""
        return IbRawExecution(
            id=record["id"],
            exec_id=record["exec_id"],
            order_id=record["order_id"],
            perm_id=record["perm_id"],
            account_id=record["account_id"],
            client_id=record["client_id"],
            symbol=record["symbol"],
            sec_type=record["sec_type"],
            exchange=record["exchange"],
            currency=record["currency"],
            expiry=record["expiry"],
            strike=record["strike"],
            right=record["right"],
            side=record["side"],
            shares=record["shares"],
            price=record["price"],
            cum_qty=record["cum_qty"],
            avg_price=record["avg_price"],
            liquidation=record["liquidation"],
            model_code=record["model_code"],
            last_liquidity=record["last_liquidity"],
            exec_time=record["exec_time"],
            raw_data=self._from_json(record["raw_data"]),
            loaded_at=record["loaded_at"],
        )

    def _to_row(self, entity: IbRawExecution) -> Dict[str, Any]:
        """Convert IbRawExecution entity to database row."""
        return {
            "exec_id": entity.exec_id,
            "order_id": entity.order_id,
            "perm_id": entity.perm_id,
            "account_id": entity.account_id,
            "client_id": entity.client_id,
            "symbol": entity.symbol,
            "sec_type": entity.sec_type,
            "exchange": entity.exchange,
            "currency": entity.currency,
            "expiry": entity.expiry,
            "strike": entity.strike,
            "right": entity.right,
            "side": entity.side,
            "shares": entity.shares,
            "price": entity.price,
            "cum_qty": entity.cum_qty,
            "avg_price": entity.avg_price,
            "liquidation": entity.liquidation,
            "model_code": entity.model_code,
            "last_liquidity": entity.last_liquidity,
            "exec_time": entity.exec_time,
            "raw_data": self._to_json(entity.raw_data),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def find_by_exec_id(
        self, exec_id: str, account_id: str
    ) -> Optional[IbRawExecution]:
        """
        Find execution by exec_id and account_id.

        Args:
            exec_id: IB execution ID.
            account_id: IB account ID.

        Returns:
            IbRawExecution if found, None otherwise.
        """
        query = """
            SELECT * FROM ib_raw_executions
            WHERE exec_id = $1 AND account_id = $2
        """
        record = await self._db.fetchrow(query, exec_id, account_id)
        return self._to_entity(record) if record else None

    async def find_by_order_id(
        self, order_id: int, account_id: str
    ) -> List[IbRawExecution]:
        """
        Find all executions for a specific order.

        Args:
            order_id: IB order ID.
            account_id: IB account ID.

        Returns:
            List of executions for the order.
        """
        query = """
            SELECT * FROM ib_raw_executions
            WHERE order_id = $1 AND account_id = $2
            ORDER BY exec_time
        """
        records = await self._db.fetch(query, order_id, account_id)
        return [self._to_entity(r) for r in records]

    async def find_by_perm_id(
        self, perm_id: int, account_id: str
    ) -> List[IbRawExecution]:
        """
        Find all executions for a specific permanent order ID.

        Args:
            perm_id: IB permanent order ID (stable across sessions).
            account_id: IB account ID.

        Returns:
            List of executions for the order.
        """
        query = """
            SELECT * FROM ib_raw_executions
            WHERE perm_id = $1 AND account_id = $2
            ORDER BY exec_time
        """
        records = await self._db.fetch(query, perm_id, account_id)
        return [self._to_entity(r) for r in records]

    async def find_by_account(
        self,
        account_id: str,
        limit: int = 1000,
    ) -> List[IbRawExecution]:
        """
        Find executions by account.

        Args:
            account_id: IB account ID.
            limit: Maximum number of records.

        Returns:
            List of executions.
        """
        query = """
            SELECT * FROM ib_raw_executions
            WHERE account_id = $1
            ORDER BY exec_time DESC
            LIMIT $2
        """
        records = await self._db.fetch(query, account_id, limit)
        return [self._to_entity(r) for r in records]

    async def find_by_date_range(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[IbRawExecution]:
        """
        Find executions within a date range.

        Args:
            account_id: IB account ID.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).

        Returns:
            List of executions within the date range.
        """
        query = """
            SELECT * FROM ib_raw_executions
            WHERE account_id = $1
              AND exec_time >= $2
              AND exec_time <= $3
            ORDER BY exec_time DESC
        """
        records = await self._db.fetch(query, account_id, start_date, end_date)
        return [self._to_entity(r) for r in records]

    async def find_by_symbol(
        self,
        account_id: str,
        symbol: str,
        sec_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[IbRawExecution]:
        """
        Find executions for a specific symbol.

        Args:
            account_id: IB account ID.
            symbol: Trading symbol.
            sec_type: Optional security type filter (STK, OPT, FUT).
            limit: Maximum number of records.

        Returns:
            List of executions for the symbol.
        """
        if sec_type:
            query = """
                SELECT * FROM ib_raw_executions
                WHERE account_id = $1 AND symbol = $2 AND sec_type = $3
                ORDER BY exec_time DESC
                LIMIT $4
            """
            records = await self._db.fetch(query, account_id, symbol, sec_type, limit)
        else:
            query = """
                SELECT * FROM ib_raw_executions
                WHERE account_id = $1 AND symbol = $2
                ORDER BY exec_time DESC
                LIMIT $3
            """
            records = await self._db.fetch(query, account_id, symbol, limit)

        return [self._to_entity(r) for r in records]

    async def find_options_by_underlying(
        self,
        account_id: str,
        symbol: str,
        expiry: Optional[str] = None,
        limit: int = 100,
    ) -> List[IbRawExecution]:
        """
        Find option executions by underlying symbol.

        Args:
            account_id: IB account ID.
            symbol: Underlying symbol.
            expiry: Optional expiry filter (YYYYMMDD format).
            limit: Maximum number of records.

        Returns:
            List of option executions.
        """
        if expiry:
            query = """
                SELECT * FROM ib_raw_executions
                WHERE account_id = $1
                  AND symbol = $2
                  AND sec_type = 'OPT'
                  AND expiry = $3
                ORDER BY exec_time DESC
                LIMIT $4
            """
            records = await self._db.fetch(query, account_id, symbol, expiry, limit)
        else:
            query = """
                SELECT * FROM ib_raw_executions
                WHERE account_id = $1
                  AND symbol = $2
                  AND sec_type = 'OPT'
                ORDER BY exec_time DESC
                LIMIT $3
            """
            records = await self._db.fetch(query, account_id, symbol, limit)

        return [self._to_entity(r) for r in records]

    async def get_latest_exec_time(
        self,
        account_id: str,
    ) -> Optional[datetime]:
        """
        Get the timestamp of the most recent execution.

        Used for incremental sync to determine where to resume.

        Args:
            account_id: IB account ID.

        Returns:
            Timestamp of the most recent execution, or None if none.
        """
        query = """
            SELECT MAX(exec_time) FROM ib_raw_executions
            WHERE account_id = $1
        """
        return await self._db.fetchval(query, account_id)

    async def get_total_volume_by_symbol(
        self,
        account_id: str,
        symbol: str,
        sec_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
    ) -> Dict[str, Decimal]:
        """
        Calculate total buy/sell volume for a symbol.

        Args:
            account_id: IB account ID.
            symbol: Trading symbol.
            sec_type: Optional security type filter.
            start_date: Optional start date filter.

        Returns:
            Dictionary with 'buy_qty', 'sell_qty', 'net_qty'.
        """
        conditions = ["account_id = $1", "symbol = $2"]
        params = [account_id, symbol]
        param_idx = 3

        if sec_type:
            conditions.append(f"sec_type = ${param_idx}")
            params.append(sec_type)
            param_idx += 1

        if start_date:
            conditions.append(f"exec_time >= ${param_idx}")
            params.append(start_date)

        query = f"""
            SELECT
                COALESCE(SUM(CASE WHEN side = 'BOT' THEN shares ELSE 0 END), 0) as buy_qty,
                COALESCE(SUM(CASE WHEN side = 'SLD' THEN shares ELSE 0 END), 0) as sell_qty
            FROM ib_raw_executions
            WHERE {' AND '.join(conditions)}
        """
        record = await self._db.fetchrow(query, *params)

        buy_qty = record["buy_qty"] or Decimal(0)
        sell_qty = record["sell_qty"] or Decimal(0)

        return {
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "net_qty": buy_qty - sell_qty,
        }

    async def get_exec_ids(
        self,
        account_id: str,
        start_date: Optional[datetime] = None,
    ) -> List[str]:
        """
        Get all execution IDs for an account.

        Used to check for duplicates before inserting.

        Args:
            account_id: IB account ID.
            start_date: Optional start date filter.

        Returns:
            List of execution IDs.
        """
        if start_date:
            query = """
                SELECT exec_id FROM ib_raw_executions
                WHERE account_id = $1 AND exec_time >= $2
            """
            records = await self._db.fetch(query, account_id, start_date)
        else:
            query = """
                SELECT exec_id FROM ib_raw_executions
                WHERE account_id = $1
            """
            records = await self._db.fetch(query, account_id)

        return [r["exec_id"] for r in records]

    # -------------------------------------------------------------------------
    # Conversion from IB API
    # -------------------------------------------------------------------------

    @classmethod
    def from_ib_execution(
        cls,
        execution: Any,
        contract: Any,
        account_id: str,
    ) -> IbRawExecution:
        """
        Convert IB API execution and contract to IbRawExecution entity.

        Args:
            execution: ib_async Execution object.
            contract: ib_async Contract object.
            account_id: IB account ID.

        Returns:
            IbRawExecution entity.
        """
        # Build raw data dict for preservation
        raw_data = {
            "execId": execution.execId,
            "orderId": execution.orderId,
            "permId": execution.permId,
            "clientId": execution.clientId,
            "acctNumber": execution.acctNumber,
            "side": execution.side,
            "shares": float(execution.shares),
            "price": float(execution.price),
            "cumQty": float(execution.cumQty) if execution.cumQty else None,
            "avgPrice": float(execution.avgPrice) if execution.avgPrice else None,
            "liquidation": execution.liquidation,
            "modelCode": execution.modelCode,
            "lastLiquidity": execution.lastLiquidity,
            "time": str(execution.time),
            "contract": {
                "symbol": contract.symbol,
                "secType": contract.secType,
                "exchange": contract.exchange,
                "currency": contract.currency,
                "lastTradeDateOrContractMonth": contract.lastTradeDateOrContractMonth,
                "strike": float(contract.strike) if contract.strike else None,
                "right": contract.right,
            },
        }

        return IbRawExecution(
            exec_id=execution.execId,
            order_id=execution.orderId,
            perm_id=execution.permId,
            account_id=account_id,
            client_id=execution.clientId,
            symbol=contract.symbol,
            sec_type=contract.secType,
            exchange=contract.exchange,
            currency=contract.currency,
            expiry=contract.lastTradeDateOrContractMonth or None,
            strike=Decimal(str(contract.strike)) if contract.strike else None,
            right=contract.right if contract.right else None,
            side=execution.side,
            shares=Decimal(str(execution.shares)),
            price=Decimal(str(execution.price)),
            cum_qty=Decimal(str(execution.cumQty)) if execution.cumQty else None,
            avg_price=Decimal(str(execution.avgPrice)) if execution.avgPrice else None,
            liquidation=execution.liquidation or 0,
            model_code=execution.modelCode,
            last_liquidity=execution.lastLiquidity,
            exec_time=parse_ib_timestamp(execution.time),
            raw_data=raw_data,
        )
