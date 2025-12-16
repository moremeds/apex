"""
Repository for Futu raw fees persistence.

Handles UPSERT operations for fees from Futu's order_fee_query() API.
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

logger = logging.getLogger(__name__)


@dataclass
class FutuRawFee:
    """Futu raw fee entity."""

    order_id: str
    account_id: str
    fee_amount: Decimal
    commission: Decimal
    platform_fee: Decimal
    settlement_fee: Decimal
    # US market fees
    sec_fee: Decimal
    taf_fee: Decimal
    orf_fee: Decimal
    occ_fee: Decimal
    # HK market fees
    stamp_duty: Decimal
    trading_fee: Decimal
    transaction_levy: Decimal
    sfc_levy: Decimal
    frc_levy: Decimal
    fee_details: Optional[Dict[str, Any]] = None
    loaded_at: Optional[datetime] = None
    id: Optional[int] = None


class FutuFeeRepository(BaseRepository[FutuRawFee]):
    """
    Repository for Futu raw fees.

    Handles persistence of fee data from Futu's order_fee_query() API.
    Uses UPSERT pattern with (order_id, account_id) as the conflict key.
    """

    def __init__(self, db: Database):
        super().__init__(db)

    @property
    def table_name(self) -> str:
        return "futu_raw_fees"

    @property
    def conflict_columns(self) -> List[str]:
        return ["order_id", "account_id"]

    def _to_entity(self, record: Record) -> FutuRawFee:
        """Convert database record to FutuRawFee entity."""
        return FutuRawFee(
            id=record["id"],
            order_id=record["order_id"],
            account_id=record["account_id"],
            fee_amount=record["fee_amount"],
            commission=record["commission"],
            platform_fee=record["platform_fee"],
            settlement_fee=record["settlement_fee"],
            sec_fee=record["sec_fee"],
            taf_fee=record["taf_fee"],
            orf_fee=record["orf_fee"],
            occ_fee=record["occ_fee"],
            stamp_duty=record["stamp_duty"],
            trading_fee=record["trading_fee"],
            transaction_levy=record["transaction_levy"],
            sfc_levy=record["sfc_levy"],
            frc_levy=record["frc_levy"],
            fee_details=self._from_json(record["fee_details"]),
            loaded_at=record["loaded_at"],
        )

    def _to_row(self, entity: FutuRawFee) -> Dict[str, Any]:
        """Convert FutuRawFee entity to database row."""
        return {
            "order_id": entity.order_id,
            "account_id": entity.account_id,
            "fee_amount": entity.fee_amount,
            "commission": entity.commission,
            "platform_fee": entity.platform_fee,
            "settlement_fee": entity.settlement_fee,
            "sec_fee": entity.sec_fee,
            "taf_fee": entity.taf_fee,
            "orf_fee": entity.orf_fee,
            "occ_fee": entity.occ_fee,
            "stamp_duty": entity.stamp_duty,
            "trading_fee": entity.trading_fee,
            "transaction_levy": entity.transaction_levy,
            "sfc_levy": entity.sfc_levy,
            "frc_levy": entity.frc_levy,
            "fee_details": self._to_json(entity.fee_details),
        }

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def find_by_order_id(
        self, order_id: str, account_id: str
    ) -> Optional[FutuRawFee]:
        """
        Find fee record by order_id and account_id.

        Args:
            order_id: Futu order ID.
            account_id: Futu account ID.

        Returns:
            FutuRawFee if found, None otherwise.
        """
        query = """
            SELECT * FROM futu_raw_fees
            WHERE order_id = $1 AND account_id = $2
        """
        record = await self._db.fetchrow(query, order_id, account_id)
        return self._to_entity(record) if record else None

    async def find_by_account(
        self, account_id: str, limit: int = 1000
    ) -> List[FutuRawFee]:
        """
        Find fee records by account.

        Args:
            account_id: Futu account ID.
            limit: Maximum number of records.

        Returns:
            List of fee records.
        """
        query = """
            SELECT * FROM futu_raw_fees
            WHERE account_id = $1
            ORDER BY loaded_at DESC
            LIMIT $2
        """
        records = await self._db.fetch(query, account_id, limit)
        return [self._to_entity(r) for r in records]

    async def find_by_order_ids(
        self, order_ids: List[str], account_id: str
    ) -> List[FutuRawFee]:
        """
        Find fee records for multiple orders.

        Args:
            order_ids: List of Futu order IDs.
            account_id: Futu account ID.

        Returns:
            List of fee records.
        """
        if not order_ids:
            return []

        query = """
            SELECT * FROM futu_raw_fees
            WHERE account_id = $1 AND order_id = ANY($2)
        """
        records = await self._db.fetch(query, account_id, order_ids)
        return [self._to_entity(r) for r in records]

    async def get_total_fees(
        self,
        account_id: str,
        order_ids: Optional[List[str]] = None,
    ) -> Dict[str, Decimal]:
        """
        Calculate total fees.

        Args:
            account_id: Futu account ID.
            order_ids: Optional list of order IDs to filter.

        Returns:
            Dictionary with fee totals by category.
        """
        if order_ids:
            query = """
                SELECT
                    COALESCE(SUM(fee_amount), 0) as total_fees,
                    COALESCE(SUM(commission), 0) as total_commission,
                    COALESCE(SUM(platform_fee), 0) as total_platform_fee,
                    COALESCE(SUM(settlement_fee), 0) as total_settlement_fee,
                    COALESCE(SUM(sec_fee), 0) as total_sec_fee,
                    COALESCE(SUM(taf_fee), 0) as total_taf_fee,
                    COALESCE(SUM(stamp_duty), 0) as total_stamp_duty
                FROM futu_raw_fees
                WHERE account_id = $1 AND order_id = ANY($2)
            """
            record = await self._db.fetchrow(query, account_id, order_ids)
        else:
            query = """
                SELECT
                    COALESCE(SUM(fee_amount), 0) as total_fees,
                    COALESCE(SUM(commission), 0) as total_commission,
                    COALESCE(SUM(platform_fee), 0) as total_platform_fee,
                    COALESCE(SUM(settlement_fee), 0) as total_settlement_fee,
                    COALESCE(SUM(sec_fee), 0) as total_sec_fee,
                    COALESCE(SUM(taf_fee), 0) as total_taf_fee,
                    COALESCE(SUM(stamp_duty), 0) as total_stamp_duty
                FROM futu_raw_fees
                WHERE account_id = $1
            """
            record = await self._db.fetchrow(query, account_id)

        return {
            "total_fees": record["total_fees"] or Decimal(0),
            "total_commission": record["total_commission"] or Decimal(0),
            "total_platform_fee": record["total_platform_fee"] or Decimal(0),
            "total_settlement_fee": record["total_settlement_fee"] or Decimal(0),
            "total_sec_fee": record["total_sec_fee"] or Decimal(0),
            "total_taf_fee": record["total_taf_fee"] or Decimal(0),
            "total_stamp_duty": record["total_stamp_duty"] or Decimal(0),
        }

    async def get_missing_fee_order_ids(
        self,
        account_id: str,
        order_ids: List[str],
    ) -> List[str]:
        """
        Find order IDs that don't have fee records.

        Used to determine which orders need fee queries.

        Args:
            account_id: Futu account ID.
            order_ids: List of order IDs to check.

        Returns:
            List of order IDs without fee records.
        """
        if not order_ids:
            return []

        query = """
            SELECT order_id FROM futu_raw_fees
            WHERE account_id = $1 AND order_id = ANY($2)
        """
        records = await self._db.fetch(query, account_id, order_ids)
        existing_ids = {r["order_id"] for r in records}

        return [oid for oid in order_ids if oid not in existing_ids]

    # -------------------------------------------------------------------------
    # Conversion from Futu API
    # -------------------------------------------------------------------------

    @classmethod
    def from_futu_fee(
        cls,
        fee_data: Dict[str, Any],
        order_id: str,
        account_id: str,
    ) -> FutuRawFee:
        """
        Convert Futu API fee data to FutuRawFee entity.

        Futu SDK returns fee_details as a list of tuples: [(title, value), ...]
        Example: [("Commission", "1.50"), ("SEC Fee", "0.02"), ...]

        This method handles both:
        - Old format: fee_list with dicts {fee_type, fee_value}
        - New format: fee_details as list of (title, value) tuples

        Args:
            fee_data: Raw fee dict from Futu API (order_fee_query result).
            order_id: Futu order ID.
            account_id: Futu account ID.

        Returns:
            FutuRawFee entity.
        """
        # Initialize all fees to 0
        fee_map = {
            "commission": Decimal(0),
            "platform_fee": Decimal(0),
            "settlement_fee": Decimal(0),
            "sec_fee": Decimal(0),
            "taf_fee": Decimal(0),
            "orf_fee": Decimal(0),
            "occ_fee": Decimal(0),
            "stamp_duty": Decimal(0),
            "trading_fee": Decimal(0),
            "transaction_levy": Decimal(0),
            "sfc_levy": Decimal(0),
            "frc_levy": Decimal(0),
        }

        # Map Futu fee names to our columns (handles both uppercase enum and display names)
        fee_name_mapping = {
            # Uppercase enum style
            "COMMISSION": "commission",
            "PLATFORM_FEE": "platform_fee",
            "SETTLEMENT_FEE": "settlement_fee",
            "SEC_FEE": "sec_fee",
            "TAF_FEE": "taf_fee",
            "ORF_FEE": "orf_fee",
            "OCC_FEE": "occ_fee",
            "STAMP_DUTY": "stamp_duty",
            "TRADING_FEE": "trading_fee",
            "TRANSACTION_LEVY": "transaction_levy",
            "SFC_LEVY": "sfc_levy",
            "FRC_LEVY": "frc_levy",
            # Display name style (from fee_details tuples)
            "Commission": "commission",
            "Platform Fee": "platform_fee",
            "Settlement Fee": "settlement_fee",
            "SEC Fee": "sec_fee",
            "TAF Fee": "taf_fee",
            "ORF Fee": "orf_fee",
            "OCC Fee": "occ_fee",
            "Stamp Duty": "stamp_duty",
            "Trading Fee": "trading_fee",
            "Transaction Levy": "transaction_levy",
            "SFC Levy": "sfc_levy",
            "FRC Levy": "frc_levy",
        }

        total_fee = Decimal(0)
        raw_fee_details = {}

        # Try to get fee_details (new SDK format - list of tuples)
        fee_details_raw = fee_data.get("fee_details", [])

        if isinstance(fee_details_raw, list) and fee_details_raw:
            # New format: [(title, value), ...]
            for item in fee_details_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    fee_name = str(item[0])
                    try:
                        # Handle N/A and empty values
                        fee_val_str = str(item[1])
                        if fee_val_str in ('N/A', '', 'None'):
                            fee_value = Decimal(0)
                        else:
                            fee_value = Decimal(fee_val_str)
                    except Exception:
                        fee_value = Decimal(0)

                    total_fee += fee_value
                    raw_fee_details[fee_name] = str(fee_value)

                    # Map to column if known name
                    if fee_name in fee_name_mapping:
                        col_name = fee_name_mapping[fee_name]
                        fee_map[col_name] = fee_value

        # Fallback: try fee_list (old format - list of dicts)
        fee_items = fee_data.get("fee_list", [])
        if fee_items and not fee_details_raw:
            for item in fee_items:
                if isinstance(item, dict):
                    fee_type = item.get("fee_type", "")
                    try:
                        fee_value = Decimal(str(item.get("fee_value", 0)))
                    except Exception:
                        fee_value = Decimal(0)

                    total_fee += fee_value
                    raw_fee_details[fee_type] = str(fee_value)

                    # Map to column if known type
                    if fee_type in fee_name_mapping:
                        col_name = fee_name_mapping[fee_type]
                        fee_map[col_name] = fee_value

        # Also check for direct fee_amount field
        if "fee_amount" in fee_data and total_fee == Decimal(0):
            try:
                fee_val_str = str(fee_data["fee_amount"])
                if fee_val_str not in ('N/A', '', 'None'):
                    total_fee = Decimal(fee_val_str)
            except Exception:
                pass

        return FutuRawFee(
            order_id=order_id,
            account_id=account_id,
            fee_amount=total_fee,
            commission=fee_map["commission"],
            platform_fee=fee_map["platform_fee"],
            settlement_fee=fee_map["settlement_fee"],
            sec_fee=fee_map["sec_fee"],
            taf_fee=fee_map["taf_fee"],
            orf_fee=fee_map["orf_fee"],
            occ_fee=fee_map["occ_fee"],
            stamp_duty=fee_map["stamp_duty"],
            trading_fee=fee_map["trading_fee"],
            transaction_levy=fee_map["transaction_levy"],
            sfc_levy=fee_map["sfc_levy"],
            frc_levy=fee_map["frc_levy"],
            fee_details=raw_fee_details if raw_fee_details else fee_data,
        )
