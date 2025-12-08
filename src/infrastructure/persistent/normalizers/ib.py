"""
IB normalizer for converting raw IB API/Flex data to unified schema.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .base import BaseNormalizer


class IbNormalizer(BaseNormalizer):
    """Normalizer for Interactive Brokers API and Flex report data."""

    @property
    def broker_name(self) -> str:
        return "IB"

    def normalize_order(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw IB order record to normalized schema.

        Handles both API orders and Flex report orders.

        Args:
            raw: Raw order dict from IB API or Flex report.
            account_id: Account identifier.

        Returns:
            Normalized order dict, or None if invalid.
        """
        # Try to get order ID from various sources
        order_id = self._get_order_id(raw)
        if not order_id:
            return None

        # Determine source (API or FLEX)
        source = raw.get("source", "API")

        # Parse instrument details
        symbol = raw.get("symbol", "")
        underlying = raw.get("underlying") or raw.get("underlyingSymbol") or symbol
        sec_type = raw.get("sec_type") or raw.get("assetCategory") or "STK"

        # Parse option details
        strike = self.safe_float(raw.get("strike"))
        expiry = self._parse_expiry(raw.get("expiry"))
        right = self._parse_right(raw.get("right") or raw.get("putCall"))

        # Parse timestamps
        create_time = self._parse_ib_datetime(raw.get("create_time") or raw.get("dateTime"))
        update_time = self._parse_ib_datetime(raw.get("update_time") or raw.get("lastExecutionTime"))

        # Map side and status
        side = self._normalize_ib_side(raw.get("side") or raw.get("buySell"))
        status = self._normalize_ib_status(raw.get("status"))

        return {
            "broker": self.broker_name,
            "account_id": account_id,
            "order_uid": self.generate_order_uid(account_id, order_id),
            "instrument_type": self.normalize_instrument_type(sec_type),
            "symbol": symbol,
            "underlying": underlying,
            "exchange": raw.get("exchange"),
            "strike": strike,
            "expiry": datetime.strptime(expiry, "%Y%m%d").date() if expiry else None,
            "option_right": "CALL" if right == "C" else "PUT" if right == "P" else None,
            "side": side,
            "qty": self.safe_float(raw.get("qty") or raw.get("quantity")),
            "limit_price": self.safe_float(raw.get("limit_price") or raw.get("limitPrice")),
            "order_type": self._normalize_ib_order_type(raw.get("order_type") or raw.get("orderType")),
            "time_in_force": raw.get("time_in_force") or raw.get("timeInForce"),
            "status": status,
            "filled_qty": self.safe_float(raw.get("filled_qty") or raw.get("filledQuantity")),
            "avg_fill_price": self.safe_float(raw.get("avg_fill_price") or raw.get("avgPrice")),
            "create_time_utc": create_time,
            "update_time_utc": update_time,
            "order_reconstructed": source == "FLEX",
            "raw_ref": {
                "table": "orders_raw_ib",
                "account": account_id,
                "order_id": order_id,
                "source": source,
            },
        }

    def normalize_trade(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw IB trade (execution) record to normalized schema.

        Handles both API executions and Flex report trades.

        Args:
            raw: Raw trade dict from IB API or Flex report.
            account_id: Account identifier.

        Returns:
            Normalized trade dict, or None if invalid.
        """
        # Get execution ID
        exec_id = raw.get("exec_id") or raw.get("trade_id") or raw.get("execId")
        if not exec_id:
            return None

        exec_id = str(exec_id)

        # Determine source
        source = raw.get("source", "API")

        # Parse instrument details
        symbol = raw.get("symbol", "")
        underlying = raw.get("underlying") or raw.get("underlyingSymbol") or symbol
        sec_type = raw.get("sec_type") or raw.get("assetCategory") or "STK"

        # Parse option details
        strike = self.safe_float(raw.get("strike"))
        expiry = self._parse_expiry(raw.get("expiry"))
        right = self._parse_right(raw.get("right") or raw.get("putCall"))

        # Parse trade time
        trade_time = self._parse_ib_datetime(
            raw.get("trade_time") or raw.get("time") or raw.get("dateTime")
        )
        if not trade_time:
            trade_time = datetime.utcnow()

        # Get order reference
        order_id = raw.get("order_id") or raw.get("orderID")
        perm_id = raw.get("perm_id") or raw.get("permId") or raw.get("ibOrderID")

        # Use perm_id as order link if order_id not available
        order_ref = order_id or (str(perm_id) if perm_id else None)

        # Map side
        side = self._normalize_ib_side(raw.get("side") or raw.get("buySell"))

        return {
            "broker": self.broker_name,
            "account_id": account_id,
            "trade_uid": self.generate_trade_uid(account_id, exec_id),
            "order_uid": self.generate_order_uid(account_id, order_ref) if order_ref else None,
            "instrument_type": self.normalize_instrument_type(sec_type),
            "symbol": symbol,
            "underlying": underlying,
            "strike": strike,
            "expiry": datetime.strptime(expiry, "%Y%m%d").date() if expiry else None,
            "option_right": "CALL" if right == "C" else "PUT" if right == "P" else None,
            "side": side,
            "qty": abs(self.safe_float(raw.get("qty") or raw.get("quantity") or raw.get("shares"))),
            "price": self.safe_float(raw.get("price") or raw.get("tradePrice")),
            "exchange": raw.get("exchange"),
            "trade_time_utc": trade_time,
            "raw_ref": {
                "table": "trades_raw_ib",
                "account": account_id,
                "exec_id": exec_id,
                "source": source,
            },
        }

    def normalize_fee(self, raw: Dict[str, Any], account_id: str) -> List[Dict[str, Any]]:
        """
        Convert a raw IB fee record to normalized schema.

        IB provides commission at execution level.

        Args:
            raw: Raw fee dict from IB API or Flex report.
            account_id: Account identifier.

        Returns:
            List of normalized fee dicts.
        """
        exec_id = raw.get("exec_id") or raw.get("execId")
        if not exec_id:
            return []

        exec_id = str(exec_id)
        commission = self.safe_float(raw.get("commission") or raw.get("ibCommission"))

        if commission == 0:
            return []

        fee_uid = self.generate_fee_uid(account_id, exec_id)
        source = raw.get("source", "API")

        fees = [{
            "broker": self.broker_name,
            "account_id": account_id,
            "fee_uid": fee_uid,
            "order_uid": None,  # IB fees are at execution level
            "trade_uid": self.generate_trade_uid(account_id, exec_id),
            "fee_type": "COMMISSION",
            "amount": abs(commission),
            "currency": raw.get("currency", "USD"),
            "raw_ref": {
                "table": "fees_raw_ib",
                "account": account_id,
                "exec_id": exec_id,
                "source": source,
            },
        }]

        # Check for additional fee components from Flex reports
        # Flex reports may include SEC fees, exchange fees, etc.
        sec_fee = self.safe_float(raw.get("secFee"))
        if sec_fee > 0:
            fees.append({
                "broker": self.broker_name,
                "account_id": account_id,
                "fee_uid": f"{fee_uid}_SEC",
                "order_uid": None,
                "trade_uid": self.generate_trade_uid(account_id, exec_id),
                "fee_type": "SEC",
                "amount": abs(sec_fee),
                "currency": raw.get("currency", "USD"),
                "raw_ref": {
                    "table": "fees_raw_ib",
                    "account": account_id,
                    "exec_id": exec_id,
                    "source": source,
                },
            })

        exchange_fee = self.safe_float(raw.get("exchangeFee"))
        if exchange_fee > 0:
            fees.append({
                "broker": self.broker_name,
                "account_id": account_id,
                "fee_uid": f"{fee_uid}_EXCH",
                "order_uid": None,
                "trade_uid": self.generate_trade_uid(account_id, exec_id),
                "fee_type": "EXCHANGE",
                "amount": abs(exchange_fee),
                "currency": raw.get("currency", "USD"),
                "raw_ref": {
                    "table": "fees_raw_ib",
                    "account": account_id,
                    "exec_id": exec_id,
                    "source": source,
                },
            })

        return fees

    def normalize_flex_trade(self, flex_trade: Any, account_id: str) -> Optional[Dict[str, Any]]:
        """
        Normalize a FlexTrade object from flex_parser.

        Args:
            flex_trade: FlexTrade dataclass instance.
            account_id: Account identifier.

        Returns:
            Normalized trade dict.
        """
        raw = {
            "exec_id": flex_trade.trade_id,
            "order_id": flex_trade.order_id,
            "perm_id": flex_trade.perm_id,
            "symbol": flex_trade.symbol,
            "underlying": flex_trade.underlying,
            "sec_type": flex_trade.sec_type,
            "side": flex_trade.side,
            "qty": flex_trade.qty,
            "price": flex_trade.price,
            "trade_time": flex_trade.trade_time,
            "exchange": flex_trade.exchange,
            "strike": flex_trade.strike,
            "expiry": flex_trade.expiry,
            "right": flex_trade.right,
            "source": "FLEX",
        }
        return self.normalize_trade(raw, account_id)

    def normalize_flex_order(self, flex_order: Any, account_id: str) -> Optional[Dict[str, Any]]:
        """
        Normalize a FlexOrder object from flex_parser.

        Args:
            flex_order: FlexOrder dataclass instance.
            account_id: Account identifier.

        Returns:
            Normalized order dict.
        """
        raw = {
            "order_id": flex_order.order_id,
            "perm_id": flex_order.perm_id,
            "symbol": flex_order.symbol,
            "underlying": flex_order.underlying,
            "sec_type": flex_order.sec_type,
            "side": flex_order.side,
            "order_type": flex_order.order_type,
            "qty": flex_order.qty,
            "limit_price": flex_order.limit_price,
            "filled_qty": flex_order.filled_qty,
            "avg_fill_price": flex_order.avg_fill_price,
            "status": flex_order.status,
            "create_time": flex_order.create_time,
            "update_time": flex_order.fill_time,
            "strike": flex_order.strike,
            "expiry": flex_order.expiry,
            "right": flex_order.right,
            "source": "FLEX",
        }
        return self.normalize_order(raw, account_id)

    def normalize_flex_fee(self, flex_trade: Any, account_id: str) -> List[Dict[str, Any]]:
        """
        Extract and normalize fees from a FlexTrade object.

        Args:
            flex_trade: FlexTrade dataclass instance.
            account_id: Account identifier.

        Returns:
            List of normalized fee dicts.
        """
        raw = {
            "exec_id": flex_trade.trade_id,
            "commission": flex_trade.commission,
            "currency": flex_trade.currency,
            "source": "FLEX",
        }

        # Check raw_data for additional fee details
        if flex_trade.raw_data:
            raw["secFee"] = flex_trade.raw_data.get("secFee")
            raw["exchangeFee"] = flex_trade.raw_data.get("exchangeFee")

        return self.normalize_fee(raw, account_id)

    def _get_order_id(self, raw: Dict[str, Any]) -> Optional[str]:
        """Extract order ID from raw data."""
        order_id = raw.get("order_id") or raw.get("orderID")
        perm_id = raw.get("perm_id") or raw.get("permId") or raw.get("ibOrderID")
        client_order_id = raw.get("client_order_id") or raw.get("clientId")

        # Prefer order_id, then perm_id, then client_order_id
        if order_id:
            return str(order_id)
        if perm_id:
            return f"perm_{perm_id}"
        if client_order_id:
            return f"client_{client_order_id}"

        return None

    def _parse_ib_datetime(self, value: Any) -> Optional[datetime]:
        """Parse IB datetime from various formats."""
        if value is None:
            return None

        if isinstance(value, datetime):
            # Assume IB times are in US/Eastern, convert to UTC
            if value.tzinfo is None:
                eastern = ZoneInfo("America/New_York")
                value = value.replace(tzinfo=eastern)
            return value.astimezone(ZoneInfo("UTC"))

        if isinstance(value, str):
            # Try various IB formats
            formats = [
                "%Y%m%d;%H%M%S",
                "%Y%m%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y%m%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    # Assume US/Eastern timezone
                    eastern = ZoneInfo("America/New_York")
                    dt = dt.replace(tzinfo=eastern)
                    return dt.astimezone(ZoneInfo("UTC"))
                except ValueError:
                    continue

        return None

    def _parse_expiry(self, value: Any) -> Optional[str]:
        """Parse expiry date to YYYYMMDD format."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.strftime("%Y%m%d")

        if isinstance(value, str):
            # Already in YYYYMMDD format
            if len(value) == 8 and value.isdigit():
                return value
            # Try to parse various formats
            for fmt in ["%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"]:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.strftime("%Y%m%d")
                except ValueError:
                    continue

        return str(value) if value else None

    def _parse_right(self, value: Any) -> Optional[str]:
        """Parse option right to C/P."""
        if value is None:
            return None

        value_str = str(value).upper()
        if value_str in ("C", "CALL"):
            return "C"
        if value_str in ("P", "PUT"):
            return "P"
        return None

    def _normalize_ib_side(self, side: Any) -> str:
        """Map IB trade side to normalized side."""
        if side is None:
            return "UNKNOWN"

        side_str = str(side).upper()
        if side_str in ("BUY", "BOT", "B"):
            return "BUY"
        if side_str in ("SELL", "SLD", "S"):
            return "SELL"
        return side_str

    def _normalize_ib_status(self, status: Any) -> str:
        """Map IB order status to normalized status."""
        if status is None:
            return "UNKNOWN"

        status_str = str(status).upper()

        mapping = {
            "FILLED": "FILLED",
            "CANCELLED": "CANCELLED",
            "CANCELED": "CANCELLED",
            "PENDINGSUBMIT": "PENDING",
            "PENDINGCANCEL": "CANCELLED",
            "PRESUBMITTED": "PENDING",
            "SUBMITTED": "WORKING",
            "APIPENDING": "PENDING",
            "APICANCELLED": "CANCELLED",
            "INACTIVE": "CANCELLED",
            "REJECTED": "REJECTED",
        }

        return mapping.get(status_str, status_str)

    def _normalize_ib_order_type(self, order_type: Any) -> str:
        """Map IB order type to normalized order type."""
        if order_type is None:
            return "UNKNOWN"

        type_str = str(order_type).upper()

        mapping = {
            "LMT": "LIMIT",
            "LIMIT": "LIMIT",
            "MKT": "MARKET",
            "MARKET": "MARKET",
            "STP": "STOP",
            "STOP": "STOP",
            "STP LMT": "STOP_LIMIT",
            "STOP_LIMIT": "STOP_LIMIT",
            "MIT": "MIT",
            "LIT": "LIT",
            "MOC": "MOC",
            "LOC": "LOC",
        }

        return mapping.get(type_str, type_str)
