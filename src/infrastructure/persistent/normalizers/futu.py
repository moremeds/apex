"""
Futu normalizer for converting raw Futu API data to unified schema.
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseNormalizer


class FutuNormalizer(BaseNormalizer):
    """Normalizer for Futu OpenD API data."""

    @property
    def broker_name(self) -> str:
        return "FUTU"

    def normalize_order(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw Futu order record to normalized schema.

        Args:
            raw: Raw order dict from Futu history_order_list_query.
            account_id: Account identifier.

        Returns:
            Normalized order dict, or None if invalid.
        """
        order_id = str(raw.get("order_id", ""))
        if not order_id:
            return None

        # Parse security code
        code = raw.get("code", "")
        asset_type, symbol, underlying, expiry, strike, right = self._parse_futu_code(code)

        # Detect market from code prefix
        market = self.detect_market_from_symbol(code)

        # Parse timestamps (Futu uses "updated_time" not "update_time")
        create_time = self.parse_timestamp(raw.get("create_time"), market)
        update_time = self.parse_timestamp(raw.get("updated_time"), market)

        # Get raw values for reference
        trd_side_raw = raw.get("trd_side")
        order_type_raw = raw.get("order_type")
        order_status_raw = raw.get("order_status")

        # Map Futu order status
        status = self._normalize_futu_status(order_status_raw)

        # Map Futu trade side
        side = self._normalize_futu_side(trd_side_raw)

        return {
            "broker": self.broker_name,
            "account_id": account_id,
            "order_uid": self.generate_order_uid(account_id, order_id),
            "instrument_type": asset_type,
            "symbol": symbol,
            "stock_name": raw.get("stock_name"),
            "underlying": underlying,
            "exchange": market,
            "strike": strike,
            "expiry": datetime.strptime(expiry, "%Y%m%d").date() if expiry else None,
            "option_right": "CALL" if right == "C" else "PUT" if right == "P" else None,
            "side": side,
            "trd_side": str(trd_side_raw) if trd_side_raw else None,
            "qty": self.safe_float(raw.get("qty")),
            "limit_price": self.safe_float(raw.get("price")) if raw.get("price") else None,
            "order_type": self._normalize_futu_order_type(order_type_raw),
            "order_type_raw": str(order_type_raw) if order_type_raw else None,
            "time_in_force": str(raw.get("time_in_force")) if raw.get("time_in_force") else None,
            # Stop/Trailing order fields
            "aux_price": self.safe_float(raw.get("aux_price")),
            "trail_type": str(raw.get("trail_type")) if raw.get("trail_type") else None,
            "trail_value": self.safe_float(raw.get("trail_value")),
            "trail_spread": self.safe_float(raw.get("trail_spread")),
            # Order options
            "fill_outside_rth": raw.get("fill_outside_rth"),
            # Status
            "status": status,
            "status_raw": str(order_status_raw) if order_status_raw else None,
            "filled_qty": self.safe_float(raw.get("dealt_qty")),
            "avg_fill_price": self.safe_float(raw.get("dealt_avg_price")) if raw.get("dealt_avg_price") else None,
            # Error/Remarks
            "last_err_msg": raw.get("last_err_msg"),
            "remark": raw.get("remark"),
            # Timestamps (UTC)
            "create_time_utc": create_time,
            "update_time_utc": update_time,
            "order_reconstructed": False,
            "raw_ref": {
                "table": "orders_raw_futu",
                "acc_id": account_id,
                "order_id": order_id,
            },
        }

    def normalize_trade(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw Futu deal record to normalized schema.

        Args:
            raw: Raw deal dict from Futu history_deal_list_query.
            account_id: Account identifier.

        Returns:
            Normalized trade dict, or None if invalid.
        """
        deal_id = str(raw.get("deal_id", ""))
        if not deal_id:
            return None

        # Parse security code
        code = raw.get("code", "")
        asset_type, symbol, underlying, expiry, strike, right = self._parse_futu_code(code)

        # Detect market from code prefix
        market = self.detect_market_from_symbol(code)

        # Parse trade time
        trade_time = self.parse_timestamp(raw.get("create_time"), market)
        if not trade_time:
            # Fallback to current time if parsing fails
            trade_time = datetime.utcnow()

        # Parse update time
        update_time = self.parse_timestamp(raw.get("updated_time"), market)

        # Map Futu trade side and determine position effect
        trd_side_raw = raw.get("trd_side")
        side = self._normalize_futu_side(trd_side_raw)
        position_effect = self._determine_position_effect(trd_side_raw)

        # Get realized PnL if available (Futu may have this in some responses)
        realized_pnl = self.safe_float(raw.get("realized_pl")) if raw.get("realized_pl") else None

        # Get order_id for linking
        order_id = str(raw.get("order_id", "")) if raw.get("order_id") else None

        return {
            "broker": self.broker_name,
            "account_id": account_id,
            "trade_uid": self.generate_trade_uid(account_id, deal_id),
            "order_uid": self.generate_order_uid(account_id, order_id) if order_id else None,
            "instrument_type": asset_type,
            "symbol": symbol,
            "stock_name": raw.get("stock_name"),
            "underlying": underlying,
            "strike": strike,
            "expiry": datetime.strptime(expiry, "%Y%m%d").date() if expiry else None,
            "option_right": "CALL" if right == "C" else "PUT" if right == "P" else None,
            "side": side,
            "trd_side": str(trd_side_raw) if trd_side_raw else None,
            "qty": self.safe_float(raw.get("qty")),
            "price": self.safe_float(raw.get("price")),
            "exchange": market,
            "status": str(raw.get("status")) if raw.get("status") else None,
            # Counter broker info
            "counter_broker_id": str(raw.get("counter_broker_id")) if raw.get("counter_broker_id") else None,
            "counter_broker_name": raw.get("counter_broker_name"),
            # Position effect and PnL
            "position_effect": position_effect,
            "realized_pnl": realized_pnl,
            # Timestamps (UTC)
            "trade_time_utc": trade_time,
            "update_time_utc": update_time,
            "raw_ref": {
                "table": "trades_raw_futu",
                "acc_id": account_id,
                "deal_id": deal_id,
            },
        }

    def _determine_position_effect(self, trd_side: Any) -> str:
        """
        Determine position effect from Futu trd_side.

        Futu trd_side values:
        - BUY: Buy to open (long)
        - SELL: Sell to close or sell to open - context dependent
        - BUY_BACK: Buy to close (close short)
        - SELL_SHORT: Sell to open (short)
        """
        if trd_side is None:
            return "OPEN"

        side_str = str(trd_side).upper()

        # Explicit closing actions
        if side_str in ("BUY_BACK", "TRDSIDE_BUY_BACK"):
            return "CLOSE"

        # For options, SELL is typically closing (selling to close)
        # For stocks, SELL could be either - default to CLOSE as it's more common
        if side_str in ("SELL", "TRDSIDE_SELL"):
            return "CLOSE"

        # Opening actions
        if side_str in ("SELL_SHORT", "TRDSIDE_SELL_SHORT"):
            return "OPEN"

        # Default BUY is opening
        return "OPEN"

    def normalize_fee(self, raw: Dict[str, Any], account_id: str) -> List[Dict[str, Any]]:
        """
        Convert a raw Futu fee record to normalized schema.

        Futu returns fee breakdown in fee_list field.

        Args:
            raw: Raw fee dict from Futu order_fee_query.
            account_id: Account identifier.

        Returns:
            List of normalized fee dicts (one per fee type).
        """
        order_id = str(raw.get("order_id", ""))
        if not order_id:
            return []

        fees = []
        fee_uid = self.generate_fee_uid(account_id, order_id)

        # Check for fee breakdown in fee_list
        fee_list = raw.get("fee_list", [])

        if fee_list and isinstance(fee_list, list):
            # Futu returns fee_list as list of [name, amount] pairs
            for idx, fee_item in enumerate(fee_list):
                if isinstance(fee_item, (list, tuple)) and len(fee_item) >= 2:
                    fee_name = str(fee_item[0])
                    fee_amount = self.safe_float(fee_item[1])

                    fee_type = self._map_futu_fee_type(fee_name)

                    fees.append({
                        "broker": self.broker_name,
                        "account_id": account_id,
                        "fee_uid": f"{fee_uid}_{idx}",
                        "order_uid": self.generate_order_uid(account_id, order_id),
                        "trade_uid": None,
                        "fee_type": fee_type,
                        "amount": fee_amount,
                        "currency": "USD",
                        "raw_ref": {
                            "table": "fees_raw_futu",
                            "acc_id": account_id,
                            "order_id": order_id,
                        },
                    })
        else:
            # Fallback: use total fee_amount as COMMISSION
            total_fee = self.safe_float(raw.get("fee_amount", 0))
            if total_fee > 0:
                fees.append({
                    "broker": self.broker_name,
                    "account_id": account_id,
                    "fee_uid": fee_uid,
                    "order_uid": self.generate_order_uid(account_id, order_id),
                    "trade_uid": None,
                    "fee_type": "COMMISSION",
                    "amount": total_fee,
                    "currency": "USD",
                    "raw_ref": {
                        "table": "fees_raw_futu",
                        "acc_id": account_id,
                        "order_id": order_id,
                    },
                })

        return fees

    def _parse_futu_code(
        self,
        code: str,
    ) -> Tuple[str, str, str, Optional[str], Optional[float], Optional[str]]:
        """
        Parse Futu security code to extract asset details.

        Futu code formats:
        - Stock: "US.AAPL", "HK.00700"
        - Option: "US.AAPL240119C190000" (underlying + YYMMDD + C/P + strike*1000)

        Returns:
            Tuple of (asset_type, symbol, underlying, expiry, strike, right)
        """
        # Remove market prefix (e.g., "US.", "HK.")
        if "." in code:
            market, ticker = code.split(".", 1)
        else:
            ticker = code

        # Check if it's an option
        option_pattern = r"^([A-Z]+)(\d{6})([CP])(\d+)$"
        match = re.match(option_pattern, ticker)

        if match:
            underlying = match.group(1)
            date_str = match.group(2)  # YYMMDD
            right = match.group(3)  # C or P
            strike_raw = match.group(4)

            # Convert YYMMDD to YYYYMMDD
            year = int(date_str[:2])
            year_full = 2000 + year if year < 50 else 1900 + year
            expiry = f"{year_full}{date_str[2:]}"

            # Strike is stored as strike * 1000
            strike = float(strike_raw) / 1000.0

            return ("OPTION", ticker, underlying, expiry, strike, right)
        else:
            return ("STOCK", ticker, ticker, None, None, None)

    def _normalize_futu_status(self, status: Any) -> str:
        """Map Futu order status to normalized status."""
        if status is None:
            return "UNKNOWN"

        status_str = str(status).upper()

        # Handle both enum name and string value
        mapping = {
            # OrderStatus enum names
            "FILLED_ALL": "FILLED",
            "FILLED_PART": "PARTIALLY_FILLED",
            "CANCELLED_ALL": "CANCELLED",
            "CANCELLED_PART": "CANCELLED",
            "SUBMITTED": "WORKING",
            "SUBMITTING": "WORKING",
            "WAITING_SUBMIT": "PENDING",
            "SUBMIT_FAILED": "REJECTED",
            "TIMEOUT": "EXPIRED",
            "FAILED": "REJECTED",
            "DISABLED": "CANCELLED",
            "DELETED": "CANCELLED",
            # Common string values
            "FILLED": "FILLED",
            "CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
            "PENDING": "PENDING",
        }

        return mapping.get(status_str, status_str)

    def _normalize_futu_side(self, side: Any) -> str:
        """Map Futu trade side to normalized side."""
        if side is None:
            return "UNKNOWN"

        side_str = str(side).upper()

        mapping = {
            "BUY": "BUY",
            "SELL": "SELL",
            "BUY_BACK": "BUY",
            "SELL_SHORT": "SELL",
            # TrdSide enum values
            "TRDSIDE_BUY": "BUY",
            "TRDSIDE_SELL": "SELL",
        }

        return mapping.get(side_str, side_str)

    def _normalize_futu_order_type(self, order_type: Any) -> str:
        """Map Futu order type to normalized order type."""
        if order_type is None:
            return "UNKNOWN"

        type_str = str(order_type).upper()

        mapping = {
            "NORMAL": "LIMIT",
            "MARKET": "MARKET",
            "ABSOLUTE_LIMIT": "LIMIT",
            "AUCTION": "MARKET",
            "AUCTION_LIMIT": "LIMIT",
            "SPECIAL_LIMIT": "LIMIT",
            "SPECIAL_LIMIT_ALL": "LIMIT",
            # Common string values
            "LIMIT": "LIMIT",
            "LMT": "LIMIT",
            "MKT": "MARKET",
            "STOP": "STOP",
            "STOP_LIMIT": "STOP_LIMIT",
        }

        return mapping.get(type_str, type_str)

    def _map_futu_fee_type(self, fee_name: str) -> str:
        """Map Futu fee name to normalized fee type."""
        name_lower = fee_name.lower()

        if "commission" in name_lower:
            return "COMMISSION"
        elif "platform" in name_lower:
            return "PLATFORM"
        elif "exchange" in name_lower:
            return "EXCHANGE"
        elif "sec" in name_lower:
            return "SEC"
        elif "taf" in name_lower:
            return "TAF"
        elif "finra" in name_lower:
            return "FINRA"
        elif "clearing" in name_lower:
            return "CLEARING"
        else:
            return "OTHER"
