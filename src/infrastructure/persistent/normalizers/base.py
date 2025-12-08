"""
Base normalizer interface for converting raw broker data to unified schema.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


# Exchange timezone mappings
EXCHANGE_TZ = {
    "US": "America/New_York",
    "HK": "Asia/Hong_Kong",
    "CN": "Asia/Shanghai",
    "SG": "Asia/Singapore",
    "JP": "Asia/Tokyo",
}


class BaseNormalizer(ABC):
    """Abstract base class for broker normalizers."""

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return the broker identifier (e.g., 'FUTU', 'IB')."""
        pass

    @abstractmethod
    def normalize_order(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw order record to normalized schema.

        Args:
            raw: Raw order dict from broker API.
            account_id: Account identifier.

        Returns:
            Normalized order dict, or None if invalid.
        """
        pass

    @abstractmethod
    def normalize_trade(self, raw: Dict[str, Any], account_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert a raw trade record to normalized schema.

        Args:
            raw: Raw trade dict from broker API.
            account_id: Account identifier.

        Returns:
            Normalized trade dict, or None if invalid.
        """
        pass

    @abstractmethod
    def normalize_fee(self, raw: Dict[str, Any], account_id: str) -> List[Dict[str, Any]]:
        """
        Convert a raw fee record to normalized schema.

        Args:
            raw: Raw fee dict from broker API.
            account_id: Account identifier.

        Returns:
            List of normalized fee dicts (may have multiple fee types).
        """
        pass

    def generate_order_uid(self, account_id: str, order_id: str) -> str:
        """Generate unified order ID."""
        return f"{self.broker_name}_{account_id}_{order_id}"

    def generate_trade_uid(self, account_id: str, trade_id: str) -> str:
        """Generate unified trade ID."""
        return f"{self.broker_name}_{account_id}_{trade_id}"

    def generate_fee_uid(self, account_id: str, reference_id: str) -> str:
        """Generate unified fee ID."""
        return f"{self.broker_name}_{account_id}_{reference_id}"

    def parse_timestamp(
        self,
        raw: Optional[str],
        market: str = "US",
        fmt: str = "%Y-%m-%d %H:%M:%S.%f",
    ) -> Optional[datetime]:
        """
        Parse broker timestamp string to timezone-aware datetime.

        Args:
            raw: Raw timestamp string.
            market: Market code for timezone lookup.
            fmt: Datetime format string.

        Returns:
            Timezone-aware datetime in UTC, or None if parsing fails.
        """
        if not raw:
            return None

        try:
            tz = ZoneInfo(EXCHANGE_TZ.get(market, "UTC"))

            # Try with milliseconds first
            try:
                dt = datetime.strptime(raw, fmt)
            except ValueError:
                # Try without milliseconds
                dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")

            # Attach local timezone and convert to UTC
            dt_local = dt.replace(tzinfo=tz)
            return dt_local.astimezone(ZoneInfo("UTC"))

        except Exception:
            return None

    def detect_market_from_symbol(self, symbol: str) -> str:
        """
        Detect market from symbol prefix.

        Args:
            symbol: Symbol string (e.g., "US.AAPL", "HK.00700").

        Returns:
            Market code (US, HK, CN, etc.).
        """
        if "." in symbol:
            prefix = symbol.split(".")[0].upper()
            if prefix in EXCHANGE_TZ:
                return prefix
        return "US"

    def parse_option_symbol(self, symbol: str) -> Tuple[str, Optional[float], Optional[str], Optional[str]]:
        """
        Parse option symbol to extract components.

        Args:
            symbol: Option symbol (various formats).

        Returns:
            Tuple of (underlying, strike, expiry, right).
            Returns (symbol, None, None, None) if not an option.
        """
        # This is a basic implementation - subclasses should override
        # for broker-specific option symbol formats
        return (symbol, None, None, None)

    def normalize_side(self, side: Any) -> str:
        """Normalize order/trade side to BUY/SELL."""
        side_str = str(side).upper()
        if side_str in ("BUY", "LONG", "B"):
            return "BUY"
        elif side_str in ("SELL", "SHORT", "S"):
            return "SELL"
        return side_str

    def normalize_order_type(self, order_type: Any) -> str:
        """Normalize order type."""
        type_str = str(order_type).upper()
        mapping = {
            "MARKET": "MARKET",
            "MKT": "MARKET",
            "LIMIT": "LIMIT",
            "LMT": "LIMIT",
            "STOP": "STOP",
            "STP": "STOP",
            "STOP_LIMIT": "STOP_LIMIT",
            "STP_LMT": "STOP_LIMIT",
        }
        return mapping.get(type_str, type_str)

    def normalize_status(self, status: Any) -> str:
        """Normalize order status."""
        status_str = str(status).upper()
        mapping = {
            "FILLED": "FILLED",
            "FILLED_ALL": "FILLED",
            "FILLED_PART": "PARTIALLY_FILLED",
            "PARTIALLY_FILLED": "PARTIALLY_FILLED",
            "CANCELLED": "CANCELLED",
            "CANCELED": "CANCELLED",
            "CANCELLED_ALL": "CANCELLED",
            "CANCELLED_PART": "CANCELLED",
            "SUBMITTED": "WORKING",
            "SUBMITTING": "WORKING",
            "WAITING_SUBMIT": "PENDING",
            "PENDING": "PENDING",
            "REJECTED": "REJECTED",
            "FAILED": "REJECTED",
            "EXPIRED": "EXPIRED",
        }
        return mapping.get(status_str, status_str)

    def normalize_instrument_type(self, asset_type: Any) -> str:
        """Normalize instrument type."""
        type_str = str(asset_type).upper()
        mapping = {
            "STOCK": "STOCK",
            "STK": "STOCK",
            "OPTION": "OPTION",
            "OPT": "OPTION",
            "FUTURE": "FUTURE",
            "FUT": "FUTURE",
            "FOREX": "FOREX",
            "FX": "FOREX",
            "CASH": "FOREX",
            "WARRANT": "WARRANT",
            "BOND": "BOND",
        }
        return mapping.get(type_str, type_str)

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        if value is None:
            return default
        try:
            f = float(value)
            if f != f:  # NaN check
                return default
            return f
        except (ValueError, TypeError):
            return default
