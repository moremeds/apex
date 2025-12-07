"""
IB Flex Report Parser for historical trade data.

Uses the ibflex library to download and parse IB Flex Query reports.
Flex reports provide full historical data that the API cannot access.

Setup in IB Account Management:
1. Performance & Reports > Flex Queries > Custom Flex Queries
2. Create query with: Trades, Orders, Transaction Fees
3. Generate API token

Dependencies:
    pip install ibflex>=0.14
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlexTrade:
    """Parsed trade from Flex report."""
    trade_id: str
    order_id: Optional[str]
    perm_id: Optional[int]
    account: str
    symbol: str
    underlying: Optional[str]
    sec_type: str  # STK, OPT, FUT
    side: str  # BUY, SELL
    qty: float
    price: float
    commission: float
    currency: str
    trade_time: datetime
    exchange: Optional[str]
    # Option-specific
    strike: Optional[float] = None
    expiry: Optional[str] = None
    right: Optional[str] = None  # C, P
    # Raw data
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class FlexOrder:
    """Parsed order from Flex report."""
    order_id: str
    perm_id: Optional[int]
    account: str
    symbol: str
    underlying: Optional[str]
    sec_type: str
    side: str
    order_type: str
    qty: float
    limit_price: Optional[float]
    filled_qty: float
    avg_fill_price: Optional[float]
    status: str
    create_time: Optional[datetime]
    fill_time: Optional[datetime]
    # Option-specific
    strike: Optional[float] = None
    expiry: Optional[str] = None
    right: Optional[str] = None
    # Raw data
    raw_data: Optional[Dict[str, Any]] = None


class FlexParser:
    """
    IB Flex Report parser.

    Downloads Flex reports using ibflex library and parses them into
    structured trade and order objects.
    """

    def __init__(self, token: str, query_id: str):
        """
        Initialize Flex parser.

        Args:
            token: Flex Web Service token from IB Account Management
            query_id: Flex Query ID
        """
        self.token = token
        self.query_id = query_id
        self._ibflex_available = self._check_ibflex()

    def _check_ibflex(self) -> bool:
        """Check if ibflex library is available."""
        try:
            import ibflex  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "ibflex library not installed. Install with: pip install ibflex>=0.14"
            )
            return False

    def fetch_and_parse(self) -> Dict[str, Any]:
        """
        Download and parse Flex report.

        Returns:
            Dict with:
                - trades: List[FlexTrade]
                - orders: List[FlexOrder]
                - raw_response: Original parsed response
        """
        if not self._ibflex_available:
            raise ImportError("ibflex library not installed")

        from ibflex import client, parser

        logger.info(f"Downloading Flex report (query_id={self.query_id})")

        try:
            # Download XML report
            xml_data = client.download(self.token, self.query_id)

            # Parse XML to Python objects
            response = parser.parse(xml_data)

            # Extract trades and orders
            trades = self._extract_trades(response)
            orders = self._extract_orders(response)

            logger.info(
                f"Parsed Flex report: {len(trades)} trades, {len(orders)} orders"
            )

            return {
                "trades": trades,
                "orders": orders,
                "raw_response": response,
            }

        except Exception as e:
            logger.error(f"Failed to fetch/parse Flex report: {e}")
            raise

    def _extract_trades(self, response) -> List[FlexTrade]:
        """Extract trades from Flex response."""
        trades = []

        for stmt in response.FlexStatements:
            account = stmt.accountId

            # Process trades (executions)
            if hasattr(stmt, 'Trades') and stmt.Trades:
                for trade in stmt.Trades:
                    try:
                        flex_trade = self._parse_trade(trade, account)
                        if flex_trade:
                            trades.append(flex_trade)
                    except Exception as e:
                        logger.warning(f"Failed to parse trade: {e}")
                        continue

        return trades

    def _extract_orders(self, response) -> List[FlexOrder]:
        """Extract orders from Flex response."""
        orders = []

        for stmt in response.FlexStatements:
            account = stmt.accountId

            # Process orders
            if hasattr(stmt, 'Orders') and stmt.Orders:
                for order in stmt.Orders:
                    try:
                        flex_order = self._parse_order(order, account)
                        if flex_order:
                            orders.append(flex_order)
                    except Exception as e:
                        logger.warning(f"Failed to parse order: {e}")
                        continue

        return orders

    def _parse_trade(self, trade, account: str) -> Optional[FlexTrade]:
        """Parse a single trade from Flex response."""
        try:
            # Extract common fields
            trade_id = str(getattr(trade, 'tradeID', '') or '')
            if not trade_id:
                return None

            # Parse datetime
            trade_time = self._parse_datetime(getattr(trade, 'dateTime', None))
            if not trade_time:
                trade_time = self._parse_datetime(getattr(trade, 'tradeDate', None))

            # Determine side
            buy_sell = getattr(trade, 'buySell', '')
            side = 'BUY' if str(buy_sell).upper() in ('BUY', 'B') else 'SELL'

            # Build raw data dict
            raw_data = {}
            for attr in dir(trade):
                if not attr.startswith('_'):
                    try:
                        raw_data[attr] = getattr(trade, attr)
                    except Exception:
                        pass

            return FlexTrade(
                trade_id=trade_id,
                order_id=str(getattr(trade, 'orderID', '') or '') or None,
                perm_id=self._safe_int(getattr(trade, 'ibOrderID', None)),
                account=account,
                symbol=str(getattr(trade, 'symbol', '')),
                underlying=str(getattr(trade, 'underlyingSymbol', '') or '') or None,
                sec_type=str(getattr(trade, 'assetCategory', 'STK')),
                side=side,
                qty=abs(float(getattr(trade, 'quantity', 0) or 0)),
                price=float(getattr(trade, 'tradePrice', 0) or 0),
                commission=abs(float(getattr(trade, 'ibCommission', 0) or 0)),
                currency=str(getattr(trade, 'currency', 'USD')),
                trade_time=trade_time or datetime.now(),
                exchange=str(getattr(trade, 'exchange', '') or '') or None,
                strike=self._safe_float(getattr(trade, 'strike', None)),
                expiry=self._parse_expiry(getattr(trade, 'expiry', None)),
                right=self._parse_right(getattr(trade, 'putCall', None)),
                raw_data=raw_data,
            )
        except Exception as e:
            logger.warning(f"Error parsing trade: {e}")
            return None

    def _parse_order(self, order, account: str) -> Optional[FlexOrder]:
        """Parse a single order from Flex response."""
        try:
            order_id = str(getattr(order, 'orderID', '') or '')
            if not order_id:
                return None

            # Parse datetime
            create_time = self._parse_datetime(getattr(order, 'dateTime', None))
            fill_time = self._parse_datetime(getattr(order, 'lastExecutionTime', None))

            # Determine side
            buy_sell = getattr(order, 'buySell', '')
            side = 'BUY' if str(buy_sell).upper() in ('BUY', 'B') else 'SELL'

            # Build raw data dict
            raw_data = {}
            for attr in dir(order):
                if not attr.startswith('_'):
                    try:
                        raw_data[attr] = getattr(order, attr)
                    except Exception:
                        pass

            return FlexOrder(
                order_id=order_id,
                perm_id=self._safe_int(getattr(order, 'ibOrderID', None)),
                account=account,
                symbol=str(getattr(order, 'symbol', '')),
                underlying=str(getattr(order, 'underlyingSymbol', '') or '') or None,
                sec_type=str(getattr(order, 'assetCategory', 'STK')),
                side=side,
                order_type=str(getattr(order, 'orderType', 'LMT')),
                qty=abs(float(getattr(order, 'quantity', 0) or 0)),
                limit_price=self._safe_float(getattr(order, 'limitPrice', None)),
                filled_qty=abs(float(getattr(order, 'filledQuantity', 0) or 0)),
                avg_fill_price=self._safe_float(getattr(order, 'avgPrice', None)),
                status=str(getattr(order, 'status', 'UNKNOWN')),
                create_time=create_time,
                fill_time=fill_time,
                strike=self._safe_float(getattr(order, 'strike', None)),
                expiry=self._parse_expiry(getattr(order, 'expiry', None)),
                right=self._parse_right(getattr(order, 'putCall', None)),
                raw_data=raw_data,
            )
        except Exception as e:
            logger.warning(f"Error parsing order: {e}")
            return None

    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse datetime from various Flex formats."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            # Try common formats
            formats = [
                "%Y%m%d;%H%M%S",
                "%Y%m%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y%m%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

        return None

    def _parse_expiry(self, value) -> Optional[str]:
        """Parse expiry date to YYYYMMDD format."""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.strftime("%Y%m%d")

        if isinstance(value, str):
            # Already in YYYYMMDD format
            if len(value) == 8 and value.isdigit():
                return value
            # Try to parse
            try:
                dt = datetime.strptime(value, "%Y-%m-%d")
                return dt.strftime("%Y%m%d")
            except ValueError:
                pass

        return str(value) if value else None

    def _parse_right(self, value) -> Optional[str]:
        """Parse option right (C/P)."""
        if value is None:
            return None

        value_str = str(value).upper()
        if value_str in ('C', 'CALL'):
            return 'C'
        if value_str in ('P', 'PUT'):
            return 'P'
        return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


def flex_trade_to_dict(trade: FlexTrade) -> Dict[str, Any]:
    """Convert FlexTrade to dict for persistence."""
    return {
        "trade_id": trade.trade_id,
        "order_id": trade.order_id,
        "perm_id": trade.perm_id,
        "account": trade.account,
        "symbol": trade.symbol,
        "underlying": trade.underlying,
        "sec_type": trade.sec_type,
        "side": trade.side,
        "qty": trade.qty,
        "price": trade.price,
        "commission": trade.commission,
        "currency": trade.currency,
        "trade_time": trade.trade_time.isoformat() if trade.trade_time else None,
        "exchange": trade.exchange,
        "strike": trade.strike,
        "expiry": trade.expiry,
        "right": trade.right,
    }


def flex_order_to_dict(order: FlexOrder) -> Dict[str, Any]:
    """Convert FlexOrder to dict for persistence."""
    return {
        "order_id": order.order_id,
        "perm_id": order.perm_id,
        "account": order.account,
        "symbol": order.symbol,
        "underlying": order.underlying,
        "sec_type": order.sec_type,
        "side": order.side,
        "order_type": order.order_type,
        "qty": order.qty,
        "limit_price": order.limit_price,
        "filled_qty": order.filled_qty,
        "avg_fill_price": order.avg_fill_price,
        "status": order.status,
        "create_time": order.create_time.isoformat() if order.create_time else None,
        "fill_time": order.fill_time.isoformat() if order.fill_time else None,
        "strike": order.strike,
        "expiry": order.expiry,
        "right": order.right,
    }
