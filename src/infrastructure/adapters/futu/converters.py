"""
Futu data converters.

Converts Futu API responses to internal domain models.
All timestamps are stored as UTC for internal consistency.
Futu returns timestamps in US Eastern time for US market.
"""

from __future__ import annotations
from typing import Optional, Dict, Union
from datetime import datetime
import logging

from ....models.position import Position, AssetType, PositionSource
from ....models.order import Order, Trade, OrderSource, OrderStatus, OrderSide, OrderType
from ....utils.timezone import now_utc, parse_futu_timestamp
from .code_parser import parse_futu_code


logger = logging.getLogger(__name__)


def convert_position(row, acc_id: Optional[int] = None) -> Optional[Position]:
    """
    Convert Futu position row to internal Position model.

    Args:
        row: pandas DataFrame row from position_list_query.
        acc_id: Account ID for the position.

    Returns:
        Position object or None if conversion fails.
    """
    try:
        code = row.get("code", "")
        qty = float(row.get("qty", 0))

        if qty == 0:
            return None

        # Parse the Futu code format
        asset_type, symbol, underlying, expiry, strike, right = parse_futu_code(code)

        # Get cost and market values
        avg_price = float(row.get("cost_price", 0) or row.get("average_cost", 0) or 0)

        return Position(
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            quantity=qty,
            avg_price=avg_price,
            multiplier=100 if asset_type == AssetType.OPTION else 1,
            expiry=expiry,
            strike=strike,
            right=right,
            source=PositionSource.FUTU,
            last_updated=now_utc(),
            account_id=str(acc_id) if acc_id else None,
        )

    except Exception as e:
        logger.warning(f"Failed to convert Futu position: {e}, row={row.to_dict()}")
        return None


def convert_order(row, acc_id: Optional[int] = None) -> Optional[Order]:
    """
    Convert Futu order row to internal Order model.

    Args:
        row: pandas DataFrame row from order_list_query.
        acc_id: Account ID for the order.

    Returns:
        Order object or None if conversion fails.
    """
    try:
        code = row.get("code", "")
        order_id = str(row.get("order_id", ""))

        # Parse the Futu code format
        asset_type_enum, symbol, underlying, expiry, strike, right = parse_futu_code(code)
        asset_type = asset_type_enum.value if asset_type_enum else "STOCK"

        # Map Futu order status
        futu_status = row.get("order_status", "")
        status_map = {
            "UNSUBMITTED": OrderStatus.PENDING,
            "WAITING_SUBMIT": OrderStatus.PENDING,
            "SUBMITTING": OrderStatus.PENDING,
            "SUBMIT_FAILED": OrderStatus.REJECTED,
            "SUBMITTED": OrderStatus.SUBMITTED,
            "FILLED_PART": OrderStatus.PARTIALLY_FILLED,
            "FILLED_ALL": OrderStatus.FILLED,
            "CANCELLING_PART": OrderStatus.PARTIALLY_FILLED,
            "CANCELLING_ALL": OrderStatus.SUBMITTED,
            "CANCELLED_PART": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED_ALL": OrderStatus.CANCELLED,
            "FAILED": OrderStatus.REJECTED,
            "DISABLED": OrderStatus.REJECTED,
            "DELETED": OrderStatus.CANCELLED,
        }
        status = status_map.get(str(futu_status), OrderStatus.PENDING)

        # Map Futu order type
        futu_order_type = row.get("order_type", "")
        order_type_map = {
            "NORMAL": OrderType.LIMIT,
            "MARKET": OrderType.MARKET,
            "ABSOLUTE_LIMIT": OrderType.LIMIT,
            "AUCTION": OrderType.MARKET,
            "AUCTION_LIMIT": OrderType.LIMIT,
            "SPECIAL_LIMIT": OrderType.LIMIT,
            "SPECIAL_LIMIT_ALL": OrderType.LIMIT,
        }
        order_type = order_type_map.get(str(futu_order_type), OrderType.LIMIT)

        # Map Futu trade side
        trd_side = row.get("trd_side", "")
        side = OrderSide.BUY if str(trd_side) in ("BUY", "BUY_BACK") else OrderSide.SELL

        # Parse timestamps (Futu returns EST for US market) -> convert to UTC
        create_time = parse_futu_timestamp(row.get("create_time"))
        updated_time = parse_futu_timestamp(row.get("updated_time"))

        # Use updated_time as filled_time for filled orders
        filled_time = updated_time if status == OrderStatus.FILLED else None

        return Order(
            order_id=order_id,
            source=OrderSource.FUTU,
            account_id=str(acc_id) if acc_id else "",
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            side=side,
            order_type=order_type,
            quantity=float(row.get("qty", 0)),
            limit_price=float(row.get("price", 0)) if row.get("price") else None,
            status=status,
            filled_quantity=float(row.get("dealt_qty", 0) or 0),
            avg_fill_price=float(row.get("dealt_avg_price", 0)) if row.get("dealt_avg_price") else None,
            created_time=create_time,
            filled_time=filled_time,
            updated_time=updated_time or now_utc(),
            expiry=expiry,
            strike=strike,
            right=right,
            exchange=row.get("exchange", None),
            time_in_force=str(row.get("time_in_force", "")) if row.get("time_in_force") else None,
        )

    except Exception as e:
        logger.warning(f"Failed to convert Futu order: {e}, row={row.to_dict() if hasattr(row, 'to_dict') else row}")
        return None


def convert_trade(row: Union[Dict, object], acc_id: Optional[int] = None) -> Optional[Trade]:
    """
    Convert Futu trade row to internal Trade model.

    Note: Futu SDK calls trades "deals" - this converts from their format.

    Args:
        row: pandas DataFrame row or dict from deal_list_query.
        acc_id: Account ID for the trade.

    Returns:
        Trade object or None if conversion fails.
    """
    try:
        # Handle both dict and pandas row
        if hasattr(row, 'get'):
            get_val = row.get
        else:
            get_val = lambda k, d=None: getattr(row, k, d)

        code = get_val("code", "")
        trade_id = str(get_val("deal_id", ""))  # Futu calls it deal_id
        order_id = str(get_val("order_id", ""))

        # Parse the Futu code format
        asset_type_enum, symbol, underlying, expiry, strike, right = parse_futu_code(code)
        asset_type = asset_type_enum.value if asset_type_enum else "STOCK"

        # Map Futu trade side
        trd_side = get_val("trd_side", "")
        side = OrderSide.BUY if str(trd_side) in ("BUY", "BUY_BACK") else OrderSide.SELL

        # Parse trade time (Futu returns EST for US market) -> convert to UTC
        trade_time = parse_futu_timestamp(get_val("create_time")) or now_utc()

        return Trade(
            trade_id=trade_id,
            order_id=order_id,
            source=OrderSource.FUTU,
            account_id=str(acc_id) if acc_id else "",
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            side=side,
            quantity=float(get_val("qty", 0)),
            price=float(get_val("price", 0)),
            commission=0.0,  # Futu doesn't return commission in trade list
            trade_time=trade_time,
            expiry=expiry,
            strike=strike,
            right=right,
        )

    except Exception as e:
        row_dict = row.to_dict() if hasattr(row, 'to_dict') else row
        logger.warning(f"Failed to convert Futu trade: {e}, row={row_dict}")
        return None


def convert_trade_with_fee(row: Dict, commission: float, acc_id: Optional[int] = None) -> Optional[Trade]:
    """
    Convert Futu deal dict to Trade model with commission.

    Args:
        row: Deal dict from deal_list_query
        commission: Commission amount for this trade
        acc_id: Account ID for the trade.

    Returns:
        Trade object or None if conversion fails
    """
    try:
        code = row.get("code", "")
        trade_id = str(row.get("deal_id", ""))
        order_id = str(row.get("order_id", ""))

        # Parse the Futu code format
        asset_type_enum, symbol, underlying, expiry, strike, right = parse_futu_code(code)
        asset_type = asset_type_enum.value if asset_type_enum else "STOCK"

        # Map trade side
        trd_side = row.get("trd_side", "")
        side = OrderSide.BUY if str(trd_side) in ("BUY", "BUY_BACK") else OrderSide.SELL

        # Parse trade time (Futu returns EST for US market) -> convert to UTC
        trade_time = parse_futu_timestamp(row.get("create_time")) or now_utc()

        return Trade(
            trade_id=trade_id,
            order_id=order_id,
            source=OrderSource.FUTU,
            account_id=str(acc_id) if acc_id else "",
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            side=side,
            quantity=float(row.get("qty", 0)),
            price=float(row.get("price", 0)),
            commission=commission,
            trade_time=trade_time,
            expiry=expiry,
            strike=strike,
            right=right,
        )

    except Exception as e:
        logger.warning(f"Failed to convert Futu trade with fee: {e}, row={row}")
        return None


def build_trade_from_order(order: Dict, acc_id: Optional[int] = None) -> Optional[Trade]:
    """
    Build a Trade object from a filled order with fees.

    For filled orders, we create a synthetic trade using the order's
    dealt_qty and dealt_avg_price. The fee is included from order_fee_query.
    Uses updated_time as the fill time (trade_time).

    Args:
        order: Order dict with fee_amount merged
        acc_id: Account ID for the trade.

    Returns:
        Trade object or None if conversion fails
    """
    try:
        code = order.get("code", "")
        order_id = str(order.get("order_id", ""))
        dealt_qty = float(order.get("dealt_qty", 0) or 0)
        dealt_avg_price = float(order.get("dealt_avg_price", 0) or 0)
        fee_amount = float(order.get("fee_amount", 0) or 0)

        if dealt_qty == 0:
            return None

        # Parse the Futu code format
        asset_type_enum, symbol, underlying, expiry, strike, right = parse_futu_code(code)
        asset_type = asset_type_enum.value if asset_type_enum else "STOCK"

        # Map trade side
        trd_side = order.get("trd_side", "")
        side = OrderSide.BUY if str(trd_side) in ("BUY", "BUY_BACK") else OrderSide.SELL

        # Use updated_time as fill time (Futu returns EST for US market) -> convert to UTC
        trade_time = parse_futu_timestamp(order.get("updated_time")) or now_utc()

        # Create trade with order_id as trade_id (synthetic ID)
        return Trade(
            trade_id=f"order_{order_id}",
            order_id=order_id,
            source=OrderSource.FUTU,
            account_id=str(acc_id) if acc_id else "",
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            side=side,
            quantity=dealt_qty,
            price=dealt_avg_price,
            commission=fee_amount,
            trade_time=trade_time,
            expiry=expiry,
            strike=strike,
            right=right,
        )

    except Exception as e:
        logger.warning(f"Failed to build trade from order: {e}, order={order}")
        return None
