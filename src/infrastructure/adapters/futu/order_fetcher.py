"""
Futu order and trade fetching.

Extracted from FutuAdapter for single-responsibility.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List

from ....models.order import Order, Trade
from ....utils.logging_setup import get_logger
from .converters import (
    build_trade_from_order,
    convert_order,
    convert_trade,
    convert_trade_with_fee,
)

if TYPE_CHECKING:
    from .adapter import FutuAdapter

logger = get_logger(__name__)


class OrderFetcher:
    """
    Handles order and trade fetching from Futu OpenD.

    Order fetching is less rate-limited than position/account queries,
    but still uses connection management and retry logic.
    """

    def __init__(self, adapter: "FutuAdapter"):
        """
        Initialize order fetcher.

        Args:
            adapter: Parent FutuAdapter for connection management.
        """
        self._adapter = adapter

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """
        Fetch order history from Futu OpenD.

        Args:
            include_open: Include currently open orders.
            include_completed: Include completed (filled/cancelled) orders.
            days_back: Number of days to look back for completed orders.

        Returns:
            List of Order objects.
        """
        await self._adapter._ensure_connected()

        from futu import RET_OK, TrdEnv

        orders = []
        trd_env_enum = getattr(TrdEnv, self._adapter.trd_env, TrdEnv.REAL)

        try:
            if include_open:
                ret, data = await self._adapter._run_blocking(
                    self._adapter._trd_ctx.order_list_query,
                    trd_env=trd_env_enum,
                    acc_id=self._adapter._acc_id,
                    refresh_cache=False,
                )
                if ret == RET_OK and not data.empty:
                    for _, row in data.iterrows():
                        order = convert_order(row, self._adapter._acc_id)
                        if order:
                            orders.append(order)
                    logger.debug(f"Fetched {len(data)} orders from Futu")

            if include_completed:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)

                ret, data = await self._adapter._run_blocking(
                    self._adapter._trd_ctx.history_order_list_query,
                    trd_env=trd_env_enum,
                    acc_id=self._adapter._acc_id,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                if ret == RET_OK and not data.empty:
                    for _, row in data.iterrows():
                        order = convert_order(row, self._adapter._acc_id)
                        if order:
                            if not any(o.order_id == order.order_id for o in orders):
                                orders.append(order)
                    logger.debug("Fetched historical orders from Futu")

            logger.info(f"Fetched {len(orders)} total orders from Futu")

        except Exception as e:
            logger.error(f"Failed to fetch orders from Futu: {e}")
            raise

        return orders

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """
        Fetch trade/execution history from Futu OpenD.

        Uses a two-step approach:
        1. Fetch filled orders via history_order_list_query with fees from order_fee_query
        2. Validate against deal_list_query to ensure no trades are missing

        Args:
            days_back: Number of days to look back.

        Returns:
            List of Trade objects.
        """
        await self._adapter._ensure_connected()

        try:
            orders_with_fees = await self._fetch_filled_orders_with_fees(days_back)
            trades_from_orders = self._build_trades_from_orders(orders_with_fees)
            deals = await self._fetch_deals(days_back)
            trades = self._validate_and_merge_trades(trades_from_orders, deals)

            logger.info(f"Fetched {len(trades)} total trades from Futu (last {days_back} days)")
            return trades

        except Exception as e:
            logger.error(f"Failed to fetch trades from Futu: {e}")
            raise

    async def _fetch_filled_orders_with_fees(self, days_back: int) -> List[Dict]:
        """Fetch filled orders and their associated fees."""
        from futu import RET_OK
        from futu import OrderStatus as FutuOrderStatus
        from futu import TrdEnv

        trd_env_enum = getattr(TrdEnv, self._adapter.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        orders_with_fees = []

        ret, data = await self._adapter._run_blocking(
            self._adapter._trd_ctx.history_order_list_query,
            trd_env=trd_env_enum,
            acc_id=self._adapter._acc_id,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            status_filter_list=[FutuOrderStatus.FILLED_ALL],
        )

        if ret != RET_OK:
            logger.warning(f"Failed to fetch filled orders: {data}")
            return []

        if data.empty:
            logger.debug("No filled orders found")
            return []

        logger.debug(f"Fetched {len(data)} filled orders from Futu history")

        order_ids = data["order_id"].astype(str).tolist()

        fees_by_order: Dict[str, float] = {}
        for i in range(0, len(order_ids), 400):
            batch_ids = order_ids[i : i + 400]
            ret_fee, fee_data = await self._adapter._run_blocking(
                self._adapter._trd_ctx.order_fee_query,
                order_id_list=batch_ids,
                trd_env=trd_env_enum,
                acc_id=self._adapter._acc_id,
            )
            if ret_fee == RET_OK and not fee_data.empty:
                for _, fee_row in fee_data.iterrows():
                    oid = str(fee_row.get("order_id", ""))
                    fee_amount = float(fee_row.get("fee_amount", 0) or 0)
                    fees_by_order[oid] = fee_amount
                logger.debug(f"Fetched fees for {len(fee_data)} orders")
            else:
                logger.warning(f"Failed to fetch order fees: {fee_data}")

        for _, row in data.iterrows():
            order_id = str(row.get("order_id", ""))
            order_dict = row.to_dict()
            order_dict["fee_amount"] = fees_by_order.get(order_id, 0.0)
            orders_with_fees.append(order_dict)

        return orders_with_fees

    async def _fetch_deals(self, days_back: int) -> List[Dict]:
        """Fetch deals (executions) from Futu."""
        from futu import RET_OK, TrdEnv

        trd_env_enum = getattr(TrdEnv, self._adapter.trd_env, TrdEnv.REAL)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        deals = []
        seen_deal_ids = set()

        # Current day deals
        ret, data = await self._adapter._run_blocking(
            self._adapter._trd_ctx.deal_list_query,
            trd_env=trd_env_enum,
            acc_id=self._adapter._acc_id,
            refresh_cache=False,
        )
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                deal_id = str(row.get("deal_id", ""))
                if deal_id and deal_id not in seen_deal_ids:
                    deals.append(row.to_dict())
                    seen_deal_ids.add(deal_id)
            logger.debug(f"Fetched {len(data)} deals (today) from Futu")

        # Historical deals
        ret, data = await self._adapter._run_blocking(
            self._adapter._trd_ctx.history_deal_list_query,
            trd_env=trd_env_enum,
            acc_id=self._adapter._acc_id,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        if ret == RET_OK and not data.empty:
            for _, row in data.iterrows():
                deal_id = str(row.get("deal_id", ""))
                if deal_id and deal_id not in seen_deal_ids:
                    deals.append(row.to_dict())
                    seen_deal_ids.add(deal_id)
            logger.debug(f"Fetched {len(data)} historical deals from Futu")

        return deals

    def _build_trades_from_orders(self, orders_with_fees: List[Dict]) -> Dict[str, Trade]:
        """Build Trade objects from filled orders with fees."""
        trades_by_order_id: Dict[str, Trade] = {}

        for order in orders_with_fees:
            trade = build_trade_from_order(order, self._adapter._acc_id)
            if trade:
                trades_by_order_id[trade.order_id] = trade

        return trades_by_order_id

    def _validate_and_merge_trades(
        self,
        trades_from_orders: Dict[str, Trade],
        deals: List[Dict],
    ) -> List[Trade]:
        """Validate trades from orders against deals and merge."""
        final_trades: List[Trade] = []
        seen_deal_ids: set = set()
        orders_with_deals: set = set()

        # Group deals by order
        deals_by_order: Dict[str, List[Dict]] = {}
        for deal in deals:
            order_id = str(deal.get("order_id", ""))
            if order_id:
                if order_id not in deals_by_order:
                    deals_by_order[order_id] = []
                deals_by_order[order_id].append(deal)

        # Process deals grouped by order
        for order_id, order_deals in deals_by_order.items():
            order_trade = trades_from_orders.get(order_id)
            total_fee = order_trade.commission if order_trade else 0.0
            fee_per_deal = total_fee / len(order_deals) if order_deals else 0.0

            for deal in order_deals:
                deal_id = str(deal.get("deal_id", ""))
                if deal_id in seen_deal_ids:
                    continue
                seen_deal_ids.add(deal_id)
                orders_with_deals.add(order_id)

                trade = convert_trade_with_fee(deal, fee_per_deal, self._adapter._acc_id)
                if trade:
                    final_trades.append(trade)

        # Add orders without deals (edge case)
        for order_id, order_trade in trades_from_orders.items():
            if order_id not in orders_with_deals:
                logger.warning(
                    f"Filled order {order_id} has no corresponding deals - "
                    f"using order data: {order_trade.symbol} qty={order_trade.quantity}"
                )
                final_trades.append(order_trade)

        # Add orphan deals (deals without matching orders)
        for deal in deals:
            order_id = str(deal.get("order_id", ""))
            deal_id = str(deal.get("deal_id", ""))
            if order_id and order_id not in trades_from_orders and deal_id not in seen_deal_ids:
                logger.warning(
                    f"Deal {deal_id} has no corresponding filled order - "
                    f"order_id={order_id}, adding without fee"
                )
                trade = convert_trade(deal, self._adapter._acc_id)
                if trade:
                    final_trades.append(trade)
                    seen_deal_ids.add(deal_id)

        return final_trades
