"""
Futu trade push notification handler.

Handles real-time trade notifications from Futu OpenD.
"""

from __future__ import annotations
from typing import Callable
import threading

from ....utils.logging_setup import get_logger


logger = get_logger(__name__)


def create_trade_handler(on_trade_callback: Callable[[dict], None]):
    """
    Factory function to create a Futu trade handler.

    Creates a handler class that inherits from Futu's TradeDealHandlerBase
    to receive real-time trade notifications (executions/fills).

    Note: Futu SDK calls executions "Deals", but we use "Trade" for consistency.

    Args:
        on_trade_callback: Callback function to invoke when a trade is received.

    Returns:
        Handler instance or None if futu library not available.
    """
    try:
        from futu import TradeDealHandlerBase

        class FutuTradeHandler(TradeDealHandlerBase):
            """
            Handler for Futu trade push notifications.

            Inherits from Futu SDK's TradeDealHandlerBase to receive
            real-time notifications when trades (executions/fills) occur.
            """

            def __init__(self, callback: Callable[[dict], None]):
                super().__init__()
                self._callback = callback
                self._lock = threading.Lock()

            def on_recv_rsp(self, rsp_str):
                """
                Called by Futu SDK when a trade notification is received.

                Args:
                    rsp_str: Response string/data from Futu SDK containing trade info.
                """
                try:
                    import pandas as pd

                    # rsp_str is typically a tuple (ret_code, data) from Futu SDK
                    # where data is a DataFrame with trade information
                    if isinstance(rsp_str, tuple) and len(rsp_str) >= 2:
                        ret_code, data = rsp_str[0], rsp_str[1]
                        if ret_code != 0:
                            logger.warning(f"Futu trade notification error: {data}")
                            return

                        # data is a pandas DataFrame with trade info
                        if isinstance(data, pd.DataFrame) and not data.empty:
                            for _, row in data.iterrows():
                                trade_data = {
                                    'trade_id': row.get('deal_id', None),  # Futu calls it deal_id
                                    'order_id': row.get('order_id', None),
                                    'code': row.get('code', None),
                                    'stock_name': row.get('stock_name', None),
                                    'qty': float(row.get('qty', 0)),
                                    'price': float(row.get('price', 0)),
                                    'trd_side': row.get('trd_side', None),
                                    'create_time': row.get('create_time', None),
                                }
                                logger.info(
                                    f"Futu trade received: {trade_data['code']} "
                                    f"qty={trade_data['qty']} price={trade_data['price']} "
                                    f"side={trade_data['trd_side']}"
                                )

                                with self._lock:
                                    if self._callback:
                                        self._callback(trade_data)

                except Exception as e:
                    logger.error(f"Error processing Futu trade notification: {e}")

        return FutuTradeHandler(on_trade_callback)

    except ImportError:
        logger.warning("futu-api library not installed, trade handler unavailable")
        return None
