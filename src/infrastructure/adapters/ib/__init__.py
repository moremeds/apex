"""
Interactive Brokers adapter package.

Provides IbAdapter for connecting to IB TWS/Gateway.
"""

from .adapter import IbAdapter
from .flex_parser import FlexParser, FlexTrade, FlexOrder, flex_trade_to_dict, flex_order_to_dict

__all__ = [
    "IbAdapter",
    "FlexParser",
    "FlexTrade",
    "FlexOrder",
    "flex_trade_to_dict",
    "flex_order_to_dict",
]
