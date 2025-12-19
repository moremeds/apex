"""
Greeks Extractor - Pure functions for extracting market data and Greeks from IB tickers.

These are pure functions with no side effects, making them easy to test.
"""

from __future__ import annotations
from math import isnan
from typing import TYPE_CHECKING

from .....models.market_data import MarketData, GreeksSource
from .....models.position import Position, AssetType
from .....utils.timezone import now_utc

if TYPE_CHECKING:
    pass  # IB Ticker type would go here


def extract_market_data(ticker, pos: Position) -> MarketData:
    """
    Extract market data from an IB ticker object.

    Pure function - no side effects.

    Args:
        ticker: IB Ticker object with bid/ask/last/close data
        pos: Position for symbol and asset type info

    Returns:
        MarketData with prices populated
    """
    # Get prices with NaN handling
    bid = _safe_float(getattr(ticker, 'bid', None))
    ask = _safe_float(getattr(ticker, 'ask', None))
    last = _safe_float(getattr(ticker, 'last', None))
    close = _safe_float(getattr(ticker, 'close', None))

    # Calculate mid price
    mid = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    elif last is not None and last > 0:
        mid = last

    return MarketData(
        symbol=pos.symbol,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        yesterday_close=close,
        timestamp=now_utc(),
        # Greeks populated separately via extract_greeks
        delta=None,
        gamma=None,
        vega=None,
        theta=None,
        iv=None,
        underlying_price=None,
        greeks_source=GreeksSource.MISSING,
    )


def extract_greeks(ticker, md: MarketData, pos: Position) -> None:
    """
    Extract Greeks from an IB ticker's modelGreeks.

    Mutates md in place to add Greeks data.
    Only extracts Greeks for options.

    Args:
        ticker: IB Ticker object with modelGreeks
        md: MarketData object to populate with Greeks
        pos: Position for asset type checking
    """
    if pos.asset_type != AssetType.OPTION:
        # Stocks have delta of 1, no other Greeks
        md.delta = 1.0
        md.greeks_source = GreeksSource.MISSING
        return

    # Extract IV (available directly on ticker)
    if hasattr(ticker, 'impliedVolatility') and ticker.impliedVolatility:
        iv = _safe_float(ticker.impliedVolatility)
        if iv is not None:
            md.iv = iv

    # Extract Greeks from modelGreeks
    if not hasattr(ticker, 'modelGreeks') or not ticker.modelGreeks:
        return

    greeks = ticker.modelGreeks

    # Extract each Greek with NaN handling
    md.delta = _safe_float(getattr(greeks, 'delta', None))
    md.gamma = _safe_float(getattr(greeks, 'gamma', None))
    md.vega = _safe_float(getattr(greeks, 'vega', None))
    md.theta = _safe_float(getattr(greeks, 'theta', None))

    # Extract underlying price (critical for delta dollars)
    if hasattr(greeks, 'undPrice') and greeks.undPrice:
        und_price = _safe_float(greeks.undPrice)
        if und_price is not None:
            md.underlying_price = und_price

    # Mark Greeks source if we got any Greeks
    if md.delta is not None or md.gamma is not None:
        md.greeks_source = GreeksSource.IBKR


def _safe_float(value) -> float | None:
    """
    Safely convert a value to float, handling NaN and None.

    Args:
        value: Value to convert

    Returns:
        Float value or None if invalid
    """
    if value is None:
        return None
    try:
        f = float(value)
        if isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
