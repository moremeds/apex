"""
P&L Calculator - Pure functions for profit/loss calculations.

Provides unrealized, daily, and intraday P&L calculations with data quality tracking.
All results are immutable frozen dataclasses for thread safety.

Usage:
    result = calculate_pnl(
        mark=155.0,
        avg_cost=150.0,
        yesterday_close=154.0,
        session_open=153.0,
        quantity=100,
        multiplier=1,
    )
    print(result.unrealized)  # 500.0
    print(result.daily)       # 100.0
    print(result.intraday)    # 200.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DataQuality(Enum):
    """Quality classification for market data ticks."""

    GOOD = "good"  # Normal, reliable data
    STALE = "stale"  # Data hasn't updated recently
    SUSPICIOUS = "suspicious"  # Unusual values (wide spread, outlier)
    ZERO_QUOTE = "zero_quote"  # Missing bid/ask


@dataclass(frozen=True, slots=True)
class PnLResult:
    """
    Immutable P&L calculation result.

    Thread-safe due to frozen=True. Use slots=True for memory efficiency
    in high-frequency streaming scenarios.

    Attributes:
        unrealized: Total unrealized P&L since position opened.
        daily: P&L since yesterday's close.
        intraday: P&L since today's session open.
        is_reliable: False if calculated from low-quality data.
        mark_price: The mark price used for calculation.
    """

    unrealized: float
    daily: float
    intraday: float
    is_reliable: bool
    mark_price: float


def calculate_pnl(
    mark: float,
    avg_cost: float,
    yesterday_close: Optional[float],
    session_open: Optional[float],
    quantity: float,
    multiplier: int,
    data_quality: DataQuality = DataQuality.GOOD,
) -> PnLResult:
    """
    Calculate all P&L metrics for a position.

    Pure function: no side effects, deterministic output.

    Args:
        mark: Current mark price (mid or last).
        avg_cost: Average cost basis per share/contract.
        yesterday_close: Previous day's closing price (None if unavailable).
        session_open: Today's session opening price (None if unavailable).
        quantity: Position size (positive for long, negative for short).
        multiplier: Contract multiplier (1 for stocks, 100 for options).
        data_quality: Quality classification of the input tick.

    Returns:
        PnLResult with all P&L metrics and reliability flag.

    Example:
        >>> result = calculate_pnl(
        ...     mark=155.0,
        ...     avg_cost=150.0,
        ...     yesterday_close=154.0,
        ...     session_open=153.0,
        ...     quantity=100,
        ...     multiplier=1,
        ... )
        >>> result.unrealized
        500.0
        >>> result.daily
        100.0
    """
    qty_mult = quantity * multiplier

    # Unrealized P&L: always calculable if we have mark and avg_cost
    unrealized = (mark - avg_cost) * qty_mult

    # Daily P&L: requires yesterday's close
    daily = 0.0
    if yesterday_close is not None and yesterday_close > 0:
        daily = (mark - yesterday_close) * qty_mult

    # Intraday P&L: requires session open price
    intraday = 0.0
    if session_open is not None and session_open > 0:
        intraday = (mark - session_open) * qty_mult

    # Reliability based on data quality
    is_reliable = data_quality not in (
        DataQuality.SUSPICIOUS,
        DataQuality.STALE,
        DataQuality.ZERO_QUOTE,
    )

    return PnLResult(
        unrealized=unrealized,
        daily=daily,
        intraday=intraday,
        is_reliable=is_reliable,
        mark_price=mark,
    )


def calculate_pnl_delta(
    old_result: PnLResult,
    new_result: PnLResult,
) -> PnLResult:
    """
    Calculate the change in P&L between two states.

    Used for streaming delta updates to avoid recalculating absolute values.

    Args:
        old_result: Previous P&L state.
        new_result: Current P&L state.

    Returns:
        PnLResult with delta values (new - old).
    """
    return PnLResult(
        unrealized=new_result.unrealized - old_result.unrealized,
        daily=new_result.daily - old_result.daily,
        intraday=new_result.intraday - old_result.intraday,
        is_reliable=new_result.is_reliable,
        mark_price=new_result.mark_price,
    )
