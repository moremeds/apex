"""
Trading calendar abstractions for purge/embargo calculations.

This module provides calendar-aware day counting for Walk-Forward and CPCV splitters.
Uses pandas_market_calendars for exchange-specific trading days when available,
with a weekday-only fallback for offline or unsupported exchanges.

Usage:
    from src.backtest.data.calendar import get_calendar, TradingCalendar

    # NYSE calendar (US equities)
    nyse = get_calendar("NYSE")
    trading_days = nyse.get_trading_days("2024-01-01", "2024-12-31")

    # Weekday fallback
    weekday = get_calendar("weekday")
    days = weekday.count_trading_days("2024-01-01", "2024-01-31")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from functools import lru_cache
from typing import List, Optional, Sequence, Union

import pandas as pd

# Try to import pandas_market_calendars (optional dependency)
try:
    import pandas_market_calendars as mcal

    MCAL_AVAILABLE = True
except ImportError:
    MCAL_AVAILABLE = False
    mcal = None


class TradingCalendar(ABC):
    """
    Abstract base class for trading calendars.

    Provides trading day operations for purge/embargo calculations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Calendar name (e.g., 'NYSE', 'weekday')."""
        ...

    @abstractmethod
    def is_trading_day(self, dt: Union[date, str]) -> bool:
        """Check if a date is a trading day."""
        ...

    @abstractmethod
    def get_trading_days(
        self, start: Union[date, str], end: Union[date, str]
    ) -> List[date]:
        """Get list of trading days in date range (inclusive)."""
        ...

    def count_trading_days(
        self, start: Union[date, str], end: Union[date, str]
    ) -> int:
        """Count trading days in date range (inclusive)."""
        return len(self.get_trading_days(start, end))

    def add_trading_days(self, dt: Union[date, str], days: int) -> date:
        """
        Add/subtract trading days from a date.

        Args:
            dt: Starting date
            days: Number of trading days to add (negative to subtract)

        Returns:
            Resulting date after adding trading days
        """
        dt = self._to_date(dt)

        if days == 0:
            return dt

        step = 1 if days > 0 else -1
        remaining = abs(days)
        current = dt

        while remaining > 0:
            current = current + timedelta(days=step)
            if self.is_trading_day(current):
                remaining -= 1

        return current

    def get_trading_day_offset(self, start: Union[date, str], end: Union[date, str]) -> int:
        """
        Get number of trading days between two dates.

        Returns positive if end > start, negative if end < start.
        """
        start = self._to_date(start)
        end = self._to_date(end)

        if start == end:
            return 0

        if end > start:
            return self.count_trading_days(start, end) - 1
        else:
            return -(self.count_trading_days(end, start) - 1)

    @staticmethod
    def _to_date(dt: Union[date, str]) -> date:
        """Convert string or date to date object."""
        if isinstance(dt, str):
            return date.fromisoformat(dt)
        return dt


class WeekdayCalendar(TradingCalendar):
    """
    Simple weekday-only calendar (Mon-Fri).

    Use as fallback when exchange calendar is not available or for testing.
    Does not account for holidays.
    """

    @property
    def name(self) -> str:
        return "weekday"

    def is_trading_day(self, dt: Union[date, str]) -> bool:
        dt = self._to_date(dt)
        return dt.weekday() < 5  # Mon=0, Fri=4

    def get_trading_days(
        self, start: Union[date, str], end: Union[date, str]
    ) -> List[date]:
        start = self._to_date(start)
        end = self._to_date(end)

        if start > end:
            return []

        days = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current)
            current = current + timedelta(days=1)

        return days


class ExchangeCalendar(TradingCalendar):
    """
    Exchange-specific trading calendar using pandas_market_calendars.

    Supports major exchanges: NYSE, NASDAQ, LSE, TSX, etc.
    """

    def __init__(self, exchange: str):
        """
        Initialize exchange calendar.

        Args:
            exchange: Exchange code (e.g., 'NYSE', 'NASDAQ', 'LSE')

        Raises:
            ImportError: If pandas_market_calendars not installed
            ValueError: If exchange not supported
        """
        if not MCAL_AVAILABLE:
            raise ImportError(
                "pandas_market_calendars required for exchange calendars. "
                "Install with: pip install pandas-market-calendars"
            )

        self._exchange = exchange.upper()
        try:
            self._calendar = mcal.get_calendar(self._exchange)
        except Exception as e:
            raise ValueError(
                f"Unsupported exchange: {exchange}. "
                f"Available: {mcal.get_calendar_names()}"
            ) from e

        # Cache for trading days (date range -> set of dates)
        self._cache: dict[tuple[str, str], set[date]] = {}

    @property
    def name(self) -> str:
        return self._exchange

    def is_trading_day(self, dt: Union[date, str]) -> bool:
        dt = self._to_date(dt)
        # Use pandas Timestamp for mcal
        ts = pd.Timestamp(dt)
        schedule = self._calendar.schedule(start_date=ts, end_date=ts)
        return len(schedule) > 0

    def get_trading_days(
        self, start: Union[date, str], end: Union[date, str]
    ) -> List[date]:
        start = self._to_date(start)
        end = self._to_date(end)

        if start > end:
            return []

        # Check cache
        cache_key = (start.isoformat(), end.isoformat())
        if cache_key not in self._cache:
            # Get trading days from exchange calendar
            schedule = self._calendar.schedule(
                start_date=pd.Timestamp(start),
                end_date=pd.Timestamp(end),
            )
            trading_dates = set(schedule.index.date)
            self._cache[cache_key] = trading_dates

        return sorted(
            d for d in self._cache[cache_key] if start <= d <= end
        )


@lru_cache(maxsize=16)
def get_calendar(name: str = "NYSE") -> TradingCalendar:
    """
    Get trading calendar by name.

    Args:
        name: Calendar name. Options:
            - 'weekday': Simple Mon-Fri calendar (no holidays)
            - Exchange codes: 'NYSE', 'NASDAQ', 'LSE', 'TSX', etc.

    Returns:
        TradingCalendar instance

    Examples:
        >>> nyse = get_calendar("NYSE")
        >>> nyse.is_trading_day("2024-01-01")  # New Year's Day
        False
        >>> nyse.is_trading_day("2024-01-02")
        True
    """
    name = name.lower()

    if name == "weekday":
        return WeekdayCalendar()

    # Try exchange calendar if mcal available
    if MCAL_AVAILABLE:
        return ExchangeCalendar(name)

    # Fallback to weekday
    import warnings

    warnings.warn(
        f"pandas_market_calendars not available, using weekday calendar for '{name}'",
        stacklevel=2,
    )
    return WeekdayCalendar()


def list_available_calendars() -> List[str]:
    """List available calendar names."""
    calendars = ["weekday"]

    if MCAL_AVAILABLE:
        calendars.extend(mcal.get_calendar_names())

    return sorted(calendars)
