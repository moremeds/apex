"""
Bar Count Calculator - Calculates expected bar counts for historical data validation.

Uses pandas_market_calendars for accurate trading days and session hours,
including holidays and early close days (Christmas Eve, day before Thanksgiving, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import ClassVar, Dict, List, Optional, Tuple

import pandas as pd

try:
    import pandas_market_calendars as mcal

    MCAL_AVAILABLE = True
except ImportError:
    MCAL_AVAILABLE = False
    mcal = None

from ...utils.logging_setup import get_logger

logger = get_logger(__name__)


# Regular session: 6.5 hours (9:30 AM - 4:00 PM ET)
REGULAR_SESSION_HOURS = 6.5

# Early close session: 3.5 hours (9:30 AM - 1:00 PM ET)
EARLY_CLOSE_HOURS = 3.5


@dataclass(frozen=True)
class ExpectedBarCount:
    """Result of expected bar calculation."""

    symbol: str
    timeframe: str
    start: date
    end: date
    trading_days: int
    early_close_days: int
    expected_bars: int
    bars_per_regular_day: float
    bars_per_early_day: float
    calendar_name: str

    @property
    def total_session_hours(self) -> float:
        """Total trading hours in the period."""
        regular_days = self.trading_days - self.early_close_days
        return (regular_days * REGULAR_SESSION_HOURS) + (self.early_close_days * EARLY_CLOSE_HOURS)


@dataclass
class TradingSession:
    """Trading session for a single day."""

    date: date
    market_open: datetime
    market_close: datetime
    is_early_close: bool

    @property
    def session_hours(self) -> float:
        """Session duration in hours."""
        delta = self.market_close - self.market_open
        return delta.total_seconds() / 3600


class BarCountCalculator:
    """
    Calculates expected bar counts for symbol/timeframe/date-range combinations.

    Uses pandas_market_calendars for:
    - Accurate trading day identification (excludes holidays)
    - Per-day session hours (handles early close days)
    - Exchange-specific schedules (NYSE, NASDAQ, etc.)

    Example:
        calc = BarCountCalculator()
        result = calc.calculate("AAPL", "1d", date(2024, 1, 1), date(2024, 12, 31))
        print(f"Expected: {result.expected_bars} bars over {result.trading_days} days")
    """

    # Bars per hour for each timeframe
    BARS_PER_HOUR: ClassVar[Dict[str, float]] = {
        "1m": 60,
        "5m": 12,
        "15m": 4,
        "30m": 2,
        "1h": 1,
        "4h": 0.25,
        "1d": 0,  # Special: 1 bar per day regardless of hours
        "1w": 0,  # Special: 1 bar per week
    }

    # Bars per regular trading day (6.5 hours)
    BARS_PER_REGULAR_DAY: ClassVar[Dict[str, float]] = {
        "1m": 390,  # 6.5 * 60
        "5m": 78,  # 6.5 * 12
        "15m": 26,  # 6.5 * 4
        "30m": 13,  # 6.5 * 2
        "1h": 7,  # ceil(6.5) = 7 hourly bars
        "4h": 2,  # Only 2 complete 4-hour bars fit in 6.5 hours
        "1d": 1,
        "1w": 0.2,  # 1/5 = one bar per week
    }

    # Bars per early close day (3.5 hours)
    BARS_PER_EARLY_DAY: ClassVar[Dict[str, float]] = {
        "1m": 210,  # 3.5 * 60
        "5m": 42,  # 3.5 * 12
        "15m": 14,  # 3.5 * 4
        "30m": 7,  # 3.5 * 2
        "1h": 4,  # ceil(3.5) = 4 hourly bars
        "4h": 1,  # Only 1 complete 4-hour bar
        "1d": 1,
        "1w": 0.2,
    }

    def __init__(
        self,
        calendar_name: str = "NYSE",
    ) -> None:
        """
        Initialize calculator.

        Args:
            calendar_name: Exchange calendar name (default: NYSE).
                          Options: NYSE, NASDAQ, LSE, TSX, etc.

        Raises:
            ImportError: If pandas_market_calendars not installed.
        """
        if not MCAL_AVAILABLE:
            raise ImportError(
                "pandas_market_calendars required for BarCountCalculator. "
                "Install with: pip install pandas-market-calendars"
            )

        self._calendar_name = calendar_name.upper()
        self._calendar = mcal.get_calendar(self._calendar_name)
        logger.debug(f"BarCountCalculator initialized with {self._calendar_name} calendar")

    @property
    def calendar_name(self) -> str:
        """Get calendar name."""
        return self._calendar_name

    def calculate(
        self,
        symbol: str,
        timeframe: str,
        start: date | datetime,
        end: date | datetime,
    ) -> ExpectedBarCount:
        """
        Calculate expected bar count for a date range.

        Args:
            symbol: Ticker symbol (used for result tracking).
            timeframe: Bar timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w).
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            ExpectedBarCount with trading_days, early_close_days, and expected_bars.

        Raises:
            ValueError: If timeframe not supported.
        """
        if timeframe not in self.BARS_PER_HOUR:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(self.BARS_PER_HOUR.keys())}"
            )

        # Normalize to date
        start_date = start.date() if isinstance(start, datetime) else start
        end_date = end.date() if isinstance(end, datetime) else end

        # Get trading sessions
        sessions = self.get_trading_sessions(start_date, end_date)

        if not sessions:
            return ExpectedBarCount(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                trading_days=0,
                early_close_days=0,
                expected_bars=0,
                bars_per_regular_day=self.BARS_PER_REGULAR_DAY.get(timeframe, 0),
                bars_per_early_day=self.BARS_PER_EARLY_DAY.get(timeframe, 0),
                calendar_name=self._calendar_name,
            )

        # Count regular vs early close days
        early_close_days = sum(1 for s in sessions if s.is_early_close)
        regular_days = len(sessions) - early_close_days

        # Calculate expected bars
        if timeframe == "1d":
            expected_bars = len(sessions)
        elif timeframe == "1w":
            # Count complete weeks
            expected_bars = self._count_weeks(sessions)
        else:
            # Sum bars per day accounting for session length
            bars_per_regular = self.BARS_PER_REGULAR_DAY[timeframe]
            bars_per_early = self.BARS_PER_EARLY_DAY[timeframe]
            expected_bars = int(regular_days * bars_per_regular + early_close_days * bars_per_early)

        return ExpectedBarCount(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            trading_days=len(sessions),
            early_close_days=early_close_days,
            expected_bars=expected_bars,
            bars_per_regular_day=self.BARS_PER_REGULAR_DAY.get(timeframe, 0),
            bars_per_early_day=self.BARS_PER_EARLY_DAY.get(timeframe, 0),
            calendar_name=self._calendar_name,
        )

    def get_trading_sessions(
        self,
        start: date,
        end: date,
    ) -> List[TradingSession]:
        """
        Get trading sessions for a date range.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of TradingSession with market hours and early-close flag.
        """
        # Get schedule from pandas_market_calendars
        schedule = self._calendar.schedule(
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end),
        )

        if schedule.empty:
            return []

        sessions = []
        for idx, row in schedule.iterrows():
            trading_date = idx.date()
            market_open = row["market_open"].to_pydatetime()
            market_close = row["market_close"].to_pydatetime()

            # Calculate session duration
            session_hours = (market_close - market_open).total_seconds() / 3600

            # Early close if less than regular session (6.5 hours)
            is_early_close = session_hours < (REGULAR_SESSION_HOURS - 0.1)  # 0.1 hour tolerance

            sessions.append(
                TradingSession(
                    date=trading_date,
                    market_open=market_open,
                    market_close=market_close,
                    is_early_close=is_early_close,
                )
            )

        return sessions

    def get_trading_days(
        self,
        start: date,
        end: date,
    ) -> List[date]:
        """
        Get list of trading days in date range.

        Args:
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of trading dates.
        """
        sessions = self.get_trading_sessions(start, end)
        return [s.date for s in sessions]

    def count_trading_days(
        self,
        start: date,
        end: date,
    ) -> int:
        """Count trading days in date range."""
        return len(self.get_trading_sessions(start, end))

    def get_early_close_days(
        self,
        start: date,
        end: date,
    ) -> List[TradingSession]:
        """
        Get early close days in date range.

        Args:
            start: Start date.
            end: End date.

        Returns:
            List of TradingSession for early close days only.
        """
        sessions = self.get_trading_sessions(start, end)
        return [s for s in sessions if s.is_early_close]

    def _count_weeks(self, sessions: List[TradingSession]) -> int:
        """Count complete weeks in trading sessions."""
        if not sessions:
            return 0

        # Group by ISO week
        weeks = set()
        for session in sessions:
            iso_cal = session.date.isocalendar()
            weeks.add((iso_cal[0], iso_cal[1]))  # (year, week_number)

        return len(weeks)

    def get_bars_per_hour(self, timeframe: str) -> float:
        """Get bars per hour for a timeframe."""
        return self.BARS_PER_HOUR.get(timeframe, 0)

    def get_bars_per_day(
        self,
        timeframe: str,
        is_early_close: bool = False,
    ) -> float:
        """
        Get expected bars per trading day for timeframe.

        Args:
            timeframe: Bar timeframe.
            is_early_close: If True, return early close day count.

        Returns:
            Expected bars per day.
        """
        if is_early_close:
            return self.BARS_PER_EARLY_DAY.get(timeframe, 0)
        return self.BARS_PER_REGULAR_DAY.get(timeframe, 0)

    def is_trading_day(self, check_date: date | datetime) -> bool:
        """
        Check if a date is a trading day.

        Args:
            check_date: Date to check.

        Returns:
            True if the date is a trading day.
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()
        sessions = self.get_trading_sessions(check_date, check_date)
        return len(sessions) > 0

    def get_previous_trading_day(
        self,
        reference_date: date | datetime | None = None,
        max_lookback_days: int = 10,
    ) -> date:
        """
        Get the most recent completed trading day.

        For backfill purposes, returns the last day where market has fully closed
        so that all data is available from sources.

        Args:
            reference_date: Date to search from (default: today).
            max_lookback_days: Maximum days to look back (default: 10).

        Returns:
            Most recent trading day before reference_date.

        Raises:
            ValueError: If no trading day found within lookback period.
        """
        if reference_date is None:
            reference_date = date.today()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()

        # Start from yesterday (today's data may be incomplete)
        check_date = reference_date - timedelta(days=1)

        for _ in range(max_lookback_days):
            if self.is_trading_day(check_date):
                return check_date
            check_date -= timedelta(days=1)

        raise ValueError(
            f"No trading day found within {max_lookback_days} days of {reference_date}"
        )

    def get_next_trading_day(
        self,
        reference_date: date | datetime | None = None,
        max_lookahead_days: int = 10,
    ) -> date:
        """
        Get the next trading day on or after the reference date.

        Useful for calculating start dates that don't land on weekends/holidays.

        Args:
            reference_date: Date to search from (default: today).
            max_lookahead_days: Maximum days to look ahead (default: 10).

        Returns:
            First trading day on or after reference_date.

        Raises:
            ValueError: If no trading day found within lookahead period.
        """
        if reference_date is None:
            reference_date = date.today()
        elif isinstance(reference_date, datetime):
            reference_date = reference_date.date()

        check_date = reference_date

        for _ in range(max_lookahead_days):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        raise ValueError(
            f"No trading day found within {max_lookahead_days} days after {reference_date}"
        )
