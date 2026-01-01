"""Historical data source protocol for downloading bar data from external sources."""

from __future__ import annotations
from typing import Protocol, List, runtime_checkable
from datetime import datetime
from dataclasses import dataclass

from ..events.domain_events import BarData


@dataclass(frozen=True, slots=True)
class DateRange:
    """Represents a date range for historical data coverage."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")

    def overlaps(self, other: DateRange) -> bool:
        """Check if this range overlaps with another."""
        return self.start <= other.end and other.start <= self.end

    def contains(self, other: DateRange) -> bool:
        """Check if this range fully contains another."""
        return self.start <= other.start and self.end >= other.end

    def merge(self, other: DateRange) -> DateRange:
        """Merge with an overlapping or adjacent range."""
        return DateRange(
            start=min(self.start, other.start),
            end=max(self.end, other.end)
        )

    def subtract(self, other: DateRange) -> List[DateRange]:
        """
        Subtract another range from this one.

        Returns list of remaining ranges (0, 1, or 2 ranges).
        """
        if not self.overlaps(other):
            return [self]

        result = []
        # Left portion
        if self.start < other.start:
            result.append(DateRange(self.start, other.start))
        # Right portion
        if self.end > other.end:
            result.append(DateRange(other.end, self.end))

        return result

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end - self.start).total_seconds()


@runtime_checkable
class HistoricalSourcePort(Protocol):
    """
    Protocol for historical data sources that download bar data.

    Unlike BarProvider (which handles live connections and subscriptions),
    this is a simpler interface focused on batch downloads for storage.

    Implementations:
    - YahooHistoricalAdapter (Yahoo Finance)
    - IbHistoricalAdapter (Interactive Brokers)
    """

    @property
    def source_name(self) -> str:
        """
        Get the source identifier (e.g., 'yahoo', 'ib').

        Used for tracking data provenance in storage.
        """
        ...

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[BarData]:
        """
        Fetch historical bars from the source.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            timeframe: Bar size ('5min', '1h', '1d').
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            List of BarData sorted by timestamp ascending.
            Empty list if no data available.

        Raises:
            ConnectionError: If source is unavailable.
            ValueError: If symbol or timeframe not supported.
        """
        ...

    def supports_timeframe(self, timeframe: str) -> bool:
        """
        Check if this source supports the given timeframe.

        Args:
            timeframe: Bar size to check.

        Returns:
            True if timeframe is supported.
        """
        ...

    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of timeframe strings (e.g., ['5min', '1h', '1d']).
        """
        ...
