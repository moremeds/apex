"""
Unified timezone utilities for the APEX project.

Conventions:
- Internal storage/processing: UTC (timezone-aware)
- Broker data: Use exchange local timezone if provided, otherwise assume exchange timezone
- UI display: Local timezone (America/New_York for US market)

All datetime objects in the system should be timezone-aware UTC for consistency.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional

from zoneinfo import ZoneInfo


# Standard timezones
UTC = timezone.utc
US_EASTERN = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")

# Market timezone mapping (for Futu and other broker timestamp parsing)
MARKET_TIMEZONE = {
    "US": ZoneInfo("America/New_York"),
    "HK": ZoneInfo("Asia/Hong_Kong"),
    "CN": ZoneInfo("Asia/Shanghai"),
    "SG": ZoneInfo("Asia/Singapore"),
    "JP": ZoneInfo("Asia/Tokyo"),
    "AU": ZoneInfo("Australia/Sydney"),
}


def now_utc() -> datetime:
    """Get current time in UTC (timezone-aware)."""
    return datetime.now(UTC)


def now_local() -> datetime:
    """Get current time in local timezone (US Eastern)."""
    return datetime.now(US_EASTERN)


def to_utc(dt: datetime, assume_tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Convert a datetime to UTC.

    Args:
        dt: The datetime to convert.
        assume_tz: If dt is naive, assume it's in this timezone.
                   Defaults to US_EASTERN if None.

    Returns:
        Timezone-aware datetime in UTC.
    """
    if dt is None:
        return now_utc()

    if dt.tzinfo is None:
        # Naive datetime - assume the given timezone (default: US Eastern)
        tz = assume_tz or US_EASTERN
        dt = dt.replace(tzinfo=tz)

    return dt.astimezone(UTC)


def to_local(dt: datetime) -> datetime:
    """
    Convert a datetime to local timezone (US Eastern) for display.

    Args:
        dt: The datetime to convert (should be timezone-aware UTC).

    Returns:
        Timezone-aware datetime in US Eastern.
    """
    if dt is None:
        return now_local()

    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=UTC)

    return dt.astimezone(US_EASTERN)


def format_local(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime for display in local timezone.

    Args:
        dt: The datetime to format (should be timezone-aware UTC).
        fmt: The format string.

    Returns:
        Formatted string in US Eastern timezone.
    """
    local_dt = to_local(dt)
    return local_dt.strftime(fmt)


def parse_timestamp(
    time_str: str,
    fmt: str = "%Y-%m-%d %H:%M:%S",
    source_tz: Optional[ZoneInfo] = None,
) -> Optional[datetime]:
    """
    Parse a timestamp string and convert to UTC.

    Args:
        time_str: The timestamp string to parse.
        fmt: The format string (default: "YYYY-MM-DD HH:MM:SS").
        source_tz: The timezone of the source data.
                   Defaults to US_EASTERN if None.

    Returns:
        Timezone-aware datetime in UTC, or None if parsing fails.
    """
    if not time_str:
        return None

    try:
        naive_dt = datetime.strptime(str(time_str), fmt)
        tz = source_tz or US_EASTERN
        local_dt = naive_dt.replace(tzinfo=tz)
        return local_dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def parse_futu_timestamp(
    time_str: Optional[str],
    market: str = "US",
    fallback: Optional[datetime] = None,
) -> datetime:
    """
    Parse Futu timestamp string to UTC datetime.

    Futu returns timestamps in exchange local time:
    - US market: America/New_York (EST/EDT)
    - HK market: Asia/Hong_Kong (HKT)
    - CN market: Asia/Shanghai (CST)

    Supports formats:
    - "YYYY-MM-DD HH:MM:SS" (standard)
    - "YYYY-MM-DD HH:MM:SS.mmm" (with milliseconds)
    - "YYYY-MM-DD" (date only)

    Args:
        time_str: Timestamp string from Futu API.
        market: Trading market code (US, HK, CN, SG, JP, AU).
        fallback: Fallback datetime if parsing fails. Defaults to now_utc().

    Returns:
        Timezone-aware datetime in UTC. Never returns None.
    """
    # Use fallback if no input or invalid input
    if fallback is None:
        fallback = now_utc()

    if not time_str or time_str in ('N/A', '', 'None', None):
        return fallback

    # Try formats: with milliseconds first, then without, then date-only
    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            naive_dt = datetime.strptime(str(time_str), fmt)
            break
        except ValueError:
            continue
    else:
        # No format matched - return fallback
        import logging
        logging.getLogger(__name__).warning(f"Could not parse Futu timestamp: {time_str}")
        return fallback

    # Apply exchange timezone
    exchange_tz = MARKET_TIMEZONE.get(market, MARKET_TIMEZONE["US"])
    local_dt = naive_dt.replace(tzinfo=exchange_tz)

    # Convert to UTC for storage
    return local_dt.astimezone(UTC)


def parse_ib_timestamp(dt: datetime) -> datetime:
    """
    Normalize IB timestamp to UTC.

    IB returns timezone-aware datetimes in UTC.

    Args:
        dt: The datetime from IB (may be timezone-aware or naive).

    Returns:
        Timezone-aware datetime in UTC.
    """
    if dt is None:
        return now_utc()

    if dt.tzinfo is not None:
        # IB returns UTC-aware datetimes
        return dt.astimezone(UTC)
    else:
        # Naive datetime - assume UTC (IB default)
        return dt.replace(tzinfo=UTC)


def ensure_utc(dt: Optional[datetime], assume_tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Ensure a datetime is timezone-aware UTC.

    This is the primary function for normalizing timestamps throughout the codebase.

    Args:
        dt: The datetime to normalize (can be None, naive, or aware).
        assume_tz: If dt is naive, assume it's in this timezone.
                   Defaults to US_EASTERN if None.

    Returns:
        Timezone-aware datetime in UTC.
    """
    if dt is None:
        return now_utc()

    return to_utc(dt, assume_tz)


def age_seconds(dt: datetime) -> float:
    """
    Calculate the age of a datetime in seconds.

    Args:
        dt: The datetime to check (should be timezone-aware UTC).

    Returns:
        Age in seconds (always positive).
    """
    if dt is None:
        return float('inf')

    # Ensure both are UTC for comparison
    utc_dt = ensure_utc(dt)
    now = now_utc()

    return (now - utc_dt).total_seconds()


def is_stale(dt: datetime, threshold_seconds: float) -> bool:
    """
    Check if a datetime is stale (older than threshold).

    Args:
        dt: The datetime to check.
        threshold_seconds: The staleness threshold in seconds.

    Returns:
        True if the datetime is older than the threshold.
    """
    return age_seconds(dt) > threshold_seconds


class DisplayTimezone:
    """
    Configurable display timezone for UI rendering.

    Used by dashboard and other UI components to display times
    in the user's preferred timezone.
    """

    def __init__(self, tz_name: str = "Asia/Hong_Kong"):
        """
        Initialize with a timezone name.

        Args:
            tz_name: IANA timezone name (e.g., "Asia/Hong_Kong", "America/New_York").
        """
        self.tz = ZoneInfo(tz_name)
        self.tz_name = tz_name

    def format_time(self, utc_dt: datetime, fmt: str = "%H:%M:%S") -> str:
        """
        Format UTC datetime for display in local timezone (time only).

        Args:
            utc_dt: UTC timezone-aware datetime.
            fmt: Time format string.

        Returns:
            Formatted time string in local timezone.
        """
        if utc_dt is None:
            utc_dt = now_utc()
        local_dt = utc_dt.astimezone(self.tz)
        return local_dt.strftime(fmt)

    def format_datetime(self, utc_dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format UTC datetime for display in local timezone (full datetime).

        Args:
            utc_dt: UTC timezone-aware datetime.
            fmt: Datetime format string.

        Returns:
            Formatted datetime string in local timezone.
        """
        if utc_dt is None:
            utc_dt = now_utc()
        local_dt = utc_dt.astimezone(self.tz)
        return local_dt.strftime(fmt)

    def format_with_tz(self, utc_dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """
        Format UTC datetime with timezone abbreviation.

        Args:
            utc_dt: UTC timezone-aware datetime.
            fmt: Format string (should include %Z for timezone).

        Returns:
            Formatted string with timezone abbreviation (e.g., "2024-01-15 09:30:00 HKT").
        """
        if utc_dt is None:
            utc_dt = now_utc()
        local_dt = utc_dt.astimezone(self.tz)
        return local_dt.strftime(fmt)

    def current_time(self, fmt: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """
        Get current time string for UI header display.

        Args:
            fmt: Format string for current time.

        Returns:
            Current time in local timezone with timezone abbreviation.
        """
        return datetime.now(self.tz).strftime(fmt)

    def to_local(self, utc_dt: datetime) -> datetime:
        """
        Convert UTC datetime to local timezone.

        Args:
            utc_dt: UTC timezone-aware datetime.

        Returns:
            Datetime in local timezone.
        """
        if utc_dt is None:
            return datetime.now(self.tz)
        return utc_dt.astimezone(self.tz)
