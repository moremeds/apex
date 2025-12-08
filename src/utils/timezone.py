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

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


# Standard timezones
UTC = timezone.utc
US_EASTERN = ZoneInfo("America/New_York")
HK_TZ = ZoneInfo("Asia/Hong_Kong")


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


def parse_futu_timestamp(time_str: str) -> Optional[datetime]:
    """
    Parse Futu timestamp (US Eastern) and convert to UTC.

    Futu returns timestamps in US Eastern time for US market orders.

    Args:
        time_str: Timestamp string in "YYYY-MM-DD HH:MM:SS" format.

    Returns:
        Timezone-aware datetime in UTC, or None if parsing fails.
    """
    return parse_timestamp(time_str, source_tz=US_EASTERN)


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
