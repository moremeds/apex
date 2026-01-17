"""
Market hours detection utility for US markets.

Handles regular trading hours, extended hours, and market holidays.
"""

from __future__ import annotations

import threading
from datetime import datetime, time
from typing import Optional, Tuple

import pytz

# Module-level cache for hot path performance
# Format: (status, expiry_timestamp)
_status_cache: Tuple[str, float] = ("", 0.0)
_cache_lock = threading.Lock()
_CACHE_TTL_SECONDS = 60.0  # Refresh every 60 seconds


class MarketHours:
    """
    US market hours detector.

    TODO -> add business day calender but low priority

    Regular Hours: 9:30 AM - 4:00 PM ET (Mon-Fri)
    Extended Hours:
    - Pre-market: 4:00 AM - 9:30 AM ET
    - After-hours: 4:00 PM - 8:00 PM ET

    Note: Does not include market holiday detection in this version.
    """

    # US Eastern timezone
    ET = pytz.timezone("US/Eastern")

    # Regular market hours (ET)
    REGULAR_OPEN = time(9, 30)
    REGULAR_CLOSE = time(16, 0)

    # Extended hours (ET)
    PREMARKET_OPEN = time(4, 0)
    AFTERHOURS_CLOSE = time(20, 0)

    @classmethod
    def is_market_open(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if regular market is currently open.

        Args:
            dt: Datetime to check (defaults to now).

        Returns:
            True if market is in regular trading hours.
        """
        if dt is None:
            dt = datetime.now(cls.ET)
        elif dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = pytz.utc.localize(dt).astimezone(cls.ET)
        else:
            dt = dt.astimezone(cls.ET)

        # Check if weekday
        if dt.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check time range
        current_time = dt.time()
        return cls.REGULAR_OPEN <= current_time < cls.REGULAR_CLOSE

    @classmethod
    def is_extended_hours(cls, dt: Optional[datetime] = None) -> bool:
        """
        Check if currently in extended hours trading (pre-market or after-hours).

        Args:
            dt: Datetime to check (defaults to now).

        Returns:
            True if in extended hours (but not regular hours).
        """
        if dt is None:
            dt = datetime.now(cls.ET)
        elif dt.tzinfo is None:
            dt = pytz.utc.localize(dt).astimezone(cls.ET)
        else:
            dt = dt.astimezone(cls.ET)

        # Not on weekends
        if dt.weekday() >= 5:
            return False

        current_time = dt.time()

        # Pre-market: 4:00 AM - 9:30 AM
        if cls.PREMARKET_OPEN <= current_time < cls.REGULAR_OPEN:
            return True

        # After-hours: 4:00 PM - 8:00 PM
        if cls.REGULAR_CLOSE <= current_time < cls.AFTERHOURS_CLOSE:
            return True

        return False

    @classmethod
    def get_market_status(cls, dt: Optional[datetime] = None) -> str:
        """
        Get current market status (cached for hot path performance).

        Uses 60-second TTL cache to avoid repeated pytz timezone conversions.
        Safe because market status only changes at fixed times (4am, 9:30am, 4pm, 8pm ET).

        Args:
            dt: Datetime to check. If None, uses cached "now" status.
                If provided, bypasses cache for exact calculation.

        Returns:
            One of: "OPEN", "EXTENDED", "CLOSED"
        """
        global _status_cache

        # If specific datetime provided, compute exactly (no cache)
        if dt is not None:
            return cls._compute_market_status(dt)

        # Check cache for current time
        import time as time_module

        now_ts = time_module.time()

        cached_status, expiry_ts = _status_cache
        if cached_status and now_ts < expiry_ts:
            return cached_status

        # Cache miss - compute and store
        with _cache_lock:
            # Double-check after acquiring lock
            cached_status, expiry_ts = _status_cache
            if cached_status and now_ts < expiry_ts:
                return cached_status

            # Compute fresh status
            status = cls._compute_market_status(None)
            _status_cache = (status, now_ts + _CACHE_TTL_SECONDS)
            return status

    @classmethod
    def _compute_market_status(cls, dt: Optional[datetime]) -> str:
        """Compute market status without caching."""
        if cls.is_market_open(dt):
            return "OPEN"
        elif cls.is_extended_hours(dt):
            return "EXTENDED"
        else:
            return "CLOSED"

    @classmethod
    def should_use_extended_hours_price(
        cls, asset_type: str, dt: Optional[datetime] = None
    ) -> bool:
        """
        Determine if extended hours price should be used for P&L calculation.

        Logic:
        - Market OPEN: Use regular price for all assets
        - Market EXTENDED: Use extended price for stocks, yesterday close for options
        - Market CLOSED: Use yesterday close for all assets

        Args:
            asset_type: Asset type (STOCK, OPTION, etc.)
            dt: Datetime to check (defaults to now).

        Returns:
            True if extended hours price should be used.
        """
        status = cls.get_market_status(dt)

        if status == "OPEN":
            # Regular hours: use current prices
            return False

        if status == "EXTENDED":
            # Extended hours: stocks trade, options don't
            return asset_type == "STOCK"

        # CLOSED: use yesterday close for all
        return False
