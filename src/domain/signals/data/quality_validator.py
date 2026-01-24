"""
Data Quality Validator - Single entry point for bar data cleaning and validation.

PR-A Deliverable: Implements "first mile" data quality gates.

Core Principles:
1. Single source of truth - all data cleaning happens here
2. Fail-fast - reject invalid data early, don't silently propagate
3. Full transparency - log and report every issue found

Usage:
    validator = DataQualityValidator()
    clean_df, quality_result = validator.validate_and_clean(df, symbol="AAPL")

    if not quality_result.valid:
        logger.warning(f"Data quality issue: {quality_result.invalid_reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

from ..schemas import BarQualityResult, InvalidValueReason

logger = get_logger(__name__)

# Sentinel value used by IB for missing data
SENTINEL_VALUE = -1.0

# OHLCV columns that must be validated
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Columns where 0.0 is invalid (price columns)
PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass
class ValidationConfig:
    """Configuration for data quality validation."""

    # Sentinel detection
    detect_sentinels: bool = True
    sentinel_value: float = SENTINEL_VALUE

    # Zero value handling
    reject_zero_close: bool = True
    reject_zero_price: bool = True

    # NaN handling
    max_nan_ratio: float = 0.1  # Max 10% NaN allowed

    # Gap detection
    detect_gaps: bool = True
    max_gap_bars: int = 5  # Max consecutive missing bars before warning

    # Timestamp validation
    require_monotonic_timestamps: bool = True


class DataQualityValidator:
    """
    Validates and cleans bar data at the pipeline entry point.

    This is the "first mile" validator - all data should pass through here
    before being used in indicators, regime detection, or reports.

    Key Features:
    - Detects and reports -1.0 sentinel values (IB missing data)
    - Rejects close=0.0 which corrupts regime calculations
    - Removes NaN values with transparency
    - Validates timestamp continuity
    - Reports detailed quality metrics
    """

    def __init__(self, config: Optional[ValidationConfig] = None) -> None:
        """
        Initialize validator.

        Args:
            config: Optional validation configuration
        """
        self._config = config or ValidationConfig()

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1d",
    ) -> Tuple[pd.DataFrame, BarQualityResult]:
        """
        Validate and clean bar data.

        This is the main entry point. Returns both the cleaned DataFrame
        and a quality report.

        Args:
            df: Raw bar data with OHLCV columns
            symbol: Symbol being validated (for logging)
            timeframe: Bar timeframe (for logging)

        Returns:
            Tuple of (cleaned_df, quality_result)
        """
        if df.empty:
            return df, BarQualityResult.invalid_result(
                reason=InvalidValueReason.NONE,
                timestamp=None,
            )

        original_len = len(df)
        clean_df = df.copy()

        # Track quality issues
        sentinel_counts: Dict[str, int] = {}
        nan_counts: Dict[str, int] = {}
        dropped_indices: List[int] = []

        # === STEP 1: Detect and handle sentinel values (-1.0) ===
        if self._config.detect_sentinels:
            clean_df, sentinel_counts = self._handle_sentinels(clean_df, symbol, timeframe)

        # === STEP 2: Detect and handle NaN values ===
        clean_df, nan_counts, nan_dropped = self._handle_nans(clean_df, symbol, timeframe)
        dropped_indices.extend(nan_dropped)

        # === STEP 3: Validate close price ===
        if self._config.reject_zero_close:
            clean_df, zero_dropped = self._validate_close_price(clean_df, symbol, timeframe)
            dropped_indices.extend(zero_dropped)

        # === STEP 4: Validate timestamp monotonicity ===
        if self._config.require_monotonic_timestamps:
            clean_df = self._validate_timestamps(clean_df, symbol, timeframe)

        # === STEP 5: Detect gaps ===
        if self._config.detect_gaps:
            self._detect_gaps(clean_df, symbol, timeframe)

        # === Build quality result ===
        usable_bars = len(clean_df)
        dropped_bars = original_len - usable_bars

        # Get last valid close and timestamp
        if clean_df.empty:
            return clean_df, BarQualityResult.invalid_result(
                reason=InvalidValueReason.ZERO_VALUE,
                timestamp=None,
                sentinel_counts=sentinel_counts,
                nan_counts=nan_counts,
            )

        last_close = float(clean_df["close"].iloc[-1])
        last_timestamp = self._get_timestamp(clean_df, -1)

        # Check if close is valid
        if last_close <= 0:
            logger.error(f"[{symbol}] Last close is invalid after cleaning: {last_close}")
            return clean_df, BarQualityResult.invalid_result(
                reason=InvalidValueReason.ZERO_VALUE,
                timestamp=last_timestamp,
                sentinel_counts=sentinel_counts,
                nan_counts=nan_counts,
            )

        return clean_df, BarQualityResult(
            valid=True,
            close=last_close,
            timestamp=last_timestamp,
            invalid_reason=InvalidValueReason.NONE,
            sentinel_counts=sentinel_counts,
            nan_counts=nan_counts,
            dropped_bar_count=dropped_bars,
            usable_bar_count=usable_bars,
        )

    def _handle_sentinels(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detect and replace sentinel values (-1.0).

        For price columns: Replace with NaN (will be handled in NaN step)
        For volume: Replace with 0 (missing volume is acceptable)

        Returns:
            Tuple of (cleaned_df, sentinel_counts_by_column)
        """
        sentinel_counts: Dict[str, int] = {}

        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                continue

            # Count sentinels
            sentinel_mask = df[col] == self._config.sentinel_value
            count = int(sentinel_mask.sum())

            if count > 0:
                sentinel_counts[col] = count
                logger.warning(
                    f"[{symbol}:{timeframe}] Found {count} sentinel values " f"in column '{col}'"
                )

                # Replace sentinels
                if col == "volume":
                    # Volume: replace with 0
                    df.loc[sentinel_mask, col] = 0
                else:
                    # Price columns: replace with NaN
                    df.loc[sentinel_mask, col] = np.nan

        if sentinel_counts:
            logger.info(f"[{symbol}:{timeframe}] Sentinel summary: {sentinel_counts}")

        return df, sentinel_counts

    def _handle_nans(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Tuple[pd.DataFrame, Dict[str, int], List[int]]:
        """
        Handle NaN values in the DataFrame.

        Strategy:
        - For close: Drop the row (can't interpolate closing price)
        - For other columns: Forward fill if possible

        Returns:
            Tuple of (cleaned_df, nan_counts_by_column, dropped_indices)
        """
        nan_counts: Dict[str, int] = {}
        dropped_indices: List[int] = []

        # Count NaNs per column
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                continue
            count = int(df[col].isna().sum())
            if count > 0:
                nan_counts[col] = count

        if nan_counts:
            logger.info(f"[{symbol}:{timeframe}] NaN counts: {nan_counts}")

        # Drop rows with NaN close (critical column)
        if "close" in df.columns:
            close_nan_mask = df["close"].isna()
            if close_nan_mask.any():
                dropped_indices = df.index[close_nan_mask].tolist()
                df = df[~close_nan_mask].copy()
                logger.warning(
                    f"[{symbol}:{timeframe}] Dropped {len(dropped_indices)} rows "
                    f"with NaN close values"
                )

        # Forward fill other columns (if any NaN remains)
        for col in ["open", "high", "low"]:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].ffill()
                # Back fill any remaining NaN at the start
                df[col] = df[col].bfill()

        return df, nan_counts, dropped_indices

    def _validate_close_price(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Validate that close prices are positive.

        Drops rows where close <= 0.

        Returns:
            Tuple of (cleaned_df, dropped_indices)
        """
        dropped_indices: List[int] = []

        if "close" not in df.columns:
            return df, dropped_indices

        invalid_mask = df["close"] <= 0
        if invalid_mask.any():
            dropped_indices = df.index[invalid_mask].tolist()
            invalid_closes = df.loc[invalid_mask, "close"].tolist()
            df = df[~invalid_mask].copy()
            logger.error(
                f"[{symbol}:{timeframe}] Dropped {len(dropped_indices)} rows "
                f"with invalid close values: {invalid_closes[:5]}..."
            )

        return df, dropped_indices

    def _validate_timestamps(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Validate that timestamps are monotonically increasing.

        Drops out-of-order rows.

        Returns:
            Cleaned DataFrame with monotonic timestamps
        """
        if df.index.is_monotonic_increasing:
            return df

        # Find and log out-of-order entries
        sorted_df = df.sort_index()
        if not df.index.equals(sorted_df.index):
            logger.warning(
                f"[{symbol}:{timeframe}] Timestamps were not monotonic, " f"sorted {len(df)} rows"
            )
            return sorted_df

        return df

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> None:
        """
        Detect gaps in the data (missing bars).

        Only logs warnings, does not modify data.
        """
        if len(df) < 2:
            return

        # Calculate expected bar interval from timeframe
        tf_seconds = self._get_timeframe_seconds(timeframe)
        if tf_seconds == 0:
            return

        # Check for gaps
        if hasattr(df.index, "to_pydatetime"):
            timestamps = df.index.to_pydatetime()
        else:
            timestamps = df.index

        gap_count = 0
        max_gap = 0

        for i in range(1, len(timestamps)):
            try:
                delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
                expected = tf_seconds
                # Allow up to 3x expected interval before considering it a gap
                if delta > expected * 3:
                    gap_bars = int(delta / expected) - 1
                    gap_count += 1
                    max_gap = max(max_gap, gap_bars)
            except (TypeError, AttributeError):
                # Index may not be datetime
                break

        if gap_count > 0:
            logger.info(
                f"[{symbol}:{timeframe}] Detected {gap_count} gaps, " f"max gap: {max_gap} bars"
            )

    def _get_timestamp(
        self,
        df: pd.DataFrame,
        idx: int,
    ) -> Optional[datetime]:
        """Get timestamp from DataFrame index."""
        try:
            ts = df.index[idx]
            if hasattr(ts, "to_pydatetime"):
                return ts.to_pydatetime()
            elif isinstance(ts, datetime):
                return ts
            else:
                return None
        except (IndexError, TypeError):
            return None

    @staticmethod
    def _get_timeframe_seconds(timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        tf_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
        }
        return tf_map.get(timeframe, 0)


def validate_close_for_regime(
    close: float,
    symbol: str,
    context: str = "regime",
) -> Tuple[bool, str]:
    """
    Validate that a close price is valid for regime calculation.

    This is a lightweight validation function for use in regime_detector.py
    and package_builder.py to catch invalid close values before they
    corrupt calculations.

    Args:
        close: Close price to validate
        symbol: Symbol for logging
        context: Context description for logging

    Returns:
        Tuple of (is_valid, error_message)
    """
    if close is None:
        return False, f"[{symbol}] {context}: close is None"

    if np.isnan(close):
        return False, f"[{symbol}] {context}: close is NaN"

    if close <= 0:
        return False, f"[{symbol}] {context}: close={close} is invalid (<=0)"

    if close == SENTINEL_VALUE:
        return False, f"[{symbol}] {context}: close={close} is sentinel value"

    return True, ""


def get_last_valid_close(
    df: pd.DataFrame,
    symbol: str,
) -> Tuple[float, Optional[datetime]]:
    """
    Get the last valid close price from a DataFrame.

    Searches backwards from the end to find the last valid (positive, non-NaN)
    close value. This is the recommended way to get close for summary.json.

    Args:
        df: DataFrame with 'close' column
        symbol: Symbol for logging

    Returns:
        Tuple of (close_value, timestamp) or (0.0, None) if no valid close found
    """
    if df.empty or "close" not in df.columns:
        logger.warning(f"[{symbol}] No close data available")
        return 0.0, None

    # Search backwards for valid close
    for i in range(len(df) - 1, -1, -1):
        close = df["close"].iloc[i]
        if close is not None and not np.isnan(close) and close > 0:
            try:
                ts = df.index[i]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                elif not isinstance(ts, datetime):
                    ts = None
            except (IndexError, TypeError):
                ts = None

            return float(close), ts

    logger.error(f"[{symbol}] No valid close found in {len(df)} bars")
    return 0.0, None
