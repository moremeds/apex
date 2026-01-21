"""
Unified Time Configuration in Bars.

All validation uses bar-based units. Conversion from days
is done once at config load time based on timeframe.

This eliminates day/bar unit confusion in multi-timeframe validation.
"""

from dataclasses import dataclass
from typing import Dict

# Standard bars per trading day for each timeframe
# Based on ~6.5 trading hours per day for US markets
BARS_PER_DAY: Dict[str, float] = {
    "1d": 1.0,
    "4h": 1.625,  # ~6.5 hours / 4
    "2h": 3.25,  # ~6.5 hours / 2
    "1h": 6.5,  # ~6.5 trading hours
    "30m": 13.0,  # ~6.5 * 2
    "15m": 26.0,  # ~6.5 * 4
    "5m": 78.0,  # ~6.5 * 12
    "1m": 390.0,  # ~6.5 * 60
}


@dataclass(frozen=True)
class ValidationTimeConfig:
    """
    Unified time configuration in BARS (not days).

    All validation uses bar-based units. Conversion from days
    is done once at config load time based on timeframe.

    Attributes:
        timeframe: Timeframe string (e.g., "1d", "4h", "1h")
        label_horizon_bars: Forward-looking window for labels (in bars)
        purge_bars: Purge zone to prevent train/test overlap (in bars)
        embargo_bars: Buffer after test period (in bars)
        horizon_bars_by_tf: Pre-computed horizon bars for each timeframe
        purge_bars_by_tf: Pre-computed purge bars for each timeframe
        embargo_bars_by_tf: Pre-computed embargo bars for each timeframe
    """

    timeframe: str
    label_horizon_bars: int
    purge_bars: int
    embargo_bars: int

    # Pre-computed bars for all timeframes (for multi-TF validation)
    horizon_bars_by_tf: Dict[str, int]
    purge_bars_by_tf: Dict[str, int]
    embargo_bars_by_tf: Dict[str, int]

    # Original day values (for reference/serialization)
    label_horizon_days: int
    purge_days: int
    embargo_days: int

    @classmethod
    def from_days(
        cls,
        timeframe: str,
        label_horizon_days: int = 20,
        purge_days: int = 5,
        embargo_days: int = 3,
        timeframes: tuple[str, ...] = ("1d", "4h", "2h"),
    ) -> "ValidationTimeConfig":
        """
        Convert day-based config to bar-based for given timeframe.

        Args:
            timeframe: Primary timeframe for this config
            label_horizon_days: Forward-looking horizon in trading days
            purge_days: Purge zone in trading days
            embargo_days: Embargo buffer in trading days
            timeframes: All timeframes to pre-compute bars for

        Returns:
            ValidationTimeConfig with bar-based units
        """
        if timeframe not in BARS_PER_DAY:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. " f"Valid options: {list(BARS_PER_DAY.keys())}"
            )

        multiplier = BARS_PER_DAY[timeframe]

        # Pre-compute bars for all requested timeframes
        horizon_bars_by_tf = {}
        purge_bars_by_tf = {}
        embargo_bars_by_tf = {}

        for tf in timeframes:
            tf_multiplier = BARS_PER_DAY.get(tf, 1.0)
            horizon_bars_by_tf[tf] = int(label_horizon_days * tf_multiplier)
            purge_bars_by_tf[tf] = int(purge_days * tf_multiplier)
            embargo_bars_by_tf[tf] = int(embargo_days * tf_multiplier)

        return cls(
            timeframe=timeframe,
            label_horizon_bars=int(label_horizon_days * multiplier),
            purge_bars=int(purge_days * multiplier),
            embargo_bars=int(embargo_days * multiplier),
            horizon_bars_by_tf=horizon_bars_by_tf,
            purge_bars_by_tf=purge_bars_by_tf,
            embargo_bars_by_tf=embargo_bars_by_tf,
            label_horizon_days=label_horizon_days,
            purge_days=purge_days,
            embargo_days=embargo_days,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "timeframe": self.timeframe,
            "label_horizon_bars": self.label_horizon_bars,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "horizon_bars_by_tf": self.horizon_bars_by_tf,
            "purge_bars_by_tf": self.purge_bars_by_tf,
            "embargo_bars_by_tf": self.embargo_bars_by_tf,
            "label_horizon_days": self.label_horizon_days,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
        }


def validate_time_config(config: ValidationTimeConfig) -> None:
    """
    Validate time config to ensure purge covers label horizon.

    This prevents look-ahead bias by ensuring training samples
    cannot have labels that extend into the test period.

    Args:
        config: ValidationTimeConfig to validate

    Raises:
        ValueError: If purge_bars < label_horizon_bars
    """
    if config.purge_bars < config.label_horizon_bars:
        raise ValueError(
            f"purge_bars ({config.purge_bars}) must be >= "
            f"label_horizon_bars ({config.label_horizon_bars}) "
            f"to prevent look-ahead bias"
        )

    # Validate for all timeframes
    for tf in config.horizon_bars_by_tf:
        horizon = config.horizon_bars_by_tf[tf]
        purge = config.purge_bars_by_tf[tf]
        if purge < horizon:
            raise ValueError(
                f"For timeframe {tf}: purge_bars ({purge}) must be >= "
                f"label_horizon_bars ({horizon}) to prevent look-ahead bias"
            )


def get_bars_per_day(timeframe: str) -> float:
    """
    Get the number of bars per trading day for a timeframe.

    Args:
        timeframe: Timeframe string (e.g., "1d", "4h")

    Returns:
        Number of bars per trading day

    Raises:
        ValueError: If timeframe is unknown
    """
    if timeframe not in BARS_PER_DAY:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. " f"Valid options: {list(BARS_PER_DAY.keys())}"
        )
    return BARS_PER_DAY[timeframe]
