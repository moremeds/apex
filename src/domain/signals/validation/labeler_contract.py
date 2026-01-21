"""
Frozen Labeler Contract for Regime Validation.

Ground truth labeler with FROZEN thresholds that define what
"trending" and "choppy" mean as research decisions, NOT tuning targets.

These thresholds must NOT participate in optimization - changing them
invalidates all previous validation results.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .time_units import BARS_PER_DAY


class RegimeLabel(Enum):
    """Ground truth regime labels for validation."""

    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# FROZEN THRESHOLDS - DO NOT MODIFY WITHOUT VERSION BUMP
# =============================================================================

LABELER_THRESHOLDS_V1: Dict[str, Dict[str, float]] = {
    # Version metadata
    "_meta": {
        "version": 1.0,
        "frozen_date": 20260120,  # YYYYMMDD
    },
    # Daily timeframe thresholds
    "1d": {
        "trending_forward_return_min": 0.10,  # 10% forward return
        "trending_sharpe_min": 1.0,  # Sharpe > 1
        "choppy_volatility_min": 0.25,  # 25% annualized vol
        "choppy_drawdown_max": -0.10,  # 10% drawdown
    },
    # 4-hour timeframe thresholds (scaled for shorter horizon)
    "4h": {
        "trending_forward_return_min": 0.06,  # 6% forward return
        "trending_sharpe_min": 0.8,  # Sharpe > 0.8
        "choppy_volatility_min": 0.30,  # 30% annualized vol
        "choppy_drawdown_max": -0.08,  # 8% drawdown
    },
    # 2-hour timeframe thresholds
    "2h": {
        "trending_forward_return_min": 0.04,  # 4% forward return
        "trending_sharpe_min": 0.6,  # Sharpe > 0.6
        "choppy_volatility_min": 0.35,  # 35% annualized vol
        "choppy_drawdown_max": -0.06,  # 6% drawdown
    },
    # 1-hour timeframe thresholds
    "1h": {
        "trending_forward_return_min": 0.03,  # 3% forward return
        "trending_sharpe_min": 0.5,  # Sharpe > 0.5
        "choppy_volatility_min": 0.40,  # 40% annualized vol
        "choppy_drawdown_max": -0.05,  # 5% drawdown
    },
}


@dataclass(frozen=True)
class RegimeLabelerConfig:
    """
    Frozen labeler configuration.

    WARNING: These thresholds define ground truth.
    Changing them invalidates all previous validation results.
    """

    version: str
    timeframe: str

    # Trending definition
    trending_forward_return_min: float
    trending_sharpe_min: float

    # Choppy definition
    choppy_volatility_min: float
    choppy_drawdown_max: float

    # Label horizon (in bars for this timeframe)
    label_horizon_bars: int

    @classmethod
    def load_v1(cls, timeframe: str = "1d", horizon_days: int = 20) -> "RegimeLabelerConfig":
        """
        Load frozen v1.0 thresholds for a specific timeframe.

        Args:
            timeframe: Timeframe to load thresholds for
            horizon_days: Forward-looking horizon in trading days

        Returns:
            Frozen config with v1.0 thresholds
        """
        if timeframe not in LABELER_THRESHOLDS_V1:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Valid: {[k for k in LABELER_THRESHOLDS_V1.keys() if not k.startswith('_')]}"
            )

        thresholds = LABELER_THRESHOLDS_V1[timeframe]
        bars_per_day = BARS_PER_DAY.get(timeframe, 1.0)
        horizon_bars = int(horizon_days * bars_per_day)

        return cls(
            version="v1.0",
            timeframe=timeframe,
            trending_forward_return_min=thresholds["trending_forward_return_min"],
            trending_sharpe_min=thresholds["trending_sharpe_min"],
            choppy_volatility_min=thresholds["choppy_volatility_min"],
            choppy_drawdown_max=thresholds["choppy_drawdown_max"],
            label_horizon_bars=horizon_bars,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "timeframe": self.timeframe,
            "trending_forward_return_min": self.trending_forward_return_min,
            "trending_sharpe_min": self.trending_sharpe_min,
            "choppy_volatility_min": self.choppy_volatility_min,
            "choppy_drawdown_max": self.choppy_drawdown_max,
            "label_horizon_bars": self.label_horizon_bars,
        }


@dataclass
class LabeledPeriod:
    """A labeled period from the labeler."""

    bar_index: int
    timestamp: pd.Timestamp
    label: RegimeLabel
    forward_return: float
    forward_sharpe: float
    forward_volatility: float
    max_drawdown: float


class RegimeLabeler:
    """
    Ground truth label generator with FROZEN thresholds.

    Thresholds are NOT optimizable. They define what "trending" and
    "choppy" mean as a research decision, not a tuning target.
    """

    def __init__(self, config: RegimeLabelerConfig):
        """
        Initialize labeler with frozen config.

        Args:
            config: Frozen labeler configuration
        """
        self.config = config

    def label_period(
        self,
        df: pd.DataFrame,
        max_end_date: Optional[date] = None,
    ) -> List[LabeledPeriod]:
        """
        Generate labels with explicit boundary protection.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            max_end_date: Labels cannot use data beyond this date (prevents leakage)

        Returns:
            List of LabeledPeriod objects

        The forward slice for label i stops at min(i + horizon, max_end_date).
        This prevents label leakage across train/test boundaries.
        """
        labels: List[LabeledPeriod] = []
        horizon = self.config.label_horizon_bars

        # Ensure we have close column
        if "close" not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        # Find max index for boundary check
        max_idx = len(df)
        if max_end_date is not None:
            # Find last index before max_end_date
            for i in range(len(df) - 1, -1, -1):
                idx_date = df.index[i]
                if hasattr(idx_date, "date"):
                    if idx_date.date() <= max_end_date:
                        max_idx = i + 1
                        break

        close = df["close"].values

        for i in range(len(df) - horizon):
            # BOUNDARY CHECK: forward slice must not exceed max_end_date
            forward_end_idx = min(i + horizon, max_idx)

            if forward_end_idx <= i:
                continue  # Not enough forward data

            # Extract forward slice
            forward_slice = close[i:forward_end_idx]

            if len(forward_slice) < 2:
                continue

            # Calculate forward metrics
            forward_return = (forward_slice[-1] - forward_slice[0]) / forward_slice[0]
            returns = np.diff(forward_slice) / forward_slice[:-1]

            # Annualization factor
            ann_factor = np.sqrt(252 * BARS_PER_DAY.get(self.config.timeframe, 1.0))

            # Forward Sharpe (annualized)
            if len(returns) > 1 and returns.std() > 0:
                forward_sharpe = (returns.mean() / returns.std()) * ann_factor
            else:
                forward_sharpe = 0.0

            # Forward volatility (annualized)
            forward_volatility = returns.std() * ann_factor if len(returns) > 1 else 0.0

            # Max drawdown in forward window
            cummax = np.maximum.accumulate(forward_slice)
            drawdowns = (forward_slice - cummax) / cummax
            max_drawdown = drawdowns.min()

            # Classify using frozen thresholds
            label = self._classify(
                forward_return=forward_return,
                forward_sharpe=forward_sharpe,
                forward_volatility=forward_volatility,
                max_drawdown=max_drawdown,
            )

            labels.append(
                LabeledPeriod(
                    bar_index=i,
                    timestamp=df.index[i],
                    label=label,
                    forward_return=forward_return,
                    forward_sharpe=forward_sharpe,
                    forward_volatility=forward_volatility,
                    max_drawdown=max_drawdown,
                )
            )

        return labels

    def _classify(
        self,
        forward_return: float,
        forward_sharpe: float,
        forward_volatility: float,
        max_drawdown: float,
    ) -> RegimeLabel:
        """
        Classify a period based on frozen thresholds.

        Priority:
        1. CHOPPY if high volatility OR significant drawdown
        2. TRENDING if high return AND good sharpe
        3. NEUTRAL otherwise
        """
        # Check CHOPPY conditions first (higher priority for risk)
        if (
            forward_volatility >= self.config.choppy_volatility_min
            or max_drawdown <= self.config.choppy_drawdown_max
        ):
            return RegimeLabel.CHOPPY

        # Check TRENDING conditions
        if (
            forward_return >= self.config.trending_forward_return_min
            and forward_sharpe >= self.config.trending_sharpe_min
        ):
            return RegimeLabel.TRENDING

        # Default to NEUTRAL
        return RegimeLabel.NEUTRAL


def load_labeler_thresholds_yaml(path: str) -> Dict[str, RegimeLabelerConfig]:
    """
    Load labeler thresholds from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Dict mapping timeframe -> RegimeLabelerConfig
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    configs = {}
    version = data.get("version", "v1.0")
    horizon_days = data.get("label_horizon_days", 20)

    for tf, thresholds in data.get("thresholds_by_tf", {}).items():
        bars_per_day = BARS_PER_DAY.get(tf, 1.0)
        horizon_bars = int(horizon_days * bars_per_day)

        configs[tf] = RegimeLabelerConfig(
            version=version,
            timeframe=tf,
            trending_forward_return_min=thresholds["trending_forward_return_min"],
            trending_sharpe_min=thresholds["trending_sharpe_min"],
            choppy_volatility_min=thresholds["choppy_volatility_min"],
            choppy_drawdown_max=thresholds["choppy_drawdown_max"],
            label_horizon_bars=horizon_bars,
        )

    return configs
