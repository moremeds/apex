"""
Turning Point Label Generation (Phase 4).

Label Definitions (No Leakage):
- Label A: ATR ZigZag Pivot - Major turning point when price reverses ≥ X × ATR20
- Label B: N-bar Reversal Risk - y=1 if future N bars have max_drawdown ≤ -k×ATR

These labels use ONLY future data for label generation, ensuring no leakage
when used with features computed from past data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class TurnType(str, Enum):
    """Type of turning point."""

    NONE = "none"
    TOP = "top"  # Peak before decline
    BOTTOM = "bottom"  # Trough before rally


@dataclass
class ZigZagPivot:
    """A pivot point detected by ZigZag algorithm."""

    index: int  # Bar index in the series
    price: float  # Price at pivot
    turn_type: TurnType  # TOP or BOTTOM
    atr_magnitude: float  # Magnitude of reversal in ATR units
    bars_since_last: int  # Bars since previous pivot


class TurningPointLabeler:
    """
    Generate turning point labels for model training.

    Two label types:
    1. ZigZag Pivot: Identifies major reversals (X × ATR threshold)
    2. N-bar Reversal Risk: Forward-looking risk label (y=1 if future drawdown)

    IMPORTANT: Labels use future data only - features must use past data only.
    """

    def __init__(
        self,
        atr_period: int = 20,
        zigzag_threshold: float = 2.0,  # ATR multiplier for ZigZag
        risk_horizon: int = 10,  # Bars to look ahead for risk label
        risk_threshold: float = 1.5,  # ATR multiplier for risk threshold
    ):
        """
        Initialize labeler.

        Args:
            atr_period: Period for ATR calculation
            zigzag_threshold: ATR multiplier for ZigZag pivot detection
            risk_horizon: Number of bars to look ahead for risk labels
            risk_threshold: ATR multiplier for risk label threshold
        """
        self.atr_period = atr_period
        self.zigzag_threshold = zigzag_threshold
        self.risk_horizon = risk_horizon
        self.risk_threshold = risk_threshold

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ATR (Average True Range).

        Args:
            df: DataFrame with high, low, close columns

        Returns:
            Series of ATR values
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def detect_zigzag_pivots(
        self,
        df: pd.DataFrame,
        atr: Optional[pd.Series] = None,
    ) -> List[ZigZagPivot]:
        """
        Detect ZigZag pivots based on ATR threshold.

        A pivot is confirmed when price reverses by at least
        zigzag_threshold × ATR from the last pivot.

        Args:
            df: DataFrame with high, low, close columns
            atr: Pre-computed ATR series (optional)

        Returns:
            List of ZigZagPivot objects
        """
        if atr is None:
            atr = self.compute_atr(df)

        pivots = []
        n = len(df)

        if n < self.atr_period + 1:
            return pivots

        # Initialize with first valid bar
        start_idx = self.atr_period
        high = df["high"].values
        low = df["low"].values
        atr_vals = atr.values

        # Track current pivot state
        last_pivot_idx = start_idx
        last_pivot_price = (high[start_idx] + low[start_idx]) / 2
        last_pivot_type = TurnType.NONE
        looking_for = TurnType.NONE  # Will determine on first significant move

        for i in range(start_idx + 1, n):
            if np.isnan(atr_vals[i]):
                continue

            threshold = self.zigzag_threshold * atr_vals[i]

            # Check for potential top (reversal from high)
            if high[i] - last_pivot_price >= threshold and last_pivot_type != TurnType.TOP:
                if looking_for in (TurnType.NONE, TurnType.TOP):
                    # New top
                    last_pivot_price = high[i]
                    last_pivot_idx = i
                    last_pivot_type = TurnType.TOP
                    looking_for = TurnType.BOTTOM
                    pivots.append(
                        ZigZagPivot(
                            index=i,
                            price=high[i],
                            turn_type=TurnType.TOP,
                            atr_magnitude=(
                                (high[i] - low[last_pivot_idx]) / atr_vals[i]
                                if last_pivot_idx > 0
                                else 0.0
                            ),
                            bars_since_last=i - last_pivot_idx,
                        )
                    )

            # Check for potential bottom (reversal from low)
            elif last_pivot_price - low[i] >= threshold and last_pivot_type != TurnType.BOTTOM:
                if looking_for in (TurnType.NONE, TurnType.BOTTOM):
                    # New bottom
                    last_pivot_price = low[i]
                    last_pivot_idx = i
                    last_pivot_type = TurnType.BOTTOM
                    looking_for = TurnType.TOP
                    pivots.append(
                        ZigZagPivot(
                            index=i,
                            price=low[i],
                            turn_type=TurnType.BOTTOM,
                            atr_magnitude=(
                                (high[last_pivot_idx] - low[i]) / atr_vals[i]
                                if last_pivot_idx > 0
                                else 0.0
                            ),
                            bars_since_last=i - last_pivot_idx,
                        )
                    )

            # Update extremes for current direction
            elif looking_for == TurnType.BOTTOM and high[i] > last_pivot_price:
                last_pivot_price = high[i]
                last_pivot_idx = i
            elif looking_for == TurnType.TOP and low[i] < last_pivot_price:
                last_pivot_price = low[i]
                last_pivot_idx = i

        return pivots

    def generate_reversal_risk_labels(
        self,
        df: pd.DataFrame,
        label_type: str = "top_risk",
        atr: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate N-bar reversal risk labels.

        For TOP_RISK (label_type="top_risk"):
            y=1 if future risk_horizon bars have drawdown ≤ -risk_threshold × ATR

        For BOTTOM_RISK (label_type="bottom_risk"):
            y=1 if future risk_horizon bars have rally ≥ risk_threshold × ATR

        IMPORTANT: This uses FUTURE data only. Features must use PAST data only.

        Args:
            df: DataFrame with high, low, close columns
            label_type: "top_risk" or "bottom_risk"
            atr: Pre-computed ATR series (optional)

        Returns:
            Series of binary labels (0 or 1)
        """
        if atr is None:
            atr = self.compute_atr(df)

        n = len(df)
        labels = pd.Series(0, index=df.index)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr_vals = atr.values

        for i in range(n - self.risk_horizon):
            if np.isnan(atr_vals[i]):
                continue

            threshold = self.risk_threshold * atr_vals[i]
            current_close = close[i]

            # Look ahead risk_horizon bars
            future_slice = slice(i + 1, i + 1 + self.risk_horizon)

            if label_type == "top_risk":
                # Check for drawdown: min future low vs current close
                future_min_low = np.nanmin(low[future_slice])
                drawdown = current_close - future_min_low
                if drawdown >= threshold:
                    labels.iloc[i] = 1

            elif label_type == "bottom_risk":
                # Check for rally: max future high vs current close
                future_max_high = np.nanmax(high[future_slice])
                rally = future_max_high - current_close
                if rally >= threshold:
                    labels.iloc[i] = 1

        return labels

    def generate_combined_labels(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, List[ZigZagPivot]]:
        """
        Generate both TOP_RISK and BOTTOM_RISK labels plus ZigZag pivots.

        Args:
            df: DataFrame with high, low, close columns

        Returns:
            Tuple of (top_risk_labels, bottom_risk_labels, zigzag_pivots)
        """
        atr = self.compute_atr(df)

        top_risk = self.generate_reversal_risk_labels(df, "top_risk", atr)
        bottom_risk = self.generate_reversal_risk_labels(df, "bottom_risk", atr)
        pivots = self.detect_zigzag_pivots(df, atr)

        return top_risk, bottom_risk, pivots

    def get_label_horizon(self) -> int:
        """
        Get the label horizon for CV purge calculation.

        This is the number of bars that labels 'look ahead',
        which must be purged from training data.
        """
        return self.risk_horizon

    def get_recent_turning_points(
        self,
        df: pd.DataFrame,
        n_recent: int = 5,
        include_current_state: bool = True,
    ) -> "TurningPointHistory":
        """
        Get recent turning point history for verification.

        Detects the most recent N turning points (ZigZag pivots)
        and current market state relative to those pivots.

        Args:
            df: DataFrame with high, low, close columns and DatetimeIndex
            n_recent: Number of recent turning points to return
            include_current_state: Include analysis of current bar state

        Returns:
            TurningPointHistory with recent pivots and state analysis
        """
        atr = self.compute_atr(df)
        pivots = self.detect_zigzag_pivots(df, atr)

        # Get the N most recent pivots
        recent_pivots = pivots[-n_recent:] if pivots else []

        # Analyze current state
        current_state = None
        if include_current_state and len(df) > 0 and len(pivots) > 0:
            last_pivot = pivots[-1]
            current_idx = len(df) - 1
            current_close = df["close"].iloc[-1]
            current_atr = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else atr.dropna().iloc[-1]

            # Distance from last pivot in ATR units
            distance_from_pivot = abs(current_close - last_pivot.price) / current_atr
            bars_since_pivot = current_idx - last_pivot.index

            # Determine if price is extended
            if last_pivot.turn_type == TurnType.TOP:
                # After a top, measure drawdown
                move_direction = "down" if current_close < last_pivot.price else "up"
            else:
                # After a bottom, measure rally
                move_direction = "up" if current_close > last_pivot.price else "down"

            current_state = CurrentTurningPointState(
                bars_since_last_pivot=bars_since_pivot,
                distance_atr_units=distance_from_pivot,
                last_pivot_type=last_pivot.turn_type,
                move_direction=move_direction,
                current_close=current_close,
                last_pivot_price=last_pivot.price,
                atr=current_atr,
            )

        # Add timestamps if available
        pivot_records = []
        for pivot in recent_pivots:
            ts = df.index[pivot.index] if hasattr(df.index, "__getitem__") else None
            pivot_records.append(
                TurningPointRecord(
                    timestamp=ts,
                    index=pivot.index,
                    price=pivot.price,
                    turn_type=pivot.turn_type,
                    atr_magnitude=pivot.atr_magnitude,
                    bars_since_last=pivot.bars_since_last,
                )
            )

        return TurningPointHistory(
            pivots=pivot_records,
            current_state=current_state,
            total_pivots_detected=len(pivots),
            data_start=df.index[0] if len(df) > 0 else None,
            data_end=df.index[-1] if len(df) > 0 else None,
        )


@dataclass
class CurrentTurningPointState:
    """Current market state relative to last turning point."""

    bars_since_last_pivot: int
    distance_atr_units: float
    last_pivot_type: TurnType
    move_direction: str  # "up" or "down" from last pivot
    current_close: float
    last_pivot_price: float
    atr: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "bars_since_last_pivot": self.bars_since_last_pivot,
            "distance_atr_units": round(self.distance_atr_units, 2),
            "last_pivot_type": self.last_pivot_type.value,
            "move_direction": self.move_direction,
            "current_close": round(self.current_close, 2),
            "last_pivot_price": round(self.last_pivot_price, 2),
            "atr": round(self.atr, 4),
        }


@dataclass
class TurningPointRecord:
    """A recorded turning point with timestamp."""

    timestamp: Optional[pd.Timestamp]
    index: int
    price: float
    turn_type: TurnType
    atr_magnitude: float
    bars_since_last: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "index": self.index,
            "price": round(self.price, 2),
            "turn_type": self.turn_type.value,
            "atr_magnitude": round(self.atr_magnitude, 2),
            "bars_since_last": self.bars_since_last,
        }


@dataclass
class TurningPointHistory:
    """
    History of recent turning points for verification.

    Provides evidence of turning point detection working correctly
    by showing the most recent pivots and current market state.
    """

    pivots: List[TurningPointRecord]
    current_state: Optional[CurrentTurningPointState]
    total_pivots_detected: int
    data_start: Optional[pd.Timestamp]
    data_end: Optional[pd.Timestamp]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "pivots": [p.to_dict() for p in self.pivots],
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "total_pivots_detected": self.total_pivots_detected,
            "data_start": self.data_start.isoformat() if self.data_start else None,
            "data_end": self.data_end.isoformat() if self.data_end else None,
        }

    def summary(self) -> str:
        """Generate human-readable summary for verification."""
        lines = [
            f"Turning Point History ({self.data_start} to {self.data_end})",
            f"Total pivots detected: {self.total_pivots_detected}",
            f"Recent pivots shown: {len(self.pivots)}",
            "",
        ]

        for i, pivot in enumerate(self.pivots):
            ts_str = (
                pivot.timestamp.strftime("%Y-%m-%d") if pivot.timestamp else f"bar {pivot.index}"
            )
            lines.append(
                f"  {i+1}. {pivot.turn_type.value.upper()} @ {ts_str}: "
                f"${pivot.price:.2f} ({pivot.atr_magnitude:.1f} ATR move)"
            )

        if self.current_state:
            lines.extend(
                [
                    "",
                    "Current State:",
                    f"  Bars since last pivot: {self.current_state.bars_since_last_pivot}",
                    f"  Distance from pivot: {self.current_state.distance_atr_units:.2f} ATR ({self.current_state.move_direction})",
                    f"  Current close: ${self.current_state.current_close:.2f}",
                    f"  Last pivot: ${self.current_state.last_pivot_price:.2f} ({self.current_state.last_pivot_type.value})",
                ]
            )

        return "\n".join(lines)
