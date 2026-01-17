"""
Walk-Forward splitter with proper purge and embargo gaps.

Walk-forward validation trains on historical data and tests on
subsequent out-of-sample periods. This prevents look-ahead bias
while testing across different market regimes.

Key features:
- Configurable train/test window sizes in trading days
- Purge gap between train and test to prevent data leakage
- Embargo period after test for model decay
- Step size for sliding/expanding windows
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ...core import TimeWindow


class SplitConfig(BaseModel):
    """Configuration for walk-forward splitting."""

    train_days: int = Field(default=252, description="Training window in trading days")
    test_days: int = Field(default=63, description="Test window in trading days")
    step_days: Optional[int] = Field(
        default=None,
        description="Step size between windows (defaults to test_days for non-overlapping)",
    )
    folds: int = Field(default=5, description="Number of walk-forward folds")

    # Gap settings for preventing look-ahead bias
    purge_days: int = Field(
        default=5,
        description="Minimum gap between train and test to prevent data leakage",
    )
    embargo_days: int = Field(
        default=2,
        description="Gap after test to allow for model decay",
    )
    label_horizon_days: int = Field(
        default=0,
        description="Trade resolution horizon in days. Effective purge = max(purge_days, label_horizon_days)",
    )

    # Window type
    expanding: bool = Field(
        default=False,
        description="If True, use expanding window (anchor at start). If False, sliding.",
    )

    @property
    def effective_purge_days(self) -> int:
        """
        Calculate effective purge gap accounting for label horizon.

        For strategies with multi-day trades (e.g., swing trading with 20-day
        holding periods), the purge gap must be at least as long as the label
        horizon to prevent leakage from future trade outcomes.
        """
        return max(self.purge_days, self.label_horizon_days)


@dataclass
class WalkForwardSplitter:
    """
    Walk-forward splitter for time-series cross-validation.

    Creates train/test splits that respect temporal ordering and prevent
    look-ahead bias through purge and embargo gaps.

    Timeline:
        |--- TRAIN ---|-- PURGE --|--- TEST ---|-- EMBARGO --|--- NEXT TRAIN ...

    Example:
        config = SplitConfig(train_days=252, test_days=63, purge_days=5)
        splitter = WalkForwardSplitter(config)

        for window in splitter.split("2020-01-01", "2024-12-31"):
            train_data = df[window.train_start:window.train_end]
            test_data = df[window.test_start:window.test_end]
            # Run backtest...
    """

    config: SplitConfig
    _trading_calendar: Optional[pd.DatetimeIndex] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize trading calendar."""
        if self._trading_calendar is None:
            # Generate a default trading calendar (weekdays only)
            # In production, would use actual exchange calendar
            self._trading_calendar = self._generate_default_calendar()

    def _generate_default_calendar(
        self, start: str = "2010-01-01", end: str = "2030-12-31"
    ) -> pd.DatetimeIndex:
        """Generate default trading calendar (weekdays only)."""
        all_days = pd.date_range(start=start, end=end, freq="D")
        # Filter to weekdays (Monday=0, Friday=4)
        return all_days[all_days.dayofweek < 5]

    def _trading_date_offset(self, from_date: date, trading_days: int) -> date:
        """
        Add trading days to a date.

        Args:
            from_date: Starting date
            trading_days: Number of trading days to add (can be negative)

        Returns:
            Date after adding trading days
        """
        # Find position in calendar
        from_ts = pd.Timestamp(from_date)

        # Calendar is always initialized in __post_init__
        calendar = self._trading_calendar
        assert calendar is not None

        if trading_days >= 0:
            # Find trading days after from_date
            mask = calendar >= from_ts
            future_days = calendar[mask]
            if len(future_days) > trading_days:
                return date(
                    future_days[trading_days].year,
                    future_days[trading_days].month,
                    future_days[trading_days].day,
                )
            return (
                date(future_days[-1].year, future_days[-1].month, future_days[-1].day)
                if len(future_days) > 0
                else from_date
            )
        else:
            # Find trading days before from_date
            mask = calendar <= from_ts
            past_days = calendar[mask]
            if len(past_days) > abs(trading_days):
                return date(
                    past_days[trading_days].year,
                    past_days[trading_days].month,
                    past_days[trading_days].day,
                )  # negative index
            return (
                date(past_days[0].year, past_days[0].month, past_days[0].day)
                if len(past_days) > 0
                else from_date
            )

    def _count_trading_days(self, start: date, end: date) -> int:
        """Count trading days between two dates (inclusive)."""
        calendar = self._trading_calendar
        assert calendar is not None
        mask = (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
        return int(mask.sum())

    def split(
        self, start_date: str | date, end_date: str | date
    ) -> Iterator[Tuple[TimeWindow, TimeWindow]]:
        """
        Generate walk-forward splits with train and test windows.

        Returns BOTH train and test windows per fold for proper Walk-Forward
        Optimization. The train window is for in-sample fitting, the test
        window for out-of-sample validation.

        Args:
            start_date: Data start date
            end_date: Data end date

        Yields:
            Tuple of (train_window, test_window) for each fold

        Example:
            for train_window, test_window in splitter.split("2020-01-01", "2024-12-31"):
                # Fit model on training data
                model.fit(data[train_window.train_start:train_window.train_end])
                # Validate on test data
                results = model.predict(data[test_window.test_start:test_window.test_end])
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        step_days = self.config.step_days or self.config.test_days
        fold = 0

        # Calculate first train window start
        current_train_start = start_date

        while fold < self.config.folds:
            # Calculate train end
            if self.config.expanding:
                train_start = start_date  # Fixed anchor
            else:
                train_start = current_train_start

            train_end = self._trading_date_offset(train_start, self.config.train_days - 1)

            # Calculate test start (after purge gap, accounting for label horizon)
            effective_purge = self.config.effective_purge_days
            test_start = self._trading_date_offset(train_end, effective_purge + 1)

            # Calculate test end
            test_end = self._trading_date_offset(test_start, self.config.test_days - 1)

            # Check if we've exceeded the data range
            if test_end > end_date:
                # Try to fit a smaller window
                test_end = end_date
                actual_test_days = self._count_trading_days(test_start, test_end)
                if actual_test_days < self.config.test_days // 2:
                    # Not enough test data, stop
                    break

            if test_start > end_date:
                break

            # Yield both train and test windows
            train_window = TimeWindow(
                window_id=f"fold_{fold}",
                fold_index=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_days=effective_purge,
                embargo_days=self.config.embargo_days,
                is_train=True,
                is_oos=False,
            )
            test_window = TimeWindow(
                window_id=f"fold_{fold}",
                fold_index=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_days=effective_purge,
                embargo_days=self.config.embargo_days,
                is_train=False,
                is_oos=True,
            )

            yield train_window, test_window

            # Move to next fold
            fold += 1
            current_train_start = self._trading_date_offset(current_train_start, step_days)

    def split_dataframe(
        self, df: pd.DataFrame, date_column: str = "date"
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, TimeWindow, TimeWindow]]:
        """
        Split a DataFrame into train/test sets.

        Args:
            df: DataFrame with date column
            date_column: Name of date column

        Yields:
            Tuples of (train_df, test_df, train_window, test_window)
        """
        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        start_date = df[date_column].min().date()
        end_date = df[date_column].max().date()

        for train_window, test_window in self.split(start_date, end_date):
            train_mask = (df[date_column] >= pd.Timestamp(train_window.train_start)) & (
                df[date_column] <= pd.Timestamp(train_window.train_end)
            )
            test_mask = (df[date_column] >= pd.Timestamp(test_window.test_start)) & (
                df[date_column] <= pd.Timestamp(test_window.test_end)
            )

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            yield train_df, test_df, train_window, test_window

    def validate_no_leakage(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, date_column: str = "date"
    ) -> bool:
        """
        Validate that there's no data leakage between train and test.

        Returns True if no leakage (train max date < test min date with gap).
        """
        train_max = train_df[date_column].max()
        test_min = test_df[date_column].min()

        gap_days = (test_min - train_max).days
        return bool(gap_days >= self.config.effective_purge_days)

    def get_minimum_data_days(self) -> int:
        """Get minimum trading days needed for one complete fold."""
        return (
            self.config.train_days
            + self.config.effective_purge_days  # Uses max(purge_days, label_horizon_days)
            + self.config.test_days
            + self.config.embargo_days
        )
