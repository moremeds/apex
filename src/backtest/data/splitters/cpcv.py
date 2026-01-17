"""
Combinatorial Purged Cross-Validation (CPCV) splitter.

CPCV is a sophisticated cross-validation method for financial time series
that generates all C(N,K) combinations of N groups with K test groups.
This provides more test paths than traditional walk-forward, enabling
better statistical analysis of strategy robustness.

Reference: Lopez de Prado, "Advances in Financial Machine Learning"
"""

from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from math import comb
from typing import Iterator, List, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ...core import TimeWindow


class CPCVConfig(BaseModel):
    """Configuration for CPCV splitting."""

    n_groups: int = Field(default=6, description="Total number of time groups to divide data into")
    n_test_groups: int = Field(default=2, description="Number of groups to use as test per path")

    # Gap settings
    purge_days: int = Field(default=5, description="Gap between train and test for each boundary")
    embargo_days: int = Field(default=2, description="Gap after test periods")


@dataclass
class CPCVPath:
    """A single CPCV path with train and test groups."""

    path_id: str
    test_group_indices: Tuple[int, ...]
    train_group_indices: Tuple[int, ...]

    # Date ranges for each group
    group_dates: List[Tuple[date, date]] = field(default_factory=list)

    @property
    def n_test_groups(self) -> int:
        return len(self.test_group_indices)

    @property
    def n_train_groups(self) -> int:
        return len(self.train_group_indices)


@dataclass
class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation splitter.

    Divides data into N groups and generates all C(N,K) combinations
    where K groups are used for testing and the rest for training.

    Key differences from walk-forward:
    1. All data is used for both training and testing (in different paths)
    2. More test paths → better statistical significance
    3. Purge/embargo applied at each train-test boundary

    Example:
        config = CPCVConfig(n_groups=6, n_test_groups=2)
        splitter = CPCVSplitter(config)

        # C(6,2) = 15 paths
        for train_windows, test_windows, path in splitter.split("2020-01-01", "2024-12-31"):
            # Combine train windows, test on test windows
            ...
    """

    config: CPCVConfig
    _trading_calendar: pd.DatetimeIndex = field(default=None, repr=False)

    def __post_init__(self):
        if self._trading_calendar is None:
            self._trading_calendar = self._generate_default_calendar()

    def _generate_default_calendar(
        self, start: str = "2010-01-01", end: str = "2030-12-31"
    ) -> pd.DatetimeIndex:
        """Generate default trading calendar (weekdays only)."""
        all_days = pd.date_range(start=start, end=end, freq="D")
        return all_days[all_days.dayofweek < 5]

    def get_path_count(self) -> int:
        """Get number of CPCV paths: C(n_groups, n_test_groups)."""
        return comb(self.config.n_groups, self.config.n_test_groups)

    def _divide_into_groups(self, start_date: date, end_date: date) -> List[Tuple[date, date]]:
        """Divide date range into equal groups."""
        # Get trading days in range
        mask = (self._trading_calendar >= pd.Timestamp(start_date)) & (
            self._trading_calendar <= pd.Timestamp(end_date)
        )
        trading_days = self._trading_calendar[mask]

        n_days = len(trading_days)
        group_size = n_days // self.config.n_groups

        groups = []
        for i in range(self.config.n_groups):
            start_idx = i * group_size
            if i == self.config.n_groups - 1:
                # Last group gets remaining days
                end_idx = n_days - 1
            else:
                end_idx = (i + 1) * group_size - 1

            groups.append((trading_days[start_idx].date(), trading_days[end_idx].date()))

        return groups

    def _is_adjacent_to_test(self, train_idx: int, test_indices: Tuple[int, ...]) -> bool:
        """Check if a training group is adjacent to any test group."""
        for test_idx in test_indices:
            if abs(train_idx - test_idx) == 1:
                return True
        return False

    def _apply_purge_embargo(
        self,
        group_dates: List[Tuple[date, date]],
        train_indices: List[int],
        test_indices: Tuple[int, ...],
    ) -> Tuple[List[Tuple[date, date]], List[Tuple[date, date]]]:
        """
        Apply purge and embargo to train/test splits.

        Removes data near train-test boundaries to prevent leakage.
        """
        train_windows = []
        test_windows = []

        for idx in train_indices:
            start, end = group_dates[idx]

            # Check if adjacent to any test group
            if self._is_adjacent_to_test(idx, test_indices):
                # Need to apply purge/embargo
                # If next group is test, purge from end
                # If previous group is test, embargo from start
                if idx + 1 in test_indices:
                    # Purge: remove days from end
                    end_ts = pd.Timestamp(end)
                    mask = self._trading_calendar <= end_ts
                    before_end = self._trading_calendar[mask]
                    if len(before_end) > self.config.purge_days:
                        end = before_end[-self.config.purge_days - 1].date()

                if idx - 1 in test_indices:
                    # Embargo: remove days from start
                    start_ts = pd.Timestamp(start)
                    mask = self._trading_calendar >= start_ts
                    after_start = self._trading_calendar[mask]
                    if len(after_start) > self.config.embargo_days:
                        start = after_start[self.config.embargo_days].date()

            if start <= end:  # Valid window after purge/embargo
                train_windows.append((start, end))

        for idx in test_indices:
            test_windows.append(group_dates[idx])

        return train_windows, test_windows

    def split(
        self, start_date: str | date, end_date: str | date
    ) -> Iterator[Tuple[List[TimeWindow], List[TimeWindow], CPCVPath]]:
        """
        Generate all CPCV paths.

        Args:
            start_date: Data start date
            end_date: Data end date

        Yields:
            Tuples of (train_windows, test_windows, path)
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        # Divide data into groups
        group_dates = self._divide_into_groups(start_date, end_date)

        # Generate all C(N,K) test group combinations
        all_indices = list(range(self.config.n_groups))
        path_num = 0

        for test_indices in combinations(all_indices, self.config.n_test_groups):
            train_indices = [i for i in all_indices if i not in test_indices]

            # Apply purge/embargo
            train_ranges, test_ranges = self._apply_purge_embargo(
                group_dates, train_indices, test_indices
            )

            # Convert to TimeWindow objects
            train_windows = [
                TimeWindow(
                    window_id=f"cpcv_{path_num}_train_{i}",
                    fold_index=path_num,
                    train_start=start,
                    train_end=end,
                    test_start=start,  # Not used for train
                    test_end=end,  # Not used for train
                    purge_days=self.config.purge_days,
                    embargo_days=self.config.embargo_days,
                    is_train=True,
                    is_oos=False,
                )
                for i, (start, end) in enumerate(train_ranges)
            ]

            test_windows = [
                TimeWindow(
                    window_id=f"cpcv_{path_num}_test_{i}",
                    fold_index=path_num,
                    train_start=group_dates[0][0],  # Reference only
                    train_end=group_dates[-1][1],  # Reference only
                    test_start=start,
                    test_end=end,
                    purge_days=self.config.purge_days,
                    embargo_days=self.config.embargo_days,
                    is_train=False,
                    is_oos=True,
                )
                for i, (start, end) in enumerate(test_ranges)
            ]

            path = CPCVPath(
                path_id=f"cpcv_path_{path_num}",
                test_group_indices=test_indices,
                train_group_indices=tuple(train_indices),
                group_dates=group_dates,
            )

            yield train_windows, test_windows, path
            path_num += 1
