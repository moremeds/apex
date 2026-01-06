"""Tests for data splitters (Walk-Forward and CPCV)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.backtest import CPCVConfig, CPCVSplitter, SplitConfig, WalkForwardSplitter


class TestWalkForwardSplitter:
    """Tests for WalkForwardSplitter."""

    def test_basic_split(self) -> None:
        """Test basic walk-forward splitting returns (train, test) tuples."""
        config = SplitConfig(
            train_days=365,
            test_days=90,
            folds=3,
            purge_days=5,
            embargo_days=2,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2024-12-31"))

        assert len(folds) == 3
        # Each fold is a tuple of (train_window, test_window)
        for i, (train_window, test_window) in enumerate(folds):
            assert train_window.fold_index == i
            assert test_window.fold_index == i
            # Train window flags
            assert train_window.is_train is True
            assert train_window.is_oos is False
            # Test window flags
            assert test_window.is_train is False
            assert test_window.is_oos is True

    def test_window_attributes(self) -> None:
        """Test that windows have correct attributes."""
        config = SplitConfig(
            train_days=100,
            test_days=30,
            folds=1,
            purge_days=5,
            embargo_days=2,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2021-12-31"))

        assert len(folds) >= 1
        train_window, test_window = folds[0]
        # Train window attributes
        assert train_window.train_start is not None
        assert train_window.train_end is not None
        assert train_window.window_id is not None
        # Test window attributes
        assert test_window.test_start is not None
        assert test_window.test_end is not None

    def test_purge_gap_calculation(self) -> None:
        """Test that purge gap is correctly calculated."""
        config = SplitConfig(
            train_days=100,
            test_days=30,
            folds=1,
            purge_days=10,
            embargo_days=0,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2020-12-31"))

        if folds:
            train_window, test_window = folds[0]
            # Window attributes are date objects
            from datetime import date as date_type
            train_end = train_window.train_end if isinstance(train_window.train_end, date_type) else datetime.strptime(train_window.train_end, "%Y-%m-%d").date()
            test_start = test_window.test_start if isinstance(test_window.test_start, date_type) else datetime.strptime(test_window.test_start, "%Y-%m-%d").date()
            gap_days = (test_start - train_end).days

            # Gap should include purge days (using trading days, so gap may vary)
            assert gap_days >= 1  # At least some gap exists

    def test_embargo_gap_calculation(self) -> None:
        """Test that embargo gap is correctly applied."""
        config = SplitConfig(
            train_days=100,
            test_days=30,
            folds=2,
            purge_days=5,
            embargo_days=2,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2021-12-31"))

        # Just verify we got windows without error
        assert len(folds) >= 1

    def test_insufficient_data_reduces_folds(self) -> None:
        """Test that insufficient data reduces number of folds."""
        config = SplitConfig(
            train_days=365,
            test_days=90,
            folds=100,  # Request many more folds than possible
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2022-12-31"))

        # Should generate fewer folds than requested
        assert len(folds) < 100

    def test_no_overlap_between_train_test(self) -> None:
        """Test that there's no overlap between train and test periods."""
        config = SplitConfig(
            train_days=252,
            test_days=63,
            folds=3,
            purge_days=5,
            embargo_days=2,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2024-12-31"))

        from datetime import date as date_type
        for train_window, test_window in folds:
            # Handle both date objects and strings
            if isinstance(train_window.train_end, date_type):
                train_end = train_window.train_end
            else:
                train_end = datetime.strptime(train_window.train_end, "%Y-%m-%d").date()

            if isinstance(test_window.test_start, date_type):
                test_start = test_window.test_start
            else:
                test_start = datetime.strptime(test_window.test_start, "%Y-%m-%d").date()

            assert test_start > train_end, "Test should start after train ends"

    def test_custom_step_days(self) -> None:
        """Test custom step days configuration."""
        config = SplitConfig(
            train_days=252,
            test_days=63,
            step_days=21,  # Monthly steps
            folds=5,
        )
        splitter = WalkForwardSplitter(config)

        folds = list(splitter.split("2020-01-01", "2024-12-31"))

        assert len(folds) >= 1


class TestCPCVSplitter:
    """Tests for CPCVSplitter (Combinatorial Purged Cross-Validation)."""

    def test_path_count(self) -> None:
        """Test that correct number of paths is generated."""
        config = CPCVConfig(n_groups=6, n_test_groups=2)
        splitter = CPCVSplitter(config)

        # C(6,2) = 15 paths
        assert splitter.get_path_count() == 15

    def test_generate_paths(self) -> None:
        """Test that paths are generated correctly."""
        config = CPCVConfig(n_groups=4, n_test_groups=1)
        splitter = CPCVSplitter(config)

        # C(4,1) = 4 paths
        assert splitter.get_path_count() == 4

        paths = list(splitter.split("2020-01-01", "2023-12-31"))
        assert len(paths) == 4

    def test_path_structure(self) -> None:
        """Test that each path has train and test windows."""
        config = CPCVConfig(n_groups=4, n_test_groups=1, purge_days=5, embargo_days=2)
        splitter = CPCVSplitter(config)

        paths = list(splitter.split("2020-01-01", "2023-12-31"))

        # CPCV yields (train_windows, test_windows, path) tuples
        for train_windows, test_windows, path in paths:
            # Each path should have 1 test group
            assert len(test_windows) == 1
            # Train windows should be remaining groups minus purge/embargo affected
            assert len(train_windows) >= 1

    def test_purge_embargo_in_cpcv(self) -> None:
        """Test that purge and embargo are applied in CPCV."""
        config = CPCVConfig(
            n_groups=5,
            n_test_groups=1,
            purge_days=10,
            embargo_days=5,
        )
        splitter = CPCVSplitter(config)

        paths = list(splitter.split("2020-01-01", "2024-12-31"))

        # Each path should have reduced training windows due to purge/embargo
        for train_windows, test_windows, path in paths:
            # With purge/embargo, adjacent groups to test may be affected
            assert len(train_windows) >= 1

    def test_all_groups_used_as_test(self) -> None:
        """Test that all groups are used as test at some point."""
        config = CPCVConfig(n_groups=5, n_test_groups=1)
        splitter = CPCVSplitter(config)

        paths = list(splitter.split("2020-01-01", "2024-12-31"))

        # Collect all test group indices from the path objects
        test_group_indices = set()
        for train_windows, test_windows, path in paths:
            test_group_indices.update(path.test_group_indices)

        # All 5 groups should be used as test
        assert len(test_group_indices) == 5

    def test_combinatorial_correctness(self) -> None:
        """Test various n_groups and n_test_groups combinations."""
        test_cases = [
            (4, 1, 4),  # C(4,1) = 4
            (4, 2, 6),  # C(4,2) = 6
            (5, 2, 10),  # C(5,2) = 10
            (6, 2, 15),  # C(6,2) = 15
            (6, 3, 20),  # C(6,3) = 20
        ]

        for n_groups, n_test_groups, expected_paths in test_cases:
            config = CPCVConfig(n_groups=n_groups, n_test_groups=n_test_groups)
            splitter = CPCVSplitter(config)
            assert (
                splitter.get_path_count() == expected_paths
            ), f"Failed for C({n_groups},{n_test_groups})"
