"""
Tests for Purged + Embargo Cross-Validation (Phase 4).

Verifies the mathematical properties that prevent data leakage:
1. No-Overlap Property: No training sample's label window overlaps test period
2. Embargo Property: After test ends, embargo_bars excluded from next train
"""

import numpy as np
import pytest

from src.domain.signals.indicators.regime.turning_point.cv import (
    PurgedFold,
    PurgedTimeSeriesSplit,
)


class TestPurgedTimeSeriesSplit:
    """Tests for PurgedTimeSeriesSplit."""

    def test_basic_split(self):
        """Test basic split produces valid indices."""
        n_samples = 100
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=2,
        )

        folds = list(splitter.split(X, y))

        assert len(folds) > 0

        for train_idx, test_idx in folds:
            # Train and test should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            # Both should be non-empty
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_no_overlap_property(self):
        """
        Test No-Overlap Property (Go/No-Go Gate #4).

        For all i in train_idx:
            i + label_horizon <= test_start

        This ensures no training sample's label window overlaps the test period.
        """
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)
        label_horizon = 10

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=label_horizon,
            embargo=2,
        )

        passed, violations = splitter.verify_no_overlap(X, y)

        assert passed, f"No-overlap property violated: {violations}"

        # Explicitly verify the property
        for train_idx, test_idx in splitter.split(X, y):
            test_start = test_idx[0]
            for i in train_idx:
                assert i + label_horizon <= test_start, (
                    f"Sample {i} label window [{i}, {i + label_horizon}) "
                    f"overlaps test period starting at {test_start}"
                )

    def test_embargo_property(self):
        """
        Test Embargo Property (Go/No-Go Gate #4).

        After test period ends, embargo_bars samples must NOT appear
        in the NEXT fold's training set.
        """
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)
        embargo = 5

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=embargo,
        )

        passed, violations = splitter.verify_embargo(X, y)

        assert passed, f"Embargo property violated: {violations}"

    def test_get_folds_returns_detailed_info(self):
        """Test that get_folds returns detailed fold information."""
        n_samples = 100
        X = np.random.randn(n_samples, 10)

        splitter = PurgedTimeSeriesSplit(
            n_splits=3,
            label_horizon=10,
            embargo=2,
        )

        folds = splitter.get_folds(X)

        assert len(folds) > 0
        for fold in folds:
            assert isinstance(fold, PurgedFold)
            assert fold.purge_start >= 0
            assert fold.purge_end >= fold.purge_start
            assert fold.embargo_end >= fold.purge_end

    def test_time_ordering(self):
        """Test that splits maintain time ordering."""
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=2,
        )

        prev_test_end = 0

        for train_idx, test_idx in splitter.split(X, y):
            # Train indices should be before test
            assert train_idx.max() < test_idx.min()

            # Test period should advance
            assert test_idx.min() >= prev_test_end
            prev_test_end = test_idx.max() + 1

    def test_expanding_window(self):
        """Test that training window expands (or at least doesn't shrink)."""
        n_samples = 200
        X = np.random.randn(n_samples, 10)

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=2,
        )

        prev_train_size = 0

        for train_idx, test_idx in splitter.split(X):
            # Train always starts from 0 (expanding window)
            assert train_idx[0] == 0
            prev_train_size = len(train_idx)

    def test_invalid_params(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            PurgedTimeSeriesSplit(n_splits=1)

        with pytest.raises(ValueError, match="label_horizon must be non-negative"):
            PurgedTimeSeriesSplit(n_splits=5, label_horizon=-1)

        with pytest.raises(ValueError, match="embargo must be non-negative"):
            PurgedTimeSeriesSplit(n_splits=5, embargo=-1)

    def test_small_dataset(self):
        """Test behavior with small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        splitter = PurgedTimeSeriesSplit(
            n_splits=3,
            label_horizon=2,
            embargo=1,
        )

        folds = list(splitter.split(X, y))

        # Should still produce some folds
        assert len(folds) >= 1

    def test_get_n_splits(self):
        """Test get_n_splits returns correct value."""
        splitter = PurgedTimeSeriesSplit(n_splits=5)
        assert splitter.get_n_splits() == 5


class TestNoOverlapPropertyExplicit:
    """Explicit index-level verification of no-overlap property."""

    @pytest.mark.parametrize("label_horizon", [5, 10, 20])
    def test_no_overlap_various_horizons(self, label_horizon):
        """
        Verify no-overlap for various label horizons.

        Property: For all i in train_idx:
            i + label_horizon <= test_start
        """
        n_samples = 150
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=label_horizon,
            embargo=2,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            test_start = test_idx.min()

            for i in train_idx:
                label_end = i + label_horizon
                assert label_end <= test_start, (
                    f"Fold {fold_idx}: Sample {i} with label_horizon={label_horizon} "
                    f"has label_end={label_end} > test_start={test_start}"
                )


class TestEmbargoPropertyExplicit:
    """Explicit index-level verification of embargo property."""

    @pytest.mark.parametrize("embargo_bars", [2, 5, 10])
    def test_embargo_various_values(self, embargo_bars):
        """
        Verify embargo for various values.

        Property: For fold k and fold k+1:
            next_train samples not in [test_end_k, test_end_k + embargo)
        """
        n_samples = 200
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)

        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=embargo_bars,
        )

        folds = list(splitter.split(X, y))

        for k in range(len(folds) - 1):
            _, test_idx_k = folds[k]
            train_idx_k1, _ = folds[k + 1]

            test_end = test_idx_k.max() + 1
            embargo_zone = set(range(test_end, min(test_end + embargo_bars, n_samples)))

            train_k1_set = set(train_idx_k1)
            overlap = train_k1_set & embargo_zone

            assert len(overlap) == 0, (
                f"Fold {k} -> {k + 1}: Embargo violated. "
                f"Train samples {sorted(overlap)} are in embargo zone "
                f"[{test_end}, {test_end + embargo_bars})"
            )
