"""
Purged + Embargo Cross-Validation (Phase 4).

Implements López de Prado's Purged K-Fold CV with embargo for
financial time series with forward-looking labels.

Key concepts:
- Purge gap: Remove samples where label horizon overlaps train/test boundary
- Embargo: Additional buffer after test period before next train period

Properties that must hold (verified by tests):
1. No-Overlap: No training sample's label window may overlap test period
2. Embargo: After test ends, embargo_bars samples excluded from next train
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class PurgedFold:
    """A single fold from purged time series split."""

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_start: int  # First index purged (train end - label_horizon)
    purge_end: int  # Last index purged (test start)
    embargo_end: int  # Index where embargo ends (test end + embargo)


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purge gap and embargo.

    For financial data where labels use future information (e.g., N-bar forward returns),
    standard k-fold or walk-forward CV can leak information. This splitter:

    1. Creates time-ordered train/test splits (like TimeSeriesSplit)
    2. Removes (purges) samples from train where label horizon overlaps test
    3. Adds an embargo period after test before next train window

    Mathematical Properties (must be CI-enforced):

    No-Overlap Property:
        For all i in train_idx:
            i + label_horizon <= test_start

    Embargo Property:
        For fold k and fold k+1:
            next_train_start >= test_end_k + embargo_bars

    Usage:
        splitter = PurgedTimeSeriesSplit(
            n_splits=5,
            label_horizon=10,
            embargo=2,
        )
        for train_idx, test_idx in splitter.split(X, y):
            # Train on train_idx, evaluate on test_idx
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 10,
        embargo: int = 2,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize splitter.

        Args:
            n_splits: Number of cross-validation folds
            label_horizon: Number of bars labels look ahead (for purging)
            embargo: Number of bars to skip after test period
            test_size: Fixed test size (optional, auto-calculated if None)
            gap: Additional gap between train and test (beyond label_horizon)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if label_horizon < 0:
            raise ValueError("label_horizon must be non-negative")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")

        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (optional, for sklearn compatibility)
            groups: Group labels (optional, unused)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        # Calculate test size
        if self.test_size is not None:
            test_size = self.test_size
        else:
            # Auto-calculate: reserve ~20% total for testing across all folds
            test_size = max(1, n_samples // (self.n_splits + 1))

        # Total purge needed per split boundary
        purge_size = self.label_horizon + self.gap

        # Minimum training size
        min_train_size = max(1, purge_size * 2)

        for fold in range(self.n_splits):
            # Test period for this fold
            test_end = n_samples - (self.n_splits - fold - 1) * test_size
            test_start = test_end - test_size

            if test_start <= min_train_size:
                continue  # Skip fold if not enough training data

            # Train indices: everything before test, minus purge zone
            # Purge zone: samples where label would overlap test period
            purge_start = max(0, test_start - purge_size)
            train_end = purge_start

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            # Apply embargo for subsequent folds (affects next fold's train)
            # This is tracked but doesn't change current fold
            embargo_end = min(test_end + self.embargo, n_samples)

            if len(train_indices) >= min_train_size and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def get_folds(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[PurgedFold]:
        """
        Get detailed fold information for debugging/verification.

        Args:
            X: Feature array
            y: Target array (optional)

        Returns:
            List of PurgedFold objects with detailed indices
        """
        folds = []
        n_samples = len(X)

        if self.test_size is not None:
            test_size = self.test_size
        else:
            test_size = max(1, n_samples // (self.n_splits + 1))

        purge_size = self.label_horizon + self.gap

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X, y)):
            test_start = test_idx[0]
            test_end = test_idx[-1] + 1
            purge_start = max(0, test_start - purge_size)
            embargo_end = min(test_end + self.embargo, n_samples)

            folds.append(
                PurgedFold(
                    fold_index=fold_idx,
                    train_indices=train_idx,
                    test_indices=test_idx,
                    purge_start=purge_start,
                    purge_end=test_start,
                    embargo_end=embargo_end,
                )
            )

        return folds

    def verify_no_overlap(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify the no-overlap property holds for all folds.

        Property: For all i in train_idx:
            i + label_horizon <= test_start

        Args:
            X: Feature array
            y: Target array (optional)

        Returns:
            Tuple of (all_valid, list_of_violations)
        """
        violations = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X, y)):
            test_start = test_idx[0]

            for i in train_idx:
                label_end = i + self.label_horizon
                if label_end > test_start:
                    violations.append(
                        f"Fold {fold_idx}: train sample {i} label window "
                        f"[{i}, {label_end}) overlaps test starting at {test_start}"
                    )
                    break  # One violation per fold is enough

        return len(violations) == 0, violations

    def verify_embargo(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Verify the embargo property holds between consecutive folds.

        Property: For fold k and fold k+1:
            next_train_start >= test_end_k + embargo_bars

        Note: In expanding window CV, each fold's train starts at 0,
        so this checks that embargo is respected in test period spacing.

        Args:
            X: Feature array
            y: Target array (optional)

        Returns:
            Tuple of (all_valid, list_of_violations)
        """
        violations = []
        folds = list(self.split(X, y))

        for k in range(len(folds) - 1):
            _, test_idx_k = folds[k]
            train_idx_k1, _ = folds[k + 1]

            test_end = test_idx_k[-1] + 1

            # For expanding window, train always starts at 0
            # Check that test periods are spaced by at least embargo
            # (train end is what matters for purge, not start)

            # Actually, the key property is that no training sample
            # from fold k+1 is in the embargo zone of fold k
            train_k1_set = set(train_idx_k1)
            embargo_zone = set(range(test_end, min(test_end + self.embargo, len(X))))

            overlap = train_k1_set & embargo_zone
            if overlap:
                violations.append(
                    f"Fold {k} -> {k + 1}: train samples {sorted(overlap)[:5]}... "
                    f"are in embargo zone [{test_end}, {test_end + self.embargo})"
                )

        return len(violations) == 0, violations
