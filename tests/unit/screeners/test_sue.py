"""Unit tests for SUE computation and multi-quarter SUE."""

from __future__ import annotations

from src.domain.screeners.pead.sue import compute_multi_quarter_sue, compute_sue

# ═══════════════════════════════════════════════════════════════════════
# Single-Quarter SUE
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSue:
    def test_sue_with_history(self) -> None:
        """SUE = raw_surprise / stdev(history) when >= 4 quarters."""
        history = [0.10, -0.05, 0.15, 0.20, -0.10, 0.05, 0.08, -0.03]
        sue = compute_sue(2.50, 2.00, history)
        # raw = 0.50, stdev ≈ 0.104
        assert sue > 4.0, f"Expected SUE > 4.0, got {sue:.2f}"

    def test_sue_no_history(self) -> None:
        """Fallback when < 4 quarters: raw / (|consensus| * 0.05 + 0.01)."""
        sue = compute_sue(2.50, 2.00, [0.1, 0.2])
        # raw = 0.50, denom = 2.0 * 0.05 + 0.01 = 0.11
        expected = 0.50 / 0.11
        assert abs(sue - expected) < 0.01

    def test_sue_zero_std(self) -> None:
        """When all historical surprises are identical, use fallback."""
        sue = compute_sue(2.50, 2.00, [0.1, 0.1, 0.1, 0.1])
        # std=0, fallback to proxy
        assert sue > 0

    def test_sue_excludes_current_quarter(self) -> None:
        """Verify that history should not contain current quarter data.

        This test documents the contract: callers must pre-filter history
        to exclude the current quarter before calling compute_sue.
        The FMP adapter does this in _merge_earning_data (Phase 0.3 fix).
        """
        # Same data with/without current quarter produces different SUE
        # because stdev changes when the outlier is included
        history_with_leak = [0.50, 0.10, -0.05, 0.15, 0.20]  # 0.50 = current Q
        history_clean = [0.10, -0.05, 0.15, 0.20]  # current Q excluded
        sue_leak = compute_sue(2.50, 2.00, history_with_leak)
        sue_clean = compute_sue(2.50, 2.00, history_clean)
        # The leaky version has different stdev, producing different SUE
        assert sue_leak != sue_clean

    def test_sue_history_order(self) -> None:
        """SUE computation uses stdev which is order-invariant, so any order works."""
        history_asc = [-0.10, -0.05, 0.05, 0.10, 0.15, 0.20]
        history_desc = [0.20, 0.15, 0.10, 0.05, -0.05, -0.10]
        sue_asc = compute_sue(2.50, 2.00, history_asc)
        sue_desc = compute_sue(2.50, 2.00, history_desc)
        assert abs(sue_asc - sue_desc) < 0.001


# ═══════════════════════════════════════════════════════════════════════
# Multi-Quarter SUE
# ═══════════════════════════════════════════════════════════════════════


class TestMultiQuarterSue:
    def test_decay_weighting(self) -> None:
        """Exponential decay gives more weight to recent quarters."""
        # All positive: more recent = higher weight
        history = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
        mq = compute_multi_quarter_sue(history, decay_lambda=0.75, min_quarters=6)
        assert mq is not None
        assert mq > 0  # All positive surprises → positive multi-Q

    def test_returns_none_when_insufficient(self) -> None:
        """Returns None when fewer than min_quarters."""
        mq = compute_multi_quarter_sue([0.1, 0.2, 0.3], min_quarters=6)
        assert mq is None

    def test_empty_list(self) -> None:
        """Empty list returns None."""
        mq = compute_multi_quarter_sue([], min_quarters=1)
        assert mq is None

    def test_deteriorating_trajectory(self) -> None:
        """Deteriorating trajectory (positive → negative) yields negative multi-Q."""
        # Recent quarters are negative, older were positive
        history = [-0.20, -0.15, -0.10, 0.05, 0.10, 0.15]
        mq = compute_multi_quarter_sue(history, decay_lambda=0.75, min_quarters=6)
        assert mq is not None
        assert mq < 0

    def test_decay_lambda_effect(self) -> None:
        """Higher decay_lambda means more weight on older quarters."""
        history = [0.30, -0.05, -0.05, -0.05, -0.05, -0.05]
        # With high decay (0.9): older negative quarters matter more → lower score
        mq_high = compute_multi_quarter_sue(history, decay_lambda=0.9, min_quarters=6)
        # With low decay (0.5): recent positive dominates → higher score
        mq_low = compute_multi_quarter_sue(history, decay_lambda=0.5, min_quarters=6)
        assert mq_high is not None and mq_low is not None
        # Low decay should give higher score (more weight to the positive recent Q)
        assert mq_low > mq_high

    def test_hand_computed(self) -> None:
        """Hand-computed expected value for 6 quarters with lambda=0.75."""
        history = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        # All identical = 0.10
        # weights: 1, 0.75, 0.5625, 0.421875, 0.316406, 0.237305
        # weighted_mean = 0.10 (all same) / 1 = 0.10
        # stdev of identical = 0, so fallback returns raw weighted_mean
        mq = compute_multi_quarter_sue(history, decay_lambda=0.75, min_quarters=6)
        assert mq is not None
        assert abs(mq - 0.10) < 0.001
