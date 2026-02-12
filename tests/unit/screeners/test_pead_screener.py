"""Unit tests for PEAD screener, scorer, and SUE computation."""

from __future__ import annotations

from datetime import date

from src.domain.screeners.pead.models import LiquidityTier
from src.domain.screeners.pead.scorer import classify_quality, score_pead_quality
from src.domain.screeners.pead.screener import PEADScreener

# ── Default config matching config/pead_screener.yaml ─────────────────────

DEFAULT_CONFIG: dict = {
    "filters": {
        "min_sue": 2.0,
        "min_earnings_day_gap": 0.02,
        "min_volume_ratio": 2.0,
        "max_entry_delay_trading_days": 5,
        "max_forward_pe": 40.0,
        "exclude_at_52w_high": True,
        "exclude_analyst_downgrade_days": 3,
        "require_revenue_beat": False,
    },
    "trade_params": {
        "default_profit_target": 0.06,
        "default_stop_loss": -0.05,
        "default_max_hold_trading_days": 25,
        "trailing_stop_atr_multiplier": 2.0,
        "trailing_stop_activation_pct": 0.03,
    },
    "regime_rules": {
        "r0_position_size_factor": 1.0,
        "r1_position_size_factor": 0.5,
        "r2_block_entirely": True,
    },
    "liquidity_tiers": {
        "large_cap_min_market_cap": 50_000_000_000,
        "mid_cap_min_market_cap": 2_000_000_000,
        "large_cap_slippage_bps": 10,
        "mid_cap_slippage_bps": 25,
        "small_cap_slippage_bps": 50,
    },
    "quality_thresholds": {"strong": 70, "moderate": 45},
}


def _make_earning(
    symbol: str = "AAPL",
    report_date: str = "2025-01-27",
    actual_eps: float = 2.50,
    consensus_eps: float = 2.00,
    historical_surprises: list[float] | None = None,
    earnings_day_gap: float = 0.05,
    earnings_day_return: float = 0.06,
    earnings_day_volume_ratio: float = 3.0,
    revenue_beat: bool = True,
    at_52w_high: bool = False,
    analyst_downgrade: bool = False,
    forward_pe: float | None = 25.0,
    current_price: float = 200.0,
    earnings_open: float = 195.0,
) -> dict:
    """Build a test earning dict matching screener input schema."""
    return {
        "symbol": symbol,
        "report_date": report_date,
        "actual_eps": actual_eps,
        "consensus_eps": consensus_eps,
        "historical_surprises": historical_surprises
        or [0.10, -0.05, 0.15, 0.20, -0.10, 0.05, 0.08, -0.03],
        "earnings_day_gap": earnings_day_gap,
        "earnings_day_return": earnings_day_return,
        "earnings_day_volume_ratio": earnings_day_volume_ratio,
        "revenue_beat": revenue_beat,
        "at_52w_high": at_52w_high,
        "analyst_downgrade": analyst_downgrade,
        "forward_pe": forward_pe,
        "current_price": current_price,
        "earnings_open": earnings_open,
    }


# Use a Monday so trading days = calendar days (no weekends) for a few days
# 2025-01-27 is a Monday
_REPORT_DATE = "2025-01-27"
_TODAY = date(2025, 1, 29)  # Wednesday = 2 trading days later
_CAPS = {"AAPL": 3_000_000_000_000, "AMZN": 2_000_000_000_000, "SMLL": 500_000_000}


# ═══════════════════════════════════════════════════════════════════════
# SUE Computation
# ═══════════════════════════════════════════════════════════════════════


class TestComputeSue:
    def test_sue_with_history(self) -> None:
        """SUE = raw_surprise / stdev(history) when >= 4 quarters."""
        history = [0.10, -0.05, 0.15, 0.20, -0.10, 0.05, 0.08, -0.03]
        sue = PEADScreener.compute_sue(2.50, 2.00, history)
        # raw = 0.50, stdev ≈ 0.104
        assert sue > 4.0, f"Expected SUE > 4.0, got {sue:.2f}"

    def test_sue_no_history(self) -> None:
        """Fallback when < 4 quarters: raw / (|consensus| * 0.05 + 0.01)."""
        sue = PEADScreener.compute_sue(2.50, 2.00, [0.1, 0.2])
        # raw = 0.50, denom = 2.0 * 0.05 + 0.01 = 0.11
        expected = 0.50 / 0.11
        assert abs(sue - expected) < 0.01

    def test_sue_zero_std(self) -> None:
        """When all historical surprises are identical, use fallback."""
        sue = PEADScreener.compute_sue(2.50, 2.00, [0.1, 0.1, 0.1, 0.1])
        # std=0, fallback to proxy
        assert sue > 0


class TestCheckEarningsReaction:
    def test_basic_reaction(self) -> None:
        gap, day_ret, vol_ratio, at_high = PEADScreener.check_earnings_reaction(
            prior_close=100.0,
            open_price=105.0,
            close_price=106.0,
            volume=5_000_000,
            avg_volume=2_000_000,
            high_52w=110.0,
        )
        assert abs(gap - 0.05) < 0.001
        assert abs(day_ret - 0.06) < 0.001
        assert abs(vol_ratio - 2.5) < 0.001
        # at_52w_high uses prior_close (pre-earnings), not close_price
        # 100 < 110*0.95=104.5 → False
        assert at_high is False

    def test_at_52w_high(self) -> None:
        """prior_close near 52w high → at_52w_high is True."""
        _, _, _, at_high = PEADScreener.check_earnings_reaction(
            prior_close=105,
            open_price=108,
            close_price=110,
            volume=1000,
            avg_volume=500,
            high_52w=110,
        )
        # 105 >= 110 * 0.95 = 104.5 → True (pre-earnings state)
        assert at_high is True


# ═══════════════════════════════════════════════════════════════════════
# Scorer
# ═══════════════════════════════════════════════════════════════════════


class TestScorer:
    def test_score_strong(self) -> None:
        """SUE=4, gap=5%, vol=3x, rev_beat → ~85 STRONG."""
        score = score_pead_quality(sue=4.0, gap_return=0.05, volume_ratio=3.0, revenue_beat=True)
        assert score >= 70, f"Expected >= 70 (STRONG), got {score}"

    def test_score_marginal(self) -> None:
        """SUE=2.1, gap=2.5%, vol=2x, no rev beat → < 70."""
        score = score_pead_quality(sue=2.1, gap_return=0.025, volume_ratio=2.0, revenue_beat=False)
        assert score < 70

    def test_score_max_100(self) -> None:
        """Score should never exceed 100."""
        score = score_pead_quality(sue=10.0, gap_return=0.10, volume_ratio=5.0, revenue_beat=True)
        assert score <= 100.0

    def test_classify_strong(self) -> None:
        assert classify_quality(75) == "STRONG"

    def test_classify_moderate(self) -> None:
        assert classify_quality(50) == "MODERATE"

    def test_classify_marginal(self) -> None:
        assert classify_quality(30) == "MARGINAL"


# ═══════════════════════════════════════════════════════════════════════
# Screener Filter Pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestScreenerFilters:
    def test_r2_blocked(self) -> None:
        """R2 regime returns empty immediately."""
        screener = PEADScreener(DEFAULT_CONFIG)
        result = screener.generate_pead_candidates(
            [_make_earning()],
            regime="R2",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0
        assert result.screened_count == 1

    def test_filters_sue(self) -> None:
        """SUE < 2.0 is excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        # Small surprise → low SUE
        earning = _make_earning(actual_eps=2.01, consensus_eps=2.00)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_filters_gap(self) -> None:
        """Gap < 2% is excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(earnings_day_gap=0.01)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_filters_volume(self) -> None:
        """Volume ratio < 2x is excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(earnings_day_volume_ratio=1.5)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_52w_high_hard_exclude(self) -> None:
        """At 52w high: hard excluded, not just scored down."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(at_52w_high=True)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_analyst_downgrade_hard_exclude(self) -> None:
        """Analyst downgrade within 3 days: hard excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(analyst_downgrade=True)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_forward_pe_filter(self) -> None:
        """PE > 40 excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(forward_pe=55.0)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0

    def test_passing_candidate(self) -> None:
        """A strong candidate passes all filters."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning()
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.symbol == "AAPL"
        assert c.quality_score > 0
        assert c.regime == "R0"

    def test_sorted_by_quality(self) -> None:
        """Candidates sorted by quality descending."""
        screener = PEADScreener(DEFAULT_CONFIG)
        strong = _make_earning(symbol="AAPL", earnings_day_gap=0.06, earnings_day_volume_ratio=3.5)
        moderate = _make_earning(
            symbol="AMZN", earnings_day_gap=0.025, earnings_day_volume_ratio=2.1
        )
        result = screener.generate_pead_candidates(
            [moderate, strong],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 2
        assert result.candidates[0].quality_score >= result.candidates[1].quality_score


class TestTradingDays:
    def test_entry_window_trading_days(self) -> None:
        """Entry window uses NYSE trading days, not calendar days."""
        screener = PEADScreener(DEFAULT_CONFIG)
        # 2025-01-27 (Mon) → 2 trading days = 2025-01-29 (Wed) ✓
        days = screener.count_trading_days(date(2025, 1, 27), date(2025, 1, 29))
        assert days == 2

    def test_weekend_not_counted(self) -> None:
        """Weekends don't count as trading days."""
        screener = PEADScreener(DEFAULT_CONFIG)
        # Friday to Monday = 1 trading day (Monday)
        days = screener.count_trading_days(date(2025, 1, 24), date(2025, 1, 27))
        assert days == 1

    def test_entry_outside_window(self) -> None:
        """Report too old (>5 trading days ago) is excluded."""
        screener = PEADScreener(DEFAULT_CONFIG)
        old_date = "2025-01-15"  # Way outside the window
        earning = _make_earning(report_date=old_date)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 0


class TestRegimeAndSizing:
    def test_r1_position_size_factor(self) -> None:
        """R1 regime: position size factor = 0.5."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning()
        result = screener.generate_pead_candidates(
            [earning],
            regime="R1",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].position_size_factor == 0.5

    def test_r0_full_size(self) -> None:
        """R0 regime: full position size (1.0)."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning()
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].position_size_factor == 1.0


class TestGapHeld:
    def test_gap_held_true(self) -> None:
        """Price >= earnings open → gap held."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(current_price=200.0, earnings_open=195.0)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert result.candidates[0].gap_held is True

    def test_gap_held_false(self) -> None:
        """Price < earnings open → gap NOT held."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(current_price=190.0, earnings_open=195.0)
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert result.candidates[0].gap_held is False


class TestLiquidityTier:
    def test_large_cap(self) -> None:
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(symbol="AAPL")
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert result.candidates[0].surprise.liquidity_tier == LiquidityTier.LARGE_CAP
        assert result.candidates[0].estimated_slippage_bps == 10

    def test_small_cap(self) -> None:
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(symbol="SMLL")
        result = screener.generate_pead_candidates(
            [earning],
            regime="R0",
            today=_TODAY,
            market_caps=_CAPS,
        )
        assert result.candidates[0].surprise.liquidity_tier == LiquidityTier.SMALL_CAP
        assert result.candidates[0].estimated_slippage_bps == 50


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases: Negative Gap + Beat-but-Closed-Down
# ═══════════════════════════════════════════════════════════════════════


class TestNegativeGapRejection:
    """Finding #1/#2: Negative gaps and beat-but-closed-down must be excluded."""

    def test_negative_gap_excluded(self) -> None:
        """Stock gapped DOWN on earnings — not a long PEAD candidate."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(earnings_day_gap=-0.03, earnings_day_return=-0.02)
        result = screener.generate_pead_candidates(
            [earning], regime="R0", today=_TODAY, market_caps=_CAPS
        )
        assert len(result.candidates) == 0

    def test_zero_gap_excluded(self) -> None:
        """Flat gap (0%) doesn't meet min_earnings_day_gap=2%."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(earnings_day_gap=0.0, earnings_day_return=0.01)
        result = screener.generate_pead_candidates(
            [earning], regime="R0", today=_TODAY, market_caps=_CAPS
        )
        assert len(result.candidates) == 0

    def test_beat_but_closed_down_excluded(self) -> None:
        """Stock gapped up 3% but closed red — market rejected the beat."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(
            earnings_day_gap=0.03,
            earnings_day_return=-0.01,  # Closed below prior close
        )
        result = screener.generate_pead_candidates(
            [earning], regime="R0", today=_TODAY, market_caps=_CAPS
        )
        assert len(result.candidates) == 0

    def test_positive_gap_positive_close_passes(self) -> None:
        """Sanity: positive gap + positive close passes both gates."""
        screener = PEADScreener(DEFAULT_CONFIG)
        earning = _make_earning(earnings_day_gap=0.04, earnings_day_return=0.03)
        result = screener.generate_pead_candidates(
            [earning], regime="R0", today=_TODAY, market_caps=_CAPS
        )
        assert len(result.candidates) == 1


class TestScorerEdgeCases:
    """Scorer must handle negative gaps correctly (no abs())."""

    def test_negative_gap_scores_zero(self) -> None:
        """Negative gap should produce 0 gap points."""
        score = score_pead_quality(sue=3.0, gap_return=-0.05, volume_ratio=3.0, revenue_beat=True)
        # Only SUE (18) + vol (25) + rev (15) = 58 max, no gap points
        assert score < 60

    def test_small_positive_gap_scores_low(self) -> None:
        """Gap < 2% should score proportionally low."""
        score = score_pead_quality(sue=3.0, gap_return=0.01, volume_ratio=3.0, revenue_beat=True)
        # gap_pct = 1% → gap_pts = 1/2 * 10 = 5
        assert score < 70
