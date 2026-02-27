"""Tests for premium strategy selection logic."""

import pytest

from src.domain.services.advisor.models import VRPResult
from src.domain.services.advisor.premium_advisor import PremiumAdvisor


@pytest.fixture
def advisor():
    return PremiumAdvisor()


@pytest.fixture
def high_vrp():
    return VRPResult(iv30=22.0, rv30=14.0, vrp=8.0, vrp_zscore=1.2, iv_percentile=72.0)


@pytest.fixture
def normal_vrp():
    return VRPResult(iv30=18.0, rv30=15.0, vrp=3.0, vrp_zscore=0.6, iv_percentile=45.0)


@pytest.fixture
def negative_vrp():
    return VRPResult(iv30=14.0, rv30=20.0, vrp=-6.0, vrp_zscore=-1.5, iv_percentile=20.0)


class TestPremiumAdvisorBlocking:
    def test_blocked_r2_regime(self, advisor, high_vrp):
        advice = advisor.advise("QQQ", 500.0, "R2", high_vrp, 0.9, None, "up")
        # R2 allows defined-risk bearish only — since trend is "up", no bearish candidates
        assert advice.action in ("BLOCKED", "HOLD")

    def test_blocked_r3_regime(self, advisor, high_vrp):
        advice = advisor.advise("QQQ", 500.0, "R3", high_vrp, 0.9, None, "up")
        assert advice.action == "BLOCKED"

    def test_blocked_negative_vrp(self, advisor, negative_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", negative_vrp, 0.9, None, "up")
        assert advice.action == "BLOCKED"
        assert "realized" in advice.reasoning[0].lower() or "vrp" in advice.reasoning[0].lower()

    def test_blocked_inverted_term_structure(self, advisor, high_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", high_vrp, 1.3, None, "up")
        assert advice.action == "BLOCKED"

    def test_hold_near_earnings(self, advisor, high_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", high_vrp, 0.9, 2, "up")
        assert advice.action == "HOLD"
        assert "earnings" in advice.reasoning[0].lower()


class TestPremiumAdvisorSell:
    def test_sell_bullish_r0(self, advisor, normal_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", normal_vrp, 0.9, None, "up")
        assert advice.action == "SELL"
        assert advice.strategy is not None
        assert len(advice.legs) >= 1

    def test_bullish_strategy_in_uptrend(self, advisor, normal_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", normal_vrp, 0.9, None, "up")
        assert advice.strategy in ("short_put", "bull_put_spread", "short_strangle", "iron_condor")

    def test_bearish_strategy_in_downtrend(self, advisor, normal_vrp):
        advice = advisor.advise("QQQ", 500.0, "R1", normal_vrp, 0.9, None, "down")
        assert advice.strategy in ("short_call", "bear_call_spread", "iron_condor")

    def test_r2_downtrend_defined_risk_only(self, advisor, normal_vrp):
        """R2 downtrend should only recommend defined-risk bearish (bear_call_spread)."""
        advice = advisor.advise("QQQ", 500.0, "R2", normal_vrp, 0.9, None, "down")
        if advice.action == "SELL":
            assert advice.strategy == "bear_call_spread"

    def test_defined_risk_in_high_iv(self, advisor):
        """When IV percentile > 75, prefer defined-risk strategies."""
        high_iv_vrp = VRPResult(iv30=28.0, rv30=18.0, vrp=10.0, vrp_zscore=1.5, iv_percentile=85.0)
        advice = advisor.advise("QQQ", 500.0, "R0", high_iv_vrp, 0.9, None, "up")
        assert advice.action == "SELL"
        if advice.strategy == "bull_put_spread":
            assert len(advice.legs) == 2

    def test_high_vrp_closer_delta(self, advisor, high_vrp):
        """High VRP z-score -> sell closer to money (higher delta)."""
        advice = advisor.advise("QQQ", 500.0, "R0", high_vrp, 0.9, None, "up")
        sell_legs = [leg for leg in advice.legs if leg.side == "sell"]
        assert sell_legs[0].target_delta >= 30

    def test_delta_capped_at_40(self, advisor):
        """Delta should never exceed 40 for sell legs."""
        extreme_vrp = VRPResult(iv30=35.0, rv30=15.0, vrp=20.0, vrp_zscore=3.0, iv_percentile=95.0)
        advice = advisor.advise("QQQ", 500.0, "R0", extreme_vrp, 0.9, None, "up")
        for leg in advice.legs:
            if leg.side == "sell":
                assert leg.target_delta <= 40

    def test_legs_have_estimated_strikes(self, advisor, normal_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", normal_vrp, 0.9, None, "up")
        for leg in advice.legs:
            assert leg.estimated_strike > 0

    def test_confidence_range(self, advisor, normal_vrp):
        advice = advisor.advise("QQQ", 500.0, "R0", normal_vrp, 0.9, None, "up")
        assert 0 <= advice.confidence <= 100

    def test_works_for_any_etf(self, advisor, normal_vrp):
        for sym in ("QQQ", "SPY", "IWM", "DIA", "GLD"):
            advice = advisor.advise(sym, 500.0, "R0", normal_vrp, 0.9, None, "up")
            assert advice.symbol == sym
            assert advice.action in ("SELL", "HOLD", "BLOCKED")
