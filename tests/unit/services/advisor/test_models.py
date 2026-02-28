"""Tests for advisor domain models."""

import pytest

from src.domain.services.advisor.models import (
    EquityAdvice,
    LegSpec,
    LegTemplate,
    MarketContext,
    PremiumAdvice,
    PremiumStrategyDef,
    VRPResult,
)


class TestVRPResult:
    def test_creation(self):
        vrp = VRPResult(iv30=18.5, rv30=15.2, vrp=3.3, vrp_zscore=0.82, iv_percentile=55.0)
        assert vrp.iv30 == 18.5
        assert vrp.vrp == pytest.approx(3.3)

    def test_is_negative(self):
        vrp = VRPResult(iv30=14.0, rv30=18.0, vrp=-4.0, vrp_zscore=-1.2, iv_percentile=30.0)
        assert vrp.vrp < 0


class TestPremiumStrategyDef:
    def test_regime_fit_filtering(self):
        strat = PremiumStrategyDef(
            name="short_put",
            display_name="Short Put",
            direction="bullish",
            risk_profile="undefined",
            regime_fit=frozenset({"R0", "R1"}),
            leg_templates=(
                LegTemplate(side="sell", option_type="put", delta_target=25, dte_target=35),
            ),
        )
        assert "R0" in strat.regime_fit
        assert "R2" not in strat.regime_fit


class TestLegSpec:
    def test_creation(self):
        leg = LegSpec(
            side="sell",
            option_type="put",
            target_delta=25,
            target_dte=35,
            estimated_strike=485.0,
        )
        assert leg.estimated_strike == 485.0


class TestPremiumAdvice:
    def test_blocked(self):
        advice = PremiumAdvice(
            symbol="QQQ",
            action="BLOCKED",
            strategy=None,
            display_name=None,
            confidence=0,
            legs=[],
            vrp_zscore=-0.5,
            iv_percentile=30.0,
            term_structure_ratio=1.3,
            regime="R2",
            earnings_warning=None,
            reasoning=["Risk-off regime"],
        )
        assert advice.action == "BLOCKED"
        assert advice.strategy is None

    def test_sell(self):
        advice = PremiumAdvice(
            symbol="SPY",
            action="SELL",
            strategy="short_put",
            display_name="Short Put",
            confidence=78,
            legs=[LegSpec("sell", "put", 25, 35, 485.0)],
            vrp_zscore=0.82,
            iv_percentile=55.0,
            term_structure_ratio=0.92,
            regime="R0",
            earnings_warning=None,
            reasoning=["VRP elevated"],
        )
        assert advice.action == "SELL"
        assert len(advice.legs) == 1


class TestEquityAdvice:
    def test_creation(self):
        advice = EquityAdvice(
            symbol="AAPL",
            sector="Technology",
            action="BUY",
            confidence=72,
            regime="R0",
            signal_summary={"bullish": 5, "bearish": 2, "neutral": 1},
            top_signals=[],
            trend_pulse=None,
            key_levels={"support": 180.0, "resistance": 195.0},
            reasoning=["5 bullish signals active"],
        )
        assert advice.action == "BUY"


class TestMarketContext:
    def test_contango(self):
        ctx = MarketContext(
            regime="R0",
            regime_name="Healthy Uptrend",
            regime_confidence=75.0,
            vix=18.5,
            vix_percentile=42.0,
            vrp_zscore=0.82,
            term_structure_ratio=0.92,
            term_structure_state="contango",
            timestamp="2026-02-27T14:30:00Z",
        )
        assert ctx.term_structure_state == "contango"
