"""Premium strategy advisor — selects optimal sell-premium strategy.

R2 gate is nuanced — blocks R3 completely, but R2 allows defined-risk bearish only.
Strategy selection uses scoring (VRP fit + risk profile + direction match) instead of
arbitrary first-match. Delta capped at 40 for sell legs.
"""

from __future__ import annotations

from src.domain.services.advisor.models import LegSpec, PremiumAdvice, PremiumStrategyDef, VRPResult
from src.domain.services.advisor.strategy_registry import get_strategies_for_regime
from src.domain.services.advisor.vrp import estimate_strike_bsm


class PremiumAdvisor:
    """Selects optimal premium strategy based on VRP, regime, and market conditions."""

    def advise(
        self,
        symbol: str,
        spot: float,
        regime: str,
        vrp: VRPResult,
        term_structure_ratio: float,
        earnings_days_away: int | None,
        trend_direction: str,
    ) -> PremiumAdvice:
        # Gate 1: R3 always blocked. R2 only allows defined-risk bearish (handled below).
        if regime == "R3":
            return self._make_advice(
                symbol=symbol,
                vrp=vrp,
                term_structure_ratio=term_structure_ratio,
                regime=regime,
                action="BLOCKED",
                reasoning=[f"Rebound regime ({regime}) — no premium selling"],
            )

        # Gate 2: VRP
        if vrp.vrp_zscore < 0:
            return self._make_advice(
                symbol=symbol,
                vrp=vrp,
                term_structure_ratio=term_structure_ratio,
                regime=regime,
                action="BLOCKED",
                reasoning=["Negative VRP — realized vol exceeds implied vol"],
            )

        # Gate 3: Term structure
        if term_structure_ratio > 1.2:
            return self._make_advice(
                symbol=symbol,
                vrp=vrp,
                term_structure_ratio=term_structure_ratio,
                regime=regime,
                action="BLOCKED",
                reasoning=["Inverted term structure (VIX/VIX3M > 1.2)"],
            )

        # Gate 4: Earnings proximity
        if earnings_days_away is not None and -3 <= earnings_days_away <= 5:
            return self._make_advice(
                symbol=symbol,
                vrp=vrp,
                term_structure_ratio=term_structure_ratio,
                regime=regime,
                action="HOLD",
                confidence=20,
                earnings_warning=f"Earnings in {earnings_days_away} days",
                reasoning=[f"Earnings proximity ({earnings_days_away} days) — hold off"],
            )

        # Select strategy
        candidates = get_strategies_for_regime(regime)

        # Filter by trend direction
        if trend_direction == "up":
            filtered = [s for s in candidates if s.direction in ("bullish", "neutral")]
        elif trend_direction == "down":
            filtered = [s for s in candidates if s.direction in ("bearish", "neutral")]
        else:
            filtered = candidates

        # In R2, direction filtering is strict — don't fallback to mismatched direction.
        # For other regimes, fall back to all regime-compatible strategies.
        if not filtered and regime != "R2":
            filtered = candidates

        # Prefer defined-risk when IV is elevated
        if vrp.iv_percentile > 75 and filtered:
            defined = [s for s in filtered if s.risk_profile == "defined"]
            if defined:
                filtered = defined

        if not filtered:
            return self._make_advice(
                symbol=symbol,
                vrp=vrp,
                term_structure_ratio=term_structure_ratio,
                regime=regime,
                action="HOLD",
                confidence=10,
                reasoning=["No strategy fits current conditions"],
            )

        # Score-based strategy selection instead of filtered[0]
        strategy = self._rank_strategies(filtered, vrp, trend_direction)

        # Delta adjustment by VRP z-score
        if vrp.vrp_zscore > 1.0:
            delta_adj = 10
        elif vrp.vrp_zscore > 0.5:
            delta_adj = 0
        else:
            delta_adj = -5

        # Build legs with estimated strikes
        iv_decimal = vrp.iv30 / 100.0
        legs: list[LegSpec] = []
        for tmpl in strategy.leg_templates:
            adjusted_delta = max(5, tmpl.delta_target + delta_adj)
            # Cap sell-leg delta at 40
            if tmpl.side == "sell":
                adjusted_delta = min(40, adjusted_delta)
            est_strike = estimate_strike_bsm(
                spot=spot,
                iv=iv_decimal,
                dte=tmpl.dte_target,
                target_delta=adjusted_delta / 100.0,
                option_type=tmpl.option_type,
            )
            legs.append(
                LegSpec(
                    side=tmpl.side,
                    option_type=tmpl.option_type,
                    target_delta=adjusted_delta,
                    target_dte=tmpl.dte_target,
                    estimated_strike=est_strike,
                )
            )

        confidence = self._compute_confidence(vrp, term_structure_ratio, regime)
        reasoning = self._build_reasoning(
            vrp, term_structure_ratio, regime, trend_direction, strategy
        )

        return PremiumAdvice(
            symbol=symbol,
            vrp_zscore=vrp.vrp_zscore,
            iv_percentile=vrp.iv_percentile,
            term_structure_ratio=term_structure_ratio,
            regime=regime,
            action="SELL",
            strategy=strategy.name,
            display_name=strategy.display_name,
            confidence=confidence,
            legs=legs,
            earnings_warning=None,
            reasoning=reasoning,
        )

    @staticmethod
    def _make_advice(
        *,
        symbol: str,
        vrp: VRPResult,
        term_structure_ratio: float,
        regime: str,
        action: str,
        reasoning: list[str],
        confidence: float = 0,
        earnings_warning: str | None = None,
    ) -> PremiumAdvice:
        """Build a non-SELL PremiumAdvice (BLOCKED / HOLD) with common fields."""
        return PremiumAdvice(
            symbol=symbol,
            vrp_zscore=vrp.vrp_zscore,
            iv_percentile=vrp.iv_percentile,
            term_structure_ratio=term_structure_ratio,
            regime=regime,
            action=action,
            strategy=None,
            display_name=None,
            confidence=confidence,
            legs=[],
            earnings_warning=earnings_warning,
            reasoning=reasoning,
        )

    def _rank_strategies(
        self,
        candidates: list[PremiumStrategyDef],
        vrp: VRPResult,
        trend: str,
    ) -> PremiumStrategyDef:
        """Score and rank strategies. Higher score = better fit."""

        def score(s: PremiumStrategyDef) -> float:
            sc = 0.0
            if s.risk_profile == "defined" and vrp.iv_percentile > 60:
                sc += 10
            if trend == "up" and s.direction == "bullish":
                sc += 5
            elif trend == "down" and s.direction == "bearish":
                sc += 5
            elif trend == "sideways" and s.direction == "neutral":
                sc += 5
            if len(s.leg_templates) >= 2:
                sc += 3
            return sc

        return max(candidates, key=score)

    def _compute_confidence(self, vrp: VRPResult, ts_ratio: float, regime: str) -> float:
        result = 50.0
        result += min(vrp.vrp_zscore * 12.5, 25.0)
        if ts_ratio < 0.9:
            result += 15.0
        elif ts_ratio < 1.0:
            result += 10.0
        if regime == "R0":
            result += 10.0
        elif regime == "R1":
            result += 5.0
        return min(max(result, 0), 100)

    def _build_reasoning(
        self,
        vrp: VRPResult,
        ts_ratio: float,
        regime: str,
        trend: str,
        strategy: PremiumStrategyDef,
    ) -> list[str]:
        reasons: list[str] = []
        if vrp.vrp_zscore > 1.0:
            reasons.append(f"VRP z-score {vrp.vrp_zscore:.2f} — strong edge for selling premium")
        elif vrp.vrp_zscore > 0.5:
            reasons.append(f"VRP z-score {vrp.vrp_zscore:.2f} — moderate edge")
        else:
            reasons.append(f"VRP z-score {vrp.vrp_zscore:.2f} — marginal edge")

        if ts_ratio < 1.0:
            reasons.append(f"Term structure in contango ({ts_ratio:.2f}) — supportive")
        else:
            reasons.append(f"Term structure flat ({ts_ratio:.2f}) — caution")

        regime_names = {"R0": "Healthy Uptrend", "R1": "Choppy/Extended", "R2": "Risk-Off"}
        reasons.append(
            f"Regime {regime} ({regime_names.get(regime, 'Unknown')})"
            f" — {strategy.direction} strategies preferred"
        )
        reasons.append(
            f"IV at {vrp.iv_percentile:.0f}th percentile"
            f" — {'elevated' if vrp.iv_percentile > 60 else 'normal'}"
        )

        return reasons
