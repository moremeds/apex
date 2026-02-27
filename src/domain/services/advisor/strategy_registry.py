"""Extensible registry of premium-selling strategies."""

from __future__ import annotations

from src.domain.services.advisor.models import LegTemplate, PremiumStrategyDef

STRATEGY_REGISTRY: list[PremiumStrategyDef] = [
    PremiumStrategyDef(
        name="short_put",
        display_name="Short Put",
        direction="bullish",
        risk_profile="undefined",
        regime_fit=frozenset({"R0", "R1"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="put", delta_target=25, dte_target=35),
        ),
    ),
    PremiumStrategyDef(
        name="short_call",
        display_name="Short Call",
        direction="bearish",
        risk_profile="undefined",
        regime_fit=frozenset({"R1"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="call", delta_target=25, dte_target=35),
        ),
    ),
    PremiumStrategyDef(
        name="bull_put_spread",
        display_name="Bull Put Spread",
        direction="bullish",
        risk_profile="defined",
        regime_fit=frozenset({"R0", "R1"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="put", delta_target=25, dte_target=35),
            LegTemplate(side="buy", option_type="put", delta_target=10, dte_target=35),
        ),
    ),
    PremiumStrategyDef(
        name="bear_call_spread",
        display_name="Bear Call Spread",
        direction="bearish",
        risk_profile="defined",
        regime_fit=frozenset({"R1", "R2"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="call", delta_target=25, dte_target=35),
            LegTemplate(side="buy", option_type="call", delta_target=10, dte_target=35),
        ),
    ),
    PremiumStrategyDef(
        name="iron_condor",
        display_name="Iron Condor",
        direction="neutral",
        risk_profile="defined",
        regime_fit=frozenset({"R0", "R1"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="put", delta_target=20, dte_target=35),
            LegTemplate(side="buy", option_type="put", delta_target=10, dte_target=35),
            LegTemplate(side="sell", option_type="call", delta_target=20, dte_target=35),
            LegTemplate(side="buy", option_type="call", delta_target=10, dte_target=35),
        ),
    ),
    PremiumStrategyDef(
        name="short_strangle",
        display_name="Short Strangle",
        direction="neutral",
        risk_profile="undefined",
        regime_fit=frozenset({"R0"}),
        leg_templates=(
            LegTemplate(side="sell", option_type="put", delta_target=20, dte_target=35),
            LegTemplate(side="sell", option_type="call", delta_target=20, dte_target=35),
        ),
    ),
]


def get_strategies_for_regime(regime: str) -> list[PremiumStrategyDef]:
    """Return strategies that fit the given regime."""
    return [s for s in STRATEGY_REGISTRY if regime in s.regime_fit]


def get_strategies_by_direction(direction: str) -> list[PremiumStrategyDef]:
    """Return strategies matching the given direction."""
    return [s for s in STRATEGY_REGISTRY if s.direction == direction]
