"""Tests for premium strategy registry."""

from src.domain.services.advisor.strategy_registry import (
    STRATEGY_REGISTRY,
    get_strategies_by_direction,
    get_strategies_for_regime,
)


class TestStrategyRegistry:
    def test_registry_not_empty(self):
        assert len(STRATEGY_REGISTRY) >= 6

    def test_no_duplicate_names(self):
        names = [s.name for s in STRATEGY_REGISTRY]
        assert len(names) == len(set(names))

    def test_all_have_legs(self):
        for s in STRATEGY_REGISTRY:
            assert len(s.leg_templates) >= 1, f"{s.name} has no legs"

    def test_regime_fit_valid(self):
        valid = {"R0", "R1", "R2", "R3"}
        for s in STRATEGY_REGISTRY:
            assert s.regime_fit.issubset(valid), f"{s.name} has invalid regime_fit"


class TestGetStrategiesForRegime:
    def test_r0_includes_bullish(self):
        strats = get_strategies_for_regime("R0")
        names = {s.name for s in strats}
        assert "short_put" in names

    def test_r0_includes_iron_condor(self):
        strats = get_strategies_for_regime("R0")
        names = {s.name for s in strats}
        assert "iron_condor" in names

    def test_r2_only_defined_risk_bearish(self):
        strats = get_strategies_for_regime("R2")
        for s in strats:
            assert s.risk_profile == "defined", f"{s.name} in R2 should be defined-risk"
            assert s.direction == "bearish", f"{s.name} in R2 should be bearish"

    def test_r2_includes_bear_call_spread(self):
        strats = get_strategies_for_regime("R2")
        names = {s.name for s in strats}
        assert "bear_call_spread" in names

    def test_r2_excludes_short_call(self):
        strats = get_strategies_for_regime("R2")
        names = {s.name for s in strats}
        assert "short_call" not in names


class TestGetStrategiesByDirection:
    def test_bullish(self):
        strats = get_strategies_by_direction("bullish")
        assert all(s.direction == "bullish" for s in strats)

    def test_neutral(self):
        strats = get_strategies_by_direction("neutral")
        assert all(s.direction == "neutral" for s in strats)
