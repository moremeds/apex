"""
Unit tests for RegimeGate.

Covers: dwell time, cooldown, R2 veto, per-regime trade caps,
size factor, policy loading, and transition tracking.
"""

from __future__ import annotations

import pytest

from src.domain.signals.indicators.regime.models import MarketRegime
from src.domain.strategy.regime_gate import RegimeGate, RegimeGateResult, RegimePolicy


def _default_policy(**overrides: object) -> RegimePolicy:
    defaults = {
        "allowed_regimes": ["R0", "R3"],
        "min_dwell_bars": 5,
        "switch_cooldown_bars": 10,
        "max_trades_per_regime": {"R0": 20, "R3": 5},
        "forced_degross_regimes": ["R2"],
        "degross_target_pct": 0.0,
        "size_factors": {"R0": 1.0, "R1": 0.5, "R3": 0.3},
    }
    defaults.update(overrides)
    return RegimePolicy(**defaults)  # type: ignore[arg-type]


class TestRegimeGateResult:
    def test_allow_creates_allowed_result(self) -> None:
        r = RegimeGateResult.allow(size_factor=0.8, reason="OK")
        assert r.allowed is True
        assert r.forced_exit is False
        assert r.size_factor == 0.8
        assert r.reason == "OK"

    def test_deny_creates_denied_result(self) -> None:
        r = RegimeGateResult.deny(reason="Not in allowed regime")
        assert r.allowed is False
        assert r.forced_exit is False
        assert r.size_factor == 0.0
        assert "Not in allowed" in r.reason

    def test_force_exit_creates_exit_result(self) -> None:
        r = RegimeGateResult.force_exit(reason="R2 veto")
        assert r.allowed is False
        assert r.forced_exit is True
        assert r.size_factor == 0.0


class TestRegimeGateBasic:
    def test_allowed_regime_after_dwell(self) -> None:
        """After enough evaluate() calls, R0 should be allowed."""
        gate = RegimeGate(_default_policy(min_dwell_bars=3, switch_cooldown_bars=3))
        # First call sets up regime as R0 (triggers on_regime_change internally)
        # Then need min_dwell + cooldown calls before allowed
        for i in range(5):
            result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)
        # After 5 calls, bars_in_regime should exceed dwell and cooldown (3 each)
        assert result.allowed is True
        assert result.size_factor == 1.0

    def test_disallowed_regime_returns_deny(self) -> None:
        gate = RegimeGate(_default_policy())
        # R1 is not in allowed_regimes ["R0", "R3"]
        result = gate.evaluate("AAPL", MarketRegime.R1_CHOPPY_EXTENDED, bar_count=100)
        assert result.allowed is False

    def test_degross_regime_returns_force_exit(self) -> None:
        gate = RegimeGate(_default_policy())
        # R2 is in forced_degross_regimes
        result = gate.evaluate("AAPL", MarketRegime.R2_RISK_OFF, bar_count=100)
        assert result.forced_exit is True


class TestRegimeGateDwellTime:
    def test_new_regime_below_dwell_time_is_denied(self) -> None:
        gate = RegimeGate(_default_policy(min_dwell_bars=5, switch_cooldown_bars=5))
        # First call: regime is new (bars_in_regime starts at 1 after increment)
        result1 = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=0)
        assert result1.allowed is False  # Not enough dwell time

        # After 3 more calls (total 4), still below dwell=5
        for i in range(1, 4):
            result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)
        assert result.allowed is False

    def test_regime_exceeds_dwell_and_cooldown(self) -> None:
        gate = RegimeGate(_default_policy(min_dwell_bars=3, switch_cooldown_bars=3))
        # Call enough times for both dwell and cooldown to be satisfied
        for i in range(10):
            result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)
        assert result.allowed is True


class TestRegimeGateCooldown:
    def test_cooldown_blocks_after_switch(self) -> None:
        gate = RegimeGate(_default_policy(min_dwell_bars=1, switch_cooldown_bars=5))
        # Start in R1 (disallowed) to establish regime tracking
        gate.evaluate("AAPL", MarketRegime.R1_CHOPPY_EXTENDED, bar_count=0)
        gate.evaluate("AAPL", MarketRegime.R1_CHOPPY_EXTENDED, bar_count=1)

        # Switch to R0 (allowed) -- first evaluate in new regime
        result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=2)
        assert result.allowed is False  # Cooldown just started

    def test_after_cooldown_is_allowed(self) -> None:
        gate = RegimeGate(_default_policy(min_dwell_bars=1, switch_cooldown_bars=3))
        # Establish R0
        for i in range(10):
            result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)
        assert result.allowed is True


class TestRegimeGateTradeCount:
    def test_trade_count_within_limit(self) -> None:
        policy = _default_policy(
            max_trades_per_regime={"R0": 3},
            min_dwell_bars=1,
            switch_cooldown_bars=1,
        )
        gate = RegimeGate(policy)

        # Build up dwell time
        for i in range(5):
            gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)

        # Record 3 trades
        for _ in range(3):
            gate.record_trade("AAPL", MarketRegime.R0_HEALTHY_UPTREND)

        # Next evaluate should be denied
        result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=10)
        assert result.allowed is False
        assert "trade" in result.reason.lower()

    def test_trade_count_resets(self) -> None:
        policy = _default_policy(
            max_trades_per_regime={"R0": 3},
            min_dwell_bars=1,
            switch_cooldown_bars=1,
        )
        gate = RegimeGate(policy)
        for i in range(5):
            gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)

        for _ in range(3):
            gate.record_trade("AAPL", MarketRegime.R0_HEALTHY_UPTREND)

        gate.reset_trade_counts("AAPL")
        result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=10)
        assert result.allowed is True


class TestRegimeGateSizeFactor:
    def test_size_factor_applied_for_regime(self) -> None:
        gate = RegimeGate(
            _default_policy(
                size_factors={"R0": 1.0, "R3": 0.3},
                min_dwell_bars=1,
                switch_cooldown_bars=1,
            )
        )
        # Build up dwell time in R3
        for i in range(5):
            result = gate.evaluate("AAPL", MarketRegime.R3_REBOUND_WINDOW, bar_count=i)
        assert result.size_factor == pytest.approx(0.3)

    def test_missing_size_factor_defaults_to_one(self) -> None:
        gate = RegimeGate(
            _default_policy(
                size_factors={},
                min_dwell_bars=1,
                switch_cooldown_bars=1,
            )
        )
        for i in range(5):
            result = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)
        assert result.size_factor == pytest.approx(1.0)


class TestRegimeGateMultiSymbol:
    def test_different_symbols_independent(self) -> None:
        gate = RegimeGate(_default_policy(min_dwell_bars=2, switch_cooldown_bars=2))
        # AAPL has been in R0 for many bars
        for i in range(10):
            gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=i)

        # NVDA just entered R0
        result_nvda = gate.evaluate("NVDA", MarketRegime.R0_HEALTHY_UPTREND, bar_count=0)
        assert result_nvda.allowed is False  # Just started, below dwell

        result_aapl = gate.evaluate("AAPL", MarketRegime.R0_HEALTHY_UPTREND, bar_count=11)
        assert result_aapl.allowed is True  # Been in R0 long enough


class TestRegimePolicy:
    def test_from_dict(self) -> None:
        data = {
            "allowed_regimes": ["R0"],
            "min_dwell_bars": 3,
            "switch_cooldown_bars": 8,
        }
        policy = RegimePolicy.from_dict(data)
        assert policy.allowed_regimes == ["R0"]
        assert policy.min_dwell_bars == 3
        assert policy.switch_cooldown_bars == 8

    def test_default_policy(self) -> None:
        policy = RegimePolicy()
        assert "R0" in policy.allowed_regimes
        assert policy.min_dwell_bars == 5
        assert "R2" in policy.forced_degross_regimes
