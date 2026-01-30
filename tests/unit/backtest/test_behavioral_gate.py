"""
Unit tests for DualMACD behavioral gate.

Tests cover:
- G1: Counterfactual exit isolation (blocked trade exit bar == baseline exit bar)
- G2: Symmetric gate logic (LONG vs SHORT)
- G4: Warmup exclusion
- Trade decision logging and resolution
- Behavioral metrics calculation
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.analysis.dual_macd.behavioral_metrics import BehavioralMetricsCalculator
from src.backtest.analysis.dual_macd.behavioral_models import (
    BehavioralMetrics,
    GatePolicy,
    TradeDecision,
)
from src.backtest.analysis.dual_macd.trade_decision_logger import TradeDecisionLogger
from src.domain.strategy.signals.dual_macd_gate import DualMACDGateSignalGenerator

# ── Fixtures ──────────────────────────────────────────────────


class StubSignalGenerator:
    """Stub base signal generator for testing."""

    def __init__(self, entry_indices: list[int], exit_indices: list[int], n: int = 500):
        self._entry_indices = entry_indices
        self._exit_indices = exit_indices
        self._n = n

    @property
    def warmup_bars(self) -> int:
        return 10

    def generate(
        self,
        data: pd.DataFrame,
        params: dict[str, Any],
        secondary_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        for i in self._entry_indices:
            if i < len(entries):
                entries.iloc[i] = True
        for i in self._exit_indices:
            if i < len(exits):
                exits.iloc[i] = True
        return entries, exits


def _make_price_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close - rng.rand(n) * 0.5,
            "high": close + rng.rand(n) * 1.0,
            "low": close - rng.rand(n) * 1.0,
            "close": close,
            "volume": rng.randint(1000, 10000, n),
        },
        index=dates,
    )


# ── G2: Symmetric Gate Logic ─────────────────────────────────


class TestSymmetricGateLogic:
    """G2: Gate rules must be explicitly directional."""

    def test_long_blocks_deteriorating(self) -> None:
        """LONG gate should block when trend is DETERIORATING."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="LONG",
        )
        # Evaluate gate for a DETERIORATING state
        result = gate._evaluate_gate(
            {"trend_state": "DETERIORATING", "tactical_signal": "NONE"}, ""
        )
        assert result[0] is False  # blocked
        assert result[1] == "BLOCK"

    def test_long_allows_bullish(self) -> None:
        """LONG gate should allow when trend is BULLISH."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="LONG",
        )
        result = gate._evaluate_gate({"trend_state": "BULLISH", "tactical_signal": "NONE"})
        assert result[0] is True
        assert result[1] == "ENTER"

    def test_long_allows_dip_buy_override(self) -> None:
        """LONG gate should allow DETERIORATING if DIP_BUY fires."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="LONG",
        )
        result = gate._evaluate_gate({"trend_state": "DETERIORATING", "tactical_signal": "DIP_BUY"})
        assert result[0] is True  # allowed via override

    def test_short_blocks_improving(self) -> None:
        """SHORT gate should block when trend is IMPROVING."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="SHORT",
        )
        result = gate._evaluate_gate({"trend_state": "IMPROVING", "tactical_signal": "NONE"})
        assert result[0] is False
        assert result[1] == "BLOCK"

    def test_short_allows_bearish(self) -> None:
        """SHORT gate should allow when trend is BEARISH."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="SHORT",
        )
        result = gate._evaluate_gate({"trend_state": "BEARISH", "tactical_signal": "NONE"})
        assert result[0] is True

    def test_short_allows_rally_sell_override(self) -> None:
        """SHORT gate should allow BULLISH if RALLY_SELL fires."""
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="SHORT",
        )
        result = gate._evaluate_gate({"trend_state": "BULLISH", "tactical_signal": "RALLY_SELL"})
        assert result[0] is True


# ── G4: Warmup Exclusion ────────────────────────────────────


class TestWarmupExclusion:
    """G4: Decisions during warmup are excluded from metrics."""

    def test_warmup_end_date_computed(self) -> None:
        data = _make_price_data(500)
        gate = DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([], []),
            direction="LONG",
        )
        warmup_end = gate.get_warmup_end_date(data)
        # Should be a date object after warmup_bars
        assert isinstance(warmup_end, date)
        assert warmup_end > data.index[0].date()

    def test_post_warmup_filter(self) -> None:
        """TradeDecisionLogger.get_post_warmup filters correctly."""
        logger = TradeDecisionLogger()
        d1 = TradeDecision(
            timestamp=date(2020, 1, 10),
            symbol="SPY",
            regime="R0",
            strategy_direction="LONG",
            intended_size=1.0,
            trend_state="BULLISH",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.0,
            gate_strength=0.0,
            allowed=True,
            action="ENTER",
            virtual_entry_price=100.0,
        )
        d2 = TradeDecision(
            timestamp=date(2020, 12, 15),
            symbol="SPY",
            regime="R0",
            strategy_direction="LONG",
            intended_size=1.0,
            trend_state="BULLISH",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.0,
            gate_strength=0.0,
            allowed=True,
            action="ENTER",
            virtual_entry_price=110.0,
        )
        logger.log_decision(d1)
        logger.log_decision(d2)

        warmup_end = date(2020, 6, 1)
        post = logger.get_post_warmup(warmup_end)
        assert len(post) == 1
        assert post[0].timestamp == date(2020, 12, 15)


# ── G1: Counterfactual Exit Isolation ────────────────────────


class TestCounterfactualExitIsolation:
    """G1: Counterfactual exits use base strategy exits only."""

    def test_resolve_counterfactual(self) -> None:
        """Blocked trade gets resolved with base exit price."""
        logger = TradeDecisionLogger()
        decision = TradeDecision(
            timestamp=date(2020, 3, 1),
            symbol="SPY",
            regime="R0",
            strategy_direction="LONG",
            intended_size=1.0,
            trend_state="DETERIORATING",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.0,
            gate_strength=0.0,
            allowed=False,
            action="BLOCK",
            block_reason="LONG blocked: trend=DETERIORATING",
            virtual_entry_price=300.0,
        )
        logger.log_decision(decision)

        # Resolve with exit price from base strategy
        logger.resolve_counterfactual(
            timestamp=date(2020, 3, 1),
            symbol="SPY",
            exit_price=280.0,  # Price went down → correct block
        )

        resolved = logger.decisions[0]
        assert resolved.virtual_exit_price == 280.0
        assert resolved.virtual_pnl_pct is not None
        assert resolved.virtual_pnl_pct < 0  # Loss → gate was correct

    def test_short_counterfactual(self) -> None:
        """SHORT blocked trade PnL calculated correctly."""
        logger = TradeDecisionLogger()
        decision = TradeDecision(
            timestamp=date(2020, 3, 1),
            symbol="SPY",
            regime="R0",
            strategy_direction="SHORT",
            intended_size=1.0,
            trend_state="IMPROVING",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.0,
            gate_strength=0.0,
            allowed=False,
            action="BLOCK",
            block_reason="SHORT blocked",
            virtual_entry_price=300.0,
        )
        logger.log_decision(decision)
        logger.resolve_counterfactual(date(2020, 3, 1), "SPY", 320.0)

        resolved = logger.decisions[0]
        # SHORT at 300, exit at 320 → loss of (300-320)/300 = -6.67%
        assert resolved.virtual_pnl_pct is not None
        assert resolved.virtual_pnl_pct < 0


# ── Behavioral Metrics ────────────────────────────────────────


class TestBehavioralMetrics:
    """Test BehavioralMetricsCalculator."""

    def test_basic_calculation(self) -> None:
        decisions = [
            TradeDecision(
                timestamp=date(2020, 6, 1),
                symbol="SPY",
                regime="R0",
                strategy_direction="LONG",
                intended_size=1.0,
                trend_state="BULLISH",
                tactical_signal="NONE",
                momentum_balance="BALANCED",
                confidence=0.0,
                gate_strength=0.0,
                allowed=True,
                action="ENTER",
                virtual_entry_price=100.0,
            ),
            TradeDecision(
                timestamp=date(2020, 7, 1),
                symbol="SPY",
                regime="R0",
                strategy_direction="LONG",
                intended_size=1.0,
                trend_state="DETERIORATING",
                tactical_signal="NONE",
                momentum_balance="BALANCED",
                confidence=0.0,
                gate_strength=0.0,
                allowed=False,
                action="BLOCK",
                block_reason="blocked",
                virtual_entry_price=110.0,
                virtual_exit_price=105.0,
                virtual_pnl_pct=-0.0455,
            ),
            TradeDecision(
                timestamp=date(2020, 8, 1),
                symbol="SPY",
                regime="R0",
                strategy_direction="LONG",
                intended_size=1.0,
                trend_state="DETERIORATING",
                tactical_signal="NONE",
                momentum_balance="BALANCED",
                confidence=0.0,
                gate_strength=0.0,
                allowed=False,
                action="BLOCK",
                block_reason="blocked",
                virtual_entry_price=108.0,
                virtual_exit_price=112.0,
                virtual_pnl_pct=0.037,
            ),
        ]

        calc = BehavioralMetricsCalculator()
        metrics = calc.calculate(decisions)

        assert metrics.baseline_trade_count == 3
        assert metrics.allowed_trade_count == 1
        assert metrics.blocked_trade_count == 2
        assert metrics.blocked_trade_loss_ratio == 0.5  # 1 of 2 blocked would have lost
        assert metrics.blocked_trade_avg_pnl < 0  # avg of -4.55% and +3.7%

    def test_g3_sharpe_cap(self) -> None:
        """G3: Anti-conservative ceiling caps Sharpe improvement."""
        calc = BehavioralMetricsCalculator()
        decisions = [
            TradeDecision(
                timestamp=date(2020, 6, 1),
                symbol="SPY",
                regime="R0",
                strategy_direction="LONG",
                intended_size=1.0,
                trend_state="BULLISH",
                tactical_signal="NONE",
                momentum_balance="BALANCED",
                confidence=0.0,
                gate_strength=0.0,
                allowed=True,
                action="ENTER",
                virtual_entry_price=100.0,
            ),
        ]
        metrics = calc.calculate(
            decisions,
            gated_sharpe=3.0,
            baseline_sharpe=1.0,
        )
        # Should be capped at 1.0 * 1.25 = 1.25
        assert metrics.allowed_trade_sharpe == 1.25

    def test_hard_constraints(self) -> None:
        calc = BehavioralMetricsCalculator()

        # Passing metrics
        good = BehavioralMetrics(
            blocked_trade_loss_ratio=0.7,
            allowed_trade_count=80,
            baseline_trade_count=100,
            max_dd_gated=0.10,
            max_dd_baseline=0.15,
        )
        assert calc.passes_hard_constraints(good) is True

        # Failing: low loss ratio
        bad = BehavioralMetrics(
            blocked_trade_loss_ratio=0.4,
            allowed_trade_count=80,
            baseline_trade_count=100,
            max_dd_gated=0.10,
            max_dd_baseline=0.15,
        )
        assert calc.passes_hard_constraints(bad) is False


# ── JSONL Export/Import ───────────────────────────────────────


class TestDecisionLoggerIO:
    def test_jsonl_roundtrip(self, tmp_path: Any) -> None:
        logger = TradeDecisionLogger()
        decision = TradeDecision(
            timestamp=date(2020, 6, 1),
            symbol="SPY",
            regime="R0",
            strategy_direction="LONG",
            intended_size=1.0,
            trend_state="BULLISH",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.5,
            gate_strength=0.0,
            allowed=True,
            action="ENTER",
            virtual_entry_price=100.0,
        )
        logger.log_decision(decision)

        path = tmp_path / "test.jsonl"
        logger.to_jsonl(path)

        loaded = TradeDecisionLogger.from_jsonl(path)
        assert len(loaded.decisions) == 1
        assert loaded.decisions[0].symbol == "SPY"
        assert loaded.decisions[0].confidence == 0.5

    def test_to_dataframe(self) -> None:
        logger = TradeDecisionLogger()
        decision = TradeDecision(
            timestamp=date(2020, 6, 1),
            symbol="SPY",
            regime="R0",
            strategy_direction="LONG",
            intended_size=1.0,
            trend_state="BULLISH",
            tactical_signal="NONE",
            momentum_balance="BALANCED",
            confidence=0.0,
            gate_strength=0.0,
            allowed=True,
            action="ENTER",
            virtual_entry_price=100.0,
        )
        logger.log_decision(decision)

        df = logger.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "SPY"


# ── Gate Policy Tests ────────────────────────────────────────


class TestGatePolicies:
    """Test per-sector gate policies (SIZE_DOWN / BYPASS / BLOCK)."""

    def _make_gate_with_policies(
        self,
        policies: Dict[str, GatePolicy],
        symbol_to_sector: Dict[str, str],
    ) -> DualMACDGateSignalGenerator:
        return DualMACDGateSignalGenerator(
            base_generator=StubSignalGenerator([300], [310]),
            direction="LONG",
            gate_policies=policies,
            symbol_to_sector=symbol_to_sector,
        )

    def test_size_down_policy(self) -> None:
        """Semiconductors sector gets SIZE_DOWN with size_factor=0.5."""
        gate = self._make_gate_with_policies(
            policies={
                "default": GatePolicy(action_on_block="BLOCK"),
                "semiconductors": GatePolicy(action_on_block="SIZE_DOWN", size_factor=0.5),
            },
            symbol_to_sector={"NVDA": "semiconductors"},
        )
        state = {"trend_state": "DETERIORATING", "tactical_signal": "NONE"}
        allowed, action, reason, size_factor = gate._evaluate_gate(state, "NVDA")
        assert allowed is True
        assert action == "SIZE_DOWN"
        assert size_factor == 0.5
        assert reason is not None

    def test_bypass_policy(self) -> None:
        """Energy sector gets BYPASS (always allowed, full size)."""
        gate = self._make_gate_with_policies(
            policies={
                "default": GatePolicy(action_on_block="BLOCK"),
                "energy": GatePolicy(action_on_block="BYPASS"),
            },
            symbol_to_sector={"XOM": "energy"},
        )
        state = {"trend_state": "BEARISH", "tactical_signal": "NONE"}
        allowed, action, reason, size_factor = gate._evaluate_gate(state, "XOM")
        assert allowed is True
        assert action == "BYPASS"
        assert size_factor == 1.0
        assert reason is None

    def test_default_policy_fallback(self) -> None:
        """Unknown sector falls back to default BLOCK policy."""
        gate = self._make_gate_with_policies(
            policies={
                "default": GatePolicy(action_on_block="BLOCK"),
                "semiconductors": GatePolicy(action_on_block="SIZE_DOWN", size_factor=0.5),
            },
            symbol_to_sector={"AAPL": "technology"},
        )
        state = {"trend_state": "DETERIORATING", "tactical_signal": "NONE"}
        allowed, action, reason, size_factor = gate._evaluate_gate(state, "AAPL")
        assert allowed is False
        assert action == "BLOCK"
        assert size_factor == 0.0

    def test_no_policy_favorable_trend(self) -> None:
        """Favorable trend always allows regardless of policy."""
        gate = self._make_gate_with_policies(
            policies={"default": GatePolicy(action_on_block="BLOCK")},
            symbol_to_sector={},
        )
        state = {"trend_state": "BULLISH", "tactical_signal": "NONE"}
        allowed, action, reason, size_factor = gate._evaluate_gate(state, "SPY")
        assert allowed is True
        assert action == "ENTER"
        assert size_factor == 1.0
