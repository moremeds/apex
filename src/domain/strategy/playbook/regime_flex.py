"""
RegimeFlex Strategy - Regime-Based Gross Exposure Scaler.

Computes and emits a regime-aware gross exposure signal with smooth
ramping on regime transitions. Does NOT place trades directly -
downstream consumers (e.g., portfolio managers or other strategies)
use the emitted gross exposure target to scale positions.

Edge: Smoothly adapts gross exposure to market regime, preventing
sudden de-grossing on transient regime flips via linear ramping.

Parameters (6):
    r0_gross_pct: Gross exposure in R0 (default 1.0)
    r1_gross_pct: Gross exposure in R1 (default 0.6)
    r3_gross_pct: Gross exposure in R3 (default 0.3)
    ramp_bars: Bars to ramp after regime change (default 10)
    min_dwell_bars: Min bars before switching (default 10)
    switch_cooldown_bars: Cooldown after switch (default 15)
"""

import logging
from typing import Dict, List, Optional

from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...signals.indicators.regime.models import MarketRegime
from ..base import Strategy, StrategyContext, TradingSignal
from ..regime_gate import RegimeGate, RegimePolicy
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy(
    "regime_flex",
    description="Regime-switched meta-strategy wrapping PulseDip + SqueezePlay",
    author="Apex",
    version="1.0",
    max_params=6,
    tier=2,
)
class RegimeFlexStrategy(Strategy):
    """
    RegimeFlex: Regime-based gross exposure signal emitter.

    Computes and emits gross exposure targets with smooth ramping:
    - R0 (Healthy): Full allocation (r0_gross_pct)
    - R1 (Choppy): Reduced allocation (r1_gross_pct)
    - R3 (Rebound): Small allocation (r3_gross_pct)
    - R2 (Risk-Off): De-gross to 0% over ramp_bars
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        r0_gross_pct: float = 1.0,
        r1_gross_pct: float = 0.6,
        r3_gross_pct: float = 0.3,
        ramp_bars: int = 10,
        min_dwell_bars: int = 10,
        switch_cooldown_bars: int = 15,
    ):
        super().__init__(strategy_id, symbols, context)

        self.r0_gross_pct = r0_gross_pct
        self.r1_gross_pct = r1_gross_pct
        self.r3_gross_pct = r3_gross_pct
        self.ramp_bars = ramp_bars

        # Gross target by regime
        self._gross_targets: Dict[str, float] = {
            "R0": r0_gross_pct,
            "R1": r1_gross_pct,
            "R2": 0.0,  # R2 = fully de-grossed
            "R3": r3_gross_pct,
        }

        # Regime gate with meta-strategy policy
        self._regime_gate = RegimeGate(
            policy=RegimePolicy(
                allowed_regimes=["R0", "R1", "R3"],
                min_dwell_bars=min_dwell_bars,
                switch_cooldown_bars=switch_cooldown_bars,
                forced_degross_regimes=["R2"],
                degross_target_pct=0.0,
                size_factors={
                    "R0": r0_gross_pct,
                    "R1": r1_gross_pct,
                    "R3": r3_gross_pct,
                },
            )
        )

        # Per-symbol tracking
        self._bar_count: Dict[str, int] = {s: 0 for s in symbols}
        self._current_gross: float = 1.0
        self._target_gross: float = 1.0
        self._ramp_start_bar: int = 0
        self._ramp_start_gross: float = 1.0
        self._last_regime: Optional[MarketRegime] = None

    def on_start(self) -> None:
        logger.info(
            f"RegimeFlex started: {self.symbols} "
            f"(R0={self.r0_gross_pct}, R1={self.r1_gross_pct}, "
            f"R3={self.r3_gross_pct})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process bar - update regime and delegate to sub-strategies."""
        symbol = bar.symbol
        if bar.close is None:
            return

        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_idx = self._bar_count[symbol]

        # Get current regime
        regime = self._get_current_regime(symbol)
        if regime is None:
            return

        # Track regime changes and compute gross scaling
        regime_result = self._regime_gate.evaluate(symbol, regime, bar_idx)

        # Detect regime transitions for ramping
        if regime != self._last_regime and self._last_regime is not None:
            self._start_ramp(regime, bar_idx)
        self._last_regime = regime

        # Compute current gross with smooth ramping
        current_gross = self._compute_ramped_gross(bar_idx)
        self._current_gross = current_gross

        # Emit regime status signal
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=symbol,
                direction="LONG" if current_gross > 0.5 else "FLAT",
                strength=current_gross,
                reason=(
                    f"RegimeFlex: regime={regime.value}, "
                    f"gross={current_gross:.0%}, "
                    f"gate={'allowed' if regime_result.allowed else 'denied'}"
                ),
                metadata={
                    "regime": regime.value,
                    "current_gross": current_gross,
                    "target_gross": self._target_gross,
                    "gate_allowed": regime_result.allowed,
                    "regime_size_factor": regime_result.size_factor,
                },
            )
        )

    def _start_ramp(self, new_regime: MarketRegime, bar_idx: int) -> None:
        """Start smooth ramping to new gross target."""
        self._target_gross = self._gross_targets.get(new_regime.value, 1.0)
        self._ramp_start_bar = bar_idx
        self._ramp_start_gross = self._current_gross

        logger.info(
            f"[{self.strategy_id}] Regime change -> {new_regime.value}, "
            f"ramping gross {self._current_gross:.0%} -> "
            f"{self._target_gross:.0%} over {self.ramp_bars} bars"
        )

    def _compute_ramped_gross(self, bar_idx: int) -> float:
        """Compute gross with linear ramping between start and target."""
        bars_since_ramp = bar_idx - self._ramp_start_bar
        if bars_since_ramp >= self.ramp_bars:
            return self._target_gross

        # Linear interpolation
        progress = bars_since_ramp / self.ramp_bars
        return self._ramp_start_gross + (self._target_gross - self._ramp_start_gross) * progress

    def _get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get current regime."""
        market_regime = self.context.get_market_regime()
        if market_regime is not None:
            return market_regime
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            return regime_output.final_regime
        return None

    def get_current_gross(self) -> float:
        """Get current gross exposure target (for external consumers)."""
        return self._current_gross

    def on_tick(self, tick: QuoteTick) -> None:
        pass

    def on_fill(self, fill: TradeFill) -> None:
        logger.debug(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )
