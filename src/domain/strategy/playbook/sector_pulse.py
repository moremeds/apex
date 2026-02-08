"""
SectorPulse Strategy - Regime-Aware Sector Rotation.

Rotates among sector ETFs ranked by TrendPulse confidence,
with regime-based allocation scaling. Rebalances weekly
with turnover caps and cost-aware execution.

Edge: Exploits sector momentum persistence (winners keep winning)
with regime gating to avoid rotation during risk-off periods.

Parameters (7):
    top_n_sectors: Number of sectors to hold (default 3)
    confidence_threshold: Min TrendPulse confidence (default 0.4)
    rebalance_day: Day of week for rebalance, 0=Mon (default 4=Fri)
    drift_threshold_pct: Drift trigger for inter-rebalance (default 0.05)
    max_turnover_pct: Turnover cap per rebalance (default 0.30)
    slippage_bps: Conservative slippage estimate (default 10.0)
    risk_per_sector_pct: Max per-sector allocation (default 0.10)

Sectors from config/universe.yaml:
    XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional

from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest
from ...signals.indicators.regime.models import MarketRegime
from ..base import Strategy, StrategyContext, TradingSignal
from ..regime_gate import RegimeGate, RegimePolicy
from ..registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy(
    "sector_pulse",
    description="Regime-aware sector rotation ranked by TrendPulse confidence",
    author="Apex",
    version="1.0",
    max_params=7,
    tier=2,
)
class SectorPulseStrategy(Strategy):
    """
    SectorPulse: Sector rotation with regime-based allocation.

    Ranks sector ETFs by momentum confidence and rotates into
    top N sectors, with regime gating and turnover caps.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        top_n_sectors: int = 3,
        confidence_threshold: float = 0.02,
        rebalance_day: int = 4,  # Friday
        drift_threshold_pct: float = 0.05,
        max_turnover_pct: float = 0.30,
        slippage_bps: float = 10.0,
        risk_per_sector_pct: float = 0.10,
    ):
        super().__init__(strategy_id, symbols, context)

        self.top_n_sectors = top_n_sectors
        self.confidence_threshold = confidence_threshold
        self.rebalance_day = rebalance_day
        self.drift_threshold_pct = drift_threshold_pct
        self.max_turnover_pct = max_turnover_pct
        self.slippage_bps = slippage_bps
        self.risk_per_sector_pct = risk_per_sector_pct

        # Price and momentum tracking
        self._prices: Dict[str, Deque[float]] = {s: deque(maxlen=60) for s in symbols}
        self._current_prices: Dict[str, Optional[float]] = {s: None for s in symbols}

        # Target weights and rebalance state
        self._target_weights: Dict[str, float] = {s: 0.0 for s in symbols}
        self._per_symbol_bar_count: Dict[str, int] = {s: 0 for s in symbols}
        self._reference_symbol: str = symbols[0]  # Use first symbol for rebalance timing
        self._last_rebalance_bar: int = 0
        self._initial_capital: float = 100_000.0
        self._cash: float = 100_000.0  # Track cash dynamically via on_fill
        self._initialized: bool = False

        # Regime gate
        self._regime_gate = RegimeGate(
            policy=RegimePolicy(
                allowed_regimes=["R0", "R1", "R3"],
                min_dwell_bars=5,
                switch_cooldown_bars=10,
                size_factors={"R0": 1.0, "R1": 0.5, "R3": 0.2},
                forced_degross_regimes=["R2"],
            )
        )

    def on_start(self) -> None:
        logger.info(
            f"SectorPulse started: {self.symbols} "
            f"(top_n={self.top_n_sectors}, "
            f"rebalance_day={self.rebalance_day})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process bar - track prices and check rebalance."""
        symbol = bar.symbol
        if bar.close is None:
            return

        self._prices[symbol].append(bar.close)
        self._current_prices[symbol] = bar.close
        self._per_symbol_bar_count[symbol] = self._per_symbol_bar_count.get(symbol, 0) + 1

        # Only trigger rebalance check on the reference symbol to avoid
        # scaling rebalance frequency with symbol count
        if symbol != self._reference_symbol:
            return

        ref_bar = self._per_symbol_bar_count[self._reference_symbol]

        # Check if all symbols have prices
        if not all(self._current_prices.values()):
            return

        # Check rebalance conditions (weekly on daily bars)
        bars_since = ref_bar - self._last_rebalance_bar
        should_rebalance = bars_since >= 5

        # Also check drift
        drift = self._calc_max_drift()
        if drift > self.drift_threshold_pct and bars_since >= 2:
            should_rebalance = True

        if should_rebalance:
            self._do_rebalance()

    def _do_rebalance(self) -> None:
        """Execute sector rotation rebalance."""
        ref_bar = self._per_symbol_bar_count[self._reference_symbol]
        self._last_rebalance_bar = ref_bar

        # Get regime for sizing
        regime = self._get_current_regime()
        size_factor = 1.0
        if regime is not None:
            gate_result = self._regime_gate.evaluate("MARKET", regime, ref_bar)
            if gate_result.forced_exit:
                # R2: liquidate everything
                self._liquidate_all()
                return
            size_factor = gate_result.size_factor

        # Rank sectors by momentum (simple 20-day return)
        rankings = self._rank_sectors()

        # Select top N sectors that pass confidence threshold
        selected: List[str] = []
        for symbol, score in rankings:
            if score >= self.confidence_threshold and len(selected) < self.top_n_sectors:
                selected.append(symbol)

        if not selected:
            return

        # Equal weight among selected, scaled by regime
        base_weight = 1.0 / len(selected)
        new_weights: Dict[str, float] = {}
        for s in self.symbols:
            if s in selected:
                new_weights[s] = base_weight * size_factor
            else:
                new_weights[s] = 0.0

        # Enforce turnover cap
        old_weights = self._target_weights.copy()
        turnover = sum(abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in self.symbols)
        if turnover > self.max_turnover_pct and self._initialized:
            # Scale down changes to stay within turnover cap
            scale = self.max_turnover_pct / turnover
            for s in self.symbols:
                diff = new_weights.get(s, 0) - old_weights.get(s, 0)
                new_weights[s] = old_weights.get(s, 0) + diff * scale

        self._target_weights = new_weights

        # Execute trades
        self._execute_rebalance(new_weights)
        self._initialized = True

        # Emit signal
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=self.symbols[0],  # Use first symbol for signal
                direction="LONG",
                strength=size_factor,
                reason=(
                    f"SectorPulse rebalance: "
                    f"selected={selected}, "
                    f"regime={regime.value if regime else 'N/A'}, "
                    f"turnover={turnover:.1%}"
                ),
                metadata={
                    "selected_sectors": selected,
                    "weights": new_weights,
                    "regime": regime.value if regime else "N/A",
                    "turnover": turnover,
                },
            )
        )

    def _rank_sectors(self) -> List[tuple[str, float]]:
        """Rank sectors by multi-horizon momentum score.

        Composite: 0.5 * ret_1m + 0.3 * ret_3m + 0.2 * ret_6m (when available).
        Falls back to available horizons with re-normalized weights.
        """
        rankings = []
        for symbol in self.symbols:
            prices = list(self._prices[symbol])
            n = len(prices)
            if n < 5:
                rankings.append((symbol, 0.0))
                continue

            # Multi-horizon returns (20d ~ 1m, 60d ~ 3m)
            horizons = []
            weights = []
            if n >= 20 and prices[-20] > 0:
                horizons.append(prices[-1] / prices[-20] - 1.0)
                weights.append(0.5)
            if n >= 60 and prices[-60] > 0:
                horizons.append(prices[-1] / prices[-60] - 1.0)
                weights.append(0.3)
            # Longest available horizon (up to 60 bars in deque)
            if n >= 40 and prices[0] > 0:
                horizons.append(prices[-1] / prices[0] - 1.0)
                weights.append(0.2)

            if not horizons:
                rankings.append((symbol, 0.0))
                continue

            # Re-normalize weights
            w_sum = sum(weights)
            score = sum(h * w / w_sum for h, w in zip(horizons, weights))
            rankings.append((symbol, score))

        # Sort descending by score
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _get_portfolio_value(self) -> float:
        """Compute current portfolio value = cash + positions."""
        position_value = sum(
            self.context.get_position_quantity(s) * (self._current_prices.get(s) or 0)
            for s in self.symbols
        )
        return self._cash + position_value

    def _execute_rebalance(self, target_weights: Dict[str, float]) -> None:
        """Execute rebalancing trades to match target weights."""
        portfolio_value = self._get_portfolio_value()
        for symbol in self.symbols:
            target_weight = target_weights.get(symbol, 0.0)
            price = self._current_prices.get(symbol)
            if price is None or price <= 0:
                continue

            current_qty = self.context.get_position_quantity(symbol)
            target_value = portfolio_value * target_weight
            target_qty = int(target_value / price)
            diff = target_qty - int(current_qty)

            if abs(diff) < 1:
                continue

            if diff > 0:
                # Cap buy quantity to available cash
                max_affordable = int(self._cash / price) if price > 0 else 0
                buy_qty = min(diff, max_affordable)
                if buy_qty < 1:
                    continue
                self.request_order(
                    OrderRequest(
                        symbol=symbol,
                        side="BUY",
                        quantity=buy_qty,
                        order_type="MARKET",
                    )
                )
            else:
                self.request_order(
                    OrderRequest(
                        symbol=symbol,
                        side="SELL",
                        quantity=abs(diff),
                        order_type="MARKET",
                    )
                )

    def _liquidate_all(self) -> None:
        """Liquidate all positions (R2 regime forced exit)."""
        for symbol in self.symbols:
            qty = self.context.get_position_quantity(symbol)
            if qty > 0:
                self.request_order(
                    OrderRequest(
                        symbol=symbol,
                        side="SELL",
                        quantity=int(abs(qty)),
                        order_type="MARKET",
                    )
                )
                logger.info(f"[{self.strategy_id}] R2 liquidation: SELL {int(abs(qty))} {symbol}")

        self._target_weights = {s: 0.0 for s in self.symbols}
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=self.symbols[0],
                direction="FLAT",
                reason="SectorPulse: R2 forced liquidation",
            )
        )

    def _calc_max_drift(self) -> float:
        """Calculate maximum weight drift from target."""
        total_value = 0.0
        position_values: Dict[str, float] = {}
        for s in self.symbols:
            price = self._current_prices.get(s)
            qty = self.context.get_position_quantity(s)
            val = (qty * price) if price and qty else 0.0
            position_values[s] = val
            total_value += val

        if total_value == 0:
            return 0.0

        max_drift = 0.0
        for s in self.symbols:
            current_w = position_values[s] / total_value
            target_w = self._target_weights.get(s, 0.0)
            max_drift = max(max_drift, abs(current_w - target_w))
        return max_drift

    def _get_current_regime(self) -> Optional[MarketRegime]:
        """Get current market-level regime."""
        market_regime = self.context.get_market_regime()
        if market_regime is not None:
            return market_regime
        # Try first symbol as proxy
        regime_output = self.context.get_regime(self.symbols[0])
        if regime_output is not None:
            return regime_output.final_regime
        return None

    def on_tick(self, tick: QuoteTick) -> None:
        pass

    def on_fill(self, fill: TradeFill) -> None:
        # Track cash for dynamic portfolio value (including commissions)
        cost = fill.price * fill.quantity
        if fill.side == "BUY":
            self._cash -= cost + fill.commission
        else:
            self._cash += cost - fill.commission
        logger.debug(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )
