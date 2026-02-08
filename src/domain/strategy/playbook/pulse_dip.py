"""
PulseDip Strategy - TrendPulse + DualMACD Confluence.

Composes APEX's TrendPulse (structural trend) with DualMACD (tactical dip
timing), gated by regime and confluence scoring.

Note: Indicators (EMA, RSI, ATR) are computed incrementally bar-by-bar
rather than via TA-Lib, since event-driven strategies process one bar at
a time. The vectorized signal generator uses TA-Lib wrappers.

Edge: Enter on RSI dips within confirmed uptrends (EMA filter + TrendPulse
bullish), exit via deterministic ExitManager priority chain.

Parameters (8, within <=8 budget):
    ema_trend_period: EMA period for trend filter (default 99)
    rsi_period: RSI calculation period (default 14)
    rsi_entry_threshold: RSI below this = dip (default 35)
    atr_stop_mult: ATR multiplier for trailing stop (default 2.5)
    hard_stop_pct: Max loss from entry before catastrophic exit (default 0.08)
    min_confluence_score: Min alignment score (default 20)
    max_hold_bars: Maximum bars to hold (default 40)
    risk_per_trade_pct: Per-trade risk fraction (default 0.02)

Entry (all must be true at bar close):
    1. Close > EMA(ema_trend_period)
    2. RSI < rsi_entry_threshold
    3. Regime in [R0, R3] via RegimeGate
    4. Confluence alignment_score >= min_confluence_score
    5. No existing position

Exit priority (split by intent):
    0. Win-target trail: if gain > 2*ATR, tighten trail to peak - 1.5*ATR
    1. Hard stop: entry * (1 - hard_stop_pct)
    2. ATR trail: peak - atr_stop_mult * ATR (default 2.5)
    3. Regime veto: R2
    4a. Profit-taking: RSI > 65 → partial exit (50%), rest continues
    4b. Deterioration: TrendPulse top_detected → full exit
    5. Time stop: max_hold_bars
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from ...events.domain_events import BarData, QuoteTick, TradeFill
from ...interfaces.execution_provider import OrderRequest
from ...signals.indicators.regime.models import MarketRegime
from ..base import Strategy, StrategyContext, TradingSignal
from ..exit_manager import ExitConditions, ExitManager
from ..position_sizer import PositionSizer
from ..regime_gate import RegimeGate, RegimePolicy
from ..registry import register_strategy

logger = logging.getLogger(__name__)

# Warmup: match signal generator (260 = TrendPulse EMA-99 + DualMACD slow_slow + buffer)
WARMUP_BARS = 260


@dataclass
class PulseDipPosition:
    """Track an open PulseDip position."""

    entry_price: float
    quantity: int
    entry_bar: int
    peak_price: float
    stop_level: float
    trail_level: float
    profit_taken: bool = False


@register_strategy(
    "pulse_dip",
    description="TrendPulse + DualMACD confluence dip-buying strategy",
    author="Apex",
    version="1.0",
    max_params=7,
    tier=1,
)
class PulseDipStrategy(Strategy):
    """
    PulseDip: Buy dips in confirmed uptrends.

    Combines TrendPulse structural trend detection with RSI dip timing,
    gated by regime classification and confluence alignment.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        ema_trend_period: int = 99,
        rsi_period: int = 14,
        rsi_entry_threshold: float = 35.0,
        atr_stop_mult: float = 2.5,
        hard_stop_pct: float = 0.08,
        min_confluence_score: int = 20,
        max_hold_bars: int = 40,
        risk_per_trade_pct: float = 0.02,
    ):
        super().__init__(strategy_id, symbols, context)

        # Validate parameters
        if not 50 <= ema_trend_period <= 200:
            raise ValueError(f"ema_trend_period must be 50-200, got {ema_trend_period}")
        if not 10 <= rsi_period <= 21:
            raise ValueError(f"rsi_period must be 10-21, got {rsi_period}")

        self.ema_trend_period = ema_trend_period
        self.rsi_period = rsi_period
        self.rsi_entry_threshold = rsi_entry_threshold
        self.atr_stop_mult = atr_stop_mult
        self.hard_stop_pct = hard_stop_pct
        self.min_confluence_score = min_confluence_score
        self.max_hold_bars = max_hold_bars
        self.risk_per_trade_pct = risk_per_trade_pct

        # Price history for indicator calculation
        self._closes: Dict[str, Deque[float]] = {
            s: deque(maxlen=max(ema_trend_period + 50, 200)) for s in symbols
        }
        self._highs: Dict[str, Deque[float]] = {s: deque(maxlen=50) for s in symbols}
        self._lows: Dict[str, Deque[float]] = {s: deque(maxlen=50) for s in symbols}

        # Per-symbol state
        self._positions: Dict[str, Optional[PulseDipPosition]] = {s: None for s in symbols}
        self._bar_count: Dict[str, int] = {s: 0 for s in symbols}
        self._ema: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._rsi: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._atr: Dict[str, Optional[float]] = {s: None for s in symbols}

        # Infrastructure
        self._exit_manager = ExitManager(
            hard_stop_pct=hard_stop_pct,
            atr_trail_mult=atr_stop_mult,
            max_hold_bars=max_hold_bars,
        )
        self._regime_gate = RegimeGate(
            policy=RegimePolicy(
                allowed_regimes=["R0", "R3"],
                min_dwell_bars=5,
                switch_cooldown_bars=10,
                size_factors={"R0": 1.0, "R3": 0.3},
                forced_degross_regimes=["R2"],
            )
        )
        self._sizer = PositionSizer(
            portfolio_value=100_000.0,
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=0.10,
        )

    def on_start(self) -> None:
        logger.info(
            f"PulseDip started: {self.symbols} "
            f"(ema={self.ema_trend_period}, rsi_thresh={self.rsi_entry_threshold})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process daily bar - main strategy logic."""
        symbol = bar.symbol
        if bar.close is None or bar.high is None or bar.low is None:
            return

        close = bar.close
        high = bar.high
        low = bar.low

        # Update history
        self._closes[symbol].append(close)
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_idx = self._bar_count[symbol]

        # Calculate indicators
        self._update_ema(symbol)
        self._update_rsi(symbol)
        self._update_atr(symbol)

        # Need warmup
        if bar_idx < WARMUP_BARS:
            return

        ema_val = self._ema[symbol]
        rsi_val = self._rsi[symbol]
        atr_val = self._atr[symbol]
        if ema_val is None or rsi_val is None or atr_val is None:
            return

        # Manage existing position
        pos = self._positions[symbol]
        if pos is not None:
            self._manage_exit(symbol, close, atr_val, rsi_val, bar_idx, pos)
            return

        # Check entry conditions
        self._check_entry(symbol, close, ema_val, rsi_val, atr_val, bar_idx)

    def _check_entry(
        self,
        symbol: str,
        close: float,
        ema_val: float,
        rsi_val: float,
        atr_val: float,
        bar_idx: int,
    ) -> None:
        """Check all entry conditions."""
        # Condition 1: Close above EMA trend filter
        if close <= ema_val:
            return

        # Condition 2: RSI below entry threshold (dip)
        if rsi_val >= self.rsi_entry_threshold:
            return

        # Condition 3: Regime gate
        regime = self._get_current_regime(symbol)
        if regime is not None:
            gate_result = self._regime_gate.evaluate(symbol, regime, bar_idx)
            if not gate_result.allowed:
                return
            size_factor = gate_result.size_factor
        else:
            size_factor = 1.0

        # Condition 4: Confluence alignment
        confluence = self.context.get_confluence(symbol, "1d")
        if confluence is not None:
            if confluence.alignment_score < self.min_confluence_score:
                return

        # Condition 5: No existing position
        if self.context.has_position(symbol):
            return

        # All conditions met - calculate size and enter
        sizing = self._sizer.calculate(
            symbol=symbol,
            price=close,
            atr=atr_val,
            stop_distance_atr_mult=self.atr_stop_mult,
            regime_size_factor=size_factor,
        )

        if not sizing.is_valid:
            return

        # Record trade in regime gate
        if regime is not None:
            self._regime_gate.record_trade(symbol, regime)

        # Calculate stop levels
        hard_stop = self._exit_manager.get_hard_stop_level(close, is_long=True)
        trail_stop = self._exit_manager.get_trailing_stop_level(close, atr_val, is_long=True)

        # Track position
        self._positions[symbol] = PulseDipPosition(
            entry_price=close,
            quantity=sizing.shares,
            entry_bar=bar_idx,
            peak_price=close,
            stop_level=hard_stop,
            trail_level=trail_stop,
        )

        # Emit signal (signal-only mode for live/TUI)
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=symbol,
                direction="LONG",
                strength=min(1.0, rsi_val / 50.0),  # Lower RSI = stronger signal
                target_quantity=float(sizing.shares),
                target_price=close,
                reason=(
                    f"PulseDip ENTRY: RSI={rsi_val:.1f}, " f"EMA_ok=True, size={sizing.shares}"
                ),
                metadata={
                    "rsi": rsi_val,
                    "ema": ema_val,
                    "atr": atr_val,
                    "regime": regime.value if regime else "N/A",
                    "size_factor": size_factor,
                },
            )
        )

        # Request order (backtest simulation mode)
        self.request_order(
            OrderRequest(
                symbol=symbol,
                side="BUY",
                quantity=sizing.shares,
                order_type="MARKET",
            )
        )

        logger.info(
            f"[{self.strategy_id}] PulseDip ENTRY: {symbol} "
            f"@ {close:.2f}, RSI={rsi_val:.1f}, "
            f"size={sizing.shares}, regime={regime.value if regime else 'N/A'}"
        )

    def _manage_exit(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        rsi_val: float,
        bar_idx: int,
        pos: PulseDipPosition,
    ) -> None:
        """Manage exit for existing position via split exit logic.

        Exit checks (in order):
        0. Win-target trail: if gain > 2*ATR, tighten trail to 1.5*ATR
        1-5. ExitManager priority chain (hard stop, ATR trail, regime, deterioration, time)
        Partial: RSI > 65 → sell 50% (profit-taking), rest continues
        """
        # Update peak price
        if close > pos.peak_price:
            pos.peak_price = close

        bars_held = bar_idx - pos.entry_bar

        # --- Pre-ExitManager: Win-target trail tightening ---
        # When profit exceeds 2*ATR, use tighter 1.5*ATR trail to lock in gains
        unrealized_gain = close - pos.entry_price
        if atr_val > 0 and unrealized_gain > 2.0 * atr_val:
            tight_trail = pos.peak_price - 1.5 * atr_val
            if close <= tight_trail:
                self._exit_full(
                    symbol,
                    close,
                    pos,
                    bars_held,
                    f"Win-target trail: {close:.2f} <= {tight_trail:.2f} "
                    f"(peak {pos.peak_price:.2f} - 1.5*ATR)",
                )
                return

        # --- Partial profit-taking: RSI > 65 → sell 50% ---
        if rsi_val > 65 and not pos.profit_taken and pos.quantity > 1:
            half_qty = pos.quantity // 2
            if half_qty > 0:
                pnl_pct = (close - pos.entry_price) / pos.entry_price
                self.emit_signal(
                    TradingSignal(
                        signal_id="",
                        symbol=symbol,
                        direction="LONG",  # still long, just reduced
                        reason=(
                            f"PulseDip PARTIAL: RSI={rsi_val:.1f} > 65, "
                            f"sell 50% (P/L: {pnl_pct:.1%})"
                        ),
                        metadata={
                            "exit_type": "profit_taking",
                            "pnl_pct": pnl_pct,
                            "sold_qty": half_qty,
                        },
                    )
                )
                self.request_order(
                    OrderRequest(
                        symbol=symbol,
                        side="SELL",
                        quantity=half_qty,
                        order_type="MARKET",
                    )
                )
                pos.quantity -= half_qty
                pos.profit_taken = True
                logger.info(
                    f"[{self.strategy_id}] PulseDip PARTIAL: {symbol} "
                    f"sell {half_qty} @ {close:.2f}, RSI={rsi_val:.1f}"
                )
                # Continue — remaining position still managed by ExitManager

        # --- ExitManager checks ---

        # Regime veto
        regime = self._get_current_regime(symbol)
        regime_is_veto = False
        if regime is not None:
            regime_is_veto = regime.value in self._regime_gate.policy.forced_degross_regimes

        # Deterioration: TrendPulse top_detected only (RSI > 65 handled as partial above)
        indicator_exit = False
        indicator_reason = ""
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            tp = getattr(regime_output, "turning_point", None)
            if tp is not None and getattr(tp, "prediction", None) == "TOP_RISK":
                indicator_exit = True
                indicator_reason = "TrendPulse top_detected"

        # Build exit conditions
        conditions = ExitConditions(
            current_price=close,
            entry_price=pos.entry_price,
            peak_price=pos.peak_price,
            current_atr=atr_val,
            bars_held=bars_held,
            regime_is_veto=regime_is_veto,
            indicator_exit=indicator_exit,
            indicator_exit_reason=indicator_reason,
            is_long=True,
        )

        exit_signal = self._exit_manager.evaluate(conditions)
        if exit_signal is not None:
            self._exit_full(
                symbol,
                close,
                pos,
                bars_held,
                exit_signal.reason,
            )

    def _exit_full(
        self,
        symbol: str,
        close: float,
        pos: PulseDipPosition,
        bars_held: int,
        reason: str,
    ) -> None:
        """Execute full position exit."""
        pnl_pct = (close - pos.entry_price) / pos.entry_price
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=symbol,
                direction="FLAT",
                reason=f"PulseDip EXIT: {reason} (P/L: {pnl_pct:.1%})",
                metadata={
                    "exit_reason": reason,
                    "pnl_pct": pnl_pct,
                    "bars_held": bars_held,
                    "partial_taken": pos.profit_taken,
                },
            )
        )
        self.request_order(
            OrderRequest(
                symbol=symbol,
                side="SELL",
                quantity=pos.quantity,
                order_type="MARKET",
            )
        )
        self._positions[symbol] = None
        logger.info(
            f"[{self.strategy_id}] PulseDip EXIT: {symbol} "
            f"@ {close:.2f}, reason={reason}, "
            f"P/L={pnl_pct:.1%}, held={bars_held} bars"
        )

    def _get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get current regime for symbol (prefer market-level)."""
        # Try market-level regime first
        market_regime = self.context.get_market_regime()
        if market_regime is not None:
            return market_regime
        # Fall back to symbol-specific
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            return regime_output.final_regime
        return None

    def _update_ema(self, symbol: str) -> None:
        """Update EMA using exponential smoothing."""
        closes = self._closes[symbol]
        if len(closes) < self.ema_trend_period:
            return
        if self._ema[symbol] is None:
            # Seed with SMA
            self._ema[symbol] = sum(list(closes)[-self.ema_trend_period :]) / self.ema_trend_period
        else:
            k = 2.0 / (self.ema_trend_period + 1)
            prev_ema = self._ema[symbol]  # guaranteed non-None in else branch
            self._ema[symbol] = closes[-1] * k + prev_ema * (1 - k)  # type: ignore[operator]

    def _update_rsi(self, symbol: str) -> None:
        """Update RSI using Wilder's smoothing."""
        closes = self._closes[symbol]
        period = self.rsi_period
        if len(closes) < period + 1:
            return

        prices = list(closes)
        gains = []
        losses = []
        for i in range(len(prices) - period, len(prices)):
            diff = prices[i] - prices[i - 1]
            gains.append(max(0, diff))
            losses.append(max(0, -diff))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            self._rsi[symbol] = 100.0
        else:
            rs = avg_gain / avg_loss
            self._rsi[symbol] = 100.0 - (100.0 / (1.0 + rs))

    def _update_atr(self, symbol: str) -> None:
        """Update ATR (14-period)."""
        closes = self._closes[symbol]
        highs = self._highs[symbol]
        lows = self._lows[symbol]
        period = 14
        if len(closes) < period + 1 or len(highs) < period or len(lows) < period:
            return

        h = list(highs)[-period:]
        l = list(lows)[-period:]
        c = list(closes)[-(period + 1) :]

        trs = []
        for i in range(period):
            tr = max(
                h[i] - l[i],
                abs(h[i] - c[i]),
                abs(l[i] - c[i]),
            )
            trs.append(tr)

        self._atr[symbol] = sum(trs) / period

    def on_tick(self, tick: QuoteTick) -> None:
        """No intraday processing - daily bars only."""
        pass

    def on_fill(self, fill: TradeFill) -> None:
        logger.debug(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )
