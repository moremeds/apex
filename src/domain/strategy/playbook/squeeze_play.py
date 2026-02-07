"""
SqueezePlay Strategy - Bollinger/Keltner Squeeze Breakout.

Detects Bollinger Bands inside Keltner Channels (squeeze), then enters
on confirmed release with directional bias from TrendPulse.

Note: Indicators (BB, KC, ATR, ADX) are computed incrementally bar-by-bar
rather than via TA-Lib, since event-driven strategies process one bar at
a time. The vectorized signal generator uses TA-Lib wrappers.

Edge: Low-volatility compression predicts expansion. The fake-release
mitigation (persistence filter) avoids premature entries.

Parameters (8, at budget cap):
    bb_period: Bollinger period (default 20)
    bb_std: Bollinger std dev multiplier (default 2.0)
    kc_multiplier: Keltner channel multiplier (default 1.5)
    release_persist_bars: Bars release must persist (default 2)
    close_outside_bars: Bars close must be outside BB (default 2)
    atr_stop_mult: ATR multiplier for trailing stop (default 2.5)
    adx_min: Minimum ADX for trend confirmation (default 20.0)
    risk_per_trade_pct: Per-trade risk fraction (default 0.02)

Fake-release mitigation:
    - Track consecutive bars where squeeze is OFF (release_count)
    - Track consecutive bars where close is outside BB (outside_count)
    - Entry requires BOTH: release_count >= release_persist_bars
      AND outside_count >= close_outside_bars

Regime policy:
    R0 + R1 allowed (squeeze works in chop). R2 = forced exit. R3 = GO_SMALL.
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

WARMUP_BARS = 50


@dataclass
class SqueezePosition:
    """Track an open SqueezePlay position."""

    entry_price: float
    quantity: int
    entry_bar: int
    peak_price: float
    direction: str  # "LONG" or "SHORT"


@register_strategy(
    "squeeze_play",
    description="Bollinger/Keltner squeeze breakout with fake-release filter",
    author="Apex",
    version="1.0",
    max_params=8,
    tier=1,
)
class SqueezePlayStrategy(Strategy):
    """
    SqueezePlay: Volatility compression -> expansion breakout.

    Enters when Bollinger Bands expand outside Keltner Channels
    (squeeze release) with persistence confirmation and directional bias.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_multiplier: float = 1.5,
        release_persist_bars: int = 2,
        close_outside_bars: int = 2,
        atr_stop_mult: float = 2.5,
        adx_min: float = 20.0,
        risk_per_trade_pct: float = 0.02,
    ):
        super().__init__(strategy_id, symbols, context)

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_multiplier = kc_multiplier
        self.release_persist_bars = release_persist_bars
        self.close_outside_bars = close_outside_bars
        self.atr_stop_mult = atr_stop_mult
        self.adx_min = adx_min
        self.risk_per_trade_pct = risk_per_trade_pct

        # Price history
        buf_len = max(bb_period + 50, 80)
        self._closes: Dict[str, Deque[float]] = {s: deque(maxlen=buf_len) for s in symbols}
        self._highs: Dict[str, Deque[float]] = {s: deque(maxlen=buf_len) for s in symbols}
        self._lows: Dict[str, Deque[float]] = {s: deque(maxlen=buf_len) for s in symbols}

        # Per-symbol squeeze tracking
        self._release_count: Dict[str, int] = {s: 0 for s in symbols}
        self._outside_count: Dict[str, int] = {s: 0 for s in symbols}
        self._squeeze_on: Dict[str, bool] = {s: False for s in symbols}

        # Position tracking
        self._positions: Dict[str, Optional[SqueezePosition]] = {s: None for s in symbols}
        self._bar_count: Dict[str, int] = {s: 0 for s in symbols}

        # Cached indicators
        self._atr: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._adx: Dict[str, Optional[float]] = {s: None for s in symbols}

        # Infrastructure
        self._exit_manager = ExitManager(
            hard_stop_pct=0.08,
            atr_trail_mult=atr_stop_mult,
            max_hold_bars=30,
        )
        self._regime_gate = RegimeGate(
            policy=RegimePolicy(
                allowed_regimes=["R0", "R1"],
                min_dwell_bars=3,
                switch_cooldown_bars=8,
                size_factors={"R0": 1.0, "R1": 0.6},
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
            f"SqueezePlay started: {self.symbols} "
            f"(bb={self.bb_period}, kc_mult={self.kc_multiplier})"
        )

    def on_bar(self, bar: BarData) -> None:
        """Process daily bar."""
        symbol = bar.symbol
        if bar.close is None or bar.high is None or bar.low is None:
            return

        close = bar.close
        high = bar.high
        low = bar.low

        self._closes[symbol].append(close)
        self._highs[symbol].append(high)
        self._lows[symbol].append(low)
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
        bar_idx = self._bar_count[symbol]

        if bar_idx < WARMUP_BARS:
            return

        # Calculate indicators
        self._update_atr(symbol)
        self._update_adx(symbol)
        bb_result = self._calc_bbands(symbol)
        kc_result = self._calc_keltner(symbol)

        if bb_result[0] is None or kc_result[0] is None:
            return

        bb_upper: float = bb_result[0]
        bb_lower: float = bb_result[1]  # type: ignore[assignment]
        kc_upper: float = kc_result[0]
        kc_lower: float = kc_result[1]  # type: ignore[assignment]

        atr_val = self._atr[symbol]
        if atr_val is None or atr_val <= 0:
            return

        # Detect squeeze state
        is_squeeze = bb_upper < kc_upper and bb_lower > kc_lower
        was_squeeze = self._squeeze_on[symbol]
        self._squeeze_on[symbol] = is_squeeze

        # Track release persistence
        if not is_squeeze:
            self._release_count[symbol] += 1
        else:
            self._release_count[symbol] = 0

        # Track close outside BB persistence
        if close > bb_upper or close < bb_lower:
            self._outside_count[symbol] += 1
        else:
            self._outside_count[symbol] = 0

        # Manage existing position
        pos = self._positions[symbol]
        if pos is not None:
            self._manage_exit(symbol, close, atr_val, bar_idx, pos)
            return

        # Check entry: squeeze release with persistence
        self._check_entry(
            symbol,
            close,
            atr_val,
            bar_idx,
            bb_upper,
            bb_lower,
            kc_upper,
        )

    def _check_entry(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        bar_idx: int,
        bb_upper: float,
        bb_lower: float,
        kc_upper: float,
    ) -> None:
        """Check squeeze release entry conditions."""
        # Persistence filter: release must persist
        if self._release_count[symbol] < self.release_persist_bars:
            return
        if self._outside_count[symbol] < self.close_outside_bars:
            return

        # ADX confirmation
        adx_val = self._adx.get(symbol)
        if adx_val is not None and adx_val < self.adx_min:
            return

        # Direction bias: close above upper BB = long, below lower = short
        if close > bb_upper:
            direction = "LONG"
        elif close < bb_lower:
            direction = "SHORT"
        else:
            return  # No clear direction

        # Regime gate
        regime = self._get_current_regime(symbol)
        size_factor = 1.0
        if regime is not None:
            gate_result = self._regime_gate.evaluate(symbol, regime, bar_idx)
            if not gate_result.allowed:
                return
            size_factor = gate_result.size_factor

        # No existing position
        if self.context.has_position(symbol):
            return

        # Size the position
        sizing = self._sizer.calculate(
            symbol=symbol,
            price=close,
            atr=atr_val,
            stop_distance_atr_mult=self.atr_stop_mult,
            regime_size_factor=size_factor,
        )
        if not sizing.is_valid:
            return

        if regime is not None:
            self._regime_gate.record_trade(symbol, regime)

        self._positions[symbol] = SqueezePosition(
            entry_price=close,
            quantity=sizing.shares,
            entry_bar=bar_idx,
            peak_price=close,
            direction=direction,
        )

        side = "BUY" if direction == "LONG" else "SELL"
        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=symbol,
                direction=direction,
                strength=1.0,
                target_quantity=float(sizing.shares),
                target_price=close,
                reason=(
                    f"SqueezePlay BREAKOUT {direction}: "
                    f"release={self._release_count[symbol]}bars, "
                    f"outside={self._outside_count[symbol]}bars"
                ),
                metadata={
                    "adx": adx_val,
                    "release_bars": self._release_count[symbol],
                    "outside_bars": self._outside_count[symbol],
                    "regime": regime.value if regime else "N/A",
                },
            )
        )
        self.request_order(
            OrderRequest(
                symbol=symbol,
                side=side,
                quantity=sizing.shares,
                order_type="MARKET",
            )
        )

        logger.info(
            f"[{self.strategy_id}] SqueezePlay {direction}: {symbol} "
            f"@ {close:.2f}, release={self._release_count[symbol]}bars, "
            f"ADX={adx_val}"
        )

    def _manage_exit(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        bar_idx: int,
        pos: SqueezePosition,
    ) -> None:
        """Manage exit via ExitManager."""
        if close > pos.peak_price and pos.direction == "LONG":
            pos.peak_price = close
        elif close < pos.peak_price and pos.direction == "SHORT":
            pos.peak_price = close

        bars_held = bar_idx - pos.entry_bar
        is_long = pos.direction == "LONG"

        # Regime veto check
        regime = self._get_current_regime(symbol)
        regime_veto = False
        if regime is not None:
            regime_veto = regime.value in self._regime_gate.policy.forced_degross_regimes

        conditions = ExitConditions(
            current_price=close,
            entry_price=pos.entry_price,
            peak_price=pos.peak_price,
            current_atr=atr_val,
            bars_held=bars_held,
            regime_is_veto=regime_veto,
            is_long=is_long,
        )

        exit_signal = self._exit_manager.evaluate(conditions)
        if exit_signal is not None:
            if is_long:
                pnl_pct = (close - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - close) / pos.entry_price

            side = "SELL" if is_long else "BUY"
            self.emit_signal(
                TradingSignal(
                    signal_id="",
                    symbol=symbol,
                    direction="FLAT",
                    reason=f"SqueezePlay EXIT: {exit_signal.reason} (P/L: {pnl_pct:.1%})",
                    metadata={
                        "exit_priority": exit_signal.priority.name,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                    },
                )
            )
            self.request_order(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    quantity=pos.quantity,
                    order_type="MARKET",
                )
            )
            self._positions[symbol] = None

    def _get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get current regime for symbol."""
        market_regime = self.context.get_market_regime()
        if market_regime is not None:
            return market_regime
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            return regime_output.final_regime
        return None

    def _calc_bbands(self, symbol: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate current Bollinger Bands values."""
        closes = list(self._closes[symbol])
        if len(closes) < self.bb_period:
            return None, None, None

        window = closes[-self.bb_period :]
        mid = sum(window) / self.bb_period
        variance = sum((x - mid) ** 2 for x in window) / self.bb_period
        std = variance**0.5

        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std
        return upper, lower, mid

    def _calc_keltner(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """Calculate current Keltner Channel values."""
        closes = list(self._closes[symbol])
        atr_val = self._atr[symbol]
        if len(closes) < self.bb_period or atr_val is None:
            return None, None

        mid = sum(closes[-self.bb_period :]) / self.bb_period
        upper = mid + self.kc_multiplier * atr_val
        lower = mid - self.kc_multiplier * atr_val
        return upper, lower

    def _update_atr(self, symbol: str) -> None:
        """Update ATR (14-period)."""
        closes = list(self._closes[symbol])
        highs = list(self._highs[symbol])
        lows = list(self._lows[symbol])
        period = 14
        if len(closes) < period + 1 or len(highs) < period or len(lows) < period:
            return

        h = highs[-period:]
        l = lows[-period:]
        c = closes[-(period + 1) :]

        trs = []
        for i in range(period):
            tr = max(h[i] - l[i], abs(h[i] - c[i]), abs(l[i] - c[i]))
            trs.append(tr)

        self._atr[symbol] = sum(trs) / period

    def _update_adx(self, symbol: str) -> None:
        """Update ADX (14-period, simplified)."""
        closes = list(self._closes[symbol])
        highs = list(self._highs[symbol])
        lows = list(self._lows[symbol])
        period = 14
        if len(highs) < period + 1 or len(lows) < period + 1:
            return

        # Simplified ADX: directional movement strength
        plus_dm_sum = 0.0
        minus_dm_sum = 0.0
        tr_sum = 0.0

        for i in range(-period, 0):
            h_diff = highs[i] - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            plus_dm = max(h_diff, 0) if h_diff > l_diff else 0
            minus_dm = max(l_diff, 0) if l_diff > h_diff else 0
            plus_dm_sum += plus_dm
            minus_dm_sum += minus_dm

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_sum += tr

        if tr_sum == 0:
            self._adx[symbol] = 0.0
            return

        plus_di = 100 * plus_dm_sum / tr_sum
        minus_di = 100 * minus_dm_sum / tr_sum
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        self._adx[symbol] = dx

    def on_tick(self, tick: QuoteTick) -> None:
        """No intraday processing."""
        pass

    def on_fill(self, fill: TradeFill) -> None:
        logger.debug(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )
