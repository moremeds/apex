"""
TrendPulse Strategy - Zig-Zag Swing + DualMACD Momentum.

Event-driven implementation of the TrendPulse signal generator for
ApexEngine and live trading. Computes zig-zag swings, DualMACD state,
and 4-factor confidence incrementally bar-by-bar.

Note: Indicators (EMA, MACD, ADX, ATR) are computed incrementally
rather than via TA-Lib, since event-driven strategies process one bar
at a time. The vectorized signal generator uses TA-Lib wrappers.

Edge: Enter on zig-zag BUY crosses within confirmed uptrends (EMA-99 +
DualMACD bullish + ADX trending), exit via ExitManager priority chain.

Parameters (8, at <=8 budget):
    zig_threshold_pct: Zig-zag reversal threshold (default 3.5)
    trend_strength_moderate: Min ADX/50 normalized strength (default 0.15)
    min_confidence: Min 4-factor confidence score (default 0.35)
    hard_stop_pct: Hard stop loss from entry (default 0.15)
    atr_stop_mult: ATR multiplier for trailing stop (default 3.5)
    exit_bearish_bars: Consecutive bearish bars for DM exit (default 3)
    adx_entry_min: Min ADX for chop filter (default 15.0)
    cooldown_bars: Post-exit cooldown bars (default 5)
    risk_per_trade_pct: Per-trade risk fraction (default 0.02)

Entry (all must be true at bar close):
    1. Zig-zag BUY cross (swing reversal up)
    2. Close > EMA-99 (daily trend bullish)
    3. ADX-normalized trend_strength >= moderate threshold
    4. DualMACD state in (BULLISH, IMPROVING, DETERIORATING)
    5. ADX >= adx_entry_min (chop filter)
    6. 4-factor confidence >= min_confidence
    7. Cooldown elapsed since last exit
    8. Regime in [R0, R1, R3] via RegimeGate
    9. No existing position

Exit priority (via ExitManager):
    1. Hard stop: entry * (1 - hard_stop_pct)
    2. ATR trail: peak - atr_stop_mult * ATR
    3. Regime veto: R2
    4. DM bearish persistence: N consecutive bearish bars
    5. Zig-zag SELL cross / top_detected
    6. Time stop: 60 bars
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
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

# Warmup: max(ema_99, macd_slow=89+signal_convergence, atr_14, adx_28) + buffer
WARMUP_BARS = 160


class ZigDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"


class DMState(Enum):
    BULLISH = "BULLISH"
    IMPROVING = "IMPROVING"
    DETERIORATING = "DETERIORATING"
    BEARISH = "BEARISH"


@dataclass
class TrendPulsePosition:
    """Track an open TrendPulse position."""

    entry_price: float
    quantity: int
    entry_bar: int
    peak_price: float
    confidence: float


@dataclass
class ZigState:
    """Incremental zig-zag swing state."""

    direction: ZigDirection = ZigDirection.UP
    pivot_price: float = 0.0
    extreme_price: float = 0.0
    cross_up: bool = False
    cross_down: bool = False


@dataclass
class MACDState:
    """Incremental DualMACD state (slow MACD: 55/89/34)."""

    ema_fast: Optional[float] = None  # EMA-55
    ema_slow: Optional[float] = None  # EMA-89
    signal_ema: Optional[float] = None  # EMA-34 of slow_macd
    prev_hist: Optional[float] = None
    slow_hist: float = 0.0
    slow_hist_delta: float = 0.0


@dataclass
class ADXState:
    """Incremental ADX/DMI state (14-period Wilder's smoothing)."""

    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    plus_dm_smooth: float = 0.0
    minus_dm_smooth: float = 0.0
    tr_smooth: float = 0.0
    dx_sum: float = 0.0
    dx_count: int = 0
    adx: Optional[float] = None
    bar_count: int = 0


@register_strategy(
    "trend_pulse",
    description="TrendPulse zig-zag swing + DualMACD momentum strategy",
    author="Apex",
    version="1.0",
    max_params=8,
    tier=1,
)
class TrendPulseStrategy(Strategy):
    """
    TrendPulse: Trade zig-zag swing reversals confirmed by DualMACD momentum.

    Enters on zig-zag BUY crosses when EMA-99 is bullish, DualMACD shows
    positive momentum, ADX indicates trending market, and 4-factor confidence
    exceeds threshold. Exits via ATR trailing stop, DM bearish persistence,
    zig-zag SELL cross, or top detection.
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        zig_threshold_pct: float = 2.5,
        trend_strength_moderate: float = 0.15,
        min_confidence: float = 0.35,
        hard_stop_pct: float = 0.15,
        atr_stop_mult: float = 5.0,
        exit_bearish_bars: int = 3,
        adx_entry_min: float = 15.0,
        cooldown_bars: int = 5,
        risk_per_trade_pct: float = 0.05,
    ):
        super().__init__(strategy_id, symbols, context)

        if not 1.0 <= zig_threshold_pct <= 10.0:
            raise ValueError(f"zig_threshold_pct must be 1-10, got {zig_threshold_pct}")
        if not 1 <= exit_bearish_bars <= 10:
            raise ValueError(f"exit_bearish_bars must be 1-10, got {exit_bearish_bars}")

        self.zig_threshold_pct = zig_threshold_pct
        self.trend_strength_moderate = trend_strength_moderate
        self.min_confidence = min_confidence
        self.hard_stop_pct = hard_stop_pct
        self.atr_stop_mult = atr_stop_mult
        self.exit_bearish_bars = exit_bearish_bars
        self.adx_entry_min = adx_entry_min
        self.cooldown_bars = cooldown_bars
        self.risk_per_trade_pct = risk_per_trade_pct

        # Price history for EMA/ATR calculation
        self._closes: Dict[str, Deque[float]] = {s: deque(maxlen=200) for s in symbols}
        self._highs: Dict[str, Deque[float]] = {s: deque(maxlen=50) for s in symbols}
        self._lows: Dict[str, Deque[float]] = {s: deque(maxlen=50) for s in symbols}

        # Per-symbol state
        self._positions: Dict[str, Optional[TrendPulsePosition]] = {s: None for s in symbols}
        self._bar_count: Dict[str, int] = {s: 0 for s in symbols}
        self._ema_99: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._atr: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._zig: Dict[str, ZigState] = {s: ZigState() for s in symbols}
        self._macd: Dict[str, MACDState] = {s: MACDState() for s in symbols}
        self._adx: Dict[str, ADXState] = {s: ADXState() for s in symbols}
        self._dm_bearish_count: Dict[str, int] = {s: 0 for s in symbols}
        self._bars_since_exit: Dict[str, int] = {s: cooldown_bars + 1 for s in symbols}

        # ADX normalization constant
        self._norm_max_adx = 50.0

        # Infrastructure
        self._exit_manager = ExitManager(
            hard_stop_pct=hard_stop_pct,
            atr_trail_mult=atr_stop_mult,
            max_hold_bars=60,
        )
        self._regime_gate = RegimeGate(
            policy=RegimePolicy(
                allowed_regimes=["R0", "R1", "R3"],
                min_dwell_bars=5,
                switch_cooldown_bars=10,
                size_factors={"R0": 1.0, "R1": 0.5, "R3": 0.3},
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
            f"TrendPulse started: {self.symbols} "
            f"(zig={self.zig_threshold_pct}%, adx_min={self.adx_entry_min})"
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

        # Update indicators
        self._update_ema_99(symbol)
        self._update_atr(symbol)
        self._update_zig(symbol, close)
        self._update_macd(symbol, close)
        self._update_adx(symbol, high, low, close)

        # Track cooldown
        self._bars_since_exit[symbol] += 1

        # Need warmup
        if bar_idx < WARMUP_BARS:
            return

        ema_val = self._ema_99[symbol]
        atr_val = self._atr[symbol]
        adx_state = self._adx[symbol]
        macd_state = self._macd[symbol]
        zig_state = self._zig[symbol]

        if ema_val is None or atr_val is None or adx_state.adx is None:
            return

        # Classify DualMACD state
        dm_state = self._classify_dm_state(macd_state)

        # Track DM bearish persistence
        if dm_state == DMState.BEARISH:
            self._dm_bearish_count[symbol] += 1
        else:
            self._dm_bearish_count[symbol] = 0

        # Manage existing position
        pos = self._positions[symbol]
        if pos is not None:
            self._manage_exit(symbol, close, atr_val, bar_idx, pos, dm_state, zig_state)
            return

        # Check entry conditions
        self._check_entry(symbol, close, ema_val, atr_val, bar_idx, dm_state, zig_state, adx_state)

    def _classify_dm_state(self, macd: MACDState) -> DMState:
        """Classify DualMACD momentum state from histogram + delta."""
        hist = macd.slow_hist
        delta = macd.slow_hist_delta
        if hist > 0 and delta >= 0:
            return DMState.BULLISH
        if hist < 0 and delta > 0:
            return DMState.IMPROVING
        if hist > 0 and delta < 0:
            return DMState.DETERIORATING
        return DMState.BEARISH

    def _compute_confidence(
        self,
        zig: ZigState,
        dm_state: DMState,
        trend_bull: bool,
        strength_ok: bool,
        atr_val: float,
    ) -> float:
        """4-factor confidence scoring (ZIG 30%, DM 25%, Trend 30%, Vol 15%)."""
        # (a) ZIG_Strength: swing amplitude relative to threshold
        if zig.pivot_price > 0:
            swing_pct = abs(zig.extreme_price - zig.pivot_price) / zig.pivot_price * 100
            zig_strength = min(1.0, swing_pct / self.zig_threshold_pct)
        else:
            zig_strength = 0.5

        # (b) DM_Health
        dm_scores = {
            DMState.BULLISH: 1.0,
            DMState.IMPROVING: 0.7,
            DMState.DETERIORATING: 0.5,
            DMState.BEARISH: 0.0,
        }
        dm_health = dm_scores[dm_state]

        # (c) Trend_Alignment (daily bullish + strength confirmed)
        trend_align = 1.0 if (trend_bull and strength_ok) else 0.0

        # (d) Vol_Quality (moderate ATR preferred - simplified)
        vol_quality = 0.5 if atr_val > 0 else 0.0

        return 0.30 * zig_strength + 0.25 * dm_health + 0.30 * trend_align + 0.15 * vol_quality

    def _check_entry(
        self,
        symbol: str,
        close: float,
        ema_val: float,
        atr_val: float,
        bar_idx: int,
        dm_state: DMState,
        zig: ZigState,
        adx_state: ADXState,
    ) -> None:
        """Check all entry conditions."""
        # 1. Zig-zag BUY cross
        if not zig.cross_up:
            return

        # 2. Close above EMA-99
        trend_bull = close > ema_val
        if not trend_bull:
            return

        # 3. Trend strength >= moderate
        adx_val = adx_state.adx or 0.0
        trend_strength = min(1.0, adx_val / self._norm_max_adx)
        strength_ok = trend_strength >= self.trend_strength_moderate
        if not strength_ok:
            return

        # 4. DualMACD allows entry (not BEARISH)
        if dm_state == DMState.BEARISH:
            return

        # 5. ADX chop filter
        if adx_val < self.adx_entry_min:
            return

        # 6. Confidence threshold
        confidence = self._compute_confidence(zig, dm_state, trend_bull, strength_ok, atr_val)
        if confidence < self.min_confidence:
            return

        # 7. Cooldown elapsed
        if self._bars_since_exit[symbol] < self.cooldown_bars:
            return

        # 8. Regime gate
        regime = self._get_current_regime(symbol)
        if regime is not None:
            gate_result = self._regime_gate.evaluate(symbol, regime, bar_idx)
            if not gate_result.allowed:
                return
            size_factor = gate_result.size_factor
        else:
            size_factor = 1.0

        # 9. No existing position
        if self.context.has_position(symbol):
            return

        # All conditions met - size and enter
        sizing = self._sizer.calculate(
            symbol=symbol,
            price=close,
            atr=atr_val,
            stop_distance_atr_mult=self.atr_stop_mult,
            confidence=confidence,
            regime_size_factor=size_factor,
        )
        if not sizing.is_valid:
            return

        if regime is not None:
            self._regime_gate.record_trade(symbol, regime)

        self._positions[symbol] = TrendPulsePosition(
            entry_price=close,
            quantity=sizing.shares,
            entry_bar=bar_idx,
            peak_price=close,
            confidence=confidence,
        )

        self.emit_signal(
            TradingSignal(
                signal_id="",
                symbol=symbol,
                direction="LONG",
                strength=confidence,
                target_quantity=float(sizing.shares),
                target_price=close,
                reason=(
                    f"TrendPulse ENTRY: conf={confidence:.2f}, "
                    f"ADX={adx_val:.1f}, DM={dm_state.value}"
                ),
                metadata={
                    "confidence": confidence,
                    "adx": adx_val,
                    "dm_state": dm_state.value,
                    "atr": atr_val,
                    "regime": regime.value if regime else "N/A",
                    "size_factor": size_factor,
                },
            )
        )

        self.request_order(
            OrderRequest(
                symbol=symbol,
                side="BUY",
                quantity=sizing.shares,
                order_type="MARKET",
            )
        )

        logger.info(
            f"[{self.strategy_id}] TrendPulse ENTRY: {symbol} "
            f"@ {close:.2f}, conf={confidence:.2f}, "
            f"ADX={adx_val:.1f}, DM={dm_state.value}, "
            f"size={sizing.shares}"
        )

    def _manage_exit(
        self,
        symbol: str,
        close: float,
        atr_val: float,
        bar_idx: int,
        pos: TrendPulsePosition,
        dm_state: DMState,
        zig: ZigState,
    ) -> None:
        """Manage exit via ExitManager + TrendPulse-specific indicators."""
        if close > pos.peak_price:
            pos.peak_price = close

        bars_held = bar_idx - pos.entry_bar

        # Regime veto
        regime = self._get_current_regime(symbol)
        regime_is_veto = False
        if regime is not None:
            regime_is_veto = regime.value in self._regime_gate.policy.forced_degross_regimes

        # TrendPulse-specific indicator exits
        indicator_exit = False
        indicator_reason = ""

        # DM bearish persistence (trigger at N+ consecutive bearish bars)
        if self._dm_bearish_count[symbol] >= self.exit_bearish_bars:
            indicator_exit = True
            indicator_reason = f"DM bearish x{self.exit_bearish_bars}"

        # Zig-zag SELL cross
        if zig.cross_down:
            indicator_exit = True
            indicator_reason = "zig_cross_down"

        # Top detection via regime provider
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            tp = getattr(regime_output, "turning_point", None)
            if tp is not None and getattr(tp, "prediction", None) == "TOP_RISK":
                indicator_exit = True
                indicator_reason = "top_detected"

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
            pnl_pct = (close - pos.entry_price) / pos.entry_price
            self.emit_signal(
                TradingSignal(
                    signal_id="",
                    symbol=symbol,
                    direction="FLAT",
                    reason=f"TrendPulse EXIT: {exit_signal.reason} (P/L: {pnl_pct:.1%})",
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
                    side="SELL",
                    quantity=pos.quantity,
                    order_type="MARKET",
                )
            )
            self._positions[symbol] = None
            self._bars_since_exit[symbol] = 0

            logger.info(
                f"[{self.strategy_id}] TrendPulse EXIT: {symbol} "
                f"@ {close:.2f}, reason={exit_signal.reason}, "
                f"P/L={pnl_pct:.1%}, held={bars_held} bars"
            )

    def _get_current_regime(self, symbol: str) -> Optional[MarketRegime]:
        """Get current regime (prefer market-level)."""
        market_regime = self.context.get_market_regime()
        if market_regime is not None:
            return market_regime
        regime_output = self.context.get_regime(symbol)
        if regime_output is not None:
            return regime_output.final_regime
        return None

    # ── Incremental Indicator Updates ────────────────────────────

    def _update_ema_99(self, symbol: str) -> None:
        """Update EMA-99 using exponential smoothing."""
        closes = self._closes[symbol]
        period = 99
        if len(closes) < period:
            return
        if self._ema_99[symbol] is None:
            self._ema_99[symbol] = sum(list(closes)[-period:]) / period
        else:
            k = 2.0 / (period + 1)
            prev = self._ema_99[symbol]
            self._ema_99[symbol] = closes[-1] * k + prev * (1 - k)  # type: ignore[operator]

    def _update_atr(self, symbol: str) -> None:
        """Update ATR (14-period simple average of true range)."""
        closes = self._closes[symbol]
        highs = self._highs[symbol]
        lows = self._lows[symbol]
        period = 14
        if len(closes) < period + 1 or len(highs) < period or len(lows) < period:
            return

        h = list(highs)[-period:]
        lo = list(lows)[-period:]
        c = list(closes)[-(period + 1) :]

        trs = []
        for i in range(period):
            tr = max(h[i] - lo[i], abs(h[i] - c[i]), abs(lo[i] - c[i]))
            trs.append(tr)

        self._atr[symbol] = sum(trs) / period

    def _update_zig(self, symbol: str, close: float) -> None:
        """Update incremental zig-zag swing detection."""
        zig = self._zig[symbol]
        zig.cross_up = False
        zig.cross_down = False

        # Initialize on first bar
        if zig.pivot_price == 0.0:
            zig.pivot_price = close
            zig.extreme_price = close
            return

        threshold = self.zig_threshold_pct / 100.0

        if zig.direction == ZigDirection.UP:
            # Track peak
            if close > zig.extreme_price:
                zig.extreme_price = close
            # Check for reversal down
            if zig.extreme_price > 0 and close < zig.extreme_price * (1 - threshold):
                zig.cross_down = True
                zig.direction = ZigDirection.DOWN
                zig.pivot_price = zig.extreme_price
                zig.extreme_price = close
        else:
            # Track trough
            if close < zig.extreme_price:
                zig.extreme_price = close
            # Check for reversal up
            if zig.extreme_price > 0 and close > zig.extreme_price * (1 + threshold):
                zig.cross_up = True
                zig.direction = ZigDirection.UP
                zig.pivot_price = zig.extreme_price
                zig.extreme_price = close

    def _update_macd(self, symbol: str, close: float) -> None:
        """Update DualMACD state (slow MACD: EMA-55/EMA-89, signal: EMA-34)."""
        st = self._macd[symbol]

        # EMA-55 (fast leg)
        if st.ema_fast is None:
            if len(self._closes[symbol]) >= 55:
                st.ema_fast = sum(list(self._closes[symbol])[-55:]) / 55
        else:
            k = 2.0 / (55 + 1)
            st.ema_fast = close * k + st.ema_fast * (1 - k)

        # EMA-89 (slow leg)
        if st.ema_slow is None:
            if len(self._closes[symbol]) >= 89:
                st.ema_slow = sum(list(self._closes[symbol])[-89:]) / 89
        else:
            k = 2.0 / (89 + 1)
            st.ema_slow = close * k + st.ema_slow * (1 - k)

        if st.ema_fast is None or st.ema_slow is None:
            return

        slow_macd = st.ema_fast - st.ema_slow

        # Signal line: EMA-34 of slow_macd
        if st.signal_ema is None:
            st.signal_ema = slow_macd
        else:
            k = 2.0 / (34 + 1)
            st.signal_ema = slow_macd * k + st.signal_ema * (1 - k)

        st.slow_hist = slow_macd - st.signal_ema
        if st.prev_hist is not None:
            st.slow_hist_delta = st.slow_hist - st.prev_hist
        else:
            st.slow_hist_delta = 0.0
        st.prev_hist = st.slow_hist

    def _update_adx(self, symbol: str, high: float, low: float, close: float) -> None:
        """Update incremental ADX (14-period Wilder's smoothing)."""
        st = self._adx[symbol]
        period = 14

        if st.prev_high is not None:
            # True Range
            tr = max(
                high - low,
                abs(high - st.prev_close),  # type: ignore[operator]
                abs(low - st.prev_close),  # type: ignore[operator]
            )
            # Directional Movement
            up_move = high - st.prev_high
            down_move = st.prev_low - low  # type: ignore[operator]
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

            st.bar_count += 1

            if st.bar_count <= period:
                # Accumulation phase
                st.plus_dm_smooth += plus_dm
                st.minus_dm_smooth += minus_dm
                st.tr_smooth += tr
            else:
                # Wilder's smoothing
                st.plus_dm_smooth = st.plus_dm_smooth - (st.plus_dm_smooth / period) + plus_dm
                st.minus_dm_smooth = st.minus_dm_smooth - (st.minus_dm_smooth / period) + minus_dm
                st.tr_smooth = st.tr_smooth - (st.tr_smooth / period) + tr

            if st.bar_count >= period and st.tr_smooth > 0:
                plus_di = 100.0 * st.plus_dm_smooth / st.tr_smooth
                minus_di = 100.0 * st.minus_dm_smooth / st.tr_smooth
                di_sum = plus_di + minus_di
                dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

                if st.adx is None:
                    st.dx_sum += dx
                    st.dx_count += 1
                    if st.dx_count >= period:
                        st.adx = st.dx_sum / period
                else:
                    st.adx = (st.adx * (period - 1) + dx) / period

        st.prev_high = high
        st.prev_low = low
        st.prev_close = close

    def on_tick(self, tick: QuoteTick) -> None:
        """No intraday processing - daily bars only."""
        pass

    def on_fill(self, fill: TradeFill) -> None:
        logger.debug(
            f"[{self.strategy_id}] FILL: "
            f"{fill.side} {fill.quantity} {fill.symbol} @ {fill.price:.2f}"
        )
