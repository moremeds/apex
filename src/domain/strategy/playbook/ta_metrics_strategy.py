"""
Technical Analysis Metrics Matrix Strategy.

A comprehensive strategy that combines multiple TA indicators:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

Creates a "metrics matrix" that scores buy/sell signals based on
the confluence of multiple indicators.

This strategy demonstrates:
- Multi-indicator signal generation
- Signal scoring and confluence
- Position sizing based on signal strength
- Walk-forward compatible design
"""

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional

from ...events.domain_events import BarData, QuoteTick
from ...interfaces.execution_provider import OrderRequest
from ..base import Strategy, StrategyContext
from ..registry import register_strategy

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Signal types from indicators."""

    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class IndicatorValues:
    """Current values of all indicators."""

    # Moving Averages
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None

    # RSI
    rsi: Optional[float] = None
    rsi_signal: Signal = Signal.NEUTRAL

    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_signal_type: Signal = Signal.NEUTRAL

    # Price action
    price: float = 0.0
    prev_price: float = 0.0


@dataclass
class MetricsMatrix:
    """
    Signal scoring matrix combining all indicators.

    Each indicator contributes a score:
    - MA Crossover: +2 (bullish cross), -2 (bearish cross)
    - RSI: +2 (oversold bounce), -2 (overbought rejection)
    - MACD: +2 (bullish crossover), -2 (bearish crossover)
    - Trend Alignment: +1 (price above all MAs), -1 (below all MAs)

    Total score range: -7 to +7
    """

    ma_crossover_score: int = 0
    rsi_score: int = 0
    macd_score: int = 0
    trend_score: int = 0

    @property
    def total_score(self) -> int:
        """Total signal score."""
        return self.ma_crossover_score + self.rsi_score + self.macd_score + self.trend_score

    @property
    def signal(self) -> Signal:
        """Convert score to signal."""
        score = self.total_score
        if score >= 4:
            return Signal.STRONG_BUY
        elif score >= 2:
            return Signal.BUY
        elif score <= -4:
            return Signal.STRONG_SELL
        elif score <= -2:
            return Signal.SELL
        return Signal.NEUTRAL

    @property
    def signal_strength(self) -> float:
        """Signal strength as 0-1 value for position sizing."""
        return min(1.0, abs(self.total_score) / 7.0)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for logging."""
        return {
            "ma_crossover": self.ma_crossover_score,
            "rsi": self.rsi_score,
            "macd": self.macd_score,
            "trend": self.trend_score,
            "total": self.total_score,
        }


@register_strategy(
    "ta_metrics",
    description="Technical Analysis Metrics Matrix Strategy",
    author="Apex",
    version="1.0",
)
class TAMetricsStrategy(Strategy):
    """
    Technical Analysis Metrics Matrix Strategy.

    Combines multiple indicators into a scoring matrix for high-probability
    trade signals. Only takes positions when multiple indicators align.

    Parameters:
        sma_fast: Fast SMA period (default: 20)
        sma_slow: Slow SMA period (default: 50)
        ema_fast: Fast EMA period (default: 12)
        ema_slow: Slow EMA period (default: 26)
        rsi_period: RSI period (default: 14)
        rsi_oversold: RSI oversold level (default: 30)
        rsi_overbought: RSI overbought level (default: 70)
        macd_fast: MACD fast period (default: 12)
        macd_slow: MACD slow period (default: 26)
        macd_signal: MACD signal period (default: 9)
        min_score: Minimum score to take position (default: 3)
        position_size: Base position size (default: 100)
        scale_by_strength: Scale position by signal strength (default: True)

    Example:
        strategy = TAMetricsStrategy(
            strategy_id="ta-spy",
            symbols=["SPY"],
            context=context,
            sma_fast=20,
            sma_slow=50,
            min_score=3,
        )
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        context: StrategyContext,
        # Moving Average params
        sma_fast: int = 20,
        sma_slow: int = 50,
        ema_fast: int = 12,
        ema_slow: int = 26,
        # RSI params
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        # MACD params
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # Trading params
        min_score: int = 3,
        position_size: float = 100,
        scale_by_strength: bool = True,
    ):
        super().__init__(strategy_id, symbols, context)

        # Store params
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.min_score = min_score
        self.position_size = position_size
        self.scale_by_strength = scale_by_strength

        # Calculate max lookback needed
        self.max_lookback = max(sma_slow, ema_slow, macd_slow + macd_signal)

        # Price history per symbol
        self._prices: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=self.max_lookback + 50) for symbol in symbols
        }

        # EMA state (requires running calculation)
        self._ema_fast: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._ema_slow: Dict[str, Optional[float]] = {s: None for s in symbols}
        self._macd_signal_ema: Dict[str, Optional[float]] = {s: None for s in symbols}

        # Previous indicator values for crossover detection
        self._prev_indicators: Dict[str, Optional[IndicatorValues]] = {s: None for s in symbols}

        # Track last action to avoid duplicate signals
        self._last_action: Dict[str, Optional[str]] = {s: None for s in symbols}

    def on_start(self) -> None:
        """Initialize strategy."""
        logger.info(
            f"TA Metrics Strategy started: {self.symbols} "
            f"(SMA: {self.sma_fast}/{self.sma_slow}, "
            f"RSI: {self.rsi_period}, "
            f"MACD: {self.macd_fast}/{self.macd_slow}/{self.macd_signal_period}, "
            f"min_score: {self.min_score})"
        )

    def on_tick(self, tick: QuoteTick) -> None:
        """Process tick - delegates to on_bar logic."""

    def on_bar(self, bar: BarData) -> None:
        """
        Process each bar.

        1. Update price history
        2. Calculate all indicators
        3. Build metrics matrix
        4. Generate signal if score exceeds threshold
        """
        symbol = bar.symbol
        price = bar.close

        if price is None or price <= 0:
            return

        # Update price history
        self._prices[symbol].append(price)

        # Need enough data for all indicators
        if len(self._prices[symbol]) < self.max_lookback:
            return

        # Calculate indicators
        indicators = self._calculate_indicators(symbol, price)

        # Build metrics matrix
        matrix = self._build_metrics_matrix(symbol, indicators)

        # Log matrix periodically (every 20 bars)
        if len(self._prices[symbol]) % 20 == 0:
            logger.debug(
                f"[{symbol}] Price: {price:.2f}, Matrix: {matrix.to_dict()}, "
                f"Signal: {matrix.signal.name}"
            )

        # Check for trade signal
        current_position = self.context.get_position_quantity(symbol)
        allow_signal = abs(matrix.total_score) >= self.min_score

        if allow_signal and matrix.signal in (Signal.STRONG_BUY, Signal.BUY):
            if current_position <= 0 and self._last_action[symbol] != "BUY":
                self._execute_buy(symbol, price, matrix)
        elif allow_signal and matrix.signal in (Signal.STRONG_SELL, Signal.SELL):
            if current_position > 0 and self._last_action[symbol] != "SELL":
                self._execute_sell(symbol, price, matrix, current_position)

        # Store for next bar
        self._prev_indicators[symbol] = indicators

    def _calculate_indicators(self, symbol: str, price: float) -> IndicatorValues:
        """Calculate all technical indicators."""
        prices = list(self._prices[symbol])

        indicators = IndicatorValues()
        indicators.price = price
        indicators.prev_price = prices[-2] if len(prices) > 1 else price

        # --- Moving Averages ---
        indicators.sma_fast = self._sma(prices, self.sma_fast)
        indicators.sma_slow = self._sma(prices, self.sma_slow)

        # EMA (exponential moving average)
        indicators.ema_fast = self._update_ema(symbol, "_ema_fast", price, self.ema_fast)
        indicators.ema_slow = self._update_ema(symbol, "_ema_slow", price, self.ema_slow)

        # --- RSI ---
        indicators.rsi = self._calculate_rsi(prices, self.rsi_period)
        if indicators.rsi is not None:
            if indicators.rsi < self.rsi_oversold:
                indicators.rsi_signal = Signal.BUY
            elif indicators.rsi > self.rsi_overbought:
                indicators.rsi_signal = Signal.SELL

        # --- MACD ---
        if indicators.ema_fast and indicators.ema_slow:
            indicators.macd_line = indicators.ema_fast - indicators.ema_slow
            indicators.macd_signal = self._update_ema(
                symbol, "_macd_signal_ema", indicators.macd_line, self.macd_signal_period
            )
            if indicators.macd_signal:
                indicators.macd_histogram = indicators.macd_line - indicators.macd_signal

        return indicators

    def _build_metrics_matrix(self, symbol: str, indicators: IndicatorValues) -> MetricsMatrix:
        """Build the signal scoring matrix."""
        matrix = MetricsMatrix()
        prev = self._prev_indicators.get(symbol)

        # --- MA Crossover Score ---
        if indicators.sma_fast and indicators.sma_slow:
            if indicators.sma_fast > indicators.sma_slow:
                # Bullish configuration
                if prev and prev.sma_fast and prev.sma_slow:
                    if prev.sma_fast <= prev.sma_slow:
                        # Fresh crossover
                        matrix.ma_crossover_score = 2
                    else:
                        matrix.ma_crossover_score = 1
                else:
                    matrix.ma_crossover_score = 1
            elif indicators.sma_fast < indicators.sma_slow:
                # Bearish configuration
                if prev and prev.sma_fast and prev.sma_slow:
                    if prev.sma_fast >= prev.sma_slow:
                        # Fresh crossover
                        matrix.ma_crossover_score = -2
                    else:
                        matrix.ma_crossover_score = -1
                else:
                    matrix.ma_crossover_score = -1

        # --- RSI Score ---
        if indicators.rsi is not None:
            if indicators.rsi < self.rsi_oversold:
                matrix.rsi_score = 2  # Oversold bounce potential
            elif indicators.rsi < 40:
                matrix.rsi_score = 1
            elif indicators.rsi > self.rsi_overbought:
                matrix.rsi_score = -2  # Overbought rejection potential
            elif indicators.rsi > 60:
                matrix.rsi_score = -1

        # --- MACD Score ---
        if indicators.macd_histogram is not None and prev:
            if prev.macd_histogram is not None:
                if indicators.macd_histogram > 0 and prev.macd_histogram <= 0:
                    # Bullish crossover
                    matrix.macd_score = 2
                elif indicators.macd_histogram > 0:
                    matrix.macd_score = 1
                elif indicators.macd_histogram < 0 and prev.macd_histogram >= 0:
                    # Bearish crossover
                    matrix.macd_score = -2
                elif indicators.macd_histogram < 0:
                    matrix.macd_score = -1

        # --- Trend Score ---
        if indicators.sma_fast and indicators.sma_slow:
            if indicators.price > indicators.sma_fast > indicators.sma_slow:
                matrix.trend_score = 1  # Strong uptrend
            elif indicators.price < indicators.sma_fast < indicators.sma_slow:
                matrix.trend_score = -1  # Strong downtrend

        return matrix

    def _execute_buy(self, symbol: str, price: float, matrix: MetricsMatrix) -> None:
        """Execute buy order."""
        # Calculate position size based on signal strength
        if self.scale_by_strength:
            size = int(self.position_size * matrix.signal_strength)
        else:
            size = int(self.position_size)

        if size <= 0:
            return

        logger.info(
            f"[{self.strategy_id}] BUY {symbol} @ {price:.2f} "
            f"(score: {matrix.total_score}, size: {size})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="BUY",
            quantity=size,
            order_type="MARKET",
        )
        self.request_order(order)
        self._last_action[symbol] = "BUY"

    def _execute_sell(
        self, symbol: str, price: float, matrix: MetricsMatrix, current_qty: float
    ) -> None:
        """Execute sell order."""
        logger.info(
            f"[{self.strategy_id}] SELL {symbol} @ {price:.2f} "
            f"(score: {matrix.total_score}, qty: {current_qty})"
        )

        order = OrderRequest(
            symbol=symbol,
            side="SELL",
            quantity=abs(current_qty),
            order_type="MARKET",
        )
        self.request_order(order)
        self._last_action[symbol] = "SELL"

    # --- Indicator Calculations ---

    def _sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def _update_ema(self, symbol: str, attr: str, value: float, period: int) -> Optional[float]:
        """Update Exponential Moving Average with new value."""
        ema_dict: Dict[str, Optional[float]] = getattr(self, attr)
        prev_ema = ema_dict.get(symbol)

        if prev_ema is None:
            # Initialize with SMA
            prices = list(self._prices[symbol])
            if len(prices) >= period:
                ema: float = sum(prices[-period:]) / period
                ema_dict[symbol] = ema
                return ema
            return None

        # EMA formula: EMA = (value - prev_ema) * multiplier + prev_ema
        multiplier = 2 / (period + 1)
        ema_result: float = (value - prev_ema) * multiplier + prev_ema
        ema_dict[symbol] = ema_result
        return ema_result

    def _calculate_rsi(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return None

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Get last 'period' changes
        recent_changes = changes[-(period):]

        gains = [c for c in recent_changes if c > 0]
        losses = [-c for c in recent_changes if c < 0]

        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_state(self) -> dict:
        """Get current strategy state for monitoring."""
        state = {}
        for symbol in self.symbols:
            prices = list(self._prices[symbol])
            if len(prices) >= self.max_lookback:
                indicators = self._calculate_indicators(symbol, prices[-1])
                matrix = self._build_metrics_matrix(symbol, indicators)
                state[symbol] = {
                    "price": indicators.price,
                    "sma_fast": indicators.sma_fast,
                    "sma_slow": indicators.sma_slow,
                    "rsi": indicators.rsi,
                    "macd_histogram": indicators.macd_histogram,
                    "matrix": matrix.to_dict(),
                    "signal": matrix.signal.name,
                    "bar_count": len(prices),
                }
            else:
                state[symbol] = {
                    "bar_count": len(prices),
                    "warmup_remaining": self.max_lookback - len(prices),
                }
        return state
