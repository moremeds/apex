"""
ATR (Average True Range) Calculator and Optimizer.

Provides:
- ATR calculation using Wilder's smoothed average
- Price level calculations (stop loss, take profit)
- Historical parameter optimization
- Session-level caching

ATR Formula (Wilder's):
- True Range = max(high - low, |high - prev_close|, |low - prev_close|)
- ATR = Wilder's smoothed average: ((prev_ATR * (period-1)) + current_TR) / period
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ATRData:
    """
    ATR calculation result with price levels.

    Contains the ATR value and derived stop loss / take profit levels
    at various ATR multipliers.
    """

    symbol: str
    current_price: float
    atr_value: float
    atr_percent: float  # ATR as % of price
    period: int
    calculated_at: datetime = field(default_factory=datetime.now)

    # Stop loss levels (below entry)
    stop_loss_1x: float = 0.0
    stop_loss_1_5x: float = 0.0
    stop_loss_2x: float = 0.0

    # Take profit levels (above entry) - 7x to 11x ATR
    take_profit_7x: float = 0.0
    take_profit_8x: float = 0.0
    take_profit_9x: float = 0.0
    take_profit_10x: float = 0.0
    take_profit_11x: float = 0.0

    def get_stop_loss(self, multiplier: float) -> float:
        """Get stop loss price at given ATR multiplier."""
        return self.current_price - (self.atr_value * multiplier)

    def get_take_profit(self, multiplier: float) -> float:
        """Get take profit price at given ATR multiplier."""
        return self.current_price + (self.atr_value * multiplier)

    def get_percent_from_entry(self, price: float) -> float:
        """Get percentage difference from current price."""
        if self.current_price == 0:
            return 0.0
        return ((price - self.current_price) / self.current_price) * 100


@dataclass
class ATROptimizationResult:
    """
    Result from historical ATR parameter optimization.

    Shows which combination of period and multipliers would have
    performed best historically.
    """

    period: int
    stop_multiplier: float
    profit_multiplier: float
    historical_win_rate: float  # % of trades that hit take profit before stop
    avg_reward_risk: float  # Average reward/risk ratio
    total_simulated_trades: int = 0
    is_recommended: bool = False

    @property
    def score(self) -> float:
        """Combined score for ranking (win_rate * reward_risk)."""
        return self.historical_win_rate * self.avg_reward_risk


class ATRCalculator:
    """
    Calculate ATR and derived price levels.

    Uses Wilder's smoothed average for ATR calculation.
    """

    @staticmethod
    def calculate_true_range(
        high: float,
        low: float,
        prev_close: float,
    ) -> float:
        """
        Calculate True Range for a single bar.

        TR = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

        Args:
            high: Bar high price.
            low: Bar low price.
            prev_close: Previous bar's close price.

        Returns:
            True Range value.
        """
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

    @staticmethod
    def calculate_true_ranges(bars: List) -> List[float]:
        """
        Calculate True Range for a list of bars.

        Args:
            bars: List of BarData objects with high, low, close attributes.

        Returns:
            List of True Range values (one fewer than input bars).
        """
        if len(bars) < 2:
            return []

        tr_values = []
        for i in range(1, len(bars)):
            tr = ATRCalculator.calculate_true_range(
                high=bars[i].high,
                low=bars[i].low,
                prev_close=bars[i - 1].close,
            )
            tr_values.append(tr)
        return tr_values

    @staticmethod
    def calculate_atr(
        bars: List,
        period: int = 14,
    ) -> Optional[float]:
        """
        Calculate ATR using Wilder's smoothed average.

        Algorithm:
        1. Calculate True Range for each bar (requires previous close)
        2. First ATR = Simple average of first {period} True Ranges
        3. Subsequent ATR = ((prev_ATR * (period-1)) + current_TR) / period

        Args:
            bars: List of BarData objects with high, low, close attributes.
                  Must have at least period + 1 bars.
            period: ATR period (default: 14).

        Returns:
            ATR value or None if insufficient data.
        """
        tr_values = ATRCalculator.calculate_true_ranges(bars)

        if len(tr_values) < period:
            logger.warning(
                f"Insufficient data for ATR({period}): have {len(tr_values)} TR values, need {period}"
            )
            return None

        # First ATR = SMA of first {period} TR values
        atr = sum(tr_values[:period]) / period

        # Apply Wilder's smoothing for remaining values
        for i in range(period, len(tr_values)):
            atr = ((atr * (period - 1)) + tr_values[i]) / period

        return atr

    @staticmethod
    def calculate_levels(
        symbol: str,
        entry_price: float,
        atr_value: float,
        period: int = 14,
    ) -> ATRData:
        """
        Calculate ATR-based price levels.

        Args:
            symbol: Symbol for the ATR data.
            entry_price: Entry/current price.
            atr_value: Calculated ATR value.
            period: ATR period used.

        Returns:
            ATRData with all levels calculated.
        """
        atr_percent = (atr_value / entry_price * 100) if entry_price > 0 else 0

        return ATRData(
            symbol=symbol,
            current_price=entry_price,
            atr_value=atr_value,
            atr_percent=atr_percent,
            period=period,
            # Stop loss levels
            stop_loss_1x=entry_price - atr_value,
            stop_loss_1_5x=entry_price - (atr_value * 1.5),
            stop_loss_2x=entry_price - (atr_value * 2.0),
            # Take profit levels
            take_profit_7x=entry_price + (atr_value * 7),
            take_profit_8x=entry_price + (atr_value * 8),
            take_profit_9x=entry_price + (atr_value * 9),
            take_profit_10x=entry_price + (atr_value * 10),
            take_profit_11x=entry_price + (atr_value * 11),
        )

    @classmethod
    def from_bars(
        cls,
        symbol: str,
        bars: List,
        period: int = 14,
        entry_price: Optional[float] = None,
    ) -> Optional[ATRData]:
        """
        Calculate ATR and levels from bar data.

        Args:
            symbol: Symbol for the ATR data.
            bars: List of BarData objects.
            period: ATR period.
            entry_price: Entry price (defaults to last bar close).

        Returns:
            ATRData or None if calculation fails.
        """
        atr_value = cls.calculate_atr(bars, period)
        if atr_value is None:
            return None

        if entry_price is None:
            entry_price = bars[-1].close if bars else 0

        return cls.calculate_levels(symbol, entry_price, atr_value, period)


class ATROptimizer:
    """
    Optimize ATR parameters using historical data analysis.

    Tests combinations of periods and multipliers against historical
    price data to find optimal stop loss / take profit settings.
    """

    DEFAULT_PERIODS = [10, 14, 20]
    DEFAULT_STOP_MULTIPLIERS = [1.0, 1.5, 2.0]
    DEFAULT_PROFIT_MULTIPLIERS = [7.0, 8.0, 9.0, 10.0, 11.0]

    @classmethod
    def analyze_historical_performance(
        cls,
        bars: List,
        periods: Optional[List[int]] = None,
        stop_multipliers: Optional[List[float]] = None,
        profit_multipliers: Optional[List[float]] = None,
    ) -> List[ATROptimizationResult]:
        """
        Analyze historical performance of different ATR parameter combinations.

        Simulates trades at each bar using different ATR periods and
        multipliers, tracking which combinations perform best.

        Args:
            bars: Historical bar data (OHLC).
            periods: ATR periods to test (default: [10, 14, 20]).
            stop_multipliers: Stop loss multipliers to test.
            profit_multipliers: Take profit multipliers to test.

        Returns:
            List of optimization results, sorted by score (best first).
        """
        periods = periods or cls.DEFAULT_PERIODS
        stop_multipliers = stop_multipliers or cls.DEFAULT_STOP_MULTIPLIERS
        profit_multipliers = profit_multipliers or cls.DEFAULT_PROFIT_MULTIPLIERS

        results = []

        for period in periods:
            for stop_mult in stop_multipliers:
                for profit_mult in profit_multipliers:
                    result = cls._simulate_combination(
                        bars=bars,
                        period=period,
                        stop_mult=stop_mult,
                        profit_mult=profit_mult,
                    )
                    if result:
                        results.append(result)

        # Sort by score (best first)
        results.sort(key=lambda r: r.score, reverse=True)

        # Mark best as recommended
        if results:
            results[0].is_recommended = True

        return results

    @classmethod
    def _simulate_combination(
        cls,
        bars: List,
        period: int,
        stop_mult: float,
        profit_mult: float,
    ) -> Optional[ATROptimizationResult]:
        """
        Simulate trades using a specific parameter combination.

        For each bar (after having enough history for ATR), simulates
        entering a long position and checks if take profit or stop loss
        would be hit first in subsequent bars.

        Args:
            bars: Historical bar data.
            period: ATR period.
            stop_mult: Stop loss multiplier.
            profit_mult: Take profit multiplier.

        Returns:
            Optimization result or None if insufficient data.
        """
        if len(bars) < period + 10:
            return None

        wins = 0
        losses = 0
        reward_risk_sum = 0.0

        # Start simulation after we have enough bars for ATR
        for i in range(period + 1, len(bars) - 5):
            # Calculate ATR at entry point
            entry_bars = bars[: i + 1]
            atr = ATRCalculator.calculate_atr(entry_bars, period)
            if atr is None or atr <= 0:
                continue

            entry_price = bars[i].close
            stop_loss = entry_price - (atr * stop_mult)
            take_profit = entry_price + (atr * profit_mult)

            # Simulate forward to see which level is hit first
            hit_tp, hit_sl = cls._check_exit(
                bars[i + 1 :], stop_loss, take_profit
            )

            if hit_tp:
                wins += 1
                reward_risk_sum += profit_mult / stop_mult
            elif hit_sl:
                losses += 1
                reward_risk_sum += 0  # Loss

        total_trades = wins + losses
        if total_trades == 0:
            return None

        win_rate = (wins / total_trades) * 100
        avg_reward_risk = reward_risk_sum / total_trades if total_trades > 0 else 0

        return ATROptimizationResult(
            period=period,
            stop_multiplier=stop_mult,
            profit_multiplier=profit_mult,
            historical_win_rate=win_rate,
            avg_reward_risk=avg_reward_risk,
            total_simulated_trades=total_trades,
        )

    @staticmethod
    def _check_exit(
        future_bars: List,
        stop_loss: float,
        take_profit: float,
        max_bars: int = 20,
    ) -> Tuple[bool, bool]:
        """
        Check if take profit or stop loss is hit in future bars.

        Args:
            future_bars: Bars after entry.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            max_bars: Maximum bars to check.

        Returns:
            Tuple of (hit_take_profit, hit_stop_loss).
        """
        for bar in future_bars[:max_bars]:
            # Check if stop loss hit (bar low goes below stop)
            if bar.low <= stop_loss:
                return False, True
            # Check if take profit hit (bar high goes above take profit)
            if bar.high >= take_profit:
                return True, False

        # Neither hit within max_bars
        return (False, False)

    @classmethod
    def get_recommendation( # TODO this is not wired!!!!
        cls,
        bars: List,
    ) -> Optional[ATROptimizationResult]:
        """
        Get the best recommended ATR parameters for given data.

        Args:
            bars: Historical bar data.

        Returns:
            Best optimization result or None.
        """
        results = cls.analyze_historical_performance(bars)
        return results[0] if results else None


@dataclass
class ATRCacheEntry:
    """Cached ATR data for a symbol."""

    data: ATRData
    bars: List  # Keep bars for recalculation with different params
    fetched_at: datetime = field(default_factory=datetime.now)
    optimization_result: Optional[ATROptimizationResult] = None


class ATRCache:
    """
    Session-level cache for ATR calculations.

    Caches both the calculated ATR data and the underlying bars,
    allowing recalculation with different parameters without
    re-fetching data.
    """

    def __init__(self):
        self._cache: Dict[str, ATRCacheEntry] = {}

    def get(self, symbol: str) -> Optional[ATRData]:
        """Get cached ATR data for a symbol."""
        entry = self._cache.get(symbol)
        return entry.data if entry else None

    def get_entry(self, symbol: str) -> Optional[ATRCacheEntry]:
        """Get full cache entry for a symbol."""
        return self._cache.get(symbol)

    def set(
        self,
        symbol: str,
        data: ATRData,
        bars: List,
        optimization: Optional[ATROptimizationResult] = None,
    ) -> None:
        """Cache ATR data and bars for a symbol."""
        self._cache[symbol] = ATRCacheEntry(
            data=data,
            bars=bars,
            optimization_result=optimization,
        )

    def has(self, symbol: str) -> bool:
        """Check if symbol is cached."""
        return symbol in self._cache

    def invalidate(self, symbol: str) -> None:
        """Remove symbol from cache."""
        self._cache.pop(symbol, None)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    def recalculate(
        self,
        symbol: str,
        period: int,
        entry_price: Optional[float] = None,
    ) -> Optional[ATRData]:
        """
        Recalculate ATR with different params using cached bars.

        Args:
            symbol: Symbol to recalculate.
            period: New ATR period.
            entry_price: Entry price (uses cached price if None).

        Returns:
            New ATRData or None if not cached.
        """
        entry = self._cache.get(symbol)
        if not entry:
            return None

        if entry_price is None:
            entry_price = entry.data.current_price

        new_data = ATRCalculator.from_bars(
            symbol=symbol,
            bars=entry.bars,
            period=period,
            entry_price=entry_price,
        )

        if new_data:
            entry.data = new_data

        return new_data

    def get_symbols(self) -> List[str]:
        """Get all cached symbols."""
        return list(self._cache.keys())
