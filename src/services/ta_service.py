"""
Technical Analysis Service using TA-Lib.

Provides technical indicators (ATR, RSI, etc.) using TA-Lib.
Delegates historical data fetching to HistoricalDataService.

Requirements:
    pip install TA-Lib
    # Note: TA-Lib requires system library installation first:
    # macOS: brew install ta-lib
    # Ubuntu: apt-get install libta-lib0-dev

Usage:
    ta = TAService(historical_service)

    # Get ATR for a symbol
    atr = await ta.get_atr("AAPL", period=14)

    # Batch ATR for multiple symbols
    atrs = await ta.get_atr_batch(["AAPL", "MSFT", "GOOGL"], period=14)

    # Get ATR levels (support/resistance based on ATR)
    levels = await ta.get_atr_levels("AAPL", spot_price=150.0, period=14)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING

from ..utils.logging_setup import get_logger
from .bar_cache_service import BarPeriod

if TYPE_CHECKING:
    from .historical_data_service import HistoricalDataService


logger = get_logger(__name__)


# Check if TA-Lib is available
try:
    import talib
    import numpy as np
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning(
        "TA-Lib not installed. Install with: pip install TA-Lib "
        "(requires system library: brew install ta-lib on macOS)"
    )


@dataclass
class ATRLevels:
    """ATR-based price levels for a symbol."""
    symbol: str
    spot: float
    atr: float
    atr_pct: float  # ATR as percentage of spot
    level_1_up: float  # spot + 1 ATR
    level_1_dn: float  # spot - 1 ATR
    level_2_up: float  # spot + 2 ATR
    level_2_dn: float  # spot - 2 ATR
    timeframe: str = "1d"
    period: int = 14


class TAService:
    """
    Technical Analysis Service using TA-Lib.

    Provides commonly used indicators:
    - ATR (Average True Range) - volatility
    - RSI (Relative Strength Index) - momentum (future)
    - SMA/EMA (Moving Averages) - trend (future)
    """

    def __init__(
        self,
        historical_service: "HistoricalDataService",
        default_atr_period: int = 14,
        default_lookback_buffer: int = 10,
    ):
        """
        Initialize TA service.

        Args:
            historical_service: Service for fetching bar data.
            default_atr_period: Default ATR period (typically 14).
            default_lookback_buffer: Extra bars to fetch beyond period.
        """
        self._historical = historical_service
        self._default_atr_period = default_atr_period
        self._lookback_buffer = default_lookback_buffer

        if not TALIB_AVAILABLE:
            logger.error("TAService initialized but TA-Lib is not available!")

    async def get_atr(
        self,
        symbol: str,
        period: int = None,
        timeframe: str = "1d",
    ) -> Optional[float]:
        """
        Calculate ATR (Average True Range) for a symbol.

        Args:
            symbol: Stock symbol.
            period: ATR period (default: 14).
            timeframe: Bar timeframe (default: "1d").

        Returns:
            ATR value or None if insufficient data.
        """
        if not TALIB_AVAILABLE:
            return None

        period = period or self._default_atr_period
        lookback = period + self._lookback_buffer

        try:
            bars = await self._historical.fetch_bars(
                symbol,
                timeframe,
                BarPeriod.bars(lookback),
            )

            if len(bars) < period:
                logger.warning(
                    f"Insufficient data for {symbol} ATR: "
                    f"got {len(bars)} bars, need {period}"
                )
                return None

            high = np.array([b.high for b in bars], dtype=np.float64)
            low = np.array([b.low for b in bars], dtype=np.float64)
            close = np.array([b.close for b in bars], dtype=np.float64)

            atr = talib.ATR(high, low, close, timeperiod=period)

            # Return the latest ATR value
            latest = atr[-1]
            if np.isnan(latest):
                return None

            return float(latest)

        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return None

    async def get_atr_batch(
        self,
        symbols: List[str],
        period: int = None,
        timeframe: str = "1d",
    ) -> Dict[str, Optional[float]]:
        """
        Calculate ATR for multiple symbols efficiently.

        Uses batch fetching from HistoricalDataService.

        Args:
            symbols: List of stock symbols.
            period: ATR period (default: 14).
            timeframe: Bar timeframe (default: "1d").

        Returns:
            Dict mapping symbol to ATR value (or None).
        """
        if not TALIB_AVAILABLE:
            return {sym: None for sym in symbols}

        period = period or self._default_atr_period
        lookback = period + self._lookback_buffer

        # Batch fetch all bars
        requests = [
            {"symbol": sym, "timeframe": timeframe, "period": BarPeriod.bars(lookback)}
            for sym in symbols
        ]
        bars_by_symbol = await self._historical.fetch_bars_batch(requests)

        # Calculate ATR for each symbol
        results: Dict[str, Optional[float]] = {}
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol, [])

            if len(bars) < period:
                results[symbol] = None
                continue

            try:
                high = np.array([b.high for b in bars], dtype=np.float64)
                low = np.array([b.low for b in bars], dtype=np.float64)
                close = np.array([b.close for b in bars], dtype=np.float64)

                atr = talib.ATR(high, low, close, timeperiod=period)
                latest = atr[-1]
                results[symbol] = None if np.isnan(latest) else float(latest)

            except Exception as e:
                logger.warning(f"ATR calculation failed for {symbol}: {e}")
                results[symbol] = None

        return results

    async def get_atr_levels(
        self,
        symbol: str,
        spot_price: float,
        period: int = None,
        timeframe: str = "1d",
    ) -> Optional[ATRLevels]:
        """
        Get ATR-based price levels for a symbol.

        Calculates support/resistance levels based on ATR multiples.

        Args:
            symbol: Stock symbol.
            spot_price: Current spot price.
            period: ATR period (default: 14).
            timeframe: Bar timeframe (default: "1d").

        Returns:
            ATRLevels dataclass or None if calculation fails.
        """
        period = period or self._default_atr_period
        atr = await self.get_atr(symbol, period, timeframe)

        if atr is None or spot_price <= 0:
            return None

        atr_pct = (atr / spot_price) * 100

        return ATRLevels(
            symbol=symbol,
            spot=spot_price,
            atr=atr,
            atr_pct=atr_pct,
            level_1_up=spot_price + atr,
            level_1_dn=spot_price - atr,
            level_2_up=spot_price + (2 * atr),
            level_2_dn=spot_price - (2 * atr),
            timeframe=timeframe,
            period=period,
        )

    async def get_atr_levels_batch(
        self,
        symbols_with_spots: Dict[str, float],
        period: int = None,
        timeframe: str = "1d",
    ) -> Dict[str, Optional[ATRLevels]]:
        """
        Get ATR levels for multiple symbols efficiently.

        Args:
            symbols_with_spots: Dict mapping symbol to spot price.
            period: ATR period (default: 14).
            timeframe: Bar timeframe.

        Returns:
            Dict mapping symbol to ATRLevels (or None).
        """
        period = period or self._default_atr_period
        symbols = list(symbols_with_spots.keys())

        atrs = await self.get_atr_batch(symbols, period, timeframe)

        results: Dict[str, Optional[ATRLevels]] = {}
        for symbol, atr in atrs.items():
            spot = symbols_with_spots.get(symbol, 0)
            if atr is None or spot <= 0:
                results[symbol] = None
                continue

            atr_pct = (atr / spot) * 100
            results[symbol] = ATRLevels(
                symbol=symbol,
                spot=spot,
                atr=atr,
                atr_pct=atr_pct,
                level_1_up=spot + atr,
                level_1_dn=spot - atr,
                level_2_up=spot + (2 * atr),
                level_2_dn=spot - (2 * atr),
                timeframe=timeframe,
                period=period,
            )

        return results

    # -------------------------------------------------------------------------
    # Future Indicators (placeholders for extensibility)
    # -------------------------------------------------------------------------

    async def get_rsi(
        self,
        symbol: str,
        period: int = 14,
        timeframe: str = "1d",
    ) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index).

        TODO: Implement when needed.
        """
        if not TALIB_AVAILABLE:
            return None

        lookback = period + self._lookback_buffer
        bars = await self._historical.fetch_bars(
            symbol, timeframe, BarPeriod.bars(lookback)
        )

        if len(bars) < period:
            return None

        close = np.array([b.close for b in bars], dtype=np.float64)
        rsi = talib.RSI(close, timeperiod=period)

        latest = rsi[-1]
        return None if np.isnan(latest) else float(latest)

    async def get_sma(
        self,
        symbol: str,
        period: int = 20,
        timeframe: str = "1d",
    ) -> Optional[float]:
        """
        Calculate Simple Moving Average.

        TODO: Implement when needed.
        """
        if not TALIB_AVAILABLE:
            return None

        bars = await self._historical.fetch_bars(
            symbol, timeframe, BarPeriod.bars(period + 5)
        )

        if len(bars) < period:
            return None

        close = np.array([b.close for b in bars], dtype=np.float64)
        sma = talib.SMA(close, timeperiod=period)

        latest = sma[-1]
        return None if np.isnan(latest) else float(latest)
