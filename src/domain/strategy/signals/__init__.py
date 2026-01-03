"""
Signal generation interfaces and TA-Lib indicator helpers.

This module provides:
- SignalGenerator protocol for vectorized entry/exit generation
- Concrete SignalGenerator implementations for each strategy
- TA-Lib wrappers for common technical indicators
"""

from .buy_and_hold import BuyAndHoldSignalGenerator
from .indicators import adx, atr, bbands, ema, macd, momentum, rsi, sma
from .ma_cross import MACrossSignalGenerator
from .momentum_breakout import MomentumBreakoutSignalGenerator
from .pairs_trading import PairsTradingSignalGenerator
from .protocol import SignalGenerator
from .rsi_mean_reversion import RSIMeanReversionSignalGenerator
from .ta_metrics import TAMetricsSignalGenerator

__all__ = [
    # Protocol
    "SignalGenerator",
    # Signal generators
    "BuyAndHoldSignalGenerator",
    "MACrossSignalGenerator",
    "MomentumBreakoutSignalGenerator",
    "PairsTradingSignalGenerator",
    "RSIMeanReversionSignalGenerator",
    "TAMetricsSignalGenerator",
    # TA-Lib wrappers
    "adx",
    "atr",
    "bbands",
    "ema",
    "macd",
    "momentum",
    "rsi",
    "sma",
]
