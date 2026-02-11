"""
Signal generation interfaces and TA-Lib indicator helpers.

This module provides:
- SignalGenerator protocol for vectorized entry/exit generation
- Concrete SignalGenerator implementations for each strategy
- TA-Lib wrappers for common technical indicators
"""

from .buy_and_hold import BuyAndHoldSignalGenerator
from .dual_macd_gate import DualMACDGateSignalGenerator
from .indicators import adx, atr, bbands, ema, macd, momentum, rsi, sma
from .protocol import DirectionalSignalGenerator, SignalGenerator
from .rsi_mean_reversion import RSIMeanReversionSignalGenerator
from .trend_pulse import TrendPulseSignalGenerator

__all__ = [
    # Protocols
    "SignalGenerator",
    "DirectionalSignalGenerator",
    # Signal generators
    "BuyAndHoldSignalGenerator",
    "DualMACDGateSignalGenerator",
    "RSIMeanReversionSignalGenerator",
    "TrendPulseSignalGenerator",
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
