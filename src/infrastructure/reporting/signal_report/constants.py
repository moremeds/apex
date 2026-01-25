"""
Signal Report Constants.

Shared constants for signal report generation.
"""

from __future__ import annotations

# Timeframe ordering for consistent display
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}

# Indicator grouping for chart layout
# Overlays: Same Y-axis as price
OVERLAY_INDICATORS = {
    "bollinger",
    "supertrend",
    "sma",
    "ema",
    "vwap",
    "keltner",
    "donchian",
    "ichimoku",
}
# Bounded oscillators (0-100 or similar fixed range)
BOUNDED_OSCILLATORS = {"rsi", "stochastic", "kdj", "williams_r", "mfi", "cci", "adx"}
# Unbounded oscillators (MACD-style, centered around 0)
UNBOUNDED_OSCILLATORS = {"macd", "momentum", "roc", "cmf", "pvo", "force_index"}
# Volume indicators
VOLUME_INDICATORS = {"obv", "volume_profile", "vwma", "ease_of_movement", "chaikin_volatility"}
