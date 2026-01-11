"""
Volatility indicators package.

Volatility indicators measure the magnitude of price fluctuations
and identify breakout opportunities.

Indicators:
- Bollinger Bands
- ATR: Average True Range
- Keltner Channels
- Donchian Channels
- Standard Deviation
- Chaikin Volatility
- Historical Volatility
- Squeeze (BB inside Keltner)
"""

# Indicators will be auto-discovered by IndicatorRegistry
VOLATILITY_INDICATORS: list[str] = []
