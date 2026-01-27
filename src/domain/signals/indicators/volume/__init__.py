"""
Volume indicators package.

Volume indicators analyze trading volume to confirm price movements
and identify accumulation/distribution patterns.

Indicators:
- Volume: Basic volume analysis with spike detection
- OBV: On-Balance Volume
- VWAP: Volume Weighted Average Price
- CVD: Cumulative Volume Delta
- Volume Ratio
- AD Line: Accumulation/Distribution
- CMF: Chaikin Money Flow
- Force Index
- VPVR: Volume Profile
"""

# Indicators will be auto-discovered by IndicatorRegistry
VOLUME_INDICATORS: list[str] = []
