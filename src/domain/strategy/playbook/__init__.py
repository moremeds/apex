"""
Strategy Playbook.

Production trading strategies registered via @register_strategy.

Available strategies:
- BuyAndHoldStrategy: Passive buy and hold for benchmarking
- RsiMeanReversionStrategy: RSI-based mean reversion with limit orders
- RegimeFlexStrategy: Regime-switched gross exposure scaling (Tier 2)
- SectorPulseStrategy: Regime-aware sector rotation (Tier 2)
- TrendPulseStrategy: Zig-zag swing + DualMACD momentum (Tier 1)
"""

from .buy_and_hold import BuyAndHoldStrategy
from .regime_flex import RegimeFlexStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .sector_pulse import SectorPulseStrategy
from .trend_pulse import TrendPulseStrategy

__all__ = [
    # Basic strategies
    "BuyAndHoldStrategy",
    # Advanced strategies
    "RsiMeanReversionStrategy",
    # Regime-aware strategies (Tier 1)
    "TrendPulseStrategy",
    # Regime-aware strategies (Tier 2)
    "RegimeFlexStrategy",
    "SectorPulseStrategy",
]
