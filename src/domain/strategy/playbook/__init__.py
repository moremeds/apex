"""
Strategy Playbook.

Production trading strategies registered via @register_strategy.

Available strategies:
- MovingAverageCrossStrategy: Classic MA crossover for trend following
- BuyAndHoldStrategy: Passive buy and hold for benchmarking
- RsiMeanReversionStrategy: RSI-based mean reversion with limit orders
- MomentumBreakoutStrategy: ATR-based breakout with trailing stops
- PairsTradingStrategy: Statistical arbitrage pairs trading
- ScheduledRebalanceStrategy: Time-based portfolio rebalancing
- TAMetricsStrategy: Multi-indicator TA with metrics matrix (MA, RSI, MACD)
- MTFRsiTrendStrategy: Multi-timeframe RSI trend + entry (daily/hourly)
- PulseDipStrategy: TrendPulse + DualMACD confluence dip-buying (Tier 1)
- SqueezePlayStrategy: Bollinger/Keltner squeeze breakout (Tier 1)
- RegimeFlexStrategy: Regime-switched gross exposure scaling (Tier 2)
- SectorPulseStrategy: Regime-aware sector rotation (Tier 2)
"""

from .buy_and_hold import BuyAndHoldStrategy
from .ma_cross import MovingAverageCrossStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .mtf_rsi_trend import MTFRsiTrendStrategy
from .pairs_trading import PairsTradingStrategy
from .pulse_dip import PulseDipStrategy
from .regime_flex import RegimeFlexStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .scheduled_rebalance import ScheduledRebalanceStrategy
from .sector_pulse import SectorPulseStrategy
from .squeeze_play import SqueezePlayStrategy
from .ta_metrics_strategy import TAMetricsStrategy

__all__ = [
    # Basic strategies
    "MovingAverageCrossStrategy",
    "BuyAndHoldStrategy",
    # Advanced strategies
    "RsiMeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "PairsTradingStrategy",
    "ScheduledRebalanceStrategy",
    # Systematic backtesting strategies
    "TAMetricsStrategy",
    # Multi-timeframe strategies
    "MTFRsiTrendStrategy",
    # Regime-aware strategies (Tier 1)
    "PulseDipStrategy",
    "SqueezePlayStrategy",
    # Regime-aware strategies (Tier 2)
    "RegimeFlexStrategy",
    "SectorPulseStrategy",
]
