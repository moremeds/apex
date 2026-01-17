"""
Example trading strategies.

This module contains example strategy implementations that demonstrate
how to use the Apex strategy framework.

Available strategies:
- MovingAverageCrossStrategy: Classic MA crossover for trend following
- BuyAndHoldStrategy: Passive buy and hold for benchmarking
- RsiMeanReversionStrategy: RSI-based mean reversion with limit orders
- MomentumBreakoutStrategy: ATR-based breakout with trailing stops
- PairsTradingStrategy: Statistical arbitrage pairs trading
- ScheduledRebalanceStrategy: Time-based portfolio rebalancing
- TAMetricsStrategy: Multi-indicator TA with metrics matrix (MA, RSI, MACD)
- MTFRsiTrendStrategy: Multi-timeframe RSI trend + entry (daily/hourly)

Feature coverage by strategy:
+---------------------------+--------+--------+-------+------+-------+---------+------+
| Strategy                  | on_bar | on_bars| Clock | Sched| Limit | Multi-  | Fill |
|                           |        | (MTF)  |       |      | Order | Symbol  | Mgmt |
+---------------------------+--------+--------+-------+------+-------+---------+------+
| MovingAverageCrossStrategy|   -    |   -    |   -   |  -   |   -   |    -    |  X   |
| BuyAndHoldStrategy        |   X    |   -    |   X   |  -   |   -   |    X    |  X   |
| RsiMeanReversionStrategy  |   X    |   -    |   -   |  -   |   X   |    -    |  X   |
| MomentumBreakoutStrategy  |   X    |   -    |   -   |  -   |   -   |    -    |  X   |
| PairsTradingStrategy      |   X    |   -    |   -   |  -   |   -   |    X    |  X   |
| ScheduledRebalanceStrategy|   X    |   -    |   X   |  X   |   -   |    X    |  X   |
| TAMetricsStrategy         |   X    |   -    |   -   |  -   |   -   |    X    |  X   |
| MTFRsiTrendStrategy       |   X    |   X    |   -   |  -   |   -   |    X    |  X   |
+---------------------------+--------+--------+-------+------+-------+---------+------+
"""

from .buy_and_hold import BuyAndHoldStrategy
from .ma_cross import MovingAverageCrossStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .mtf_rsi_trend import MTFRsiTrendStrategy
from .pairs_trading import PairsTradingStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .scheduled_rebalance import ScheduledRebalanceStrategy
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
]
