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

Feature coverage by strategy:
+---------------------------+--------+-------+------+-------+---------+------+
| Strategy                  | on_bar | Clock | Sched| Limit | Multi-  | Fill |
|                           |        |       |      | Order | Symbol  | Mgmt |
+---------------------------+--------+-------+------+-------+---------+------+
| MovingAverageCrossStrategy|   -    |   -   |  -   |   -   |    -    |  X   |
| BuyAndHoldStrategy        |   X    |   X   |  -   |   -   |    X    |  X   |
| RsiMeanReversionStrategy  |   X    |   -   |  -   |   X   |    -    |  X   |
| MomentumBreakoutStrategy  |   X    |   -   |  -   |   -   |    -    |  X   |
| PairsTradingStrategy      |   X    |   -   |  -   |   -   |    X    |  X   |
| ScheduledRebalanceStrategy|   X    |   X   |  X   |   -   |    X    |  X   |
+---------------------------+--------+-------+------+-------+---------+------+
"""

from .ma_cross import MovingAverageCrossStrategy
from .buy_and_hold import BuyAndHoldStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .pairs_trading import PairsTradingStrategy
from .scheduled_rebalance import ScheduledRebalanceStrategy

__all__ = [
    # Basic strategies
    "MovingAverageCrossStrategy",
    "BuyAndHoldStrategy",
    # Advanced strategies
    "RsiMeanReversionStrategy",
    "MomentumBreakoutStrategy",
    "PairsTradingStrategy",
    "ScheduledRebalanceStrategy",
]
