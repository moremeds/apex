"""
Textual widgets for the Apex Dashboard.
"""

from .positions_table import PositionsTable
from .summary_panel import SummaryPanel
from .alerts_list import AlertsList
from .signals_table import SignalsTable
from .health_bar import HealthBar
from .atr_panel import ATRPanel
from .strategy_list import StrategyList
from .strategy_config import StrategyConfigPanel
from .backtest_results import BacktestResultsPanel
from .orders_panel import OrdersPanel
from .header import HeaderWidget

__all__ = [
    "PositionsTable",
    "SummaryPanel",
    "AlertsList",
    "SignalsTable",
    "HealthBar",
    "ATRPanel",
    "StrategyList",
    "StrategyConfigPanel",
    "BacktestResultsPanel",
    "OrdersPanel",
    "HeaderWidget",
]
