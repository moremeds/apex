"""
Textual widgets for the Apex Dashboard.
"""

from .alerts_list import AlertsList
from .atr_panel import ATRPanel
from .backtest_results import BacktestResultsPanel
from .header import HeaderWidget
from .health_bar import HealthBar
from .orders_panel import OrdersPanel
from .positions_table import PositionsTable
from .signals_table import SignalsTable
from .strategy_config import StrategyConfigPanel
from .strategy_list import StrategyList
from .summary_panel import SummaryPanel

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
