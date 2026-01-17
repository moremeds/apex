"""
TUI ViewModels - Framework-agnostic data transformation layer.

ViewModels handle:
- Business logic (grouping, aggregation, sorting)
- Diff computation for incremental updates
- State caching for efficient rendering

ViewModels MUST NOT:
- Import Textual modules
- Hold widget references
- Contain display/rendering logic
"""

from .alert_vm import AlertViewModel  # OPT-011
from .base import BaseViewModel, CellUpdate, RowUpdate
from .indicator_status_vm import IndicatorRow, IndicatorStatusViewModel, RowType
from .order_vm import OrderViewModel  # OPT-011
from .position_vm import PositionViewModel
from .signal_vm import SignalViewModel
from .strategy_vm import StrategyDisplayState, StrategyViewModel
from .summary_vm import SummaryViewModel
from .trading_signal_vm import TradingSignalViewModel

__all__ = [
    "BaseViewModel",
    "CellUpdate",
    "RowUpdate",
    "PositionViewModel",
    "SignalViewModel",
    "TradingSignalViewModel",
    "StrategyDisplayState",
    "StrategyViewModel",
    "SummaryViewModel",
    "AlertViewModel",
    "OrderViewModel",
    "IndicatorStatusViewModel",
    "IndicatorRow",
    "RowType",
]
