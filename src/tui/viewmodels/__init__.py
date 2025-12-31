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

from .base import BaseViewModel, CellUpdate, RowUpdate
from .position_vm import PositionViewModel
from .signal_vm import SignalViewModel
from .strategy_vm import StrategyDisplayState, StrategyViewModel
from .summary_vm import SummaryViewModel
from .alert_vm import AlertViewModel  # OPT-011
from .order_vm import OrderViewModel  # OPT-011

__all__ = [
    "BaseViewModel",
    "CellUpdate",
    "RowUpdate",
    "PositionViewModel",
    "SignalViewModel",
    "StrategyDisplayState",
    "StrategyViewModel",
    "SummaryViewModel",
    "AlertViewModel",
    "OrderViewModel",
]
