"""
Pure calculation functions for risk metrics.

These calculators are stateless, pure functions that take inputs and return
immutable results. They are designed for:
- Easy unit testing (no mocks required)
- Thread safety (immutable outputs)
- Streaming architecture (can be called per-tick)

Modules:
- pnl_calculator: Unrealized, daily, intraday P&L
- greeks_calculator: Position Greeks scaling
- notional_calculator: Notional and concentration metrics
"""

from .greeks_calculator import GreeksResult, calculate_position_greeks
from .notional_calculator import NotionalResult, calculate_notional
from .pnl_calculator import PnLResult, calculate_pnl

__all__ = [
    "PnLResult",
    "calculate_pnl",
    "GreeksResult",
    "calculate_position_greeks",
    "NotionalResult",
    "calculate_notional",
]
