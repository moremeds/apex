"""Panel rendering modules for the Terminal Dashboard."""

from .header import render_header
from .summary import render_portfolio_summary
from .alerts import render_market_alerts, update_persistent_alerts
from .signals import (
    render_breaches,
    render_risk_signals,
    render_risk_signals_fullscreen,
    update_persistent_risk_signals,
)
from .positions import (
    render_consolidated_positions,
    render_broker_positions,
    render_positions_profile,
)
from .health import render_health
from .history import (
    render_position_history_today,
    render_open_orders,
    render_position_history_recent,
)
from .atr_levels import (
    render_atr_levels,
    render_atr_loading,
    render_atr_empty,
    render_atr_compact,
)
from .strategies import (
    render_strategy_list,
    render_strategy_params,
    render_backtest_performance,
    render_lab_main,
)

__all__ = [
    # Header
    "render_header",
    # Summary
    "render_portfolio_summary",
    # Alerts
    "render_market_alerts",
    "update_persistent_alerts",
    # Signals
    "render_breaches",
    "render_risk_signals",
    "render_risk_signals_fullscreen",
    "update_persistent_risk_signals",
    # Positions
    "render_consolidated_positions",
    "render_broker_positions",
    "render_positions_profile",
    # Health
    "render_health",
    # History
    "render_position_history_today",
    "render_open_orders",
    "render_position_history_recent",
    # ATR Levels
    "render_atr_levels",
    "render_atr_loading",
    "render_atr_empty",
    "render_atr_compact",
    # Strategies / Lab
    "render_strategy_list",
    "render_strategy_params",
    "render_backtest_performance",
    "render_lab_main",
]
