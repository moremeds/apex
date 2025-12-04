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

__all__ = [
    "render_header",
    "render_portfolio_summary",
    "render_market_alerts",
    "update_persistent_alerts",
    "render_breaches",
    "render_risk_signals",
    "render_risk_signals_fullscreen",
    "update_persistent_risk_signals",
    "render_consolidated_positions",
    "render_broker_positions",
    "render_positions_profile",
    "render_health",
    "render_position_history_today",
    "render_open_orders",
    "render_position_history_recent",
]
