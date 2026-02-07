"""
Strategy comparison dashboard builder.

Generates a self-contained interactive HTML report comparing multiple
strategies side-by-side with equity curves, metrics, regime performance,
per-symbol heatmaps, and trade analysis.

Uses Plotly for interactive charts. Output is a single HTML file.

Usage:
    builder = StrategyComparisonBuilder()
    builder.add_strategy("pulse_dip", result_pulse)
    builder.add_strategy("squeeze_play", result_squeeze)
    builder.add_strategy("buy_and_hold", result_baseline)
    builder.build("results/comparison_report.html")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .templates import render_comparison_html

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""

    name: str
    tier: str = ""
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0
    avg_trade_pnl: float = 0.0
    equity_curve: List[List[float]] = field(default_factory=list)
    drawdown_curve: List[List[float]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    per_symbol_sharpe: Dict[str, float] = field(default_factory=dict)
    per_regime_sharpe: Dict[str, float] = field(default_factory=dict)
    per_regime_return: Dict[str, float] = field(default_factory=dict)
    stress_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    rolling_sharpe: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "total_return": round(self.total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 3),
            "profit_factor": round(self.profit_factor, 3),
            "trade_count": self.trade_count,
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve,
            "monthly_returns": self.monthly_returns,
            "per_symbol_sharpe": self.per_symbol_sharpe,
            "per_regime_sharpe": self.per_regime_sharpe,
            "per_regime_return": self.per_regime_return,
            "stress_results": self.stress_results,
            "rolling_sharpe": self.rolling_sharpe,
        }


class StrategyComparisonBuilder:
    """
    Builds a multi-strategy comparison HTML dashboard.

    Collects metrics from multiple strategy backtest runs and generates
    an interactive HTML report with 5 tabs.
    """

    def __init__(
        self,
        title: str = "APEX Strategy Comparison Dashboard",
        universe_name: str = "",
        period: str = "",
    ) -> None:
        self._title = title
        self._universe_name = universe_name
        self._period = period
        self._strategies: Dict[str, StrategyMetrics] = {}
        self._symbols: List[str] = []
        self._generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def add_strategy(self, name: str, metrics: StrategyMetrics) -> None:
        """Add a strategy's metrics to the comparison."""
        self._strategies[name] = metrics
        logger.info(f"Added strategy '{name}' to comparison dashboard")

    def set_symbols(self, symbols: List[str]) -> None:
        """Set the symbol universe."""
        self._symbols = symbols

    def build(self, output_path: str) -> str:
        """
        Build the HTML dashboard and write to file.

        Args:
            output_path: Path to write the HTML file.

        Returns:
            Path to the generated file.
        """
        if not self._strategies:
            logger.warning("No strategies to compare")
            return output_path

        # Prepare data for template
        data = {
            "title": self._title,
            "generated_at": self._generated_at,
            "universe_name": self._universe_name,
            "period": self._period,
            "strategy_count": len(self._strategies),
            "symbols": self._symbols,
            "strategies": {name: m.to_dict() for name, m in self._strategies.items()},
        }

        # Render HTML
        html = render_comparison_html(data)

        # Write to file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

        logger.info(
            f"Strategy comparison dashboard written to {output_path} "
            f"({len(self._strategies)} strategies)"
        )
        return output_path
