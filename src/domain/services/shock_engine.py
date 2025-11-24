"""
Simple Shock Engine - Spot shock scenarios only.

Applies spot price shocks to portfolio and re-calculates Greeks.
Defers IV shocks and combined shocks to v1.2.
"""

from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass
from copy import deepcopy
from ...models.position import Position
from ...models.market_data import MarketData
from ...models.risk_snapshot import RiskSnapshot
from .risk_engine import RiskEngine


@dataclass
class ShockScenario:
    """Shock scenario result."""
    name: str
    spot_shock: float  # Percentage (e.g., -0.05 for -5%)
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float
    total_unrealized_pnl: float
    pnl_change: float  # vs base case


class SimpleShockEngine:
    """
    Simple spot shock scenario engine (MVP).

    Applies spot price shocks only. Greeks are NOT recalculated
    (assumes Greeks remain constant for small shocks).
    """

    def __init__(self, risk_engine: RiskEngine, config: Dict[str, any]):
        """
        Initialize shock engine.

        Args:
            risk_engine: RiskEngine instance for recalculation.
            config: Scenarios configuration dict.
        """
        self.risk_engine = risk_engine
        self.config = config

    def run_spot_shocks(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        base_snapshot: RiskSnapshot,
    ) -> List[ShockScenario]:
        """
        Run spot shock scenarios.

        Args:
            positions: Current positions.
            market_data: Current market data.
            base_snapshot: Base case snapshot (no shock).

        Returns:
            List of ShockScenario results.
        """
        results = []
        spot_shocks = self.config.get("scenarios", {}).get("spot_shocks", [])

        for shock_pct in spot_shocks:
            shocked_md = self._apply_spot_shock(market_data, shock_pct)
            shocked_snapshot = self.risk_engine.build_snapshot(
                positions, shocked_md, base_snapshot  # Reuse account info
            )

            scenario = ShockScenario(
                name=f"Spot_{shock_pct:+.0%}",
                spot_shock=shock_pct,
                portfolio_delta=shocked_snapshot.portfolio_delta,
                portfolio_gamma=shocked_snapshot.portfolio_gamma,
                portfolio_vega=shocked_snapshot.portfolio_vega,
                portfolio_theta=shocked_snapshot.portfolio_theta,
                total_unrealized_pnl=shocked_snapshot.total_unrealized_pnl,
                pnl_change=shocked_snapshot.total_unrealized_pnl - base_snapshot.total_unrealized_pnl,
            )
            results.append(scenario)

        return results

    def _apply_spot_shock(
        self, market_data: Dict[str, MarketData], shock_pct: float
    ) -> Dict[str, MarketData]:
        """
        Apply spot price shock to market data.

        Args:
            market_data: Original market data.
            shock_pct: Shock percentage (e.g., -0.05 for -5%).

        Returns:
            New dict with shocked prices. Greeks remain unchanged (simplification).
        """
        shocked = {}
        for symbol, md in market_data.items():
            shocked_md = deepcopy(md)
            if shocked_md.last is not None:
                shocked_md.last *= (1 + shock_pct)
            if shocked_md.bid is not None:
                shocked_md.bid *= (1 + shock_pct)
            if shocked_md.ask is not None:
                shocked_md.ask *= (1 + shock_pct)
            if shocked_md.mid is not None:
                shocked_md.mid *= (1 + shock_pct)
            shocked[symbol] = shocked_md

        return shocked
