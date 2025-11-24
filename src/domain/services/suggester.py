"""
Simple Suggester - Top contributors diagnosis.

Identifies positions contributing most to limit breaches.
Defers optimization/hedging efficiency to v1.2.
"""

from __future__ import annotations
from typing import Dict, List, Any
from dataclasses import dataclass
from ...models.position import Position
from ...models.market_data import MarketData
from ...models.risk_snapshot import RiskSnapshot


@dataclass
class Contributor:
    """Position contributing to a risk metric."""
    symbol: str
    underlying: str
    contribution: float
    contribution_pct: float


class SimpleSuggester:
    """
    Simple diagnosis service for breach analysis.

    Identifies top contributors to breached metrics.
    MVP scope: diagnosis only (no hedge suggestions).
    """

    def __init__(self):
        """Initialize suggester."""
        pass

    def top_delta_contributors(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        snapshot: RiskSnapshot,
        top_n: int = 5,
    ) -> List[Contributor]:
        """
        Identify top delta contributors.

        Args:
            positions: All positions.
            market_data: Market data dict.
            snapshot: Current snapshot.
            top_n: Number of top contributors to return.

        Returns:
            List of Contributor objects sorted by absolute delta contribution.
        """
        contributions = []

        for pos in positions:
            md = market_data.get(pos.symbol)
            if not md or not md.has_greeks():
                continue

            delta_contrib = (md.delta or 0.0) * pos.quantity * pos.multiplier
            pct = (abs(delta_contrib) / abs(snapshot.portfolio_delta) * 100) if snapshot.portfolio_delta != 0 else 0

            contributions.append(
                Contributor(
                    symbol=pos.symbol,
                    underlying=pos.underlying,
                    contribution=delta_contrib,
                    contribution_pct=pct,
                )
            )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions[:top_n]

    def top_vega_contributors(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        snapshot: RiskSnapshot,
        top_n: int = 5,
    ) -> List[Contributor]:
        """Identify top vega contributors."""
        contributions = []

        for pos in positions:
            md = market_data.get(pos.symbol)
            if not md or not md.has_greeks():
                continue

            vega_contrib = (md.vega or 0.0) * pos.quantity * pos.multiplier
            pct = (abs(vega_contrib) / abs(snapshot.portfolio_vega) * 100) if snapshot.portfolio_vega != 0 else 0

            contributions.append(
                Contributor(
                    symbol=pos.symbol,
                    underlying=pos.underlying,
                    contribution=vega_contrib,
                    contribution_pct=pct,
                )
            )

        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions[:top_n]

    def top_notional_contributors(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        snapshot: RiskSnapshot,
        top_n: int = 5,
    ) -> List[Contributor]:
        """Identify top notional exposure contributors."""
        contributions = []

        for pos in positions:
            md = market_data.get(pos.symbol)
            if not md:
                continue

            mark = md.effective_mid()
            if mark is None:
                continue

            notional = mark * pos.quantity * pos.multiplier
            pct = (abs(notional) / snapshot.total_gross_notional * 100) if snapshot.total_gross_notional != 0 else 0

            contributions.append(
                Contributor(
                    symbol=pos.symbol,
                    underlying=pos.underlying,
                    contribution=notional,
                    contribution_pct=pct,
                )
            )

        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions[:top_n]
