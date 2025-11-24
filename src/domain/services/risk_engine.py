"""
Risk Engine - Core portfolio risk calculation.

Aggregates positions with market data to compute:
- Unrealized P&L and daily P&L
- Portfolio Greeks (delta, gamma, vega, theta)
- Notional exposure and concentration
- Expiry bucket analysis
"""

from __future__ import annotations
from typing import Dict, List, Any
from datetime import date
from ...models.position import Position
from ...models.market_data import MarketData
from ...models.account import AccountInfo
from ...models.risk_snapshot import RiskSnapshot


class RiskEngine:
    """
    Core risk calculation engine.

    Builds RiskSnapshot from positions, market data, and account info.
    Uses IBKR Greeks exclusively (no BSM fallback in MVP).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk engine with configuration.

        Args:
            config: Risk configuration dict (from ConfigManager).
        """
        self.config = config

    def build_snapshot(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        account_info: AccountInfo,
    ) -> RiskSnapshot:
        """
        Build complete risk snapshot from current state.

        Args:
            positions: List of all positions.
            market_data: Dict mapping symbol -> MarketData.
            account_info: Current account information.

        Returns:
            RiskSnapshot with aggregated metrics.

        Performance target: <100ms for <100 positions.
        """
        snapshot = RiskSnapshot()
        snapshot.total_positions = len(positions)

        # Initialize expiry buckets
        expiry_buckets = {
            "0DTE": self._empty_bucket(),
            "1_7D": self._empty_bucket(),
            "8_30D": self._empty_bucket(),
            "31_90D": self._empty_bucket(),
            "90D_PLUS": self._empty_bucket(),
            "NO_EXPIRY": self._empty_bucket(),
        }

        # Aggregate by underlying
        by_underlying: Dict[str, Dict[str, float]] = {}

        for pos in positions:
            md = market_data.get(pos.symbol)

            # Track missing market data
            if md is None:
                snapshot.positions_with_missing_md += 1
                continue

            mark = md.effective_mid()
            if mark is None:
                snapshot.positions_with_missing_md += 1
                continue

            # Calculate position notional
            notional = mark * pos.quantity * pos.multiplier

            # Aggregate Greeks (use IBKR only, skip if missing)
            if md.has_greeks():
                delta_contribution = (md.delta or 0.0) * pos.quantity * pos.multiplier
                gamma_contribution = (md.gamma or 0.0) * pos.quantity * pos.multiplier
                vega_contribution = (md.vega or 0.0) * pos.quantity * pos.multiplier
                theta_contribution = (md.theta or 0.0) * pos.quantity * pos.multiplier
            else:
                # For stocks without Greeks, delta = 1.0
                if pos.asset_type.value == "STOCK":
                    delta_contribution = 1.0 * pos.quantity * pos.multiplier
                    gamma_contribution = 0.0
                    vega_contribution = 0.0
                    theta_contribution = 0.0
                else:
                    snapshot.missing_greeks_count += 1
                    continue

            # Update portfolio Greeks
            snapshot.portfolio_delta += delta_contribution
            snapshot.portfolio_gamma += gamma_contribution
            snapshot.portfolio_vega += vega_contribution
            snapshot.portfolio_theta += theta_contribution

            # Update notional
            snapshot.total_gross_notional += abs(notional)
            snapshot.total_net_notional += notional

            # Group by underlying
            if pos.underlying not in by_underlying:
                by_underlying[pos.underlying] = {
                    "notional": 0.0,
                    "delta": 0.0,
                }
            by_underlying[pos.underlying]["notional"] += notional
            by_underlying[pos.underlying]["delta"] += delta_contribution

            # Expiry bucket classification
            bucket_key = pos.expiry_bucket()
            bucket = expiry_buckets[bucket_key]
            bucket["count"] += 1
            bucket["notional"] += notional
            bucket["delta"] += delta_contribution
            bucket["gamma"] += gamma_contribution
            bucket["vega"] += vega_contribution
            bucket["theta"] += theta_contribution

            # Near-term Greeks concentration
            dte = pos.days_to_expiry()
            if dte is not None:
                if dte <= 7:
                    # Gamma notional = gamma * mark^2 * 0.01 * qty * mult
                    gamma_notional = (md.gamma or 0.0) * (mark ** 2) * 0.01 * pos.quantity * pos.multiplier
                    snapshot.gamma_notional_near_term += abs(gamma_notional)
                if dte <= 30:
                    vega_notional = (md.vega or 0.0) * pos.quantity * pos.multiplier
                    snapshot.vega_notional_near_term += abs(vega_notional)

            # P&L calculation
            cost_basis = pos.avg_price * pos.quantity * pos.multiplier
            unrealized = (mark - pos.avg_price) * pos.quantity * pos.multiplier
            snapshot.total_unrealized_pnl += unrealized

            if md.yesterday_close:
                daily_pnl = (mark - md.yesterday_close) * pos.quantity * pos.multiplier
                snapshot.total_daily_pnl += daily_pnl

        # Finalize expiry buckets
        snapshot.expiry_buckets = expiry_buckets

        # Finalize by-underlying
        snapshot.delta_by_underlying = {k: v["delta"] for k, v in by_underlying.items()}
        snapshot.notional_by_underlying = {k: v["notional"] for k, v in by_underlying.items()}

        # Concentration metrics
        if by_underlying:
            max_underlying = max(by_underlying.items(), key=lambda x: abs(x[1]["notional"]))
            snapshot.max_underlying_symbol = max_underlying[0]
            snapshot.max_underlying_notional = abs(max_underlying[1]["notional"])
            if snapshot.total_gross_notional > 0:
                snapshot.concentration_pct = (
                    snapshot.max_underlying_notional / snapshot.total_gross_notional
                )

        # Account metrics
        snapshot.margin_utilization = account_info.margin_utilization()
        snapshot.buying_power = account_info.buying_power

        return snapshot

    def _empty_bucket(self) -> Dict[str, Any]:
        """Create empty expiry bucket structure."""
        return {
            "count": 0,
            "notional": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
        }
