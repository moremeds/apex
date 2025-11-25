"""
Risk Engine - Core portfolio risk calculation with performance optimizations.

Aggregates positions with market data to compute:
- Unrealized P&L and daily P&L
- Portfolio Greeks (delta, gamma, vega, theta)
- Notional exposure and concentration
- Expiry bucket analysis

Performance features:
- Concurrent processing for large portfolios (>50 positions)
- Works with Greeks caching in MarketDataStore
"""

from __future__ import annotations
from typing import Dict, List, Any
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...models.position import Position
from ...models.market_data import MarketData
from ...models.account import AccountInfo
from ...models.risk_snapshot import RiskSnapshot, PositionRisk
from ...utils.market_hours import MarketHours


class RiskEngine:
    """
    Core risk calculation engine.

    Builds RiskSnapshot from positions, market data, and account info.
    Uses IBKR Greeks exclusively (no BSM fallback in MVP).

    Performance:
    - Sequential processing for <50 positions (low overhead)
    - Parallel processing for >=50 positions (ThreadPoolExecutor)
    """

    def __init__(self, config: Dict[str, Any], parallel_threshold: int = 50, max_workers: int = 4):
        """
        Initialize risk engine with configuration.

        Args:
            config: Risk configuration dict (from ConfigManager).
            parallel_threshold: Number of positions to trigger parallel processing.
            max_workers: Maximum worker threads for parallel processing.
        """
        self.config = config
        self.parallel_threshold = parallel_threshold
        self.max_workers = max_workers

    def build_snapshot(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        account_info: AccountInfo,
    ) -> RiskSnapshot:
        """
        Build complete risk snapshot from current state.

        Uses parallel processing for portfolios >= parallel_threshold positions.

        Args:
            positions: List of all positions.
            market_data: Dict mapping symbol -> MarketData.
            account_info: Current account information.

        Returns:
            RiskSnapshot with aggregated metrics.

        Performance target: <100ms for <100 positions, <250ms for 100-250, <500ms for 250-500.
        """
        snapshot = RiskSnapshot()
        snapshot.total_positions = len(positions)

        # Choose processing strategy based on portfolio size
        if len(positions) < self.parallel_threshold:
            # Sequential processing for small portfolios (lower overhead)
            metrics_list = [self._calculate_position_metrics(pos, market_data) for pos in positions]
        else:
            # Parallel processing for large portfolios
            metrics_list = self._calculate_metrics_parallel(positions, market_data)

        # Aggregate metrics into snapshot
        self._aggregate_metrics(snapshot, metrics_list, account_info)

        return snapshot

    def _calculate_position_metrics(
        self, pos: Position, market_data: Dict[str, MarketData]
    ) -> PositionRisk | None:
        """
        Calculate metrics for a single position.

        Args:
            pos: Position to calculate.
            market_data: Market data dictionary.

        Returns:
            PositionRisk or None if missing data.
        """
        md = market_data.get(pos.symbol)

        # Track missing market data
        if md is None:
            return PositionRisk(
                symbol=pos.symbol,
                underlying=pos.underlying,
                expiry=pos.expiry,
                strike=pos.strike,
                right=pos.right,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                mark=None,
                market_value=0.0,
                notional=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                delta_contribution=0.0,
                gamma_contribution=0.0,
                vega_contribution=0.0,
                theta_contribution=0.0,
                delta_dollars=0.0,
                expiry_bucket=pos.expiry_bucket(),
                days_to_expiry=pos.days_to_expiry(),
                gamma_notional_near_term=0.0,
                vega_notional_near_term=0.0,
                has_missing_md=True,
                has_missing_greeks=False,
            )

        mark = md.effective_mid()
        if mark is None:
            return PositionRisk(
                symbol=pos.symbol,
                underlying=pos.underlying,
                expiry=pos.expiry,
                strike=pos.strike,
                right=pos.right,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                mark=None,
                market_value=0.0,
                notional=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                delta_contribution=0.0,
                gamma_contribution=0.0,
                vega_contribution=0.0,
                theta_contribution=0.0,
                delta_dollars=0.0,
                expiry_bucket=pos.expiry_bucket(),
                days_to_expiry=pos.days_to_expiry(),
                gamma_notional_near_term=0.0,
                vega_notional_near_term=0.0,
                has_missing_md=True,
                has_missing_greeks=False,
            )

        # Calculate position notional using live mark (no market hours adjustments)
        notional = mark * pos.quantity * pos.multiplier

        # Aggregate Greeks (use IBKR only, skip if missing)
        if md.has_greeks():
            delta_contribution = (md.delta or 0.0) * pos.quantity * pos.multiplier
            gamma_contribution = (md.gamma or 0.0) * pos.quantity * pos.multiplier
            vega_contribution = (md.vega or 0.0) * pos.quantity * pos.multiplier
            theta_contribution = (md.theta or 0.0) * pos.quantity * pos.multiplier
            has_missing_greeks = False
        else:
            # For stocks without Greeks, delta = 1.0
            if pos.asset_type.value == "STOCK":
                delta_contribution = 1.0 * pos.quantity * pos.multiplier
                gamma_contribution = 0.0
                vega_contribution = 0.0
                theta_contribution = 0.0
                has_missing_greeks = False
            else:
                return PositionRisk(
                    symbol=pos.symbol,
                    underlying=pos.underlying,
                    expiry=pos.expiry,
                    strike=pos.strike,
                    right=pos.right,
                    quantity=pos.quantity,
                    avg_price=pos.avg_price,
                    mark=mark,
                    market_value=0.0,
                    notional=notional,
                    unrealized_pnl=0.0,
                    daily_pnl=0.0,
                    delta_contribution=0.0,
                    gamma_contribution=0.0,
                    vega_contribution=0.0,
                    theta_contribution=0.0,
                    delta_dollars=0.0,
                    expiry_bucket=pos.expiry_bucket(),
                    days_to_expiry=pos.days_to_expiry(),
                    gamma_notional_near_term=0.0,
                    vega_notional_near_term=0.0,
                    has_missing_md=False,
                    has_missing_greeks=True,
                )

        # Near-term Greeks concentration
        dte = pos.days_to_expiry()
        gamma_notional_near_term = 0.0
        vega_notional_near_term = 0.0

        if dte is not None:
            if dte <= 7:
                # Gamma notional = gamma * mark^2 * 0.01 * qty * mult
                gamma_notional_near_term = abs((md.gamma or 0.0) * (mark ** 2) * 0.01 * pos.quantity * pos.multiplier)
            if dte <= 30:
                vega_notional_near_term = abs((md.vega or 0.0) * pos.quantity * pos.multiplier)

        # P&L calculation with market hours logic
        # Market status determines which price to use
        market_status = MarketHours.get_market_status()

        # Determine price to use for unrealized P&L
        if market_status == "OPEN":
            # Regular hours: use current mark
            pnl_price = mark
            calculate_daily_pnl = True
        elif market_status == "EXTENDED":
            # Extended hours: stocks use current price, options use yesterday close
            if pos.asset_type.value == "STOCK":
                pnl_price = mark  # Stocks trade in extended hours
            else:
                pnl_price = md.yesterday_close if md.yesterday_close else mark  # Options don't trade extended hours
            calculate_daily_pnl = False  # Skip daily P&L when market not in regular hours
        else:
            # Market closed: use yesterday close for all assets
            pnl_price = md.yesterday_close if md.yesterday_close else mark
            calculate_daily_pnl = False

        # Calculate market value using market-hours-aware price
        market_value = pnl_price * pos.quantity * pos.multiplier

        # Calculate unrealized P&L
        # For options: avg_price is per contract, pnl_price is per share
        # For stocks: both are per share
        if pos.asset_type.value == "OPTION":
            current_value = pnl_price * pos.multiplier  # Convert to per-contract value
            unrealized = (current_value - pos.avg_price) * pos.quantity
        else:
            unrealized = (pnl_price - pos.avg_price) * pos.quantity * pos.multiplier

        # Daily P&L: only calculated during regular market hours
        daily_pnl = 0.0
        if calculate_daily_pnl and md.yesterday_close:
            daily_pnl = (mark - md.yesterday_close) * pos.quantity * pos.multiplier

        delta_dollars = delta_contribution * (mark if mark is not None else 0.0)

        return PositionRisk(
            symbol=pos.symbol,
            underlying=pos.underlying,
            expiry=pos.expiry,
            strike=pos.strike,
            right=pos.right,
            quantity=pos.quantity,
            avg_price=pos.avg_price,
            mark=mark,
            market_value=market_value,
            notional=notional,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            delta_contribution=delta_contribution,
            gamma_contribution=gamma_contribution,
            vega_contribution=vega_contribution,
            theta_contribution=theta_contribution,
            delta_dollars=delta_dollars,
            expiry_bucket=pos.expiry_bucket(),
            days_to_expiry=dte,
            gamma_notional_near_term=gamma_notional_near_term,
            vega_notional_near_term=vega_notional_near_term,
            has_missing_md=False,
            has_missing_greeks=has_missing_greeks,
        )

    def _calculate_metrics_parallel(
        self, positions: List[Position], market_data: Dict[str, MarketData]
    ) -> List[PositionRisk | None]:
        """
        Calculate position metrics in parallel using ThreadPoolExecutor.

        Args:
            positions: List of positions to calculate.
            market_data: Market data dictionary.

        Returns:
            List of PositionRisk.
        """
        metrics_list: List[PositionRisk | None] = [None] * len(positions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._calculate_position_metrics, pos, market_data): idx
                for idx, pos in enumerate(positions)
            }

            # Collect results as they complete, preserving input order
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    metrics_list[idx] = future.result()
                except Exception as e:
                    pos = positions[idx]
                    import logging
                    logging.error(f"Error calculating metrics for {pos.symbol}: {e}")
                    metrics_list[idx] = None

        return metrics_list

    def _aggregate_metrics(
        self, snapshot: RiskSnapshot, metrics_list: List[PositionRisk | None], account_info: AccountInfo
    ) -> None:
        """
        Aggregate position metrics into snapshot.

        Args:
            snapshot: Snapshot to populate.
            metrics_list: List of position metrics.
            account_info: Account information.
        """
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

        # Persist per-position metrics for presentation
        snapshot.positions = [m for m in metrics_list if m is not None]

        for metrics in snapshot.positions:
            if metrics is None:
                continue

            # Track missing data
            if metrics.has_missing_md:
                snapshot.positions_with_missing_md += 1
                continue

            if metrics.has_missing_greeks:
                snapshot.missing_greeks_count += 1
                continue

            # Update portfolio Greeks
            snapshot.portfolio_delta += metrics.delta_contribution
            snapshot.portfolio_gamma += metrics.gamma_contribution
            snapshot.portfolio_vega += metrics.vega_contribution
            snapshot.portfolio_theta += metrics.theta_contribution

            # Update notional
            snapshot.total_gross_notional += abs(metrics.notional)
            snapshot.total_net_notional += metrics.notional

            # Group by underlying
            if metrics.underlying not in by_underlying:
                by_underlying[metrics.underlying] = {
                    "notional": 0.0,
                    "delta": 0.0,
                }
            by_underlying[metrics.underlying]["notional"] += metrics.notional
            by_underlying[metrics.underlying]["delta"] += metrics.delta_contribution

            # Expiry bucket classification
            bucket = expiry_buckets[metrics.expiry_bucket]
            bucket["count"] += 1
            bucket["notional"] += metrics.notional
            bucket["delta"] += metrics.delta_contribution
            bucket["gamma"] += metrics.gamma_contribution
            bucket["vega"] += metrics.vega_contribution
            bucket["theta"] += metrics.theta_contribution

            # Near-term Greeks concentration
            snapshot.gamma_notional_near_term += metrics.gamma_notional_near_term
            snapshot.vega_notional_near_term += metrics.vega_notional_near_term

            # P&L
            snapshot.total_unrealized_pnl += metrics.unrealized_pnl
            snapshot.total_daily_pnl += metrics.daily_pnl

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
