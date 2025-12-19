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
from typing import Dict, List, Any, Tuple, Optional, Set, TYPE_CHECKING
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
import time

from src.utils.logging_setup import get_logger
from src.models.position import Position
from src.models.market_data import MarketData
from src.models.account import AccountInfo
from src.models.risk_snapshot import RiskSnapshot
from src.models.position_risk import PositionRisk
from src.utils.market_hours import MarketHours
from src.utils.timezone import now_utc

if TYPE_CHECKING:
    from src.domain.interfaces.event_bus import EventBus
    from src.infrastructure.adapters.yahoo import YahooFinanceAdapter
    from src.infrastructure.observability.risk_metrics import RiskMetrics

logger = get_logger(__name__)

# Defaults for near-term Greeks concentration thresholds (overridable via config)
DEFAULT_NEAR_TERM_GAMMA_DTE = 7  # Days to expiry threshold for near-term gamma
DEFAULT_NEAR_TERM_VEGA_DTE = 30  # Days to expiry threshold for near-term vega
GAMMA_NOTIONAL_FACTOR = 0.01  # Multiplier for gamma notional calculation


@dataclass
class PositionMetrics:
    """Metrics calculated for a single position."""
    symbol: str
    underlying: str
    notional: float
    unrealized_pnl: float
    daily_pnl: float
    delta_contribution: float
    gamma_contribution: float
    vega_contribution: float
    theta_contribution: float
    expiry_bucket: str
    days_to_expiry: int | None
    gamma_notional_near_term: float
    vega_notional_near_term: float
    has_missing_md: bool
    has_missing_greeks: bool


class RiskEngine:
    """
    Core risk calculation engine.

    Builds RiskSnapshot from positions, market data, and account info.
    Uses IBKR Greeks exclusively (no BSM fallback in MVP).

    Performance:
    - Sequential processing for <50 positions (low overhead)
    - Parallel processing for >=50 positions (ThreadPoolExecutor)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        parallel_threshold: int = 50,
        max_workers: int = 4,
        yahoo_adapter: Optional["YahooFinanceAdapter"] = None,
        risk_metrics: Optional["RiskMetrics"] = None,
    ):
        """
        Initialize risk engine with configuration.

        Args:
            config: Risk configuration dict (from ConfigManager).
            parallel_threshold: Number of positions to trigger parallel processing.
            max_workers: Maximum worker threads for parallel processing.
            yahoo_adapter: Optional YahooFinanceAdapter for dynamic beta lookup.
                          If None, falls back to static config lookup.
            risk_metrics: Optional RiskMetrics for Prometheus observability.
                         If provided, metrics are recorded during snapshot builds.
        """
        self.config = config
        self.parallel_threshold = parallel_threshold
        self.max_workers = max_workers
        self._yahoo_adapter = yahoo_adapter
        self._risk_metrics = risk_metrics

        # Extract near-term DTE thresholds from config (with defaults)
        risk_engine_cfg = config.get("risk_engine", {})
        self.near_term_gamma_dte = risk_engine_cfg.get("near_term_gamma_dte", DEFAULT_NEAR_TERM_GAMMA_DTE)
        self.near_term_vega_dte = risk_engine_cfg.get("near_term_vega_dte", DEFAULT_NEAR_TERM_VEGA_DTE)

        # Persistent executor for parallel position metrics calculation
        # Created once and reused across snapshots (not per-call)
        #
        # GIL Analysis (HYG-004): ThreadPoolExecutor is correct here, not ProcessPoolExecutor.
        # Reasons:
        # 1. Calculations are simple arithmetic (notional, P&L, Greeks) - not numpy-heavy
        # 2. ProcessPoolExecutor IPC overhead (pickling Position/MarketData) exceeds GIL cost
        # 3. Some I/O involved (Yahoo adapter) where ThreadPool releases GIL
        # 4. Measured latency is well within performance targets (<100ms for <100 positions)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="risk_")

        # Event-driven state tracking
        self._lock = Lock()
        self._needs_rebuild = False
        # Tracks which underlyings have changed since last rebuild (for future incremental rebuild)
        # TODO: Phase 2 - Use _dirty_underlyings to scope rebuilds:
        #   1. Cache previous PositionRisk objects by symbol
        #   2. Only recalculate positions with dirty underlyings
        #   3. Merge recalculated with cached results
        #   Currently, build_snapshot() always recomputes all positions.
        self._dirty_underlyings: Set[str] = set()

    def build_snapshot(
        self,
        positions: List[Position],
        market_data: Dict[str, MarketData],
        account_info: AccountInfo,
        ib_account: Optional[AccountInfo] = None,
        futu_account: Optional[AccountInfo] = None,
    ) -> RiskSnapshot:
        """
        Build complete risk snapshot from current state.

        Uses parallel processing for portfolios >= parallel_threshold positions.

        Args:
            positions: List of all positions.
            market_data: Dict mapping symbol -> MarketData.
            account_info: Aggregated account information.
            ib_account: Optional IB-specific account info for separate display.
            futu_account: Optional Futu-specific account info for separate display.

        Returns:
            RiskSnapshot with aggregated metrics.

        Performance target: <100ms for <100 positions, <250ms for 100-250, <500ms for 250-500.
        """
        start_time = time.perf_counter()

        snapshot = RiskSnapshot()
        snapshot.total_positions = len(positions)

        # Store per-broker account info for separate display
        if ib_account:
            snapshot.ib_net_liquidation = ib_account.net_liquidation
            snapshot.ib_buying_power = ib_account.buying_power
        if futu_account:
            snapshot.futu_net_liquidation = futu_account.net_liquidation
            snapshot.futu_buying_power = futu_account.buying_power
        snapshot.total_net_liquidation = account_info.net_liquidation

        # Prefetch betas for all unique underlyings (batch fetch to avoid N+1 API calls)
        beta_cache: Dict[str, float] = {}
        if self._yahoo_adapter is not None:
            unique_underlyings = list(set(pos.underlying for pos in positions))
            beta_cache = self._yahoo_adapter.get_betas(unique_underlyings)

        # Cache market status once per snapshot (avoid repeated calls per-position)
        market_status = MarketHours.get_market_status()

        # Choose processing strategy based on portfolio size
        if len(positions) < self.parallel_threshold:
            # Sequential processing for small portfolios (lower overhead)
            metrics_list = [self._calculate_position_metrics(pos, market_data, market_status) for pos in positions]
        else:
            # Parallel processing for large portfolios
            metrics_list = self._calculate_metrics_parallel(positions, market_data, market_status)

        # Create PositionRisk objects for each position (single source of truth)
        position_risks = []
        for pos, metrics in zip(positions, metrics_list):
            if metrics is not None:
                pos_risk = self._create_position_risk(pos, market_data.get(pos.symbol), metrics, beta_cache)
                position_risks.append(pos_risk)

        snapshot.position_risks = position_risks

        # Aggregate metrics into snapshot
        self._aggregate_metrics(snapshot, metrics_list, account_info)

        # Record observability metrics
        if self._risk_metrics:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._risk_metrics.record_snapshot_build_duration(duration_ms)
            self._risk_metrics.record_snapshot(snapshot)
            logger.debug(f"Snapshot build completed in {duration_ms:.2f}ms")

        return snapshot

    def _calculate_position_metrics(
        self, pos: Position, market_data: Dict[str, MarketData], market_status: str
    ) -> PositionMetrics | None:
        """
        Calculate metrics for a single position.

        When market data is missing or stale, uses fallback strategies to preserve
        risk visibility rather than zeroing out exposure (which hides risk).

        Fallback order:
        1. Live mid/last price from primary provider
        2. Yesterday's close from market data
        3. Yesterday's close from Yahoo Finance (stocks only)
        4. Average cost basis (last resort - preserves notional visibility)

        Args:
            pos: Position to calculate.
            market_data: Market data dictionary.

        Returns:
            PositionMetrics with calculated values. has_missing_md=True when using fallback.
        """
        md = market_data.get(pos.symbol)
        has_missing_md = False
        has_missing_greeks = False

        # Determine mark price with fallback chain
        mark = None
        fallback_source = None

        if md is not None:
            # Try live price first
            if md.has_live_price():
                mark = md.effective_mid()
            # Fall back to yesterday's close from market data
            elif md.yesterday_close is not None and md.yesterday_close > 0:
                mark = md.yesterday_close
                has_missing_md = True
                fallback_source = "md_yesterday_close"

        # If still no mark, try Yahoo Finance for yesterday's close (stocks only)
        if mark is None and self._yahoo_adapter is not None:
            # For stocks, try to get yesterday's close from Yahoo
            if pos.asset_type.value == "STOCK":
                yahoo_md = self._yahoo_adapter.get_latest(pos.symbol)
                if yahoo_md and yahoo_md.yesterday_close and yahoo_md.yesterday_close > 0:
                    mark = yahoo_md.yesterday_close
                    has_missing_md = True
                    fallback_source = "yahoo_yesterday_close"

        # Last resort: use average cost basis to preserve notional visibility
        # This ensures risk doesn't silently disappear when data is unavailable
        if mark is None and pos.avg_price and pos.avg_price > 0:
            mark = pos.avg_price
            has_missing_md = True
            fallback_source = "avg_cost_basis"

        # If still no mark, return with explicit missing data flag
        # but calculate what we can (expiry bucket, DTE)
        if mark is None:
            return PositionMetrics(
                symbol=pos.symbol,
                underlying=pos.underlying,
                notional=0.0,
                unrealized_pnl=0.0,
                daily_pnl=0.0,
                delta_contribution=0.0,
                gamma_contribution=0.0,
                vega_contribution=0.0,
                theta_contribution=0.0,
                expiry_bucket=pos.expiry_bucket(),
                days_to_expiry=pos.days_to_expiry(),
                gamma_notional_near_term=0.0,
                vega_notional_near_term=0.0,
                has_missing_md=True,
                has_missing_greeks=True,
            )

        if fallback_source:
            logger.debug(f"Using fallback price for {pos.symbol}: {fallback_source}=${mark:.2f}")

        # Calculate position notional
        notional = mark * pos.quantity * pos.multiplier

        # Aggregate Greeks (use IBKR only, but still calculate P&L even if Greeks missing)
        qty_mult = pos.quantity * pos.multiplier
        if md is not None and md.has_greeks():
            delta_contribution = (md.delta or 0.0) * qty_mult
            gamma_contribution = (md.gamma or 0.0) * qty_mult
            vega_contribution = (md.vega or 0.0) * qty_mult
            theta_contribution = (md.theta or 0.0) * qty_mult
        elif pos.asset_type.value == "STOCK":
            # Stocks without Greeks: delta = 1.0
            delta_contribution = qty_mult
            gamma_contribution = 0.0
            vega_contribution = 0.0
            theta_contribution = 0.0
        else:
            # Options without Greeks during off-hours: flag missing
            delta_contribution = 0.0
            gamma_contribution = 0.0
            vega_contribution = 0.0
            theta_contribution = 0.0
            has_missing_greeks = True

        # Near-term Greeks concentration
        dte = pos.days_to_expiry()
        gamma_notional_near_term = 0.0
        vega_notional_near_term = 0.0

        if dte is not None and md is not None:
            if dte <= self.near_term_gamma_dte:
                # Gamma notional = gamma * mark^2 * factor * qty * mult
                gamma_notional_near_term = abs((md.gamma or 0.0) * (mark ** 2) * GAMMA_NOTIONAL_FACTOR * pos.quantity * pos.multiplier)
            if dte <= self.near_term_vega_dte:
                vega_notional_near_term = abs((md.vega or 0.0) * pos.quantity * pos.multiplier)

        # P&L calculation with market hours logic
        yesterday_close = md.yesterday_close if md is not None else None
        is_stock = pos.asset_type.value == "STOCK"

        # Use current mark when market is open, or extended hours for stocks
        # Otherwise use yesterday's close (options don't trade extended hours)
        use_live_price = market_status == "OPEN" or (market_status == "EXTENDED" and is_stock)
        pnl_price = mark if use_live_price else (yesterday_close or mark)

        unrealized = (pnl_price - pos.avg_price) * qty_mult

        # Daily P&L: change from yesterday's close
        daily_pnl = (mark - yesterday_close) * qty_mult if yesterday_close and yesterday_close > 0 else 0.0

        return PositionMetrics(
            symbol=pos.symbol,
            underlying=pos.underlying,
            notional=notional,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            delta_contribution=delta_contribution,
            gamma_contribution=gamma_contribution,
            vega_contribution=vega_contribution,
            theta_contribution=theta_contribution,
            expiry_bucket=pos.expiry_bucket(),
            days_to_expiry=dte,
            gamma_notional_near_term=gamma_notional_near_term,
            vega_notional_near_term=vega_notional_near_term,
            has_missing_md=has_missing_md,
            has_missing_greeks=has_missing_greeks,
        )

    def _calculate_metrics_parallel(
        self, positions: List[Position], market_data: Dict[str, MarketData], market_status: str
    ) -> List[PositionMetrics | None]:
        """
        Calculate position metrics in parallel using persistent ThreadPoolExecutor.

        Args:
            positions: List of positions to calculate.
            market_data: Market data dictionary.
            market_status: Market status string (cached once per snapshot).

        Returns:
            List of PositionMetrics.
        """
        metrics_list: List[PositionMetrics | None] = [None] * len(positions)

        # Use persistent executor (created in __init__, not per-call)
        # Submit all tasks and track original position index to preserve order
        future_to_idx = {
            self._executor.submit(self._calculate_position_metrics, pos, market_data, market_status): idx
            for idx, pos in enumerate(positions)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                metrics = future.result()
                metrics_list[idx] = metrics
            except Exception as e:
                pos = positions[idx]
                logger.error(f"Error calculating metrics for {pos.symbol}: {e}")
                metrics_list[idx] = None

        return metrics_list

    def _create_position_risk(
        self,
        pos: Position,
        md: MarketData | None,
        metrics: PositionMetrics,
        beta_cache: Dict[str, float] | None = None,
    ) -> PositionRisk:
        """
        Create PositionRisk object from position, market data, and calculated metrics.

        This is the SINGLE SOURCE OF TRUTH for all position-level calculations.
        The dashboard should use these pre-calculated values without recalculation.

        Args:
            pos: The position.
            md: Market data for the position (may be None).
            metrics: Calculated position metrics.
            beta_cache: Pre-fetched beta values by underlying symbol.

        Returns:
            PositionRisk object with all calculated fields.
        """
        # Extract market data fields
        mark_price = None
        iv = None
        has_market_data = md is not None and not metrics.has_missing_md
        has_greeks = False
        is_stale = False
        is_using_close = False  # Track if using yesterday's close instead of live data

        if md is not None:
            # Check if we have valid live price data
            if md.has_live_price():
                mark_price = md.effective_mid()
            elif md.yesterday_close is not None and md.yesterday_close > 0:
                # No valid live data, fall back to yesterday's close
                mark_price = md.yesterday_close
                is_using_close = True

            iv = md.iv
            has_greeks = md.has_greeks()
            is_stale = md.is_stale()

        # Calculate delta dollars (delta * underlying_price * quantity * multiplier)
        delta_dollars = 0.0
        if md is not None:
            # Get per-share delta
            if md.delta is not None:
                per_share_delta = md.delta
            elif pos.asset_type.value == "STOCK":
                per_share_delta = 1.0
            else:
                per_share_delta = 0.0

            # Determine price to use for delta dollars calculation
            if pos.asset_type.value == "OPTION":
                # For options, use underlying price (NOT option price)
                # Delta dollars = exposure to underlying price movement
                price_for_delta = md.underlying_price if md.underlying_price else mark_price
            else:
                # For stocks, use stock price
                price_for_delta = mark_price

            if price_for_delta:
                delta_dollars = per_share_delta * price_for_delta * pos.quantity * pos.multiplier

        # Get beta from pre-fetched cache (default to 1.0 if not available)
        if beta_cache is not None:
            beta = beta_cache.get(pos.underlying, 1.0)
        else:
            beta = 1.0  # Default beta when no cache available

        # Calculate beta-adjusted delta (SPY-equivalent exposure)
        # Use beta or 1.0 to handle beta=0 case (beta=0 means no correlation, not no exposure)
        beta_adjusted_delta = metrics.delta_contribution * (beta if beta is not None else 1.0)

        # Create PositionRisk object
        return PositionRisk(
            position=pos,
            mark_price=mark_price,
            iv=iv,
            market_value=metrics.notional,
            unrealized_pnl=metrics.unrealized_pnl,
            daily_pnl=metrics.daily_pnl,
            delta=metrics.delta_contribution,
            gamma=metrics.gamma_contribution,
            vega=metrics.vega_contribution,
            theta=metrics.theta_contribution,
            beta=beta,
            delta_dollars=delta_dollars,
            notional=metrics.notional,
            beta_adjusted_delta=beta_adjusted_delta,
            has_market_data=has_market_data and not metrics.has_missing_md,
            has_greeks=has_greeks and not metrics.has_missing_greeks,
            is_stale=is_stale,
            is_using_close=is_using_close,
            calculated_at=now_utc(),
        )

    def _aggregate_metrics(
        self, snapshot: RiskSnapshot, metrics_list: List[PositionMetrics | None], account_info: AccountInfo
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

        for metrics in metrics_list:
            if metrics is None:
                continue

            # Track missing data
            if metrics.has_missing_md:
                snapshot.positions_with_missing_md += 1
                continue

            if metrics.has_missing_greeks:
                snapshot.missing_greeks_count += 1

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
        # Note: snapshot is mutated in place, no return needed

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

    # ========== Event-Driven Methods ==========

    def subscribe_to_events(self, event_bus: "EventBus") -> None:
        """Subscribe to data change events."""
        from src.domain.interfaces.event_bus import EventType

        event_bus.subscribe(EventType.POSITIONS_BATCH, self._on_data_changed)
        event_bus.subscribe(EventType.POSITION_UPDATED, self._on_data_changed)
        event_bus.subscribe(EventType.MARKET_DATA_BATCH, self._on_data_changed)
        event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_tick)
        event_bus.subscribe(EventType.ACCOUNT_UPDATED, self._on_data_changed)
        event_bus.subscribe(EventType.TIMER_TICK, self._on_timer)
        logger.debug("RiskEngine subscribed to events")

    def _on_data_changed(self, payload: dict) -> None:
        """
        Handle data change events (positions, market data batch, account).

        Marks that full snapshot rebuild is needed.

        Args:
            payload: Event payload.
        """
        with self._lock:
            self._needs_rebuild = True
        logger.debug(f"RiskEngine marked dirty: {payload.get('source', 'unknown')} data changed")

    def _on_market_tick(self, payload: dict) -> None:
        """
        Handle single market data tick (from streaming).

        Only marks the affected underlying as dirty for incremental rebuild.

        Args:
            payload: Event payload with 'symbol' and 'data'.
        """
        symbol = payload.get("symbol", "")
        data = payload.get("data")

        # Get underlying from market data or symbol
        underlying = symbol
        if data and hasattr(data, "underlying") and data.underlying:
            underlying = data.underlying
        elif " " in symbol:  # Option symbol format: "AAPL  240119C..."
            underlying = symbol.split()[0]

        with self._lock:
            self._dirty_underlyings.add(underlying)

    def _on_timer(self, payload: dict) -> None:
        """
        Handle timer tick for periodic reconciliation.

        Forces rebuild if any dirty state exists.

        Args:
            payload: Event payload.
        """
        with self._lock:
            if self._dirty_underlyings:
                self._needs_rebuild = True
        logger.debug("RiskEngine timer tick")

    def mark_dirty(self, underlying: str | None = None) -> None:
        """
        Mark the risk engine as needing a rebuild.

        Args:
            underlying: Optional specific underlying to mark dirty.
                       If None, marks all as needing rebuild.
        """
        with self._lock:
            if underlying:
                self._dirty_underlyings.add(underlying)
            else:
                self._needs_rebuild = True

    def needs_rebuild(self) -> bool:
        """
        Check if snapshot needs rebuild.

        Returns:
            True if rebuild is needed.
        """
        with self._lock:
            return self._needs_rebuild or bool(self._dirty_underlyings)

    def clear_dirty_state(self) -> None:
        """Clear dirty state after rebuild."""
        with self._lock:
            self._needs_rebuild = False
            self._dirty_underlyings.clear()

    def set_risk_metrics(self, risk_metrics: "RiskMetrics") -> None:
        """
        Set or replace the risk metrics instance for observability.

        Useful for dependency injection after construction.

        Args:
            risk_metrics: RiskMetrics instance for Prometheus export.
        """
        self._risk_metrics = risk_metrics
        logger.info("RiskMetrics attached to RiskEngine")

    def close(self) -> None:
        """Shutdown the persistent executor."""
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            logger.info("RiskEngine executor shutdown complete")
