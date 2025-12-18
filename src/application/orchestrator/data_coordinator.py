"""
DataCoordinator - Handles position/market data fetching and reconciliation.

Responsibilities:
- Fetching positions from all brokers
- Reconciling positions across sources
- Subscribing to market data
- Managing account info with TTL caching
"""

from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...utils.perf_logger import log_timing_async
from ...models.position import Position
from ...models.account import AccountInfo
from ...infrastructure.monitoring import HealthStatus

if TYPE_CHECKING:
    from ...infrastructure.adapters.broker_manager import BrokerManager
    from ...infrastructure.adapters.market_data_manager import MarketDataManager
    from ...infrastructure.stores import PositionStore, MarketDataStore, AccountStore
    from ...domain.services.pos_reconciler import Reconciler
    from ...domain.services.mdqc import MDQC
    from ...infrastructure.monitoring import HealthMonitor

logger = get_logger(__name__)


class DataCoordinator:
    """
    Coordinates data fetching and reconciliation across all sources.

    Handles:
    - Position fetching from brokers
    - Position reconciliation
    - Market data subscription management
    - Account info with TTL caching
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        market_data_manager: MarketDataManager,
        position_store: PositionStore,
        market_data_store: MarketDataStore,
        account_store: AccountStore,
        reconciler: Reconciler,
        mdqc: MDQC,
        health_monitor: HealthMonitor,
        config: Dict[str, Any],
    ):
        self.broker_manager = broker_manager
        self.market_data_manager = market_data_manager
        self.position_store = position_store
        self.market_data_store = market_data_store
        self.account_store = account_store
        self.reconciler = reconciler
        self.mdqc = mdqc
        self.health_monitor = health_monitor

        # Account TTL caching
        self._account_ttl_sec: float = config.get("account_ttl_sec", 30.0)
        self._last_account_fetch: Optional[datetime] = None
        self._cached_account_info: Optional[AccountInfo] = None

        # Background task tracking
        self._background_tasks: set[asyncio.Task] = set()

    @log_timing_async("fetch_and_reconcile", warn_threshold_ms=500)
    async def fetch_and_reconcile(
        self,
        on_dirty_callback: Optional[callable] = None,
    ) -> tuple[List[Position], AccountInfo]:
        """
        Fetch positions and accounts from all sources and reconcile.

        Args:
            on_dirty_callback: Called when data changes (to mark risk engine dirty)

        Returns:
            Tuple of (merged_positions, account_info)
        """
        # Parallel fetch of positions and accounts
        positions_task = asyncio.create_task(self.broker_manager.fetch_positions_by_broker())
        accounts_task = asyncio.create_task(self.broker_manager.fetch_account_info_by_broker())

        positions_by_broker, accounts_by_broker = await asyncio.gather(
            positions_task, accounts_task
        )

        # Reconcile positions
        merged_positions = self._reconcile_positions(positions_by_broker)

        # Aggregate account info
        account_info = self._aggregate_account_info(accounts_by_broker)

        # Update stores
        self.position_store.upsert_positions(merged_positions)
        self.account_store.update(account_info)

        # Subscribe to market data for new symbols
        await self._subscribe_market_data(merged_positions)

        # Validate market data quality
        market_data = self.market_data_store.get_all()
        self.mdqc.validate_all(market_data)

        # Notify dirty callback if provided
        if on_dirty_callback:
            on_dirty_callback()

        return merged_positions, account_info

    def _reconcile_positions(
        self, positions_by_broker: Dict[str, List[Position]]
    ) -> List[Position]:
        """Reconcile positions from all brokers."""
        # Flatten all positions
        all_positions = []
        for broker_name, positions in positions_by_broker.items():
            all_positions.extend(positions)

        if not all_positions:
            return []

        # Use reconciler to merge positions (not reconcile which returns issues)
        merged = self.reconciler.merge_positions(
            ib_positions=positions_by_broker.get("ib", []),
            manual_positions=[],
            futu_positions=positions_by_broker.get("futu", []),
        )

        return merged

    def _aggregate_account_info(
        self, accounts_by_broker: Dict[str, AccountInfo]
    ) -> AccountInfo:
        """Aggregate account info from all brokers."""
        if not accounts_by_broker:
            return AccountInfo(
                net_liquidation=0.0,
                total_cash=0.0,
                buying_power=0.0,
                margin_used=0.0,
                margin_available=0.0,
                maintenance_margin=0.0,
                init_margin_req=0.0,
                excess_liquidity=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
                account_id="AGGREGATED",
            )

        # Start with zero aggregates
        aggregated = AccountInfo(
            net_liquidation=0.0,
            total_cash=0.0,
            buying_power=0.0,
            margin_used=0.0,
            margin_available=0.0,
            maintenance_margin=0.0,
            init_margin_req=0.0,
            excess_liquidity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(),
            account_id="AGGREGATED",
        )

        for broker_name, account in accounts_by_broker.items():
            aggregated = AccountInfo(
                net_liquidation=aggregated.net_liquidation + account.net_liquidation,
                total_cash=aggregated.total_cash + account.total_cash,
                buying_power=aggregated.buying_power + account.buying_power,
                margin_used=aggregated.margin_used + account.margin_used,
                margin_available=aggregated.margin_available + account.margin_available,
                maintenance_margin=aggregated.maintenance_margin + account.maintenance_margin,
                init_margin_req=aggregated.init_margin_req + account.init_margin_req,
                excess_liquidity=aggregated.excess_liquidity + account.excess_liquidity,
                realized_pnl=aggregated.realized_pnl + account.realized_pnl,
                unrealized_pnl=aggregated.unrealized_pnl + account.unrealized_pnl,
                timestamp=datetime.now(),
                account_id="AGGREGATED",
            )

        return aggregated

    async def _subscribe_market_data(self, positions: List[Position]) -> None:
        """Subscribe to market data for positions."""
        all_symbols = [p.symbol for p in positions]
        new_symbols = [s for s in all_symbols if not self.market_data_store.has_fresh_data(s)]

        if new_symbols and self.market_data_manager.is_connected():
            positions_to_subscribe = [p for p in positions if p.symbol in new_symbols]
            # Track background task
            task = asyncio.create_task(
                self._subscribe_market_data_background(positions_to_subscribe, len(all_symbols))
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        elif not self.market_data_manager.is_connected():
            logger.warning("No market data providers connected")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.UNHEALTHY,
                "No market data providers connected"
            )

    async def _subscribe_market_data_background(
        self, positions: List[Position], total_symbols: int
    ) -> None:
        """Subscribe to market data in background."""
        try:
            symbols = [p.symbol for p in positions]
            await self.market_data_manager.subscribe(symbols)

            # Prime the market data cache + create provider-side streaming subscriptions.
            # IB's MarketDataFetcher uses reqMktData() during fetch_market_data() to create
            # live ticker subscriptions; streaming updates then flow via callbacks.
            market_data_list = await self.market_data_manager.fetch_market_data(positions)
            if market_data_list:
                # Ensure store is populated immediately (do not rely on EventBus wiring order).
                self.market_data_store.upsert(market_data_list)

            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.HEALTHY if market_data_list else HealthStatus.DEGRADED,
                (
                    f"Streaming {total_symbols} symbols (primed {len(market_data_list)})"
                    if market_data_list
                    else f"Subscribed {total_symbols} symbols (no data yet)"
                ),
            )
        except Exception as e:
            logger.error(f"Failed to subscribe market data: {e}")
            self.health_monitor.update_component_health(
                "market_data_feed",
                HealthStatus.DEGRADED,
                f"Subscription error: {str(e)[:50]}"
            )

    async def fetch_account_info_cached(self) -> AccountInfo:
        """Fetch account info with TTL caching."""
        now = datetime.now()

        # Return cached if fresh
        if (
            self._cached_account_info is not None
            and self._last_account_fetch is not None
            and (now - self._last_account_fetch).total_seconds() < self._account_ttl_sec
        ):
            return self._cached_account_info

        # Fetch fresh
        accounts_by_broker = await self.broker_manager.fetch_account_info_by_broker()
        account_info = self._aggregate_account_info(accounts_by_broker)

        # Update cache
        self._cached_account_info = account_info
        self._last_account_fetch = now
        self.account_store.update(account_info)

        return account_info

    async def cleanup(self) -> None:
        """Cancel all background tasks."""
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
