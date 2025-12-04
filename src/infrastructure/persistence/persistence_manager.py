"""Persistence Manager - Coordinates all persistence operations with batching."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from threading import Lock
import time

from .duckdb_adapter import DuckDBAdapter
from .repositories.position_repository import PositionRepository
from .repositories.portfolio_repository import PortfolioRepository
from .repositories.alert_repository import AlertRepository
from .repositories.order_repository import OrderRepository
from src.models.risk_snapshot import RiskSnapshot
from src.models.position_risk import PositionRisk
from src.models.risk_signal import RiskSignal
from src.models.position import PositionSource
from src.models.order import Order, Trade, OrderSource
from src.domain.interfaces.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

# Only persist positions from real brokers (not manual file entries)
BROKER_SOURCES = {PositionSource.IB, PositionSource.FUTU}


@dataclass
class PersistenceConfig:
    """Configuration for persistence manager."""
    db_path: str = "./data/apex_risk.duckdb"
    snapshot_interval_sec: int = 60
    change_detection: bool = True
    batch_size: int = 50
    batch_timeout_ms: int = 100
    position_snapshots_days: int = 90
    portfolio_snapshots_days: int = 365
    alerts_days: int = 365
    orders_days: int = 365
    trades_days: int = 365
    order_sync_days_back: int = 30  # Days to sync on startup


@dataclass
class PositionState:
    """Tracks position state for change detection."""
    symbol: str
    quantity: float
    avg_price: Optional[float]
    source: Optional[str]


class PersistenceManager:
    """
    Manages all persistence operations with batching and change detection.

    Features:
    - Batched writes to reduce I/O
    - Position change detection
    - Async-friendly (non-blocking) operations
    - Daily P&L tracking
    """

    def __init__(self, config: Optional[PersistenceConfig] = None):
        """
        Initialize persistence manager.

        Args:
            config: Persistence configuration
        """
        self.config = config or PersistenceConfig()
        self.db = DuckDBAdapter(self.config.db_path)

        # Initialize repositories
        self.positions = PositionRepository(self.db)
        self.portfolio = PortfolioRepository(self.db)
        self.alerts = AlertRepository(self.db)
        self.orders = OrderRepository(self.db)

        # State tracking for change detection
        self._position_states: Dict[str, PositionState] = {}
        self._last_snapshot_time: Optional[datetime] = None
        self._lock = Lock()

        # Batch queues
        self._alert_batch: List[RiskSignal] = []
        self._change_batch: List[Dict[str, Any]] = []

        # Stats
        self._stats = {
            "snapshots_saved": 0,
            "changes_detected": 0,
            "alerts_saved": 0,
            "last_persist_ms": 0,
            "orders_synced": 0,
            "trades_synced": 0,
        }

        logger.info(f"PersistenceManager initialized: {self.config.db_path}")

        # Load previous position state from database on startup
        self._load_position_state_from_db()

    def persist_snapshot(self, snapshot: RiskSnapshot) -> None:
        """
        Persist a risk snapshot (positions + portfolio).

        Args:
            snapshot: RiskSnapshot to persist
        """
        start_time = time.time()

        with self._lock:
            now = datetime.now()

            # Check if enough time has passed since last snapshot
            if self._last_snapshot_time:
                elapsed = (now - self._last_snapshot_time).total_seconds()
                if elapsed < self.config.snapshot_interval_sec:
                    return

            # Filter to only broker positions (IB, FUTU - exclude MANUAL)
            # Check both source and all_sources fields
            broker_position_risks = []
            for pr in (snapshot.position_risks or []):
                if not pr.position:
                    continue
                pos = pr.position
                # Check primary source
                if pos.source in BROKER_SOURCES:
                    broker_position_risks.append(pr)
                # Also check all_sources (reconciler populates this)
                elif pos.all_sources and any(s in BROKER_SOURCES for s in pos.all_sources):
                    broker_position_risks.append(pr)

            # Debug logging (print to console for visibility)
            total_positions = len(snapshot.position_risks) if snapshot.position_risks else 0
            if total_positions > 0:
                sources = {}
                for pr in snapshot.position_risks:
                    src = pr.position.source.value if pr.position and pr.position.source else "None"
                    sources[src] = sources.get(src, 0) + 1
                msg = f"💾 Persistence: {total_positions} positions by source: {sources}, {len(broker_position_risks)} to save"
                print(msg, flush=True)
                logger.info(msg)
            else:
                print("💾 Persistence: No positions in snapshot", flush=True)
                logger.info("Persistence: No positions in snapshot")

            # Save portfolio snapshot (uses full snapshot metrics which are already broker-based)
            self.portfolio.save_snapshot(snapshot)

            # Save position snapshots (only broker positions)
            if broker_position_risks:
                self.positions.save_snapshots(broker_position_risks, now)
                print(f"💾 Saved {len(broker_position_risks)} position snapshots to database", flush=True)

                # Detect position changes (only broker positions)
                if self.config.change_detection:
                    changes = self._detect_changes(broker_position_risks)
                    if changes:
                        self._change_batch.extend(changes)
                        self._flush_changes()
                        print(f"💾 Detected {len(changes)} position changes", flush=True)
            else:
                print("💾 No broker positions to save (all MANUAL?)", flush=True)

            # Update daily P&L tracking
            self._update_daily_pnl(snapshot)

            self._last_snapshot_time = now
            self._stats["snapshots_saved"] += 1

        self._stats["last_persist_ms"] = (time.time() - start_time) * 1000
        logger.debug(f"Snapshot persisted in {self._stats['last_persist_ms']:.1f}ms")

    def _detect_changes(self, broker_position_risks: List[PositionRisk]) -> List[Dict[str, Any]]:
        """
        Detect position changes compared to last known state.

        Only tracks broker positions (IB, FUTU). Manual positions are ignored.
        """
        changes = []
        current_symbols: Set[str] = set()

        for pr in broker_position_risks:
            pos = pr.position
            symbol = pr.symbol
            current_symbols.add(symbol)

            prev_state = self._position_states.get(symbol)

            if prev_state is None:
                # New position opened
                changes.append({
                    "change_type": "OPEN",
                    "symbol": symbol,
                    "underlying": pos.underlying,
                    "quantity_before": None,
                    "quantity_after": pos.quantity,
                    "avg_price_after": pos.avg_price,
                    "source": pos.source.value if pos.source else None,
                })
                self._stats["changes_detected"] += 1

            elif prev_state.quantity != pos.quantity:
                if pos.quantity == 0:
                    # Position closed
                    changes.append({
                        "change_type": "CLOSE",
                        "symbol": symbol,
                        "underlying": pos.underlying,
                        "quantity_before": prev_state.quantity,
                        "quantity_after": 0,
                        "avg_price_before": prev_state.avg_price,
                        "source": pos.source.value if pos.source else None,
                    })
                else:
                    # Position modified (added/reduced)
                    changes.append({
                        "change_type": "MODIFY",
                        "symbol": symbol,
                        "underlying": pos.underlying,
                        "quantity_before": prev_state.quantity,
                        "quantity_after": pos.quantity,
                        "avg_price_before": prev_state.avg_price,
                        "avg_price_after": pos.avg_price,
                        "source": pos.source.value if pos.source else None,
                    })
                self._stats["changes_detected"] += 1

            # Update state tracking
            self._position_states[symbol] = PositionState(
                symbol=symbol,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                source=pos.source.value if pos.source else None,
            )

        # Detect closed positions (existed before but not in current broker positions)
        for symbol, prev_state in list(self._position_states.items()):
            if symbol not in current_symbols:
                # Only record close if it was a broker position
                if prev_state.source in {s.value for s in BROKER_SOURCES}:
                    changes.append({
                        "change_type": "CLOSE",
                        "symbol": symbol,
                        "underlying": symbol.split()[0] if " " in symbol else symbol,
                        "quantity_before": prev_state.quantity,
                        "quantity_after": 0,
                        "avg_price_before": prev_state.avg_price,
                        "source": prev_state.source,
                    })
                    self._stats["changes_detected"] += 1
                del self._position_states[symbol]

        return changes

    def _flush_changes(self) -> None:
        """Flush change batch to database."""
        if self._change_batch:
            # Count opens and closes for daily stats
            opens = sum(1 for c in self._change_batch if c["change_type"] == "OPEN")
            closes = sum(1 for c in self._change_batch if c["change_type"] == "CLOSE")

            self.positions.save_changes_batch(self._change_batch)

            today = date.today()
            if opens:
                self.portfolio.increment_positions_opened(today, opens)
            if closes:
                self.portfolio.increment_positions_closed(today, closes)

            self._change_batch.clear()

    def _update_daily_pnl(self, snapshot: RiskSnapshot) -> None:
        """Update daily P&L tracking."""
        today = date.today()

        # Save/update daily P&L
        self.portfolio.save_daily_pnl(
            trade_date=today,
            unrealized_pnl=snapshot.total_unrealized_pnl,
            daily_pnl=snapshot.total_daily_pnl,
            is_open=False,  # Always update close values
            total_positions=snapshot.total_positions,
        )

        # Update drawdown tracking
        if snapshot.total_daily_pnl is not None:
            self.portfolio.update_daily_drawdown(today, snapshot.total_daily_pnl)

    def persist_alerts(self, signals: List[RiskSignal]) -> None:
        """
        Persist risk alerts.

        Args:
            signals: List of RiskSignal to persist
        """
        if not signals:
            return

        with self._lock:
            self._alert_batch.extend(signals)

            # Flush if batch is full
            if len(self._alert_batch) >= self.config.batch_size:
                self._flush_alerts()

    def _flush_alerts(self) -> None:
        """Flush alert batch to database."""
        if self._alert_batch:
            count = self.alerts.save_alerts_batch(self._alert_batch)
            self._stats["alerts_saved"] += count
            self._alert_batch.clear()

    def flush(self) -> None:
        """Flush all pending batches."""
        with self._lock:
            self._flush_changes()
            self._flush_alerts()

    def cleanup(self) -> Dict[str, int]:
        """Run cleanup of old records based on retention settings."""
        results = {}

        results["position_snapshots"] = self.positions.cleanup_old_snapshots(
            self.config.position_snapshots_days
        )
        results["portfolio_snapshots"] = self.portfolio.cleanup_old_snapshots(
            self.config.portfolio_snapshots_days
        )
        results["alerts"] = self.alerts.cleanup_old_alerts(
            self.config.alerts_days
        )

        # Vacuum database
        self.db.vacuum()

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return {
            **self._stats,
            "db_stats": self.db.get_stats(),
            "pending_alerts": len(self._alert_batch),
            "pending_changes": len(self._change_batch),
            "tracked_positions": len(self._position_states),
        }

    def close(self) -> None:
        """Close persistence manager and flush pending data."""
        self.flush()
        self.db.close()
        logger.info("PersistenceManager closed")

    def _load_position_state_from_db(self) -> None:
        """
        Load previous position state from database on startup.

        This enables proper change detection by comparing against the last
        known positions from a previous session (reconciliation on startup).
        """
        try:
            # Get the most recent position snapshot for each symbol (DuckDB compatible)
            rows = self.db.fetch_all("""
                WITH ranked AS (
                    SELECT
                        symbol, quantity, avg_price, source,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY snapshot_time DESC) as rn
                    FROM position_snapshots
                    WHERE source IN ('IB', 'FUTU')
                )
                SELECT symbol, quantity, avg_price, source
                FROM ranked
                WHERE rn = 1
            """)

            if rows:
                for row in rows:
                    row_dict = row
                    symbol = row_dict.get("symbol")
                    if symbol:
                        self._position_states[symbol] = PositionState(
                            symbol=symbol,
                            quantity=row_dict.get("quantity", 0),
                            avg_price=row_dict.get("avg_price"),
                            source=row_dict.get("source"),
                        )

                logger.info(f"Loaded {len(self._position_states)} positions from database for change detection")
            else:
                logger.info("No previous positions found in database - starting fresh")

        except Exception as e:
            logger.warning(f"Failed to load position state from database: {e}")
            # Continue with empty state - will treat all positions as new

    def get_all_position_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all position snapshots from database (most recent per symbol).

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of position snapshot records
        """
        try:
            # Use window function to get latest snapshot per symbol (DuckDB compatible)
            rows = self.db.fetch_all(f"""
                WITH ranked AS (
                    SELECT
                        symbol, underlying, asset_type, quantity, avg_price,
                        mark_price, unrealized_pnl, daily_pnl, source,
                        snapshot_time, expiry, strike, option_type,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY snapshot_time DESC) as rn
                    FROM position_snapshots
                )
                SELECT
                    symbol, underlying, asset_type, quantity, avg_price,
                    mark_price, unrealized_pnl, daily_pnl, source,
                    snapshot_time, expiry, strike, option_type
                FROM ranked
                WHERE rn = 1
                LIMIT {limit}
            """)
            return rows if rows else []
        except Exception as e:
            logger.warning(f"Failed to get position snapshots: {e}")
            return []

    def get_position_count(self) -> int:
        """Get total count of unique positions in database."""
        try:
            result = self.db.fetch_one("""
                SELECT COUNT(DISTINCT symbol) as count
                FROM position_snapshots
            """)
            return result["count"] if result else 0
        except Exception as e:
            logger.warning(f"Failed to get position count: {e}")
            return 0

    def __repr__(self) -> str:
        return f"PersistenceManager(db={self.config.db_path}, snapshots={self._stats['snapshots_saved']})"

    # ========== Event-Driven Methods ==========

    def subscribe_to_events(self, event_bus: EventBus) -> None:
        """
        Subscribe to persistence-related events.

        Args:
            event_bus: Event bus to subscribe to.
        """
        event_bus.subscribe(EventType.SNAPSHOT_READY, self._on_snapshot_ready)
        event_bus.subscribe(EventType.RISK_SIGNAL, self._on_risk_signal)
        event_bus.subscribe(EventType.POSITION_UPDATED, self._on_position_change)
        event_bus.subscribe(EventType.ORDERS_BATCH, self._on_orders_batch)
        event_bus.subscribe(EventType.TRADES_BATCH, self._on_trades_batch)
        event_bus.subscribe(EventType.ORDER_UPDATED, self._on_order_updated)
        event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        logger.debug("PersistenceManager subscribed to events")

    def _on_snapshot_ready(self, payload: dict) -> None:
        """
        Handle snapshot ready event.

        Persists the snapshot asynchronously using a thread pool to avoid blocking.

        Args:
            payload: Event payload with 'snapshot'.
        """
        snapshot = payload.get("snapshot")
        if snapshot:
            # Run persistence in a thread pool to avoid blocking the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self._persist_snapshot_sync, snapshot)
            except RuntimeError:
                # No event loop running, fall back to sync
                self._persist_snapshot_sync(snapshot)

    def _persist_snapshot_sync(self, snapshot: RiskSnapshot) -> None:
        """Synchronous snapshot persistence (runs in thread pool)."""
        try:
            self.persist_snapshot(snapshot)
        except Exception as e:
            logger.error(f"Failed to persist snapshot from event: {e}")

    def _on_risk_signal(self, payload: dict) -> None:
        """
        Handle risk signal event.

        Persists the signal to the database asynchronously via thread pool
        to avoid blocking the event loop during DuckDB writes.

        Args:
            payload: Event payload with 'signal'.
        """
        signal = payload.get("signal")
        if signal:
            try:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self._persist_alert_sync, signal)
            except RuntimeError:
                # No event loop running, fall back to sync
                self._persist_alert_sync(signal)

    def _persist_alert_sync(self, signal) -> None:
        """Synchronous alert persistence (runs in thread pool)."""
        try:
            self.persist_alerts([signal])
        except Exception as e:
            logger.error(f"Failed to persist risk signal from event: {e}")

    def _on_position_change(self, payload: dict) -> None:
        """
        Handle position update event (e.g., from trade deal push).

        Logs the position change for audit trail.

        Args:
            payload: Event payload with position update info.
        """
        symbol = payload.get("symbol", "unknown")
        source = payload.get("source", "unknown")
        logger.info(f"Position change event: {symbol} from {source}")

    def _on_orders_batch(self, payload: dict) -> None:
        """
        Handle batch of orders from broker sync.

        Args:
            payload: Event payload with 'orders' list.
        """
        orders = payload.get("orders", [])
        if not orders:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._sync_orders_sync, orders)
        except RuntimeError:
            # No event loop running, fall back to sync
            self._sync_orders_sync(orders)

    def _sync_orders_sync(self, orders: List[Order]) -> None:
        """Synchronous order sync (runs in thread pool)."""
        try:
            result = self.sync_orders(orders)
            logger.info(
                f"Synced {result['inserted']} new, {result['updated']} updated orders"
            )
        except Exception as e:
            logger.error(f"Failed to sync orders from event: {e}")

    def _on_trades_batch(self, payload: dict) -> None:
        """
        Handle batch of trades from broker sync.

        Args:
            payload: Event payload with 'trades' list.
        """
        trades = payload.get("trades", [])
        if not trades:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._sync_trades_sync, trades)
        except RuntimeError:
            # No event loop running, fall back to sync
            self._sync_trades_sync(trades)

    def _sync_trades_sync(self, trades: List[Trade]) -> None:
        """Synchronous trade sync (runs in thread pool)."""
        try:
            result = self.sync_trades(trades)
            logger.info(
                f"Synced {result['inserted']} new, {result['updated']} updated trades"
            )
        except Exception as e:
            logger.error(f"Failed to sync trades from event: {e}")

    def _on_order_updated(self, payload: dict) -> None:
        """
        Handle single order update event (e.g., status change).

        Args:
            payload: Event payload with 'order'.
        """
        order = payload.get("order")
        if not order:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._sync_orders_sync, [order])
        except RuntimeError:
            self._sync_orders_sync([order])

    def _on_trade_executed(self, payload: dict) -> None:
        """
        Handle single trade execution event (e.g., fill).

        Args:
            payload: Event payload with 'trade'.
        """
        trade = payload.get("trade")
        if not trade:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._sync_trades_sync, [trade])
        except RuntimeError:
            self._sync_trades_sync([trade])

    # ========== Order History Methods ==========

    def sync_orders(self, orders: List[Order]) -> Dict[str, int]:
        """
        Synchronize orders to the database (upsert).

        Args:
            orders: List of Order objects to sync

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        if not orders:
            return {"inserted": 0, "updated": 0}

        result = self.orders.upsert_orders(orders)
        self._stats["orders_synced"] += result["inserted"] + result["updated"]
        return result

    def sync_trades(self, trades: List[Trade]) -> Dict[str, int]:
        """
        Synchronize trades to the database (upsert).

        Args:
            trades: List of Trade objects to sync

        Returns:
            Dict with counts: {"inserted": N, "updated": M}
        """
        if not trades:
            return {"inserted": 0, "updated": 0}

        result = self.orders.upsert_trades(trades)
        self._stats["trades_synced"] += result["inserted"] + result["updated"]
        return result

    async def sync_order_history_from_brokers(
        self,
        ib_adapters: Optional[List] = None,
        futu_adapter: Optional[Any] = None,
        days_back: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Synchronize order and trade history from all brokers on startup.

        This method fetches historical orders and trades from both IB and Futu
        adapters and upserts them into the database.

        Args:
            ib_adapters: List of IB adapter instances (supports multiple accounts)
            futu_adapter: Futu adapter instance
            days_back: Number of days to look back (default from config)

        Returns:
            Dict with sync statistics
        """
        days_back = days_back or self.config.order_sync_days_back
        results = {
            "ib_orders": {"inserted": 0, "updated": 0},
            "ib_trades": {"inserted": 0, "updated": 0},
            "futu_orders": {"inserted": 0, "updated": 0},
            "futu_trades": {"inserted": 0, "updated": 0},
            "errors": [],
        }

        # Sync from IB adapters (can have multiple accounts)
        if ib_adapters:
            for idx, ib_adapter in enumerate(ib_adapters):
                try:
                    if not ib_adapter.is_connected():
                        logger.warning(f"IB adapter {idx} not connected, skipping order sync")
                        continue

                    # Fetch orders
                    logger.info(f"Fetching order history from IB adapter {idx}...")
                    orders = await ib_adapter.fetch_orders(include_open=True, include_completed=True)
                    order_result = self.sync_orders(orders)
                    results["ib_orders"]["inserted"] += order_result["inserted"]
                    results["ib_orders"]["updated"] += order_result["updated"]

                    # Fetch trades
                    logger.info(f"Fetching trade history from IB adapter {idx}...")
                    trades = await ib_adapter.fetch_trades(days_back=days_back)
                    trade_result = self.sync_trades(trades)
                    results["ib_trades"]["inserted"] += trade_result["inserted"]
                    results["ib_trades"]["updated"] += trade_result["updated"]

                    logger.info(
                        f"IB adapter {idx} sync complete: "
                        f"orders={order_result['inserted']}+{order_result['updated']}, "
                        f"trades={trade_result['inserted']}+{trade_result['updated']}"
                    )

                except Exception as e:
                    error_msg = f"Failed to sync from IB adapter {idx}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

        # Sync from Futu adapter
        if futu_adapter:
            try:
                if not futu_adapter.is_connected():
                    logger.warning("Futu adapter not connected, skipping order sync")
                else:
                    # Fetch orders
                    logger.info("Fetching order history from Futu...")
                    orders = await futu_adapter.fetch_orders(
                        include_open=True,
                        include_completed=True,
                        days_back=days_back,
                    )
                    order_result = self.sync_orders(orders)
                    results["futu_orders"]["inserted"] = order_result["inserted"]
                    results["futu_orders"]["updated"] = order_result["updated"]

                    # Fetch trades
                    logger.info("Fetching trade history from Futu...")
                    trades = await futu_adapter.fetch_trades(days_back=days_back)
                    trade_result = self.sync_trades(trades)
                    results["futu_trades"]["inserted"] = trade_result["inserted"]
                    results["futu_trades"]["updated"] = trade_result["updated"]

                    logger.info(
                        f"Futu sync complete: "
                        f"orders={order_result['inserted']}+{order_result['updated']}, "
                        f"trades={trade_result['inserted']}+{trade_result['updated']}"
                    )

            except Exception as e:
                error_msg = f"Failed to sync from Futu: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Log summary
        total_orders = (
            results["ib_orders"]["inserted"] + results["ib_orders"]["updated"] +
            results["futu_orders"]["inserted"] + results["futu_orders"]["updated"]
        )
        total_trades = (
            results["ib_trades"]["inserted"] + results["ib_trades"]["updated"] +
            results["futu_trades"]["inserted"] + results["futu_trades"]["updated"]
        )
        logger.info(
            f"Order history sync complete: {total_orders} orders, {total_trades} trades "
            f"({len(results['errors'])} errors)"
        )

        return results

    def get_order_stats(self) -> Dict[str, Any]:
        """Get order and trade statistics."""
        return {
            "orders_synced": self._stats["orders_synced"],
            "trades_synced": self._stats["trades_synced"],
            "total_orders": len(self.orders.get_orders(limit=10000)),
            "total_trades": len(self.orders.get_trades(limit=10000)),
            "open_orders": len(self.orders.get_open_orders()),
        }

    def get_recent_trades(
        self,
        days: int = 7,
        source: Optional[OrderSource] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades for display/analysis.

        Args:
            days: Number of days to look back
            source: Filter by source (IB/FUTU)

        Returns:
            List of trade records
        """
        start_time = datetime.now() - timedelta(days=days)
        return self.orders.get_trades(
            source=source,
            start_time=start_time,
            limit=500,
        )

    def reconcile_orders_with_trades(self) -> Dict[str, Any]:
        """
        Reconcile orders with trades to identify discrepancies.

        Checks for:
        - Orders marked FILLED but no corresponding trades
        - Trades without parent orders
        - Fill quantity mismatches

        Returns:
            Dict with reconciliation results
        """
        issues = []

        # Get all filled orders from last 30 days
        start_time = datetime.now() - timedelta(days=30)
        from src.models.order import OrderStatus
        filled_orders = self.orders.get_orders(
            status=OrderStatus.FILLED,
            start_time=start_time,
            limit=1000,
        )

        for order in filled_orders:
            source = OrderSource(order["source"])
            trades = self.orders.get_trades_by_order(
                source=source,
                order_id=order["order_id"],
                account_id=order["account_id"],
            )

            if not trades:
                issues.append({
                    "type": "MISSING_TRADES",
                    "order_id": order["order_id"],
                    "source": order["source"],
                    "symbol": order["symbol"],
                    "message": f"Order {order['order_id']} is FILLED but has no trades",
                })
            else:
                # Check fill quantity
                total_trade_qty = sum(t["quantity"] for t in trades)
                if abs(total_trade_qty - order["filled_quantity"]) > 0.01:
                    issues.append({
                        "type": "QUANTITY_MISMATCH",
                        "order_id": order["order_id"],
                        "source": order["source"],
                        "symbol": order["symbol"],
                        "order_qty": order["filled_quantity"],
                        "trade_qty": total_trade_qty,
                        "message": f"Order filled qty ({order['filled_quantity']}) != trades ({total_trade_qty})",
                    })

        return {
            "issues_count": len(issues),
            "issues": issues,
            "orders_checked": len(filled_orders),
        }
