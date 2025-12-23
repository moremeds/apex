"""
SnapshotCoordinator - Handles risk snapshot building and dispatching.

Responsibilities:
- Dirty tracking (when to rebuild snapshots)
- Debounced snapshot dispatching
- Snapshot readiness gating
- Risk signal evaluation
"""

from __future__ import annotations
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...utils.trace_context import new_cycle
from ...utils.perf_logger import log_timing, log_timing_async
from ...models.risk_snapshot import RiskSnapshot
from ...models.risk_signal import RiskSignal
from ...domain.interfaces.event_bus import EventType
from ...domain.events.domain_events import SnapshotReadyEvent

if TYPE_CHECKING:
    from ...domain.services.risk.risk_engine import RiskEngine
    from ...domain.services.risk.rule_engine import RuleEngine
    from ...domain.services.risk.risk_signal_engine import RiskSignalEngine
    from ...domain.services.risk.risk_alert_logger import RiskAlertLogger
    from ...domain.interfaces.event_bus import EventBus
    from ...infrastructure.stores import PositionStore, MarketDataStore, AccountStore
    from ...infrastructure.monitoring import HealthMonitor
    from ...infrastructure.observability import RiskMetrics

logger = get_logger(__name__)


class SnapshotCoordinator:
    """
    Coordinates risk snapshot building and dispatching.

    Handles:
    - Dirty tracking to trigger rebuilds
    - Debounced snapshot dispatching
    - Snapshot readiness gating for startup
    - Risk signal evaluation and logging
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        rule_engine: RuleEngine,
        position_store: PositionStore,
        market_data_store: MarketDataStore,
        account_store: AccountStore,
        event_bus: EventBus,
        health_monitor: HealthMonitor,
        config: Dict[str, Any],
        risk_signal_engine: Optional[RiskSignalEngine] = None,
        risk_alert_logger: Optional[RiskAlertLogger] = None,
        risk_metrics: Optional[RiskMetrics] = None,
    ):
        self.risk_engine = risk_engine
        self.rule_engine = rule_engine
        self.position_store = position_store
        self.market_data_store = market_data_store
        self.account_store = account_store
        self.event_bus = event_bus
        self.health_monitor = health_monitor
        self.risk_signal_engine = risk_signal_engine
        self.risk_alert_logger = risk_alert_logger
        self._risk_metrics = risk_metrics

        # Configuration
        dashboard_cfg = config.get("dashboard", {})
        self._snapshot_min_interval_sec: float = dashboard_cfg.get("snapshot_min_interval_sec", 1.0)
        self._snapshot_ready_ratio: float = dashboard_cfg.get("snapshot_ready_ratio", 0.9)
        self._snapshot_ready_timeout_sec: float = dashboard_cfg.get("snapshot_ready_timeout_sec", 30.0)

        # State
        self._latest_snapshot: Optional[RiskSnapshot] = None
        self._latest_risk_signals: List[RiskSignal] = []
        self._snapshot_ready: asyncio.Event = asyncio.Event()
        self._snapshot_readiness_achieved: bool = False
        self._snapshot_startup_time: Optional[datetime] = None
        self._running: bool = False
        self._dispatcher_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the snapshot dispatcher."""
        self._running = True
        self._snapshot_startup_time = datetime.now()
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info("SnapshotCoordinator started")

    async def stop(self) -> None:
        """Stop the snapshot dispatcher."""
        self._running = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        # M7: Stop risk alert logger to flush remaining logs
        if self.risk_alert_logger:
            self.risk_alert_logger.stop()

        logger.info("SnapshotCoordinator stopped")

    async def _dispatch_loop(self) -> None:
        """
        Main dispatch loop - rebuilds snapshots when risk engine is dirty.

        Uses debouncing to prevent excessive rebuilds during rapid updates.
        """
        last_snapshot_time = 0.0

        while self._running:
            try:
                # Wait for dirty flag or timeout
                await asyncio.sleep(0.05)  # 50ms poll interval

                if not self.risk_engine.needs_rebuild():
                    continue

                # Debounce: ensure minimum interval between snapshots
                now = time.time()
                elapsed = now - last_snapshot_time
                if elapsed < self._snapshot_min_interval_sec:
                    await asyncio.sleep(self._snapshot_min_interval_sec - elapsed)

                # Build snapshot
                await self._build_and_dispatch_snapshot()
                last_snapshot_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot dispatch error: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error

    @log_timing_async("snapshot_dispatch", warn_threshold_ms=300, error_threshold_ms=1000)
    async def _build_and_dispatch_snapshot(self) -> None:
        """Build a new risk snapshot and dispatch it."""
        new_cycle()  # New trace ID for this snapshot cycle

        # Gather data
        positions = self.position_store.get_all()  # Returns List[Position]
        market_data = self.market_data_store.get_all()
        account_info = self.account_store.get()

        if not positions:
            return

        # Position updates can arrive before the first account fetch completes.
        # Avoid crashing snapshot building on startup; use a safe zeroed AccountInfo.
        if account_info is None:
            from ...models.account import AccountInfo

            account_info = AccountInfo(
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
                account_id="missing",
            )

        # Build snapshot (runs in thread to avoid blocking)
        snapshot = await asyncio.to_thread(
            self.risk_engine.build_snapshot,
            positions,
            market_data,
            account_info
        )

        if snapshot is None:
            return

        # Clear dirty state after successful build
        self.risk_engine.clear_dirty_state()

        self._latest_snapshot = snapshot

        # Evaluate risk signals
        if self.risk_signal_engine:
            try:
                self._latest_risk_signals = self.risk_signal_engine.evaluate(snapshot)
                snapshot.risk_signals = self._latest_risk_signals
            except Exception as e:
                logger.error(f"Risk signal evaluation error: {e}")

        # Log alerts
        if self.risk_alert_logger:
            self._log_risk_alerts(snapshot)

        # Check readiness gating
        self._check_snapshot_readiness(snapshot)

        # Publish snapshot ready event (typed event instead of dict payload)
        event = SnapshotReadyEvent(
            snapshot_id=snapshot.snapshot_id if hasattr(snapshot, 'snapshot_id') else "",
            position_count=snapshot.total_positions,
            coverage_pct=getattr(snapshot, 'market_data_coverage', 0.0),
            portfolio_delta=snapshot.portfolio_delta,
            unrealized_pnl=snapshot.total_unrealized_pnl,
        )
        self.event_bus.publish(EventType.SNAPSHOT_READY, event)
        self._snapshot_ready.set()

    @log_timing("log_risk_alerts", warn_threshold_ms=50)
    def _log_risk_alerts(self, snapshot: RiskSnapshot) -> None:
        """Log risk alerts to audit file."""
        if not self.risk_alert_logger:
            return

        # Log breaches (convert to RiskSignal for logging)
        breaches = self.rule_engine.evaluate(snapshot)
        for breach in breaches:
            signal = RiskSignal.from_breach(breach, layer=1)
            self.risk_alert_logger.log_risk_signal(signal, snapshot)

        # Log signals
        for signal in self._latest_risk_signals:
            self.risk_alert_logger.log_risk_signal(signal, snapshot)

    def _check_snapshot_readiness(self, snapshot: RiskSnapshot) -> None:
        """Check if snapshot meets readiness criteria."""
        if self._snapshot_readiness_achieved:
            return

        # Calculate coverage
        total_positions = len(snapshot.position_risks) if snapshot.position_risks else 0
        if total_positions == 0:
            return

        positions_with_md = sum(
            1 for pr in snapshot.position_risks
            if pr.has_market_data
        )
        coverage = positions_with_md / total_positions

        # Check timeout
        if self._snapshot_startup_time:
            elapsed = (datetime.now() - self._snapshot_startup_time).total_seconds()
            if elapsed > self._snapshot_ready_timeout_sec:
                self._snapshot_readiness_achieved = True
                logger.warning(
                    f"Snapshot readiness timeout ({elapsed:.1f}s), "
                    f"proceeding with {coverage:.1%} coverage"
                )
                return

        # Check coverage threshold
        if coverage >= self._snapshot_ready_ratio:
            self._snapshot_readiness_achieved = True
            logger.info(f"Snapshot readiness achieved: {coverage:.1%} coverage")

    def get_latest_snapshot(self) -> Optional[RiskSnapshot]:
        """Get the latest risk snapshot."""
        return self._latest_snapshot

    def get_latest_risk_signals(self) -> List[RiskSignal]:
        """Get the latest risk signals."""
        return self._latest_risk_signals

    async def wait_for_snapshot(self, timeout: float = 5.0) -> Optional[RiskSnapshot]:
        """Wait for a new snapshot to be ready."""
        self._snapshot_ready.clear()
        try:
            await asyncio.wait_for(self._snapshot_ready.wait(), timeout=timeout)
            return self._latest_snapshot
        except asyncio.TimeoutError:
            return self._latest_snapshot

    def is_ready(self) -> bool:
        """Check if snapshot coordinator is ready (has valid snapshot)."""
        return self._snapshot_readiness_achieved and self._latest_snapshot is not None
