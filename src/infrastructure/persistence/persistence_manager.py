"""Persistence Manager - Coordinates all persistence operations with batching."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from threading import Lock
import time

from .duckdb_adapter import DuckDBAdapter
from .repositories.position_repository import PositionRepository
from .repositories.portfolio_repository import PortfolioRepository
from .repositories.alert_repository import AlertRepository
from src.models.risk_snapshot import RiskSnapshot
from src.models.position_risk import PositionRisk
from src.models.risk_signal import RiskSignal
from src.models.position import PositionSource

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
        }

        logger.info(f"PersistenceManager initialized: {self.config.db_path}")

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
            broker_position_risks = [
                pr for pr in (snapshot.position_risks or [])
                if pr.position.source in BROKER_SOURCES
            ]

            # Save portfolio snapshot (uses full snapshot metrics which are already broker-based)
            self.portfolio.save_snapshot(snapshot)

            # Save position snapshots (only broker positions)
            if broker_position_risks:
                self.positions.save_snapshots(broker_position_risks, now)

                # Detect position changes (only broker positions)
                if self.config.change_detection:
                    changes = self._detect_changes(broker_position_risks)
                    if changes:
                        self._change_batch.extend(changes)
                        self._flush_changes()

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

    def __repr__(self) -> str:
        return f"PersistenceManager(db={self.config.db_path}, snapshots={self._stats['snapshots_saved']})"
