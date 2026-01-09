"""
Repository for Technical Analysis signals and indicators.

Implements SignalPersistencePort with PostgreSQL/TimescaleDB storage.
Handles:
- Trading signals (ta_signals table)
- Indicator values (indicator_values table)
- Confluence scores (confluence_scores table)

All tables are TimescaleDB hypertables with automatic compression.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from asyncpg import Record

from src.infrastructure.persistence.database import Database
from src.infrastructure.persistence.repositories.base import BaseRepository
from src.domain.interfaces.signal_persistence import SignalPersistencePort

if TYPE_CHECKING:
    from src.domain.signals.models import TradingSignal

logger = logging.getLogger(__name__)


# =============================================================================
# Entity Classes
# =============================================================================


@dataclass
class TASignalEntity:
    """Database entity for TA trading signals."""

    time: datetime
    signal_id: str
    symbol: str
    timeframe: str
    category: str
    indicator: str
    direction: str
    strength: int
    priority: str
    trigger_rule: str
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    previous_value: Optional[float] = None
    message: Optional[str] = None
    cooldown_until: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IndicatorValueEntity:
    """Database entity for indicator values."""

    time: datetime
    symbol: str
    timeframe: str
    indicator: str
    state: Dict[str, Any]
    previous_state: Optional[Dict[str, Any]] = None
    bar_close: Optional[float] = None


@dataclass
class ConfluenceScoreEntity:
    """Database entity for confluence scores."""

    time: datetime
    symbol: str
    timeframe: str
    alignment_score: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_indicators: int
    dominant_direction: Optional[str] = None


# =============================================================================
# Repository Implementation
# =============================================================================


class TASignalRepository(SignalPersistencePort):
    """
    Repository for TA signals, indicators, and confluence scores.

    Implements SignalPersistencePort for domain layer decoupling.
    Uses TimescaleDB hypertables for efficient time-series storage.

    Usage:
        db = Database(config)
        await db.connect()

        repo = TASignalRepository(db)
        await repo.save_signal(signal)
        signals = await repo.get_recent_signals(limit=100)
    """

    def __init__(self, db: Database):
        """
        Initialize repository with database connection.

        Args:
            db: Database connection manager.
        """
        self._db = db

    # -------------------------------------------------------------------------
    # Signal Operations (implements SignalPersistencePort)
    # -------------------------------------------------------------------------

    async def save_signal(self, signal: "TradingSignal") -> None:
        """
        Persist a trading signal.

        Uses INSERT (no UPSERT) since signals are append-only time-series.
        TimescaleDB handles partitioning automatically.
        """
        query = """
            INSERT INTO ta_signals (
                time, signal_id, symbol, timeframe, category, indicator,
                direction, strength, priority, trigger_rule, current_value,
                threshold, previous_value, message, cooldown_until, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
        """
        await self._db.execute(
            query,
            signal.timestamp,
            signal.signal_id,
            signal.symbol,
            signal.timeframe,
            signal.category.value if hasattr(signal.category, "value") else signal.category,
            signal.indicator,
            signal.direction.value if hasattr(signal.direction, "value") else signal.direction,
            signal.strength,
            signal.priority.value if hasattr(signal.priority, "value") else signal.priority,
            signal.trigger_rule,
            signal.current_value,
            signal.threshold,
            signal.previous_value,
            signal.message,
            signal.cooldown_until,
            json.dumps(signal.metadata) if signal.metadata else None,
        )

    async def get_recent_signals(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List["TradingSignal"]:
        """Retrieve recent signals for TUI startup."""
        from src.domain.signals.models import (
            TradingSignal,
            SignalCategory,
            SignalDirection,
            SignalPriority,
        )

        conditions = []
        params = []
        param_idx = 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        if timeframe:
            conditions.append(f"timeframe = ${param_idx}")
            params.append(timeframe)
            param_idx += 1

        if category:
            conditions.append(f"category = ${param_idx}")
            params.append(category)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        query = f"""
            SELECT * FROM ta_signals
            {where_clause}
            ORDER BY time DESC
            LIMIT ${param_idx}
        """
        records = await self._db.fetch(query, *params)

        return [
            TradingSignal(
                signal_id=r["signal_id"],
                symbol=r["symbol"],
                category=SignalCategory(r["category"]),
                indicator=r["indicator"],
                direction=SignalDirection(r["direction"]),
                strength=r["strength"],
                priority=SignalPriority(r["priority"]),
                timeframe=r["timeframe"],
                trigger_rule=r["trigger_rule"],
                current_value=r["current_value"],
                threshold=r["threshold"],
                previous_value=r["previous_value"],
                timestamp=r["time"],
                cooldown_until=r["cooldown_until"],
                message=r["message"] or "",
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
            )
            for r in records
        ]

    async def get_signals_since(
        self,
        since: datetime,
        symbol: Optional[str] = None,
        indicator: Optional[str] = None,
        limit: int = 1000,
    ) -> List["TradingSignal"]:
        """Retrieve signals since a given timestamp."""
        from src.domain.signals.models import (
            TradingSignal,
            SignalCategory,
            SignalDirection,
            SignalPriority,
        )

        conditions = ["time > $1"]
        params = [since]
        param_idx = 2

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        if indicator:
            conditions.append(f"indicator = ${param_idx}")
            params.append(indicator)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT * FROM ta_signals
            WHERE {' AND '.join(conditions)}
            ORDER BY time ASC
            LIMIT ${param_idx}
        """
        records = await self._db.fetch(query, *params)

        return [
            TradingSignal(
                signal_id=r["signal_id"],
                symbol=r["symbol"],
                category=SignalCategory(r["category"]),
                indicator=r["indicator"],
                direction=SignalDirection(r["direction"]),
                strength=r["strength"],
                priority=SignalPriority(r["priority"]),
                timeframe=r["timeframe"],
                trigger_rule=r["trigger_rule"],
                current_value=r["current_value"],
                threshold=r["threshold"],
                previous_value=r["previous_value"],
                timestamp=r["time"],
                cooldown_until=r["cooldown_until"],
                message=r["message"] or "",
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
            )
            for r in records
        ]

    # -------------------------------------------------------------------------
    # Indicator Operations (implements SignalPersistencePort)
    # -------------------------------------------------------------------------

    async def save_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        timestamp: datetime,
        state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
        bar_close: Optional[float] = None,
    ) -> None:
        """Persist indicator state."""
        query = """
            INSERT INTO indicator_values (
                time, symbol, timeframe, indicator, state, previous_state, bar_close
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        await self._db.execute(
            query,
            timestamp,
            symbol,
            timeframe,
            indicator,
            json.dumps(state),
            json.dumps(previous_state) if previous_state else None,
            bar_close,
        )

    async def get_indicator_history(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Retrieve indicator history for charting."""
        conditions = ["symbol = $1", "timeframe = $2", "indicator = $3"]
        params = [symbol, timeframe, indicator]
        param_idx = 4

        if start:
            conditions.append(f"time >= ${param_idx}")
            params.append(start)
            param_idx += 1

        if end:
            conditions.append(f"time <= ${param_idx}")
            params.append(end)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT time, state, previous_state, bar_close
            FROM indicator_values
            WHERE {' AND '.join(conditions)}
            ORDER BY time ASC
            LIMIT ${param_idx}
        """
        records = await self._db.fetch(query, *params)

        return [
            {
                "time": r["time"],
                "state": json.loads(r["state"]),
                "previous_state": json.loads(r["previous_state"]) if r["previous_state"] else None,
                "bar_close": r["bar_close"],
            }
            for r in records
        ]

    async def get_latest_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
    ) -> Optional[Dict[str, Any]]:
        """Get the latest indicator value."""
        query = """
            SELECT time, state, previous_state, bar_close
            FROM indicator_values
            WHERE symbol = $1 AND timeframe = $2 AND indicator = $3
            ORDER BY time DESC
            LIMIT 1
        """
        record = await self._db.fetchrow(query, symbol, timeframe, indicator)

        if not record:
            return None

        return {
            "time": record["time"],
            "state": json.loads(record["state"]),
            "previous_state": json.loads(record["previous_state"]) if record["previous_state"] else None,
            "bar_close": record["bar_close"],
        }

    # -------------------------------------------------------------------------
    # Confluence Operations (implements SignalPersistencePort)
    # -------------------------------------------------------------------------

    async def save_confluence(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        alignment_score: float,
        bullish_count: int,
        bearish_count: int,
        neutral_count: int,
        total_indicators: int,
        dominant_direction: Optional[str] = None,
    ) -> None:
        """Persist confluence score."""
        query = """
            INSERT INTO confluence_scores (
                time, symbol, timeframe, alignment_score, bullish_count,
                bearish_count, neutral_count, total_indicators, dominant_direction
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        await self._db.execute(
            query,
            timestamp,
            symbol,
            timeframe,
            alignment_score,
            bullish_count,
            bearish_count,
            neutral_count,
            total_indicators,
            dominant_direction,
        )

    async def get_confluence_history(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve confluence score history."""
        conditions = ["symbol = $1", "timeframe = $2"]
        params = [symbol, timeframe]
        param_idx = 3

        if start:
            conditions.append(f"time >= ${param_idx}")
            params.append(start)
            param_idx += 1

        if end:
            conditions.append(f"time <= ${param_idx}")
            params.append(end)
            param_idx += 1

        params.append(limit)

        query = f"""
            SELECT time, alignment_score, bullish_count, bearish_count,
                   neutral_count, total_indicators, dominant_direction
            FROM confluence_scores
            WHERE {' AND '.join(conditions)}
            ORDER BY time DESC
            LIMIT ${param_idx}
        """
        records = await self._db.fetch(query, *params)

        return [
            {
                "time": r["time"],
                "alignment_score": r["alignment_score"],
                "bullish_count": r["bullish_count"],
                "bearish_count": r["bearish_count"],
                "neutral_count": r["neutral_count"],
                "total_indicators": r["total_indicators"],
                "dominant_direction": r["dominant_direction"],
            }
            for r in records
        ]

    # -------------------------------------------------------------------------
    # Additional Query Methods (not in port, but useful)
    # -------------------------------------------------------------------------

    async def get_signal_counts_by_indicator(
        self,
        since: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get count of signals grouped by indicator."""
        conditions = []
        params = []
        param_idx = 1

        if since:
            conditions.append(f"time >= ${param_idx}")
            params.append(since)
            param_idx += 1

        if symbol:
            conditions.append(f"symbol = ${param_idx}")
            params.append(symbol)
            param_idx += 1

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT indicator, COUNT(*) as count
            FROM ta_signals
            {where_clause}
            GROUP BY indicator
            ORDER BY count DESC
        """
        records = await self._db.fetch(query, *params)
        return {r["indicator"]: r["count"] for r in records}

    async def get_indicator_coverage(
        self,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Dict[str, datetime]]:
        """
        Get date coverage for each indicator.

        Returns dict of indicator -> {min_time, max_time, count}.
        """
        query = """
            SELECT indicator,
                   MIN(time) as min_time,
                   MAX(time) as max_time,
                   COUNT(*) as count
            FROM indicator_values
            WHERE symbol = $1 AND timeframe = $2
            GROUP BY indicator
        """
        records = await self._db.fetch(query, symbol, timeframe)

        return {
            r["indicator"]: {
                "min_time": r["min_time"],
                "max_time": r["max_time"],
                "count": r["count"],
            }
            for r in records
        }

    async def delete_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """
        Delete data older than retention_days.

        Returns count of deleted rows per table.
        """
        cutoff = datetime.utcnow() - __import__("datetime").timedelta(days=retention_days)

        counts = {}

        # Delete old signals
        result = await self._db.execute(
            "DELETE FROM ta_signals WHERE time < $1",
            cutoff,
        )
        counts["ta_signals"] = int(result.split()[-1]) if result else 0

        # Delete old indicator values
        result = await self._db.execute(
            "DELETE FROM indicator_values WHERE time < $1",
            cutoff,
        )
        counts["indicator_values"] = int(result.split()[-1]) if result else 0

        # Delete old confluence scores
        result = await self._db.execute(
            "DELETE FROM confluence_scores WHERE time < $1",
            cutoff,
        )
        counts["confluence_scores"] = int(result.split()[-1]) if result else 0

        logger.info(f"Deleted old data (>{retention_days} days): {counts}")
        return counts

    # -------------------------------------------------------------------------
    # TUI Tab 7 Support Methods
    # -------------------------------------------------------------------------

    async def get_indicator_summary(self) -> List[Dict[str, Any]]:
        """
        Get high-level summary per indicator type for Tab 7 display.

        Returns list of dicts with:
        - indicator: Indicator name
        - symbol_count: Number of unique symbols with this indicator
        - last_update: Most recent update timestamp
        - oldest_update: Oldest update timestamp
        """
        query = """
            SELECT
                indicator,
                COUNT(DISTINCT symbol) as symbol_count,
                MAX(time) as last_update,
                MIN(time) as oldest_update
            FROM indicator_values
            GROUP BY indicator
            ORDER BY last_update DESC
        """
        records = await self._db.fetch(query)

        return [
            {
                "indicator": r["indicator"],
                "symbol_count": r["symbol_count"],
                "last_update": r["last_update"],
                "oldest_update": r["oldest_update"],
            }
            for r in records
        ]

    async def get_indicator_details(self, indicator: str) -> List[Dict[str, Any]]:
        """
        Get detailed per-symbol info for drill-down in Tab 7.

        Args:
            indicator: Indicator name to get details for.

        Returns list of dicts with:
        - symbol: Symbol name
        - timeframe: Timeframe (e.g., "1h", "1d")
        - last_update: Most recent update timestamp
        - state: Latest indicator state dict
        """
        query = """
            SELECT DISTINCT ON (symbol, timeframe)
                symbol, timeframe, time as last_update, state
            FROM indicator_values
            WHERE indicator = $1
            ORDER BY symbol, timeframe, time DESC
        """
        records = await self._db.fetch(query, indicator)

        return [
            {
                "symbol": r["symbol"],
                "timeframe": r["timeframe"],
                "last_update": r["last_update"],
                "state": json.loads(r["state"]) if r["state"] else {},
            }
            for r in records
        ]
