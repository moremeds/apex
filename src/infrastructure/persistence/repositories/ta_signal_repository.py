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
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.domain.interfaces.signal_persistence import SignalPersistencePort
from src.infrastructure.persistence.database import Database

if TYPE_CHECKING:
    from src.domain.signals.models import TradingSignal

logger = logging.getLogger(__name__)

# Type variable for async function return type
T = TypeVar("T")


def db_retry() -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Retry decorator for database operations.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s) on:
    - Connection errors
    - Timeout errors
    - Transient database errors

    Logs each retry attempt with context.
    """
    import asyncpg

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (
                    asyncpg.PostgresConnectionError,
                    asyncpg.InterfaceError,
                    asyncpg.InternalClientError,
                    ConnectionError,
                    TimeoutError,
                )
            ),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Database operation failed, retrying ({retry_state.attempt_number}/3)",
                extra={
                    "function": func.__name__,
                    "error": str(retry_state.outcome.exception()) if retry_state.outcome else None,
                },
            ),
        )
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


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

    @db_retry()
    async def save_signal(self, signal: Any) -> None:  # type: ignore[override]
        """
        Persist a trading signal.

        Handles both TradingSignal and TradingSignalEvent objects.
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
        # Extract values handling both TradingSignal and TradingSignalEvent
        category = getattr(signal, "category", "")
        if hasattr(category, "value"):
            category = category.value
        direction = getattr(signal, "direction", "")
        if hasattr(direction, "value"):
            direction = direction.value
        priority = getattr(signal, "priority", "")
        if hasattr(priority, "value"):
            priority = priority.value
        metadata = getattr(signal, "metadata", None)

        await self._db.execute(
            query,
            signal.timestamp,
            signal.signal_id,
            signal.symbol,
            getattr(signal, "timeframe", ""),
            category,
            getattr(signal, "indicator", ""),
            direction,
            float(signal.strength),
            priority,
            getattr(signal, "trigger_rule", ""),
            getattr(signal, "current_value", None),
            getattr(signal, "threshold", None),
            getattr(signal, "previous_value", None),
            getattr(signal, "message", ""),
            getattr(signal, "cooldown_until", None),
            json.dumps(metadata) if metadata else None,
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
            SignalCategory,
            SignalDirection,
            SignalPriority,
            TradingSignal,
        )

        conditions: List[str] = []
        params: List[Any] = []
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
            SignalCategory,
            SignalDirection,
            SignalPriority,
            TradingSignal,
        )

        conditions = ["time > $1"]
        params: List[Any] = [since]
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

    @db_retry()
    async def save_indicator(  # type: ignore[override]
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
        params: List[Any] = [symbol, timeframe, indicator]
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
            "previous_state": (
                json.loads(record["previous_state"]) if record["previous_state"] else None
            ),
            "bar_close": record["bar_close"],
        }

    # -------------------------------------------------------------------------
    # Confluence Operations (implements SignalPersistencePort)
    # -------------------------------------------------------------------------

    @db_retry()
    async def save_confluence(  # type: ignore[override]
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
        params: List[Any] = [symbol, timeframe]
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
        params: List[Any] = []
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

    @db_retry()
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

    # Indicator metadata: (category, full_name, description)
    INDICATOR_METADATA: Dict[str, tuple] = {
        # Momentum indicators
        "rsi": (
            "momentum",
            "Relative Strength Index",
            "Measures speed/change of price movements (0-100)",
        ),
        "macd": (
            "momentum",
            "Moving Avg Convergence Divergence",
            "Trend-following momentum using EMA crossovers",
        ),
        "kdj": ("momentum", "KDJ Stochastic", "Enhanced stochastic with J line for divergence"),
        "mfi": (
            "momentum",
            "Money Flow Index",
            "Volume-weighted RSI measuring buying/selling pressure",
        ),
        "cci": (
            "momentum",
            "Commodity Channel Index",
            "Measures price deviation from statistical mean",
        ),
        "roc": (
            "momentum",
            "Rate of Change",
            "Percentage change between current and N-period price",
        ),
        "momentum": ("momentum", "Momentum", "Simple price difference over N periods"),
        "tsi": ("momentum", "True Strength Index", "Double-smoothed momentum oscillator"),
        "ultimate": (
            "momentum",
            "Ultimate Oscillator",
            "Multi-timeframe momentum with weighted averages",
        ),
        "williams_r": ("momentum", "Williams %R", "Overbought/oversold oscillator (-100 to 0)"),
        "awesome": (
            "momentum",
            "Awesome Oscillator",
            "Difference between 5 and 34-period SMA of midpoints",
        ),
        "trix": ("momentum", "TRIX", "Triple-smoothed EMA rate of change"),
        "rsi_harmonics": ("momentum", "RSI Harmonics", "RSI with harmonic pattern detection"),
        # Trend indicators
        "adx": (
            "trend",
            "Average Directional Index",
            "Measures trend strength regardless of direction",
        ),
        "aroon": ("trend", "Aroon", "Identifies trend changes and strength using high/low timing"),
        "supertrend": ("trend", "SuperTrend", "ATR-based trend indicator with clear signals"),
        "psar": ("trend", "Parabolic SAR", "Stop and reverse points for trend following"),
        "ichimoku": (
            "trend",
            "Ichimoku Cloud",
            "Multi-component system showing support/resistance/trend",
        ),
        "vortex": (
            "trend",
            "Vortex Indicator",
            "Identifies trend direction using +VI/-VI crossovers",
        ),
        "trendline": ("trend", "Trendline", "Automated trendline detection and analysis"),
        "zerolag": ("trend", "Zero Lag EMA", "Reduced-lag exponential moving average"),
        # Volatility indicators
        "atr": (
            "volatility",
            "Average True Range",
            "Measures market volatility using price ranges",
        ),
        "bollinger": ("volatility", "Bollinger Bands", "Volatility bands using standard deviation"),
        "keltner": ("volatility", "Keltner Channel", "ATR-based volatility channel around EMA"),
        "donchian": ("volatility", "Donchian Channel", "Highest high/lowest low over N periods"),
        "stddev": ("volatility", "Standard Deviation", "Statistical measure of price dispersion"),
        "hvol": ("volatility", "Historical Volatility", "Annualized standard deviation of returns"),
        "chaikin_vol": ("volatility", "Chaikin Volatility", "Rate of change of ATR"),
        # Volume indicators
        "obv": ("volume", "On-Balance Volume", "Cumulative volume based on price direction"),
        "ad": ("volume", "Accumulation/Distribution", "Volume-weighted price location indicator"),
        "cmf": ("volume", "Chaikin Money Flow", "Volume-weighted accumulation over N periods"),
        "vwap": ("volume", "Volume Weighted Avg Price", "Average price weighted by volume"),
        "vpvr": ("volume", "Volume Profile", "Volume distribution across price levels"),
        "cvd": ("volume", "Cumulative Volume Delta", "Net buying vs selling volume over time"),
        "force": ("volume", "Force Index", "Price change multiplied by volume"),
        # Moving averages
        "sma": ("moving_avg", "Simple Moving Average", "Arithmetic mean of prices over N periods"),
        "ema": (
            "moving_avg",
            "Exponential Moving Average",
            "Weighted average favoring recent prices",
        ),
        # Patterns & Levels
        "pivot": ("pattern", "Pivot Points", "Support/resistance levels from prior period"),
        "fibonacci": ("pattern", "Fibonacci Retracement", "Key levels based on Fibonacci ratios"),
        "support_resistance": (
            "pattern",
            "Support/Resistance",
            "Detected price levels with high reaction",
        ),
        "candlestick": (
            "pattern",
            "Candlestick Patterns",
            "Japanese candlestick pattern recognition",
        ),
        "chart_patterns": (
            "pattern",
            "Chart Patterns",
            "Geometric pattern detection (H&S, triangles)",
        ),
    }

    def _get_indicator_info(self, indicator: str) -> Dict[str, str]:
        """Get full metadata for an indicator."""
        if indicator in self.INDICATOR_METADATA:
            cat, name, desc = self.INDICATOR_METADATA[indicator]
            return {"category": cat, "full_name": name, "description": desc}
        return {"category": "other", "full_name": indicator.title(), "description": ""}

    async def get_indicator_summary(self) -> List[Dict[str, Any]]:
        """
        Get high-level summary per indicator type for Tab 7 display.

        Returns list of dicts with:
        - category: Category name (momentum, trend, volatility, volume, etc.)
        - indicator: Indicator short name
        - full_name: Full indicator name
        - description: Short description of indicator
        - symbol_count: Number of unique symbols with this indicator
        - last_update: Most recent update timestamp
        - oldest_update: The oldest "latest update" among all symbols (most stale symbol)
        """
        # This query finds, for each indicator:
        # 1. The latest update per symbol (subquery)
        # 2. Then MIN of those to find the most stale symbol's latest update
        query = """
            WITH latest_per_symbol AS (
                SELECT indicator, symbol, MAX(time) as latest
                FROM indicator_values
                GROUP BY indicator, symbol
            )
            SELECT
                indicator,
                COUNT(DISTINCT symbol) as symbol_count,
                MAX(latest) as last_update,
                MIN(latest) as oldest_update
            FROM latest_per_symbol
            GROUP BY indicator
            ORDER BY indicator
        """
        records = await self._db.fetch(query)

        result = []
        for r in records:
            info = self._get_indicator_info(r["indicator"])
            result.append(
                {
                    "category": info["category"],
                    "indicator": r["indicator"],
                    "full_name": info["full_name"],
                    "description": info["description"],
                    "symbol_count": r["symbol_count"],
                    "last_update": r["last_update"],
                    "oldest_update": r["oldest_update"],
                }
            )
        return result

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
