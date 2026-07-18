"""Signal Service — long-running daemon that consumes IB ticks and writes to PostgreSQL.

Replaces the FastAPI server lifespan + WebBridge for signal persistence.

Entry point: python -m src.services.signal_service
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _db_config_from_env(pg_url: str, base: Any) -> Any:
    """
    Build a DatabaseConfig from a libpq DSN URL, preserving pool/timescale settings
    from the base config. Used when APEX_PG_URL overrides config/base.yaml.
    """
    from urllib.parse import unquote, urlparse

    from config.models import DatabaseConfig

    parsed = urlparse(pg_url)
    return DatabaseConfig(
        type=base.type,
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/") or base.database,
        user=unquote(parsed.username) if parsed.username else base.user,
        password=unquote(parsed.password) if parsed.password else "",
        pool=base.pool,
        timescale=base.timescale,
    )


# ── Event handlers (extracted for testability) ────────────────


async def _on_bar_close(
    event: Any,
    repo: Any,
    signal_engine: Any | None = None,
    db: Any | None = None,
) -> None:
    """Handle BAR_CLOSE: write bar to PG, save score + summary + NOTIFY on daily close."""
    try:
        await repo.insert_bar(
            event.symbol,
            event.timeframe,
            event.open,
            event.high,
            event.low,
            event.close,
            event.volume,
            event.timestamp,
        )
    except Exception:
        logger.warning("Bar insert failed for %s/%s", event.symbol, event.timeframe, exc_info=True)

    if event.timeframe == "1d" and signal_engine:
        try:
            ie = signal_engine.indicator_engine
            all_states = ie.get_all_indicator_states(timeframe="1d")
            for (sym, _tf, ind), state in all_states.items():
                if ind == "regime_detector" and sym == event.symbol and state:
                    await repo.save_score_snapshot(
                        event.symbol,
                        event.timestamp,
                        state.get("composite_score", 0.0),
                        state.get("trend_state", "unknown"),
                        state.get("regime", "R0"),
                    )

                    if db is not None:
                        # Upsert summary row (Xenon's single-source-of-truth view)
                        import json as _json

                        try:
                            await db.execute(
                                """INSERT INTO summary
                                       (symbol, regime, regime_name, confidence,
                                        composite_score, component_states, updated_at)
                                   VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                                   ON CONFLICT (symbol) DO UPDATE SET
                                       regime=$2, regime_name=$3, confidence=$4,
                                       composite_score=$5, component_states=$6::jsonb,
                                       updated_at=NOW()""",
                                event.symbol,
                                state.get("regime", "R0"),
                                state.get("regime_name", "Unknown"),
                                state.get("confidence", 50.0),
                                state.get("composite_score", 0.0),
                                _json.dumps(state.get("component_states", {})),
                            )
                        except Exception:
                            logger.debug(
                                "Summary upsert failed for %s",
                                event.symbol,
                                exc_info=True,
                            )

                        await db.notify(
                            "apex_regime",
                            {
                                "symbol": event.symbol,
                                "regime": state.get("regime", "R0"),
                                "score": state.get("composite_score", 0.0),
                                "trend_state": state.get("trend_state", "unknown"),
                                "ts": str(event.timestamp),
                            },
                        )
                    break
        except Exception:
            logger.debug("Score snapshot failed for %s", event.symbol, exc_info=True)


async def _on_trading_signal(event: Any, repo: Any, db: Any | None = None) -> None:
    """Handle TRADING_SIGNAL: write signal to PG, fire apex_signal NOTIFY."""
    direction_map = {
        "bullish": "bullish",
        "bearish": "bearish",
        "long": "bullish",
        "short": "bearish",
    }
    direction = direction_map.get(event.direction, "neutral")
    rule = getattr(event, "trigger_rule", None) or event.indicator
    try:
        await repo.insert_signal(
            symbol=event.symbol,
            rule=rule,
            direction=direction,
            strength=event.strength,
            timeframe=event.timeframe,
            indicator=event.indicator,
            ts=event.timestamp,
        )
    except Exception:
        logger.warning("Signal insert failed for %s", event.symbol, exc_info=True)
        return

    if db is not None:
        await db.notify(
            "apex_signal",
            {
                "symbol": event.symbol,
                "rule": rule,
                "direction": direction,
                "strength": event.strength,
                "timeframe": event.timeframe,
                "indicator": event.indicator,
                "ts": str(event.timestamp),
            },
        )


# ── Main daemon ───────────────────────────────────────────────


async def run_signal_service() -> None:
    """Main entry point for the signal service daemon."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    from config.config_manager import ConfigManager
    from src.domain.events.event_types import EventType
    from src.domain.events.priority_event_bus import PriorityEventBus
    from src.domain.signals.signal_engine import SignalEngine
    from src.infrastructure.persistence.database import Database
    from src.infrastructure.persistence.pg_repositories import PgRepositories
    from src.infrastructure.persistence.pg_schema import ensure_schema

    cm = ConfigManager(config_dir="config", env="dev")
    config = cm.load()
    timeframes = ["30m", "1h", "4h", "1d"]

    assert config.database is not None, "database config required in config/base.yaml"

    # APEX_PG_URL env var overrides config (unifies with _pg_publish.py / Xenon).
    pg_url = os.environ.get("APEX_PG_URL")
    db_config = _db_config_from_env(pg_url, config.database) if pg_url else config.database
    db = Database(db_config)
    await db.connect()
    await ensure_schema(db)
    repo = PgRepositories(db)
    logger.info("PostgreSQL connected and schema ensured")

    event_bus = PriorityEventBus(fast_lane_max_size=50000)
    await event_bus.start()

    signal_engine = SignalEngine(
        event_bus=event_bus,
        timeframes=timeframes,
        max_workers=4,
    )
    signal_engine.start()
    logger.info("SignalEngine started (timeframes=%s)", timeframes)

    loop = asyncio.get_running_loop()

    def _bar_handler(event: Any) -> None:
        asyncio.run_coroutine_threadsafe(
            _on_bar_close(event, repo, signal_engine=signal_engine, db=db),
            loop,
        )

    def _signal_handler(event: Any) -> None:
        asyncio.run_coroutine_threadsafe(
            _on_trading_signal(event, repo, db=db),
            loop,
        )

    event_bus.subscribe(EventType.BAR_CLOSE, _bar_handler)
    event_bus.subscribe(EventType.TRADING_SIGNAL, _signal_handler)

    ib_adapter = None
    try:
        from src.infrastructure.adapters.ib.live_adapter import IbLiveAdapter

        ib_port = config.ibkr.port if hasattr(config, "ibkr") and config.ibkr else 4001
        ib_host = config.ibkr.host if hasattr(config, "ibkr") and config.ibkr else "127.0.0.1"
        ib_adapter = IbLiveAdapter(host=ib_host, port=ib_port, client_id=20)
        await ib_adapter.connect()
        ib_adapter.set_quote_callback(signal_engine.on_tick)
        logger.info("IB Gateway connected (%s:%d)", ib_host, ib_port)
    except Exception:
        logger.warning("IB Gateway not available — running without live ticks", exc_info=True)

    logger.info("Signal service running — Ctrl+C to stop")

    stop_event = asyncio.Event()

    def _handle_signal(sig: int, frame: Any) -> None:
        logger.info("Received signal %d, shutting down...", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    await stop_event.wait()

    event_bus.unsubscribe(EventType.BAR_CLOSE, _bar_handler)
    event_bus.unsubscribe(EventType.TRADING_SIGNAL, _signal_handler)

    if ib_adapter:
        try:
            await ib_adapter.disconnect()
        except Exception:
            pass

    signal_engine.stop()
    await event_bus.stop()
    await db.close()
    logger.info("Signal service stopped")


def main() -> None:
    asyncio.run(run_signal_service())


if __name__ == "__main__":
    main()
