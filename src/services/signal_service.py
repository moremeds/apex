"""Signal Service — long-running daemon that consumes IB ticks and writes to PostgreSQL.

Replaces the FastAPI server lifespan + WebBridge for signal persistence.

Entry point: python -m src.services.signal_service
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Event handlers (extracted for testability) ────────────────


async def _on_bar_close(
    event: Any,
    repo: Any,
    signal_engine: Any | None = None,
) -> None:
    """Handle BAR_CLOSE: write bar to PG, save score on daily close."""
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
        logger.warning(
            "Bar insert failed for %s/%s", event.symbol, event.timeframe, exc_info=True
        )

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
                    break
        except Exception:
            logger.debug("Score snapshot failed for %s", event.symbol, exc_info=True)


async def _on_trading_signal(event: Any, repo: Any) -> None:
    """Handle TRADING_SIGNAL: write signal to PG."""
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


# ── Bootstrap helpers ─────────────────────────────────────────


async def _bootstrap_from_r2(
    signal_engine: Any,
    r2_client: Any,
    symbols: list[str],
    timeframes: list[str],
) -> int:
    """Inject R2 history into SignalEngine and compute indicators."""
    import time as _time

    ie = signal_engine.indicator_engine
    t0 = _time.monotonic()
    warmed = 0

    for tf in timeframes:
        for symbol in symbols:
            try:
                key = f"parquet/historical/{tf}/{symbol}.parquet"
                df = await asyncio.to_thread(r2_client.get_parquet, key)
                if df is not None and not df.empty:
                    if len(df) > 500:
                        df = df.tail(500)
                    bar_dicts = []
                    for idx, row in df.iterrows():
                        ts = row.get("timestamp") or row.get("date") or idx
                        if hasattr(ts, "to_pydatetime"):
                            ts = ts.to_pydatetime()
                        if (
                            ts is not None
                            and hasattr(ts, "tzinfo")
                            and ts.tzinfo is not None
                        ):
                            ts = ts.replace(tzinfo=None)
                        bar_dicts.append(
                            {
                                "timestamp": ts,
                                "open": float(row.get("open", 0)),
                                "high": float(row.get("high", 0)),
                                "low": float(row.get("low", 0)),
                                "close": float(row.get("close", 0)),
                                "volume": (
                                    int(row.get("volume", 0))
                                    if row.get("volume")
                                    else 0
                                ),
                            }
                        )
                    n = ie.inject_historical_bars(symbol, tf, bar_dicts)
                    if n > 0:
                        warmed += 1
            except Exception as e:
                logger.debug("Bootstrap skip %s/%s: %s", symbol, tf, e)

    for tf in timeframes:
        for symbol in symbols:
            try:
                await ie.compute_on_history(symbol, tf)
            except Exception:
                logger.debug(
                    "Compute on history failed for %s/%s", symbol, tf, exc_info=True
                )

    logger.info(
        "Bootstrap: %d symbol/tf pairs warmed in %.1fs", warmed, _time.monotonic() - t0
    )
    return warmed


async def _fill_data_gap(
    engine: Any,
    symbols: list[str],
    tf: str = "1d",
) -> int:
    """Fill gap between R2-bootstrapped history and today using FMP/Yahoo."""
    from src.domain.services.bar_count_calculator import BarCountCalculator
    from src.services.bar_loader import load_bars

    cal = BarCountCalculator("NYSE")
    today = date.today()
    filled = 0

    for sym in symbols:
        bar_deque = engine.get_history(sym, tf)
        if not bar_deque:
            continue
        latest_ts = bar_deque[-1].get("timestamp")
        if latest_ts is None:
            continue
        latest_date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
        missing_days = cal.get_trading_days(latest_date + timedelta(days=1), today)
        if not missing_days:
            continue

        try:
            import pandas as pd

            result = await asyncio.to_thread(
                load_bars,
                [sym],
                tf,
                (today - latest_date).days + 1,
                today,
            )
            df = result.get(sym)
            if df is not None and not df.empty:
                df = df[df.index > pd.Timestamp(latest_ts)]
                if not df.empty:
                    bar_dicts = []
                    for idx, row in df.iterrows():
                        ts = (
                            idx.to_pydatetime()
                            if hasattr(idx, "to_pydatetime")
                            else idx
                        )
                        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                        bar_dicts.append(
                            {
                                "timestamp": ts,
                                "open": float(row.get("open", 0)),
                                "high": float(row.get("high", 0)),
                                "low": float(row.get("low", 0)),
                                "close": float(row.get("close", 0)),
                                "volume": (
                                    int(row.get("volume", 0))
                                    if row.get("volume")
                                    else 0
                                ),
                            }
                        )
                    n = engine.inject_historical_bars(sym, tf, bar_dicts)
                    if n > 0:
                        filled += n
        except Exception:
            logger.warning("Gap fill failed for %s/%s", sym, tf, exc_info=True)

    return filled


# ── Main daemon ───────────────────────────────────────────────


async def run_signal_service() -> None:
    """Main entry point for the signal service daemon."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
    )

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
    db = Database(config.database)
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
            _on_bar_close(event, repo, signal_engine=signal_engine),
            loop,
        )

    def _signal_handler(event: Any) -> None:
        asyncio.run_coroutine_threadsafe(
            _on_trading_signal(event, repo),
            loop,
        )

    event_bus.subscribe(EventType.BAR_CLOSE, _bar_handler)
    event_bus.subscribe(EventType.TRADING_SIGNAL, _signal_handler)

    ib_adapter = None
    try:
        from src.infrastructure.adapters.ib.live_adapter import IbLiveAdapter

        ib_port = config.ibkr.port if hasattr(config, "ibkr") and config.ibkr else 4001
        ib_host = (
            config.ibkr.host if hasattr(config, "ibkr") and config.ibkr else "127.0.0.1"
        )
        ib_adapter = IbLiveAdapter(host=ib_host, port=ib_port, client_id=20)
        await ib_adapter.connect()
        ib_adapter.set_quote_callback(signal_engine.on_tick)
        logger.info("IB Gateway connected (%s:%d)", ib_host, ib_port)
    except Exception:
        logger.warning(
            "IB Gateway not available — running without live ticks", exc_info=True
        )

    from src.domain.services.regime.universe_loader import load_universe

    universe = load_universe()
    all_symbols = sorted(list(universe.all_symbols))
    logger.info("Loaded %d symbols from universe", len(all_symbols))

    r2_client = None
    try:
        from src.infrastructure.adapters.r2.client import R2Client

        r2_client = R2Client()
    except (ValueError, ImportError):
        logger.info("R2 client not available")

    if r2_client:
        await _bootstrap_from_r2(signal_engine, r2_client, all_symbols, timeframes)

        ie = signal_engine.indicator_engine
        for gap_tf in timeframes:
            gap_count = await _fill_data_gap(ie, all_symbols, gap_tf)
            if gap_count > 0:
                logger.info("Gap fill [%s]: %d new bars", gap_tf, gap_count)
                for sym in all_symbols:
                    try:
                        await ie.compute_on_history(sym, gap_tf)
                    except Exception:
                        pass

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
