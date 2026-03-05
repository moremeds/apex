"""APEX Live Dashboard — FastAPI entrypoint with full lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.services.regime.universe_loader import load_universe
from src.domain.signals.signal_engine import SignalEngine
from src.server.config import load_server_config
from src.server.persistence import ServerPersistence
from src.server.routes.advisor import create_advisor_router
from src.server.routes.jobs import create_jobs_router
from src.server.routes.monitor import create_monitor_router
from src.server.routes.portfolio import create_portfolio_router
from src.server.routes.screeners import _CachedProxy, create_screeners_router
from src.server.routes.symbols import create_symbols_router
from src.server.routes.ws import create_ws_router
from src.server.signal_helpers import tick_to_dict as _tick_to_dict
from src.server.web_bridge import WebBridge
from src.server.ws_hub import WebSocketHub

logger = logging.getLogger(__name__)

_start_time = time.monotonic()


async def _periodic_flush(persistence: ServerPersistence, interval_sec: int) -> None:
    """Periodically flush buffered ticks to DuckDB."""
    while True:
        await asyncio.sleep(interval_sec)
        try:
            count = persistence.flush_to_duckdb()
            if count > 0:
                logger.info("Periodic flush: %d ticks → DuckDB", count)
        except Exception:
            logger.exception("Periodic flush failed")


def _get_prev_close(engine: Any, symbol: str) -> float | None:
    """Get previous trading day's close from 1d bar history.

    If the latest 1d bar is today's bar, prev_close is bar[-2].
    Otherwise the latest bar IS the previous close (weekend/holiday).
    """
    bar_deque = engine.get_history(symbol, "1d")
    if not bar_deque or len(bar_deque) < 2:
        return None
    today = date.today()
    latest = bar_deque[-1]
    latest_ts = latest.get("timestamp")
    latest_date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    if latest_date == today:
        return float(bar_deque[-2]["close"])
    return float(latest["close"])


async def _fill_data_gap(engine: Any, symbols: list[str], tf: str = "1d") -> int:
    """Fill gap between R2-bootstrapped history and today using FMP/Yahoo waterfall."""
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
                load_bars, [sym], tf, (today - latest_date).days + 1, today
            )
            df = result.get(sym)
            if df is not None and not df.empty:
                df = df[df.index > pd.Timestamp(latest_ts)]
                if not df.empty:
                    bar_dicts = []
                    for idx, row in df.iterrows():
                        ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
                        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                            ts = ts.replace(tzinfo=None)
                        bar_dicts.append(
                            {
                                "timestamp": ts,
                                "open": float(row.get("open", 0)),
                                "high": float(row.get("high", 0)),
                                "low": float(row.get("low", 0)),
                                "close": float(row.get("close", 0)),
                                "volume": int(row.get("volume", 0)) if row.get("volume") else 0,
                            }
                        )
                    n = engine.inject_historical_bars(sym, tf, bar_dicts)
                    if n > 0:
                        filled += n
        except Exception:
            logger.warning("Gap fill failed for %s/%s", sym, tf, exc_info=True)

    return filled


def _compute_and_persist_summary(
    signal_engine: SignalEngine, persistence: ServerPersistence, universe: Any
) -> dict | None:
    """Build summary from signal engine indicator state and persist to DuckDB."""
    ie = signal_engine.indicator_engine
    all_states = ie.get_all_indicator_states(timeframe="1d")
    regime_data: dict[str, dict] = {}
    for (sym, tf, ind), state in all_states.items():
        if ind == "regime_detector" and state:
            regime_data[sym] = state
    if not regime_data:
        logger.info("No regime data available for summary computation")
        return None

    tickers = []
    for symbol, state in regime_data.items():
        bar_deque = ie.get_history(symbol, "1d")
        close = bar_deque[-1]["close"] if bar_deque else 0.0
        prev_close = _get_prev_close(ie, symbol) or 0.0
        daily_change_pct = 0.0
        if prev_close > 0:
            daily_change_pct = round(((close - prev_close) / prev_close) * 100, 2)

        sector = ""
        if universe:
            sector_etf = (
                universe.get_sector_for_symbol(symbol)
                if hasattr(universe, "get_sector_for_symbol")
                else ""
            )
            if sector_etf:
                sector_names = getattr(universe, "sector_names", {})
                sector = sector_names.get(sector_etf, sector_etf)

        # Include component_states for Signals page RegimeSection
        comp_states: dict = {}
        cs = state.get("component_states")
        if cs and hasattr(cs, "to_dict"):
            comp_states = cs.to_dict()
        elif isinstance(cs, dict):
            comp_states = cs

        tickers.append(
            {
                "symbol": symbol,
                "close": close,
                "prev_close": prev_close,
                "daily_change_pct": daily_change_pct,
                "regime": state.get("regime", "R1"),
                "regime_name": state.get("regime_name", "Unknown"),
                "confidence": state.get("confidence", 50),
                "composite_score_avg": round(state.get("composite_score", 0), 1),
                "sector": sector,
                "component_states": comp_states,
            }
        )

    summary = {"tickers": tickers, "generated_at": datetime.now().isoformat()}
    persistence.save_summary(summary)
    logger.info("Computed and persisted summary for %d tickers", len(tickers))

    # Seed score_history table so /api/score-history returns data
    now = datetime.now(timezone.utc)
    for ticker in tickers:
        persistence.save_score_snapshot(
            symbol=ticker["symbol"],
            ts=now,
            score=ticker.get("composite_score_avg", 0),
            trend_state=ticker.get("regime_name", "Unknown"),
            regime=ticker.get("regime", "R1"),
        )
    logger.info("Saved score snapshots for %d tickers", len(tickers))

    return summary


async def _init_app_container(
    config: Any,
    event_bus: PriorityEventBus | None = None,
    signal_engine: SignalEngine | None = None,
) -> Any:
    """Initialize AppContainer from base.yaml config (same as TUI).

    Accepts optional shared event_bus and signal_engine for server mode.
    Returns the container or None if initialization fails.
    """
    try:
        from config.config_manager import ConfigManager
        from src.application.bootstrap import AppContainer
        from src.utils import StructuredLogger

        base_config_path = config.base_config_path
        if not Path(base_config_path).exists():
            logger.warning("Base config not found at %s — skipping AppContainer", base_config_path)
            return None

        config_dir = str(Path(base_config_path).parent)
        cm = ConfigManager(config_dir=config_dir, env="dev")
        app_config = cm.load()

        container = AppContainer(
            config=app_config,
            env="server",
            metrics_port=0,
            no_dashboard=True,
            shared_event_bus=event_bus,
            shared_signal_engine=signal_engine,
        )

        slog = StructuredLogger(logger)
        await container.initialize(slog)
        await container.start()

        logger.info("AppContainer initialized and started (brokers wiring complete)")
        return container
    except Exception:
        logger.exception("Failed to initialize AppContainer — portfolio features disabled")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle — startup and shutdown orchestration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    config = load_server_config()

    # ── 1. Core components ────────────────────────────────────
    event_bus = PriorityEventBus(fast_lane_max_size=50000)
    hub = WebSocketHub()
    persistence = ServerPersistence(config.duckdb_path)

    # ── 2. Start event bus ────────────────────────────────────
    await event_bus.start()

    # ── 3. Create and start SignalEngine ──────────────────────
    signal_engine = SignalEngine(
        event_bus=event_bus,
        timeframes=config.timeframes,
        max_workers=4,
    )
    signal_engine.start()

    # ── 4. Create and start WebBridge ─────────────────────────
    loop = asyncio.get_running_loop()
    web_bridge = WebBridge(
        event_bus=event_bus,
        signal_engine=signal_engine,
        hub=hub,
        persistence=persistence,
        loop=loop,
    )
    web_bridge.start()

    # ── Longbridge adapter (optional) ─────────────────────────
    quote_adapter = None
    historical_adapter = None
    r2_client = None

    lb_enabled = config.providers.get("longbridge")
    has_lb_creds = bool(os.environ.get("LONGPORT_APP_KEY"))

    # ── R2 client (optional) ──────────────────────────────────
    try:
        from src.infrastructure.adapters.r2.client import R2Client

        r2_client = R2Client()
        logger.info("R2 client initialized")
    except (ValueError, ImportError):
        logger.info("R2 client not available (no credentials)")
    except Exception:
        logger.exception("Failed to initialize R2 client")

    # ── Load symbol universe ──────────────────────────────────
    universe = load_universe()
    all_symbols = list(universe.all_symbols)
    logger.info("Loaded %d symbols from universe.yaml", len(all_symbols))

    if r2_client is not None and config.from_screener:
        try:
            screener_data = r2_client.get_json("screeners.json") or {}
            base_set = set(all_symbols)
            for source, entries in screener_data.items():
                if isinstance(entries, list):
                    for e in entries:
                        sym = e.get("symbol") if isinstance(e, dict) else None
                        if sym and sym not in base_set:
                            all_symbols.append(sym)
                            base_set.add(sym)
            all_symbols = sorted(all_symbols)[: config.max_symbols]
        except Exception:
            logger.exception("Failed to load screener symbols from R2")

    # ── 5. AppContainer (portfolio/risk/broker — optional) ────
    container = await _init_app_container(config, event_bus, signal_engine)
    if container is not None:
        web_bridge.wire_portfolio_events(container)

    # ── Longbridge connection (deferred) ──────────────────────
    async def _connect_longbridge() -> None:
        nonlocal quote_adapter, historical_adapter
        try:
            from src.infrastructure.adapters.longbridge.historical_adapter import (
                LongbridgeHistoricalAdapter,
            )
            from src.infrastructure.adapters.longbridge.quote_adapter import (
                LongbridgeQuoteAdapter,
            )

            qa = LongbridgeQuoteAdapter()
            await qa.connect()

            ha = LongbridgeHistoricalAdapter()
            if hasattr(qa, "_ctx") and qa._ctx:
                ha._ctx = qa._ctx
                ha._connected = True

            # Wire tick callback
            def on_tick(tick: Any) -> None:
                if not signal_engine.is_started:
                    return  # Engine already stopped
                signal_engine.on_tick(tick)
                persistence.buffer_tick(tick)
                tick_dict = _tick_to_dict(tick)
                try:
                    pc = _get_prev_close(signal_engine.indicator_engine, tick.symbol)
                except (AssertionError, AttributeError):
                    pc = None  # Engine stopping — skip prev_close
                if pc is not None:
                    tick_dict["prev_close"] = pc
                coro = hub.broadcast_quote(tick.symbol, tick_dict)
                asyncio.run_coroutine_threadsafe(coro, loop)

            qa.set_quote_callback(on_tick)
            await qa.subscribe_quotes(all_symbols)

            quote_adapter = qa
            historical_adapter = ha
            app.state.quote_adapter = qa
            app.state.historical_adapter = ha

            logger.info("Longbridge connected — %d symbols streaming", len(all_symbols))
        except Exception:
            logger.exception("Failed to connect Longbridge adapter")

    lb_task = None
    if lb_enabled and lb_enabled.enabled and has_lb_creds:
        lb_task = asyncio.create_task(_connect_longbridge())

    # ── R2 history bootstrap ──────────────────────────────────
    async def _bootstrap_pipeline() -> None:
        """Warm up indicators from R2 history after Longbridge connects."""
        if lb_task:
            try:
                await lb_task
            except Exception:
                pass

        if r2_client is None:
            logger.info("Skipping bootstrap — no R2 client")
            return

        import time as _time

        t0 = _time.monotonic()
        warmed = 0
        warmup_tfs = config.timeframes
        ie = signal_engine.indicator_engine

        for tf in warmup_tfs:
            for symbol in all_symbols:
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
                            if ts is not None and hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                                ts = ts.replace(tzinfo=None)
                            bar_dicts.append(
                                {
                                    "timestamp": ts,
                                    "open": float(row.get("open", 0)),
                                    "high": float(row.get("high", 0)),
                                    "low": float(row.get("low", 0)),
                                    "close": float(row.get("close", 0)),
                                    "volume": (
                                        int(row.get("volume", 0)) if row.get("volume") else 0
                                    ),
                                }
                            )
                        n = ie.inject_historical_bars(symbol, tf, bar_dicts)
                        if n > 0:
                            warmed += 1
                            persistence.bulk_insert_bars(symbol, tf, bar_dicts)
                except Exception as e:
                    logger.debug("Bootstrap skip %s/%s: %s", symbol, tf, e)

        # Compute indicators on injected history
        for tf in warmup_tfs:
            for symbol in all_symbols:
                try:
                    await ie.compute_on_history(symbol, tf)
                except Exception:
                    logger.debug("Compute on history failed for %s/%s", symbol, tf, exc_info=True)

        logger.info("Bootstrap: %d symbol/tf pairs warmed in %.1fs", warmed, _time.monotonic() - t0)

        # ── Fill gap between R2 EOD and today via FMP/Yahoo ──
        try:
            total_gap = 0
            for gap_tf in config.timeframes:
                gap_count = await _fill_data_gap(
                    engine=ie,
                    symbols=all_symbols,
                    tf=gap_tf,
                )
                total_gap += gap_count
                if gap_count > 0:
                    logger.info(
                        "Gap fill [%s]: %d new bars injected from FMP/Yahoo", gap_tf, gap_count
                    )
            if total_gap > 0:
                recomputed = 0
                for gap_tf in config.timeframes:
                    for symbol in all_symbols:
                        try:
                            await ie.compute_on_history(symbol, gap_tf)
                            recomputed += 1
                        except Exception:
                            logger.debug(
                                "Gap fill recompute failed for %s/%s",
                                symbol,
                                gap_tf,
                                exc_info=True,
                            )
                logger.info("Gap fill recompute: %d symbol/tf pairs updated", recomputed)
        except Exception:
            logger.exception("Gap fill failed")

        # ── Compute and persist summary ──
        _compute_and_persist_summary(signal_engine, persistence, universe)

        # ── Fetch VIX + VIX3M for advisor ──
        from datetime import datetime, timedelta, timezone

        import pandas as pd

        vix_data: dict[str, Any] = {}
        for vix_sym in ("^VIX", "^VIX3M", "SPY"):
            try:
                from src.infrastructure.adapters.yahoo.historical_adapter import (
                    YahooHistoricalAdapter,
                )

                yahoo = YahooHistoricalAdapter()
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=400)
                bars = await yahoo.fetch_bars(vix_sym, "1d", start, end)
                if bars:
                    vix_data[vix_sym] = pd.Series(
                        [b.close for b in bars],
                        index=pd.DatetimeIndex([b.timestamp for b in bars]),
                    )
                    logger.info("Fetched %d bars for %s", len(bars), vix_sym)
            except Exception:
                logger.debug("Failed to fetch %s for advisor", vix_sym)
        app.state.vix_data = vix_data

        # ── Create AdvisorService ──
        from src.domain.services.advisor.advisor_service import AdvisorService

        etf_syms = ["SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT"]

        sector_map: dict[str, str] = {}
        if universe:
            sector_names = getattr(universe, "sector_names", {})
            for sym in all_symbols:
                sector_etf = universe.get_sector_for_symbol(sym)
                if sector_etf and sector_etf in sector_names:
                    sector_map[sym] = sector_names[sector_etf]
                elif sector_etf:
                    sector_map[sym] = sector_etf

        def _get_vix() -> tuple[Any, Any]:
            vd = getattr(app.state, "vix_data", {})
            return vd.get("^VIX"), vd.get("^VIX3M")

        def _get_underlying(sym: str) -> Any:
            vd = getattr(app.state, "vix_data", {})
            if sym in vd:
                return vd[sym]
            bar_list = ie.get_history(sym, "1d")
            if bar_list and len(bar_list) > 0:
                closes = pd.Series(
                    [b["close"] for b in bar_list],
                    index=pd.DatetimeIndex([b["timestamp"] for b in bar_list]),
                )
                return closes
            return None

        advisor_svc = AdvisorService(
            get_regime_states=signal_engine.get_regime_states,
            get_indicator_states=ie.get_all_indicator_states,
            get_vix_data=_get_vix,
            get_underlying_close=_get_underlying,
            get_recent_signals=web_bridge.get_recent_signals,
            etf_symbols=etf_syms,
            universe_symbols=all_symbols,
            sector_map=sector_map,
        )
        app.state.advisor_service = advisor_svc
        web_bridge.set_advisor_service(advisor_svc)
        logger.info("AdvisorService created and wired to WebBridge")

    bootstrap_task = asyncio.create_task(_bootstrap_pipeline())

    # ── Periodic flush task ──
    flush_task = asyncio.create_task(_periodic_flush(persistence, config.r2_flush_interval_sec))

    # ── Wire R2 into the screener cache ──
    if hasattr(app.state, "screener_cache"):
        app.state.screener_cache._r2 = r2_client

    # ── Store refs on app.state for routes ──
    app.state.hub = hub
    app.state.signal_engine = signal_engine
    app.state.web_bridge = web_bridge
    app.state.persistence = persistence
    app.state.quote_adapter = quote_adapter
    app.state.historical_adapter = historical_adapter
    app.state.r2_client = r2_client
    app.state.container = container

    logger.info(
        "APEX Live Dashboard started (symbols=%d, timeframes=%s, longbridge=%s, r2=%s, portfolio=%s)",
        len(all_symbols),
        config.timeframes,
        quote_adapter is not None,
        r2_client is not None,
        container is not None,
    )

    yield

    # ── Shutdown (reverse order) ──────────────────────────────
    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass

    if bootstrap_task and not bootstrap_task.done():
        bootstrap_task.cancel()
        try:
            await bootstrap_task
        except asyncio.CancelledError:
            pass

    if lb_task and not lb_task.done():
        lb_task.cancel()
        try:
            await lb_task
        except asyncio.CancelledError:
            pass

    # 1. Longbridge (stop tick source FIRST to prevent post-stop callbacks)
    if quote_adapter is not None:
        try:
            await quote_adapter.disconnect()
            logger.info("Longbridge disconnected")
        except Exception:
            logger.exception("Longbridge disconnect error")

    # 2. SignalEngine (flush final bars while WebBridge still subscribed)
    signal_engine.stop()

    # 3. WebBridge (now safe to unsubscribe — no more events coming)
    web_bridge.stop()

    # 4. AppContainer
    if container is not None:
        try:
            await container.cleanup()
            logger.info("AppContainer cleaned up")
        except Exception:
            logger.exception("AppContainer cleanup error")

    # 5. Event bus
    await event_bus.stop()

    # 6. Persistence
    persistence.close()
    logger.info("APEX Live Dashboard stopped")


def create_app(config_path: str = "config/server.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="APEX Live Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:
        hub = getattr(app.state, "hub", None)
        container = getattr(app.state, "container", None)
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count if hub else 0,
            "portfolio_enabled": container is not None,
        }

    screener_cache = _CachedProxy(r2_client=None)
    app.state.screener_cache = screener_cache

    app.include_router(create_ws_router())
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router(proxy=screener_cache))
    app.include_router(create_monitor_router())
    app.include_router(create_advisor_router())
    app.include_router(create_jobs_router())
    app.include_router(create_portfolio_router())

    dist_path = Path("web/dist")
    if dist_path.is_dir():
        index_html = dist_path / "index.html"

        @app.get("/{full_path:path}", response_model=None)
        async def spa_catchall(request: Request, full_path: str) -> Response:
            static_file = dist_path / full_path
            if static_file.is_file() and not full_path.startswith("api/"):
                return FileResponse(static_file)
            if "." in full_path.split("/")[-1]:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return FileResponse(index_html)

    return app


def create_test_app() -> FastAPI:
    """Create a minimal app without lifespan for unit testing."""
    app = FastAPI(title="APEX Live Dashboard", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    hub = WebSocketHub()
    app.state.hub = hub
    app.state.container = None

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count,
            "portfolio_enabled": False,
        }

    screener_cache = _CachedProxy(r2_client=None)
    app.state.screener_cache = screener_cache

    app.include_router(create_ws_router())
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router(proxy=screener_cache))
    app.include_router(create_monitor_router())
    app.include_router(create_advisor_router())
    app.include_router(create_jobs_router())
    app.include_router(create_portfolio_router())

    dist_path = Path("web/dist")
    if dist_path.is_dir():
        index_html = dist_path / "index.html"

        @app.get("/{full_path:path}", response_model=None)
        async def spa_catchall_test(request: Request, full_path: str) -> Response:
            static_file = dist_path / full_path
            if static_file.is_file() and not full_path.startswith("api/"):
                return FileResponse(static_file)
            if "." in full_path.split("/")[-1]:
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return FileResponse(index_html)

    return app


# Module-level app for `uvicorn src.server.main:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "src.server.main:app",
        host="0.0.0.0",  # nosec B104
        port=8080,
        reload=True,
        ws_ping_interval=60,
        ws_ping_timeout=60,
    )
