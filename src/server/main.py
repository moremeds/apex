"""APEX Live Dashboard — FastAPI entrypoint with full lifecycle orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.domain.events.domain_events import QuoteTick
from src.domain.services.regime.universe_loader import load_universe
from src.server.config import load_server_config
from src.server.persistence import ServerPersistence
from src.server.pipeline import ServerPipeline
from src.server.routes.advisor import create_advisor_router
from src.server.routes.monitor import create_monitor_router
from src.server.routes.screeners import create_screeners_router
from src.server.routes.symbols import create_symbols_router
from src.server.routes.ws import create_ws_router
from src.server.ws_hub import WebSocketHub

logger = logging.getLogger(__name__)

_start_time = time.monotonic()


def _tick_to_dict(tick: QuoteTick) -> dict[str, Any]:
    """Convert QuoteTick to JSON-serializable dict for WS broadcast."""
    return {
        "last": tick.last,
        "bid": tick.bid,
        "ask": tick.ask,
        "volume": tick.volume,
        "ts": tick.timestamp.isoformat() if tick.timestamp else None,
    }


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle — startup and shutdown orchestration."""
    # Ensure app loggers are visible (uvicorn only configures its own loggers)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    config = load_server_config()

    # ── Core components ────────────────────────────────────
    hub = WebSocketHub()
    persistence = ServerPersistence(config.duckdb_path)
    pipeline = ServerPipeline(hub=hub, timeframes=config.timeframes, config=config)

    # ── Longbridge adapter (optional — only if enabled + env vars present) ──
    quote_adapter = None
    historical_adapter = None
    r2_client = None

    lb_enabled = config.providers.get("longbridge")
    has_lb_creds = bool(os.environ.get("LONGPORT_APP_KEY"))

    # ── R2 client (optional — tries env vars then config/secrets.yaml) ──
    try:
        from src.infrastructure.adapters.r2.client import R2Client

        r2_client = R2Client()
        logger.info("R2 client initialized")
    except (ValueError, ImportError):
        logger.info("R2 client not available (no credentials)")
    except Exception:
        logger.exception("Failed to initialize R2 client")

    # ── Load symbol universe (universe.yaml + R2 screeners) ──
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
            # Cap total but always keep the full base universe
            all_symbols = sorted(all_symbols)[: config.max_symbols]
        except Exception:
            logger.exception("Failed to load screener symbols from R2")

    # ── Longbridge connection (deferred — SDK takes ~13s to connect) ──
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
            loop = asyncio.get_running_loop()

            def on_tick(tick: Any) -> None:
                pipeline.on_tick(tick)
                persistence.buffer_tick(tick)
                coro = hub.broadcast_quote(tick.symbol, _tick_to_dict(tick))
                asyncio.run_coroutine_threadsafe(coro, loop)

            qa.set_quote_callback(on_tick)

            # Subscribe all universe symbols
            await qa.subscribe_quotes(all_symbols)

            # Update refs
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

    # ── Start pipeline ──
    await pipeline.start()

    # ── R2 history bootstrap (warm up indicators from historical data) ──
    async def _bootstrap_pipeline() -> None:
        """Warm up indicators from R2 history after Longbridge connects."""
        if lb_task:
            try:
                await lb_task
            except Exception:
                pass  # Longbridge failure doesn't block warmup

        if r2_client is None:
            logger.info("Skipping bootstrap — no R2 client")
            return

        import time as _time

        t0 = _time.monotonic()
        warmed = 0
        warmup_tfs = [tf for tf in config.timeframes if tf in {"1d", "1h", "4h"}]

        for tf in warmup_tfs:
            for symbol in all_symbols:
                try:
                    key = f"parquet/historical/{tf}/{symbol}.parquet"
                    df = await asyncio.to_thread(r2_client.get_parquet, key)
                    if df is not None and not df.empty:
                        if len(df) > 500:
                            df = df.tail(500)
                        bar_dicts = []
                        for _, row in df.iterrows():
                            ts = row.get("timestamp") or row.get("date")
                            if hasattr(ts, "to_pydatetime"):
                                ts = ts.to_pydatetime()
                            # Strip timezone to match yfinance convention
                            # (regime detector benchmark data is tz-naive)
                            if ts is not None and hasattr(ts, "tzinfo") and ts.tzinfo is not None:
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
                        n = pipeline._indicator_engine.inject_historical_bars(symbol, tf, bar_dicts)
                        if n > 0:
                            warmed += 1
                except Exception as e:
                    logger.debug("Bootstrap skip %s/%s: %s", symbol, tf, e)

        # Compute indicators on injected history
        for tf in warmup_tfs:
            for symbol in all_symbols:
                try:
                    await pipeline._indicator_engine.compute_on_history(symbol, tf)
                except Exception:
                    pass

        logger.info("Bootstrap: %d symbol/tf pairs warmed in %.1fs", warmed, _time.monotonic() - t0)

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

        # Build sector map from universe
        sector_map: dict[str, str] = {}
        if universe:
            sector_names = getattr(universe, "sector_names", {})
            for sym in all_symbols:
                sector_etf = universe.get_sector_for_symbol(sym)
                if sector_etf and sector_etf in sector_names:
                    sector_map[sym] = sector_names[sector_etf]
                elif sector_etf:
                    sector_map[sym] = sector_etf

        def _get_vix() -> tuple[pd.Series | None, pd.Series | None]:
            vd = getattr(app.state, "vix_data", {})
            return vd.get("^VIX"), vd.get("^VIX3M")

        def _get_underlying(sym: str) -> pd.Series | None:
            # Prefer Yahoo-fetched data (longer history for VRP z-score)
            vd = getattr(app.state, "vix_data", {})
            if sym in vd:
                return vd[sym]
            # Fall back to indicator engine bar history
            engine = pipeline._indicator_engine
            bar_deque = engine._history.get((sym, "1d"))
            if bar_deque and len(bar_deque) > 0:
                closes = pd.Series(
                    [b["close"] for b in bar_deque],
                    index=pd.DatetimeIndex([b["timestamp"] for b in bar_deque]),
                )
                return closes
            return None

        advisor_svc = AdvisorService(
            get_regime_states=pipeline.get_regime_states,
            get_indicator_states=pipeline._indicator_engine.get_all_indicator_states,
            get_vix_data=_get_vix,
            get_underlying_close=_get_underlying,
            get_recent_signals=pipeline.get_recent_signals,
            etf_symbols=etf_syms,
            universe_symbols=all_symbols,
            sector_map=sector_map,
        )
        app.state.advisor_service = advisor_svc
        pipeline.set_advisor_service(advisor_svc)
        logger.info("AdvisorService created and wired to pipeline")

    bootstrap_task = asyncio.create_task(_bootstrap_pipeline())

    # ── Periodic flush task ──
    flush_task = asyncio.create_task(_periodic_flush(persistence, config.r2_flush_interval_sec))

    # ── Store refs on app.state for routes ──
    app.state.hub = hub
    app.state.pipeline = pipeline
    app.state.persistence = persistence
    app.state.quote_adapter = quote_adapter
    app.state.historical_adapter = historical_adapter
    app.state.r2_client = r2_client

    logger.info(
        "APEX Live Dashboard started (symbols=%d, timeframes=%s, longbridge=%s, r2=%s)",
        len(all_symbols),
        config.timeframes,
        quote_adapter is not None,
        r2_client is not None,
    )

    yield

    # ── Shutdown ────────────────────────────────────────────
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

    await pipeline.stop()

    if quote_adapter is not None:
        await quote_adapter.disconnect()

    persistence.close()
    logger.info("APEX Live Dashboard stopped")


def create_app(config_path: str = "config/server.yaml") -> FastAPI:
    """Create and configure the FastAPI application.

    For testing, call with no lifespan and inject mocks.
    For production, the lifespan handles all wiring.
    """
    app = FastAPI(
        title="APEX Live Dashboard",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health endpoint (always available, reads from app.state)
    @app.get("/api/health")
    async def health() -> dict:
        hub = getattr(app.state, "hub", None)
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count if hub else 0,
        }

    # Routes — dependencies resolved lazily from app.state
    # This allows the lifespan to wire real adapters after startup
    hub_placeholder = WebSocketHub()  # used before lifespan runs (e.g., tests)
    app.include_router(create_ws_router(hub_placeholder))
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router())
    app.include_router(create_monitor_router(hub=hub_placeholder))
    app.include_router(create_advisor_router())

    # SPA catch-all: serve index.html for any non-API, non-WS GET request.
    # This enables direct navigation to /signals, /monitor, etc.
    dist_path = Path("web/dist")
    if dist_path.is_dir():
        index_html = dist_path / "index.html"

        @app.get("/{full_path:path}")
        async def spa_catchall(request: Request, full_path: str) -> FileResponse:
            # Serve actual static files (JS, CSS, images) if they exist
            static_file = dist_path / full_path
            if static_file.is_file() and not full_path.startswith("api/"):
                return FileResponse(static_file)
            # Otherwise serve index.html for client-side routing
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

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "uptime": round(time.monotonic() - _start_time, 1),
            "ws_clients": hub.client_count,
        }

    app.include_router(create_ws_router(hub))
    app.include_router(create_symbols_router())
    app.include_router(create_screeners_router())
    app.include_router(create_monitor_router(hub=hub))
    app.include_router(create_advisor_router())

    dist_path = Path("web/dist")
    if dist_path.is_dir():
        index_html = dist_path / "index.html"

        @app.get("/{full_path:path}")
        async def spa_catchall_test(request: Request, full_path: str) -> FileResponse:
            static_file = dist_path / full_path
            if static_file.is_file() and not full_path.startswith("api/"):
                return FileResponse(static_file)
            return FileResponse(index_html)

    return app


# Module-level app for `uvicorn src.server.main:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8080, reload=True)
