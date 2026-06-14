"""APEX API Server — FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, cast

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.jobs.manager import JobManager
from src.api.routes.backtest import router as backtest_router
from src.api.routes.health import router as health_router
from src.api.routes.regime import router as regime_router
from src.api.routes.screener import router as screener_router
from src.api.routes.strategy import router as strategy_router

if TYPE_CHECKING:
    from src.domain.interfaces.event_bus import EventBus

logger = logging.getLogger(__name__)

# xenon's ib_realtime WS server defaults to port 8765 (DEFAULT_IB_REALTIME_PORT on
# the xenon side). Bake the same default in so apex connects to a local xenon out
# of the box; override with APEX_XENON_WS_URL.
DEFAULT_XENON_WS_URL = "ws://127.0.0.1:8765"


def _build_database(pg_url: str) -> Any:
    """Build a Database from a libpq DSN (APEX_PG_URL), with default pool/timescale.

    Parsed directly here (rather than via config/base.yaml) so the API server can
    point at any Postgres via a single env var.
    """
    from urllib.parse import unquote, urlparse

    from config.models import DatabaseConfig
    from src.infrastructure.persistence.database import Database

    parsed = urlparse(pg_url)
    cfg = DatabaseConfig(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/").lstrip("/") or "apex_signals",
        user=unquote(parsed.username) if parsed.username else "apex_app",
        password=unquote(parsed.password) if parsed.password else "",
    )
    return Database(cfg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the asyncpg pool on startup, close it on shutdown."""
    pg_url = os.environ.get("APEX_PG_URL")
    if pg_url:
        try:
            app.state.pg_pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=5)
            app.state.pg_connected = True
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.warning("PG connection failed: %s", e)
            app.state.pg_pool = None
            app.state.pg_connected = False
    else:
        app.state.pg_pool = None
        app.state.pg_connected = False
        logger.info("No APEX_PG_URL set — running without PG")

    # Streaming TA signal surface (Phase 3). Guarded with `getattr(..., None) is None`
    # so tests can pre-inject fakes that the lifespan must not clobber.
    from src.api.ws.hub import SignalHub

    if getattr(app.state, "signal_hub", None) is None:
        app.state.signal_hub = SignalHub()
    # Signal repository: powers the WS initial snapshot, the REST backfill, AND
    # persistence of fired signals (passed to TASignalService below). Built when
    # APEX_PG_URL is set; guarded so tests can pre-inject a fake repo. A failed
    # connection degrades gracefully (live WS push still works without PG).
    if getattr(app.state, "signal_repo", None) is None:
        app.state.signal_repo = None
        repo_pg_url = os.environ.get("APEX_PG_URL")
        if repo_pg_url:
            from src.infrastructure.persistence.repositories.ta_signal_repository import (
                TASignalRepository,
            )

            try:
                signal_db = _build_database(repo_pg_url)
                await signal_db.connect()
                app.state.signal_db = signal_db
                app.state.signal_repo = TASignalRepository(signal_db)
                logger.info("Signal repository connected (snapshot + REST backfill enabled)")
            except Exception as e:  # pragma: no cover - depends on a live PG
                logger.warning("Signal repository unavailable, snapshot/REST disabled: %s", e)
                app.state.signal_repo = None
                app.state.signal_db = None
    # NOTE (env-gated): the real SubscriptionManager (livewire provider +
    # TASignalService) and the SignalEmitter(bus) are constructed here when their
    # dependencies are present. Left unbuilt in environments without livewire/PG;
    # the WS route requires app.state.subscription_manager to be set before use.

    # Full streaming pipeline (env-gated on APEX_LIVEWIRE_ROOT): construct the
    # event bus, TA compute service, signal emitter, and subscription manager so
    # the server itself streams signals. Guarded so tests can pre-inject fakes.
    if getattr(app.state, "subscription_manager", None) is None:
        livewire_root = os.environ.get("APEX_LIVEWIRE_ROOT")
        if livewire_root:
            from pathlib import Path

            from src.api.ws.emitter import SignalEmitter
            from src.application.services.ta_signal_service import TASignalService
            from src.application.subscriptions.manager import SubscriptionManager
            from src.domain.events.priority_event_bus import PriorityEventBus
            from src.infrastructure.adapters.livewire.ohlc_provider import (
                LivewireOhlcProvider,
            )

            timeframes = [
                tf.strip()
                for tf in os.environ.get("APEX_TIMEFRAMES", "1d").split(",")
                if tf.strip()
            ]

            bus = PriorityEventBus()
            await bus.start()
            app.state.event_bus = bus

            service = TASignalService(
                event_bus=cast("EventBus", bus),
                timeframes=timeframes,
                persistence=getattr(app.state, "signal_repo", None),
            )
            await service.start()
            app.state.ta_service = service

            # Fan fired signals out to connected argon WS clients (Phase 3 emitter).
            emitter = SignalEmitter(app.state.signal_hub)
            emitter.subscribe(bus)
            app.state.signal_emitter = emitter

            provider = LivewireOhlcProvider(bronze_root=Path(livewire_root))
            app.state.subscription_manager = SubscriptionManager(
                provider=provider, compute=service, timeframes=timeframes
            )
            logger.info("Streaming TA pipeline started (timeframes=%s)", timeframes)

    # Phase 4 live-in (env-gated): connect to xenon's tick feed when configured.
    # Mirrors the pre-injection guard above so tests can inject a fake/spy client.
    if getattr(app.state, "xenon_client", None) is None:
        xenon_url = os.environ.get("APEX_XENON_WS_URL", DEFAULT_XENON_WS_URL)
        xenon_bus = getattr(app.state, "event_bus", None)
        if xenon_url and xenon_bus is not None:
            from src.infrastructure.adapters.xenon.client import XenonTickClient

            client = XenonTickClient(xenon_url, event_bus=xenon_bus)
            app.state.xenon_client = client
            sm = getattr(app.state, "subscription_manager", None)
            if sm is not None and hasattr(sm, "set_live_feed"):
                sm.set_live_feed(client)
            await client.connect()  # non-blocking: launches the background loop
        else:
            app.state.xenon_client = None

    try:
        yield
    finally:
        xenon_client = getattr(app.state, "xenon_client", None)
        if xenon_client is not None:
            await xenon_client.close()
        ta_service = getattr(app.state, "ta_service", None)
        if ta_service is not None:
            await ta_service.stop()
        event_bus = getattr(app.state, "event_bus", None)
        if event_bus is not None and hasattr(event_bus, "stop"):
            await event_bus.stop()
        signal_db = getattr(app.state, "signal_db", None)
        if signal_db is not None:
            await signal_db.close()
        if app.state.pg_pool is not None:
            await app.state.pg_pool.close()
            logger.info("PostgreSQL pool closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="APEX Signal Server",
        description="Signal generation, backtesting, and strategy management API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.job_manager = JobManager()

    app.include_router(health_router)
    app.include_router(strategy_router)
    app.include_router(regime_router)
    app.include_router(screener_router)
    app.include_router(backtest_router)

    # Streaming TA signal surface (Phase 3): REST pull + WS push to argon.
    from src.api.routes.signals import router as signals_router
    from src.api.ws.signals_ws import router as signals_ws_router

    app.include_router(signals_router)
    app.include_router(signals_ws_router)

    return app


def main() -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    port = int(os.environ.get("APEX_API_PORT", "8322"))
    workers = int(os.environ.get("APEX_API_WORKERS", "1"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    logger.info("Starting APEX API server on port %d", port)

    uvicorn.run(
        "src.api.server:create_app",
        host="0.0.0.0",  # nosec B104 - backend service intentionally listens on all interfaces for Xenon consumers
        port=port,
        workers=workers,
        factory=True,
    )


if __name__ == "__main__":
    main()
