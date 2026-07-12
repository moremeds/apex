"""APEX API Server — FastAPI application factory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator, cast

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
    from src.infrastructure.adapters.livewire.ohlc_provider import PriceMode
    from src.infrastructure.persistence.database import Database

logger = logging.getLogger(__name__)


def _apex_version() -> str:
    """Real running version, from the installed dist metadata (set from pyproject at
    build). Falls back to the VERSION file for an editable checkout that isn't installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("apex-risk")
    except PackageNotFoundError:  # pragma: no cover - only hit in a bare checkout
        from pathlib import Path

        try:
            return Path(__file__).resolve().parents[2].joinpath("VERSION").read_text().strip()
        except OSError:
            return "unknown"


APEX_VERSION = _apex_version()

# xenon's ib_realtime WS server defaults to port 8765 (DEFAULT_IB_REALTIME_PORT on
# the xenon side). Bake the same default in so apex connects to a local xenon out
# of the box; override with APEX_XENON_WS_URL.
DEFAULT_XENON_WS_URL = "ws://127.0.0.1:8765"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect PG + build the streaming pipeline on startup; tear it down on shutdown."""
    livewire_root = os.environ.get("APEX_LIVEWIRE_ROOT")
    silver_root = os.environ.get("APEX_LIVEWIRE_SILVER_ROOT")
    livewire_price_mode = os.environ.get("APEX_LIVEWIRE_PRICE_MODE", "raw")
    if livewire_price_mode not in ("raw", "adjusted"):
        raise ValueError(f"unsupported Livewire price mode: {livewire_price_mode!r}")
    app.state.livewire_price_mode = livewire_price_mode

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

    # Everything constructed below is torn down in the `finally`, even if startup
    # fails partway, so a half-built pipeline never leaks the pool or bus tasks.
    try:
        # Streaming TA signal surface (Phase 3). Guarded with `getattr(..., None)
        # is None` so tests can pre-inject fakes that the lifespan must not clobber.
        from src.api.ws.hub import SignalHub

        if getattr(app.state, "signal_hub", None) is None:
            app.state.signal_hub = SignalHub()
        # Signal repository over the SAME asyncpg pool (no second pool; the original
        # DSN is preserved by reusing pg_pool). Powers the WS initial snapshot, the
        # REST backfill, AND persistence of fired signals (handed to the service).
        if getattr(app.state, "signal_repo", None) is None:
            app.state.signal_repo = None
            pool = getattr(app.state, "pg_pool", None)
            if pool is not None:
                from src.infrastructure.persistence.repositories.ta_signal_repository import (
                    TASignalRepository,
                )

                # asyncpg.Pool proxies fetch/execute/fetchrow, so it satisfies the
                # repo's Database dependency without opening a second pool.
                app.state.signal_repo = TASignalRepository(cast("Database", pool))
                logger.info("Signal repository ready (snapshot + REST backfill enabled)")

        # Chart read surface (bars + compute-on-read indicators + confluence): expose
        # the bar provider + indicator registry on app.state so /bars, /indicators and
        # /confluence work even without a live subscription. Reused by the pipeline below.
        if getattr(app.state, "ohlc_provider", None) is None:
            app.state.ohlc_provider = None
            if livewire_root:
                from pathlib import Path

                from src.infrastructure.adapters.livewire.ohlc_provider import (
                    LivewireOhlcProvider,
                )

                app.state.ohlc_provider = LivewireOhlcProvider(
                    bronze_root=Path(livewire_root),
                    silver_root=Path(silver_root) if silver_root else None,
                    price_mode=cast("PriceMode", livewire_price_mode),
                )
                logger.info("Bar provider ready (chart read surface enabled)")
        if getattr(app.state, "indicator_registry", None) is None:
            from src.domain.signals.indicators.registry import get_indicator_registry

            app.state.indicator_registry = get_indicator_registry()

        # Full streaming pipeline (env-gated on APEX_LIVEWIRE_ROOT): construct the
        # event bus, TA compute service, signal emitter, and subscription manager so
        # the server itself streams signals. Guarded so tests can pre-inject fakes.
        if getattr(app.state, "subscription_manager", None) is None:
            if livewire_root:
                from src.api.ws.emitter import SignalEmitter
                from src.application.services.ta_signal_service import TASignalService
                from src.application.subscriptions.manager import SubscriptionManager
                from src.domain.events.priority_event_bus import PriorityEventBus

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

                # Fan fired signals out to connected argon WS clients (Phase 3).
                emitter = SignalEmitter(app.state.signal_hub)
                emitter.subscribe(bus)
                app.state.signal_emitter = emitter

                app.state.subscription_manager = SubscriptionManager(
                    provider=app.state.ohlc_provider,
                    compute=service,
                    timeframes=timeframes,
                )
                logger.info("Streaming TA pipeline started (timeframes=%s)", timeframes)

        # Phase 4 live-in: connect to xenon's tick feed. The URL is baked in
        # (DEFAULT_XENON_WS_URL) so apex reaches a local xenon out of the box;
        # override with APEX_XENON_WS_URL. Only runs once a bus exists.
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

        if getattr(app.state, "revision_watcher", None) is None:
            app.state.revision_watcher = None
            subscription_manager = getattr(app.state, "subscription_manager", None)
            if silver_root and subscription_manager is not None:
                from pathlib import Path

                from src.application.subscriptions.revision_watcher import RevisionWatcher
                from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader

                poll_seconds = float(os.environ.get("APEX_LIVEWIRE_REVISION_POLL_SECONDS", "30"))
                app.state.revision_watcher = RevisionWatcher(
                    RevisionManifestReader(Path(silver_root)),
                    subscription_manager,
                    poll_seconds,
                )
                await app.state.revision_watcher.start()
                logger.info("Silver revision watcher started (interval=%ss)", poll_seconds)

        yield
    finally:
        # Best-effort teardown, each step isolated so one failure can't strand the
        # rest. Order matters: stop ingest (xenon) first, then drain the bus while
        # the service is still subscribed (bus.stop() dispatches queued events), then
        # stop the service (drains its persistence tasks), then close the shared pool.
        revision_watcher = getattr(app.state, "revision_watcher", None)
        if revision_watcher is not None:
            try:
                await revision_watcher.stop()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("teardown: revision watcher stop failed: %s", e)
        xenon_client = getattr(app.state, "xenon_client", None)
        if xenon_client is not None:
            try:
                await xenon_client.close()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("teardown: xenon close failed: %s", e)
        event_bus = getattr(app.state, "event_bus", None)
        if event_bus is not None and hasattr(event_bus, "stop"):
            try:
                await event_bus.stop()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("teardown: event bus stop failed: %s", e)
        ta_service = getattr(app.state, "ta_service", None)
        if ta_service is not None:
            try:
                await ta_service.stop()
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("teardown: ta_service stop failed: %s", e)
        if getattr(app.state, "pg_pool", None) is not None:
            try:
                await app.state.pg_pool.close()
                logger.info("PostgreSQL pool closed")
            except Exception as e:  # pragma: no cover - best-effort teardown
                logger.warning("teardown: pg pool close failed: %s", e)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="APEX Signal Server",
        description="Signal generation, backtesting, and strategy management API",
        version=APEX_VERSION,
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

    # Chart read surface: bars + compute-on-read indicators + confluence.
    from src.api.routes.chart import router as chart_router

    app.include_router(chart_router)

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
