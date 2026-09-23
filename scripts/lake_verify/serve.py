"""Lake-only apex server for real-lake verification (plan §4.1).

Runs the real registered REST routes with the production lifespan OFF, so no PG pool,
xenon socket, event bus or subscription manager starts; lake state is injected the
way the route tests inject it. Refuses to start with any non-lake APEX_* variable in
its environment, so a production env file cannot leak credentials into a candidate.

    APEX_LIVEWIRE_ROOT=... uv run python scripts/lake_verify/serve.py --port 8342

Works against the baseline tree too: readers that do not exist there are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path.cwd()))

LAKE_ENV = (
    "APEX_LIVEWIRE_ROOT",
    "APEX_LIVEWIRE_SILVER_ROOT",
    "APEX_LIVEWIRE_PRICE_MODE",
    "APEX_LIVEWIRE_COVERAGE_DB",
    "APEX_LIVEWIRE_LAKE_ROOT",
    "APEX_LIVEWIRE_DELISTED_ROOT",
    "APEX_LIVEWIRE_REPAIRS_ROOT",
    "APEX_LAKE_QUERY_TIMEOUT_SECONDS",
    "APEX_VERIFY_LAKE_ROOT",  # the matrix oracle's lake path, read by scripts only
)


def build_app(price_mode: str | None = None):  # type: ignore[no-untyped-def]
    """The real REST app with lake state injected; ``price_mode`` overrides the
    process configuration (the matrix runs a raw and an adjusted app side by side)."""
    stray = sorted(k for k in os.environ if k.startswith("APEX_") and k not in LAKE_ENV)
    if stray:
        raise SystemExit(f"refusing non-lake environment: {stray}")
    from src.api.server import create_app
    from src.infrastructure.adapters.livewire.coverage import CoverageCatalog
    from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider

    env = os.environ.get
    silver = env("APEX_LIVEWIRE_SILVER_ROOT")
    delisted = env("APEX_LIVEWIRE_DELISTED_ROOT")
    app = create_app()
    app.state.ohlc_provider = LivewireOhlcProvider(
        bronze_root=Path(env("APEX_LIVEWIRE_ROOT", "")),
        silver_root=Path(silver) if silver else None,
        price_mode=price_mode or env("APEX_LIVEWIRE_PRICE_MODE", "raw"),  # type: ignore[arg-type]
        delisted_root=Path(delisted) if delisted else None,
    )
    catalog = env("APEX_LIVEWIRE_COVERAGE_DB")
    app.state.coverage_catalog = CoverageCatalog(Path(catalog)) if catalog else None
    try:  # candidate-only readers; absent on the baseline tree
        from src.infrastructure.adapters.livewire.repairs import RepairsReader
        from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader

        app.state.pit_reader = PitRevisionReader(Path(silver)) if silver else None
        app.state.repairs_reader = RepairsReader.from_env()
    except ImportError:
        pass
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, lifespan="off", log_level="warning")


if __name__ == "__main__":
    main()
