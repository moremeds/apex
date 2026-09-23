"""Candidate-side adapter: run the shared lake queries in-process under the BOUNDED
output policy (the MCP policy), rendered by the same payload builders REST uses.

This is the only verification module that imports the candidate; the oracle
(lake.py) never does.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.api.errors import STATUS_BY_CODE, ApiErrorCode
from src.api.payload.chart import bars_payload_from, bulk_bars_payload_from, rates_payload_from
from src.application.lake.bars import query_bars, query_rates
from src.application.lake.bulk import query_bulk_bars
from src.application.lake.errors import LakeError
from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader

_SERVICES: Dict[str, LakeServices] = {}


def _services(process: str) -> LakeServices:
    if process not in _SERVICES:
        env = os.environ.get
        silver = Path(env("APEX_LIVEWIRE_SILVER_ROOT", ""))
        delisted = env("APEX_LIVEWIRE_DELISTED_ROOT")
        provider = LivewireOhlcProvider(
            bronze_root=Path(env("APEX_LIVEWIRE_ROOT", "")),
            silver_root=silver,
            price_mode=process,  # type: ignore[arg-type]
            delisted_root=Path(delisted) if delisted else None,
        )
        _SERVICES[process] = LakeServices(
            provider=provider,
            silver=RevisionManifestReader(silver),
            pit=PitRevisionReader(silver),
        )
    return _SERVICES[process]


def _when(value: Optional[str]) -> Optional[datetime]:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


async def call(request: Dict[str, Any]) -> Tuple[int, Any]:
    kwargs = dict(request["kwargs"])
    for key in ("start", "end"):
        if key in kwargs:
            kwargs[key] = _when(kwargs[key])
    services = _services(request["process"])
    now = datetime.now(timezone.utc)
    try:
        if request["call"] == "bars":
            result = await query_bars(services, policy="bounded", **kwargs)
            return 200, bars_payload_from(result, generated_at=now)
        if request["call"] == "bulk":
            bulk = await query_bulk_bars(services, policy="bounded", **kwargs)
            return 200, bulk_bars_payload_from(bulk, generated_at=now)
        if request["call"] == "rates":
            rates = await query_rates(services, policy="bounded", **kwargs)
            return 200, rates_payload_from(rates, generated_at=now, bounded=True)
    except LakeError as exc:
        return STATUS_BY_CODE[ApiErrorCode(exc.code)], {
            "error": {"code": exc.code, "message": exc.message, "details": exc.details}
        }
    raise ValueError(f"unknown in-process call {request['call']!r}")
