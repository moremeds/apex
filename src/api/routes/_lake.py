"""Build the per-request ``LakeServices`` from ``app.state``.

Reading app.state on every request (instead of one container built in the lifespan)
keeps the existing injection seam: tests set ``app.state.ohlc_provider`` or
``coverage_catalog`` directly and never run the production lifespan.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Request

from src.application.lake.errors import LakeError
from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.membership import MembershipReader
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.reference import LivewireReferenceReader
from src.infrastructure.adapters.livewire.repairs import RepairsReader
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader


def repairs_from_env() -> RepairsReader:
    root = os.environ.get("APEX_LIVEWIRE_REPAIRS_ROOT", "").strip()
    return RepairsReader(Path(root).expanduser() if root else None)


def lake_services(request: Request) -> LakeServices:
    state = request.app.state
    provider: Any = getattr(state, "ohlc_provider", None)
    silver_root = getattr(provider, "silver_root", None) if provider is not None else None
    pit = getattr(state, "pit_reader", None)
    if pit is None and silver_root is not None:
        pit = PitRevisionReader(silver_root)
    return LakeServices(
        provider=provider,
        catalog=getattr(state, "coverage_catalog", None),
        membership=MembershipReader.from_env(),
        reference=LivewireReferenceReader.from_env(),
        silver=RevisionManifestReader(silver_root) if silver_root is not None else None,
        pit=pit,
        repairs=getattr(state, "repairs_reader", None) or repairs_from_env(),
    )


def provider_or_raise(request: Request) -> Any:
    """The shared bar provider, for routes that have not moved to a lake query."""
    provider = getattr(request.app.state, "ohlc_provider", None)
    if provider is None:
        raise LakeError("provider_not_configured", "bar provider not configured")
    return provider
