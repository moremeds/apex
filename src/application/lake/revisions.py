"""Silver and PIT revision discovery and detail.

Silver: one numbered-manifest listing, newest first, with ``current`` flagged; detail
by explicit number (omitted -> current, resolved once). PIT: every retained manifest
with its index and publisher status, plus the latest per index -- never
``current.json``, which is one pointer shared by all indexes (design §3.4). Detail
pages long lists (affected symbols, member scopes); the artifact manifest is never
dumped.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.application.lake.errors import LakeError
from src.application.lake.services import LakeServices, Page, check_page, page_of
from src.infrastructure.adapters.livewire.pit_revisions import (
    PitRevision,
    PitRevisionNotFound,
    PitRevisionReader,
    PitRevisionSummary,
    PitScope,
    PitUnavailable,
)
from src.infrastructure.adapters.livewire.revisions import (
    AffectedSymbol,
    RevisionManifestError,
    RevisionManifestReader,
    RevisionNotFound,
    SilverRevision,
)

logger = logging.getLogger(__name__)


def _silver(services: LakeServices) -> RevisionManifestReader:
    if services.silver is None:
        raise LakeError("provider_not_configured", "Silver root is not configured")
    return services.silver


def _pit(services: LakeServices) -> PitRevisionReader:
    if services.pit is None:
        raise LakeError("provider_not_configured", "Silver root is not configured")
    return services.pit


@dataclass(frozen=True)
class SilverRevisionList:
    current: Optional[int]
    current_error: Optional[str]
    page: Page[int]


async def list_silver_revisions(
    services: LakeServices, *, limit: Optional[int] = None, offset: Optional[int] = None
) -> SilverRevisionList:
    size, skip = check_page(limit, offset)
    reader = _silver(services)
    numbers = await asyncio.to_thread(reader.list_revisions)
    try:
        current: Optional[int] = await asyncio.to_thread(reader.current_revision_number)
        error = None
    except RevisionManifestError as exc:
        # Retained numbered manifests are still listable when the pointer is broken;
        # the listing says so rather than guessing a current revision.
        logger.warning("Silver current pointer unreadable: %s", exc)
        current, error = None, "current pointer unreadable or inconsistent"
    return SilverRevisionList(current, error, page_of(numbers, size, skip))


@dataclass(frozen=True)
class SilverRevisionDetail:
    revision: SilverRevision
    is_current: bool
    affected: Page[AffectedSymbol]


async def silver_revision_detail(
    services: LakeServices,
    revision: Optional[int] = None,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> SilverRevisionDetail:
    """Omitted ``revision`` resolves ``current`` once. A broken current pointer never
    blocks an explicit numbered read; it only leaves ``is_current`` false."""
    size, skip = check_page(limit, offset)
    reader = _silver(services)
    try:
        if revision is None:
            manifest = await asyncio.to_thread(reader.read_current)
            return SilverRevisionDetail(manifest, True, page_of(manifest.affected, size, skip))
        manifest = await asyncio.to_thread(reader.read_revision, revision)
    except RevisionNotFound as exc:
        raise LakeError("unknown_revision", str(exc)) from exc
    except RevisionManifestError as exc:
        raise LakeError("adjusted_unavailable", str(exc)) from exc
    try:
        current: Optional[int] = await asyncio.to_thread(reader.current_revision_number)
    except RevisionManifestError as exc:
        logger.warning("Silver current pointer unreadable; is_current=false: %s", exc)
        current = None
    return SilverRevisionDetail(
        manifest, manifest.revision == current, page_of(manifest.affected, size, skip)
    )


@dataclass(frozen=True)
class PitRevisionList:
    # Any manifest on disk, before the index filter: a filter that matches nothing is
    # not "PIT was never published".
    available: bool
    latest_per_index: Dict[str, int]
    page: Page[PitRevisionSummary]


async def list_pit_revisions(
    services: LakeServices,
    *,
    index_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> PitRevisionList:
    """No manifest on disk is an empty list, not an error (discovery says available
    is false); an unreadable manifest is ``pit_unavailable``."""
    size, skip = check_page(limit, offset)
    try:
        summaries: List[PitRevisionSummary] = await asyncio.to_thread(_pit(services).list_revisions)
    except PitUnavailable as exc:
        raise LakeError("pit_unavailable", str(exc)) from exc
    available = bool(summaries)
    latest: Dict[str, int] = {}
    for summary in summaries:  # newest first
        latest.setdefault(summary.index_id, summary.revision)
    if index_id is not None:
        summaries = [s for s in summaries if s.index_id == index_id]
        latest = {k: v for k, v in latest.items() if k == index_id}
    return PitRevisionList(available, latest, page_of(summaries, size, skip))


@dataclass(frozen=True)
class PitRevisionDetail:
    manifest: PitRevision
    daily_artifact_count: int
    members: Page[PitScope]


async def pit_revision_detail(
    services: LakeServices,
    revision: int,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> PitRevisionDetail:
    size, skip = check_page(limit, offset)
    try:
        manifest = await asyncio.to_thread(_pit(services).read, revision)
    except PitRevisionNotFound as exc:
        raise LakeError("unknown_revision", str(exc)) from exc
    except PitUnavailable as exc:
        raise LakeError("pit_unavailable", str(exc)) from exc
    return PitRevisionDetail(
        manifest, len(manifest.daily_artifacts), page_of(manifest.members, size, skip)
    )


def pit_summary_dict(summary: PitRevisionSummary) -> Dict[str, Any]:
    return {
        "revision": summary.revision,
        "index_id": summary.index_id,
        "publisher_status": summary.status,
        "as_of": summary.as_of.isoformat(),
        "published_at": summary.published_at.isoformat(),
        "daily_bar_cutoff": summary.daily_bar_cutoff.isoformat(),
        "silver_revision": summary.silver_revision,
        "membership_revision": summary.membership_revision,
        "member_count": summary.member_count,
    }


def scope_dict(scope: PitScope) -> Dict[str, Any]:
    return {
        "symbol": scope.symbol,
        "security_id": scope.security_id,
        "session_from": scope.session_from.isoformat(),
        "session_to": None if scope.session_to is None else scope.session_to.isoformat(),
        "effective_from": scope.effective_from,
        "effective_to": scope.effective_to,
    }
