"""Identity queries: corporate actions, delisting intervals, security resolution and
point-in-time index membership.

Moved out of the REST routes so MCP reuses the exact behaviour; REST keeps its
envelopes. Dates are UTC calendar days (membership ``as_of``/``known_at`` evaluate at
UTC end-of-day, design §3.5) -- no instant precision is promised.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional

from src.application.lake.errors import LakeError
from src.application.lake.services import LakeServices, Page, check_page, page_of
from src.infrastructure.adapters.livewire.membership import (
    Member,
    MembershipDataError,
    MembershipEvent,
    MembershipReader,
)
from src.infrastructure.adapters.livewire.reference import (
    ACTION_TYPES,
    CorporateAction,
    IdentityInterval,
    ReferenceDataError,
)

# livewire writes these ids for securities it could not map to the security master.
UNRESOLVED_PREFIX = "unresolved:"


def today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _membership(services: LakeServices) -> MembershipReader:
    if services.membership is None:
        raise LakeError(
            "provider_not_configured",
            "index membership is not configured; set APEX_LIVEWIRE_LAKE_ROOT",
        )
    return services.membership


# -- corporate actions / delisting ------------------------------------------------


@dataclass(frozen=True)
class ActionsResult:
    symbol: str
    provider: Optional[str]
    page: Page[CorporateAction]


async def corporate_actions(
    services: LakeServices,
    symbol: str,
    *,
    action_type: Optional[str] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    paged: bool = False,
) -> ActionsResult:
    """Ticker-keyed actions (not security-keyed: a reused ticker's log may belong to a
    different, living company). ``paged=False`` returns every match (legacy REST)."""
    if action_type is not None and action_type not in ACTION_TYPES:
        raise LakeError(
            "invalid_parameter",
            f"unknown type {action_type!r} (have {list(ACTION_TYPES)})",
            symbol=symbol,
            asset_class="equity",
        )
    if start is not None and end is not None and start > end:
        raise LakeError(
            "invalid_parameter",
            f"start {start.isoformat()} is after end {end.isoformat()}",
            symbol=symbol,
            asset_class="equity",
        )
    size, skip = check_page(limit, offset) if paged else (0, 0)
    reader = services.reference
    if reader.bronze_root is None:
        raise LakeError(
            "provider_not_configured",
            "corporate actions are not configured; set APEX_LIVEWIRE_ROOT",
        )
    ticker = symbol.upper()
    try:
        actions = await asyncio.to_thread(
            reader.fetch_actions, ticker, action_type=action_type, start=start, end=end
        )
        provider = await asyncio.to_thread(reader.fetch_provider, ticker)
    except ReferenceDataError as exc:
        raise LakeError(
            "provider_not_configured", str(exc), symbol=symbol, asset_class="equity"
        ) from exc
    if actions is None:
        # No log at all is an unknown ticker; a log with nothing matching is a 200.
        raise LakeError(
            "unknown_symbol",
            f"no corporate-action log for {ticker}",
            symbol=symbol,
            asset_class="equity",
        )
    page = page_of(actions, size, skip) if paged else Page(actions, len(actions), 0, False)
    return ActionsResult(ticker, provider, page)


async def delisting(
    services: LakeServices,
    symbol: str,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    paged: bool = False,
) -> Page[IdentityInterval]:
    """Today's standing identity intervals for a ticker. No delisting reason exists in
    the security master, so none is returned or inferred."""
    size, skip = check_page(limit, offset) if paged else (0, 0)
    reader = services.reference
    if reader.lake_root is None:
        raise LakeError(
            "provider_not_configured",
            "the security master is not configured; set APEX_LIVEWIRE_LAKE_ROOT",
        )
    ticker = symbol.upper()
    try:
        intervals = await asyncio.to_thread(reader.fetch_identity, ticker)
    except ReferenceDataError as exc:
        raise LakeError(
            "provider_not_configured", str(exc), symbol=symbol, asset_class="equity"
        ) from exc
    if intervals is None:
        raise LakeError(
            "provider_not_configured",
            "security master artifact is missing under APEX_LIVEWIRE_LAKE_ROOT",
            symbol=symbol,
            asset_class="equity",
        )
    if not intervals:
        raise LakeError(
            "unknown_symbol",
            f"no verified security-master record for {ticker}",
            symbol=symbol,
            asset_class="equity",
        )
    return page_of(intervals, size, skip) if paged else Page(intervals, len(intervals), 0, False)


# -- security resolution ----------------------------------------------------------


@dataclass(frozen=True)
class SecurityResolution:
    symbol: str
    as_of: date
    known_at: Optional[date]
    security_id: str


async def resolve_security(
    services: LakeServices, symbol: str, as_of: date, known_at: Optional[date] = None
) -> SecurityResolution:
    """Map a ticker to one ``security_id`` at ``as_of`` (optionally as known at
    ``known_at``). Ambiguity is an error, never a pick. No interval from today's
    security master is attached: that would leak later knowledge into a historical
    ``known_at`` answer."""
    reader = _membership(services)
    ticker = symbol.strip().upper()
    if not ticker:
        raise LakeError("invalid_parameter", "symbol is required")
    if not await asyncio.to_thread(reader.security_master_path.is_file):
        raise LakeError("membership_unavailable", "the security master is not present in the lake")
    try:
        resolution = await asyncio.to_thread(
            reader.resolve_symbol, ticker, as_of, known_at=known_at
        )
    except MembershipDataError as exc:
        raise LakeError("membership_unavailable", str(exc)) from exc
    if resolution.ambiguous:
        raise LakeError(
            "ambiguous_security",
            f"symbol {ticker!r} maps to more than one security on {as_of.isoformat()}; "
            "apex will not guess which",
            symbol=ticker,
        )
    if resolution.security_id is None:
        raise LakeError(
            "unknown_symbol",
            f"symbol {ticker!r} is not in the security master on {as_of.isoformat()}"
            + ("" if known_at is None else f" as known at {known_at.isoformat()}"),
            symbol=ticker,
        )
    return SecurityResolution(ticker, as_of, known_at, resolution.security_id)


# -- index membership -------------------------------------------------------------


async def list_indices(services: LakeServices) -> List[str]:
    """Index ids discovered on disk, sorted; never a hardcoded list."""
    reader = _membership(services)
    try:
        return await asyncio.to_thread(reader.list_indices)
    except MembershipDataError as exc:
        raise LakeError("membership_unavailable", str(exc)) from exc


@dataclass(frozen=True)
class MembersResult:
    index_id: str
    as_of: date
    known_at: Optional[date]
    include_candidates: bool
    unresolved_count: int
    total: int
    page: Page[Member]


async def index_members(
    services: LakeServices,
    index_id: str,
    as_of: date,
    *,
    known_at: Optional[date] = None,
    include_candidates: bool = False,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    paged: bool = False,
) -> MembersResult:
    """Members replayed from the event log. An index whose log holds no row under the
    requested status reading fails closed (not yet published); rows that exist but do
    not apply at ``as_of`` make an empty answer that is the truth."""
    reader = _membership(services)
    size, skip = check_page(limit, offset) if paged else (0, 0)
    if await asyncio.to_thread(reader.events_path, index_id) is None:
        raise LakeError("unknown_index", f"unknown index {index_id!r}")
    try:
        members = await asyncio.to_thread(
            reader.members_with_symbols,
            index_id,
            as_of,
            known_at=known_at,
            include_candidates=include_candidates,
        )
        if not members and not await asyncio.to_thread(
            reader.has_events_for_status,
            index_id,
            include_candidates=include_candidates,
        ):
            raise LakeError(
                "membership_unavailable",
                f"membership data is not yet available for index {index_id!r} "
                f"as of {as_of.isoformat()}",
            )
    except MembershipDataError as exc:
        raise LakeError("membership_unavailable", str(exc)) from exc
    unresolved = sum(1 for m in members if m.security_id.startswith(UNRESOLVED_PREFIX))
    page = page_of(members, size, skip) if paged else Page(members, len(members), 0, False)
    return MembersResult(
        index_id, as_of, known_at, include_candidates, unresolved, len(members), page
    )


@dataclass(frozen=True)
class HistoryResult:
    symbol: str
    security_id: str
    page: Page[MembershipEvent]


async def membership_history(
    services: LakeServices,
    symbol: str,
    as_of: date,
    *,
    index_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    paged: bool = False,
) -> HistoryResult:
    """Today's effective timeline for a ticker; ``as_of`` only selects which security
    the ticker names. Unions the resolved id with livewire's ``unresolved:<TICKER>``
    placeholder, because pre-identity-floor events stay logged under the placeholder
    even once the ticker resolves."""
    reader = _membership(services)
    size, skip = check_page(limit, offset) if paged else (0, 0)
    ticker = symbol.strip().upper()
    if not ticker:
        raise LakeError("invalid_parameter", "symbol is required")
    try:
        resolution = await asyncio.to_thread(reader.resolve_symbol, ticker, as_of)
    except MembershipDataError as exc:
        raise LakeError("membership_unavailable", str(exc)) from exc
    if resolution.ambiguous:
        raise LakeError(
            "ambiguous_security",
            f"symbol {ticker!r} maps to more than one security on {as_of.isoformat()}; "
            "apex will not guess which",
            symbol=ticker,
        )
    if index_id is not None and await asyncio.to_thread(reader.events_path, index_id) is None:
        raise LakeError("unknown_index", f"unknown index {index_id!r}")
    security_id = resolution.security_id
    placeholder = f"{UNRESOLVED_PREFIX}{ticker}"
    ids = [placeholder] if security_id is None else [security_id, placeholder]
    try:
        events: List[MembershipEvent] = await asyncio.to_thread(
            reader.history_for_security, ids, index_id
        )
    except MembershipDataError as exc:
        raise LakeError("membership_unavailable", str(exc)) from exc
    if security_id is None and events:
        security_id = placeholder
    if security_id is None:
        raise LakeError(
            "unknown_symbol",
            f"symbol {ticker!r} is not in the security master on {as_of.isoformat()}"
            " and has no unresolved membership events",
            symbol=ticker,
        )
    page = page_of(events, size, skip) if paged else Page(events, len(events), 0, False)
    return HistoryResult(ticker, security_id, page)
