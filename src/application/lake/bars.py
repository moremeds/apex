"""Shared bars, bulk-bars and rates queries.

One implementation serves two output policies (design §3.2):

- ``legacy`` -- the existing REST contract: an explicit ``start`` returns the whole
  window, no ``start`` tail-slices to ``limit`` (default 2000), ``limit <= 0`` is all.
- ``bounded`` -- MCP and new REST callers: ``limit`` (default 250) applies even with an
  explicit start, the last N rows of the window are returned, and ``truncated`` is
  known exactly because N+1 rows are selected.

Either way the tail is pushed into the DuckDB read, never sliced from a full file.

Revision pins (design §3.4) are mutually exclusive, imply adjusted, and require
equity / 1d / listing=listed: an immutable-history claim cannot rest on mutable
Bronze intraday or on the unadjusted delisted archive.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Dict, List, Literal, Optional

from src.application.lake.errors import LakeError
from src.application.lake.guards import (
    DEFAULT_BARS,
    artifact_exists,
    check_listing,
    check_timeframe,
    contract_identity,
    require_bars_payload,
    resolve_window,
    silver_revision,
    spec_or_raise,
)
from src.application.lake.services import LakeServices
from src.domain.events.domain_events import BarData
from src.infrastructure.adapters.livewire.asset_classes import AssetClassSpec
from src.infrastructure.adapters.livewire.ohlc_provider import (
    AdjustedDataUnavailable,
    RatePoint,
)
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout
from src.infrastructure.adapters.livewire.paths import parquet_path
from src.infrastructure.adapters.livewire.pit_revisions import (
    PitRevision,
    PitRevisionNotFound,
    PitScope,
    PitUnavailable,
)
from src.infrastructure.adapters.livewire.revisions import (
    RevisionManifestError,
    RevisionNotFound,
)

OutputPolicy = Literal["legacy", "bounded"]

BOUNDED_BARS_DEFAULT = 250
BOUNDED_RATES_DEFAULT = 500
BOUNDED_SERIES_MAX = 5000


@dataclass(frozen=True)
class PitProvenance:
    revision: int
    index_id: str
    publisher_status: str
    policy_version: str
    as_of: datetime
    published_at: datetime
    daily_bar_cutoff: date
    silver_revision: int
    membership_revision: int
    scopes: tuple[PitScope, ...]


@dataclass(frozen=True)
class BarsResult:
    symbol: str
    asset_class: str
    timeframe: str
    price_mode: str
    listing_status: str
    bars: List[BarData]
    window_start: datetime
    window_end: datetime
    truncated: bool
    adjustment_revision: Optional[int]
    contract: Optional[Dict[str, Any]]
    immutable_history: bool
    pinned_silver_revision: Optional[int] = None
    pit: Optional[PitProvenance] = None


@dataclass(frozen=True)
class RatesResult:
    symbol: str
    points: List[RatePoint]
    window_start: datetime
    window_end: datetime
    truncated: bool


def check_price_mode(price_mode: Optional[str], spec: AssetClassSpec, symbol: str) -> None:
    if price_mode is not None and price_mode not in ("raw", "adjusted"):
        raise LakeError(
            "invalid_parameter",
            f"unknown price_mode {price_mode!r} (have raw, adjusted)",
            symbol=symbol,
            asset_class=spec.name,
        )
    if price_mode == "adjusted" and not spec.supports_adjusted:
        raise LakeError(
            "adjusted_not_supported",
            f"Silver exists only for equity; {spec.name} is served raw",
            symbol=symbol,
            asset_class=spec.name,
        )


def check_pins(
    spec: AssetClassSpec,
    symbol: str,
    timeframe: str,
    listing: str,
    price_mode: Optional[str],
    silver_rev: Optional[int],
    pit_rev: Optional[int],
) -> Optional[str]:
    """Validate revision pins; return which one is set ("silver" / "pit" / None)."""
    if silver_rev is not None and pit_rev is not None:
        raise LakeError(
            "invalid_parameter",
            "silver_revision and pit_revision are mutually exclusive",
            symbol=symbol,
            asset_class=spec.name,
        )
    kind = "silver" if silver_rev is not None else "pit" if pit_rev is not None else None
    if kind is None:
        return None
    number = silver_rev if kind == "silver" else pit_rev
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise LakeError("invalid_parameter", f"{kind}_revision must be a positive integer")
    if price_mode == "raw":
        raise LakeError(
            "invalid_parameter",
            f"{kind}_revision pins adjusted Silver; it cannot be combined with price_mode=raw",
            symbol=symbol,
            asset_class=spec.name,
        )
    if spec.name != "equity" or timeframe != "1d" or listing != "listed":
        raise LakeError(
            "revision_not_supported",
            f"{kind}_revision requires asset_class=equity, timeframe=1d and listing=listed "
            "(historical pins cannot rest on mutable intraday Bronze or the raw archive)",
            symbol=symbol,
            asset_class=spec.name,
        )
    return kind


def tail_for(
    policy: OutputPolicy,
    limit: Optional[int],
    default_bounded: int,
    maximum: int,
    legacy_default: int,
) -> tuple[Optional[int], int]:
    """``(requested_tail_or_None_for_all, limit_passed_to_resolve_window)``."""
    if policy == "bounded":
        resolved = default_bounded if limit is None else limit
        if not 1 <= resolved <= maximum:
            raise LakeError("invalid_parameter", f"limit must be between 1 and {maximum}")
        return resolved, resolved
    resolved = legacy_default if limit is None else limit
    return (resolved if resolved > 0 else None), resolved


def trim_tail(rows: List[Any], tail: Optional[int]) -> tuple[List[Any], bool]:
    if tail is None or len(rows) <= tail:
        return rows, False
    return rows[-tail:], True


async def query_bars(
    services: LakeServices,
    *,
    symbol: str,
    asset_class: str = "equity",
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    price_mode: Optional[str] = None,
    listing: str = "listed",
    silver_revision_pin: Optional[int] = None,
    pit_revision: Optional[int] = None,
    policy: OutputPolicy = "legacy",
) -> BarsResult:
    # Request validation first: a malformed request is malformed whether or not the
    # provider happens to be up, and answering it with 503 tells the caller to retry
    # something that can never succeed.
    spec = spec_or_raise(asset_class)
    require_bars_payload(spec, symbol)
    check_timeframe(spec, timeframe)
    check_price_mode(price_mode, spec, symbol)
    pin = check_pins(
        spec, symbol, timeframe, listing, price_mode, silver_revision_pin, pit_revision
    )
    provider = services.require_provider()
    tail, window_limit = tail_for(
        policy, limit, BOUNDED_BARS_DEFAULT, BOUNDED_SERIES_MAX, DEFAULT_BARS
    )
    if pin == "pit":
        assert pit_revision is not None
        return await _pit_bars(
            services, symbol, start, end, tail, window_limit, pit_revision, policy
        )

    # An explicit price_mode is a REQUEST, not a filter: raw is always satisfiable;
    # adjusted only where Silver exists (checked above). A pin implies adjusted.
    effective = "adjusted" if pin else (price_mode or provider.effective_price_mode(spec.name))
    # After the mode is resolved: the listing check rejects adjusted-over-delisted.
    listing_status = check_listing(provider, listing, symbol, spec.name, timeframe, effective)
    window_start, window_end, legacy_tail = resolve_window(
        timeframe, start, end, window_limit, from_epoch=listing_status != "listed"
    )
    if policy == "legacy":
        tail = legacy_tail
    try:
        if pin == "silver":
            provider = await pin_silver(services, provider, silver_revision_pin)
        elif effective == "adjusted":
            provider = await asyncio.to_thread(provider.pin_snapshot)
        bars = await provider.fetch_bars(
            symbol,
            timeframe,
            window_start,
            window_end,
            asset_class=spec.name,
            price_mode=effective,
            listing=listing_status,
            tail=None if tail is None else tail + 1,
        )
    except AdjustedDataUnavailable as exc:
        raise LakeError(
            "adjusted_unavailable", str(exc), symbol=symbol, asset_class=spec.name
        ) from exc
    except QueryTimeout as exc:
        raise LakeError("query_timeout", str(exc), symbol=symbol, asset_class=spec.name) from exc
    if not bars and not artifact_exists(
        provider, symbol, timeframe, spec, effective, listing_status
    ):
        # "No artifact" is a 404; "artifact exists, window is empty" is a quiet 200.
        raise LakeError(
            "unknown_symbol",
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    bars, truncated = trim_tail(bars, tail)
    adjusted = effective == "adjusted"
    return BarsResult(
        symbol=symbol,
        asset_class=spec.name,
        timeframe=timeframe,
        price_mode=effective,
        listing_status=listing_status,
        bars=bars,
        window_start=window_start,
        window_end=window_end,
        truncated=truncated,
        adjustment_revision=silver_revision(provider) if adjusted else None,
        contract=contract_identity(spec, bars),
        # Adjusted daily comes from an immutable Silver artifact; raw Bronze and
        # adjusted intraday (mutable Bronze x factors) do not.
        immutable_history=adjusted and timeframe == "1d",
        pinned_silver_revision=silver_revision_pin,
    )


async def pin_silver(services: LakeServices, provider: Any, revision: Optional[int]) -> Any:
    if services.silver is None:
        raise LakeError("provider_not_configured", "Silver root is not configured")
    assert revision is not None
    try:
        manifest = await asyncio.to_thread(services.silver.read_revision, revision)
    except RevisionNotFound as exc:
        raise LakeError("unknown_revision", str(exc)) from exc
    except RevisionManifestError as exc:
        raise LakeError("adjusted_unavailable", str(exc)) from exc
    return await asyncio.to_thread(provider.pin_snapshot, manifest)


async def _pit_bars(
    services: LakeServices,
    symbol: str,
    start: Optional[datetime],
    end: Optional[datetime],
    tail: Optional[int],
    window_limit: int,
    revision: int,
    policy: OutputPolicy,
) -> BarsResult:
    if services.pit is None:
        raise LakeError("provider_not_configured", "Silver root is not configured")
    manifest = await _read_pit(services, revision)
    scopes = manifest.scopes_for(symbol)
    summary = manifest.summary
    if not scopes:
        raise LakeError(
            "invalid_parameter",
            f"{symbol} is not a member of {summary.index_id} in PIT revision {revision}",
            symbol=symbol,
            asset_class="equity",
        )
    # No start reads the scope from its beginning (tail-sliced): a PIT scope can end
    # years before today, so a lookback measured back from now would miss it.
    window_start, window_end, legacy_tail = resolve_window(
        "1d", start, end, window_limit, from_epoch=True
    )
    if policy == "legacy":
        tail = legacy_tail
    first, last = window_start.date(), min(window_end.date(), summary.daily_bar_cutoff)
    hits = tuple(scope for scope in scopes if first <= last and scope.intersects(first, last))
    scope_details = {
        "scopes": [
            {
                "security_id": s.security_id,
                "session_from": s.session_from.isoformat(),
                "session_to": None if s.session_to is None else s.session_to.isoformat(),
            }
            for s in scopes
        ],
        "daily_bar_cutoff": summary.daily_bar_cutoff.isoformat(),
    }
    if not hits:
        raise LakeError(
            "invalid_parameter",
            f"the requested window is outside {symbol}'s scope in PIT revision {revision}",
            symbol=symbol,
            asset_class="equity",
            details=scope_details,
        )
    if len({scope.security_id for scope in hits}) > 1:
        raise LakeError(
            "ambiguous_symbol",
            f"{symbol} maps to more than one security in PIT revision {revision} within "
            "the requested window; narrow the window to one scope",
            symbol=symbol,
            asset_class="equity",
            details=scope_details,
        )
    try:
        path = await asyncio.to_thread(manifest.daily_artifact_path, symbol)
    except PitUnavailable as exc:
        raise LakeError("pit_unavailable", str(exc), symbol=symbol, asset_class="equity") from exc
    clipped_end = min(window_end, datetime.combine(last, time.max, tzinfo=timezone.utc))
    try:
        bars = await services.require_provider().fetch_artifact_daily(
            path,
            symbol,
            window_start,
            clipped_end,
            tail=None if tail is None else tail + 1,
            date_ranges=tuple((s.session_from, s.session_to) for s in hits),
        )
    except QueryTimeout as exc:
        raise LakeError("query_timeout", str(exc), symbol=symbol, asset_class="equity") from exc
    bars, truncated = trim_tail(bars, tail)
    return BarsResult(
        symbol=symbol,
        asset_class="equity",
        timeframe="1d",
        price_mode="adjusted",
        listing_status="listed",
        bars=bars,
        window_start=window_start,
        window_end=clipped_end,
        truncated=truncated,
        adjustment_revision=summary.silver_revision,
        contract=None,
        immutable_history=True,
        pit=_provenance(manifest, hits),
    )


async def _read_pit(services: LakeServices, revision: int) -> PitRevision:
    assert services.pit is not None
    try:
        return await asyncio.to_thread(services.pit.read, revision)
    except PitRevisionNotFound as exc:
        raise LakeError("unknown_revision", str(exc)) from exc
    except PitUnavailable as exc:
        raise LakeError("pit_unavailable", str(exc)) from exc


def _provenance(manifest: PitRevision, scopes: tuple[PitScope, ...]) -> PitProvenance:
    summary = manifest.summary
    return PitProvenance(
        revision=summary.revision,
        index_id=summary.index_id,
        publisher_status=summary.status,
        policy_version=manifest.policy_version,
        as_of=summary.as_of,
        published_at=summary.published_at,
        daily_bar_cutoff=summary.daily_bar_cutoff,
        silver_revision=summary.silver_revision,
        membership_revision=summary.membership_revision,
        scopes=scopes,
    )


async def query_rates(
    services: LakeServices,
    *,
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    policy: OutputPolicy = "legacy",
) -> RatesResult:
    """A yield series. Legacy REST without a limit keeps its full-history default; a
    positive limit (or the bounded policy) returns the last N points of the window."""
    spec = spec_or_raise("rates")
    provider = services.require_provider()
    if policy == "legacy" and limit is None:
        tail: Optional[int] = None
    else:
        tail, _ = tail_for(
            "bounded",
            limit,
            BOUNDED_RATES_DEFAULT,
            BOUNDED_SERIES_MAX,
            BOUNDED_RATES_DEFAULT,
        )
    window_start, window_end, _ = resolve_window("1d", start, end, 0)
    try:
        points = await provider.fetch_rate_series(
            symbol, window_start, window_end, tail=None if tail is None else tail + 1
        )
    except QueryTimeout as exc:
        raise LakeError("query_timeout", str(exc), symbol=symbol, asset_class=spec.name) from exc
    if not points and not parquet_path(provider.bronze_root, symbol, "1d", spec.name).exists():
        raise LakeError(
            "unknown_symbol",
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    points, truncated = trim_tail(points, tail)
    return RatesResult(symbol, points, window_start, window_end, truncated)
