"""Discovery queries: asset classes, instrument search/detail, catalog coverage,
futures contracts and lake status.

Discovery reads the coverage catalog (a daily snapshot) or bounded exact-file probes;
nothing here walks a lake tree. The one directory listing is the futures partition
(114 contract dirs on 2026-09-23), because the catalog carries only 14 of them.
Status never exposes absolute host paths or secret env values.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.application.lake.errors import LakeError
from src.application.lake.guards import delisted_artifact_exists, spec_or_raise
from src.application.lake.services import LakeServices, Page, check_page
from src.infrastructure.adapters.livewire.asset_classes import ASSET_CLASSES
from src.infrastructure.adapters.livewire.coverage import (
    CatalogIdentity,
    CoverageRow,
    CoverageUnavailable,
    InstrumentRow,
)
from src.infrastructure.adapters.livewire.membership import MembershipDataError
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.paths import parquet_path
from src.infrastructure.adapters.livewire.pit_revisions import PitUnavailable
from src.infrastructure.adapters.livewire.revisions import RevisionManifestError

logger = logging.getLogger(__name__)

_FUTURES_ROOT_RE = re.compile(r"^[A-Z0-9]{1,8}$")


def asset_classes() -> List[Dict[str, Any]]:
    """The registry: what each class publishes and whether Silver exists for it."""
    return [
        {
            "asset_class": spec.name,
            "payload": spec.payload,
            "timeframes": list(spec.timeframes),
            "supports_adjusted": spec.supports_adjusted,
            "extra_bar_fields": list(spec.extra_bar_fields),
        }
        for spec in ASSET_CLASSES.values()
    ]


async def search_instruments(
    services: LakeServices, *, q: Optional[str], asset_class: Optional[str], limit: int
) -> List[InstrumentRow]:
    catalog = services.require_catalog()
    if asset_class is not None:
        spec_or_raise(asset_class)
    try:
        # DuckDB is synchronous and the catalog shares the external volume with the
        # lake: running it inline would stall every other request on this worker.
        return await asyncio.to_thread(
            catalog.list_instruments, asset_class=asset_class, query=q, limit=limit
        )
    except CoverageUnavailable as exc:
        # An unreadable catalog is NOT an empty universe.
        raise LakeError("provider_not_configured", str(exc)) from exc


@dataclass(frozen=True)
class InstrumentDetail:
    symbol: str
    asset_class: str
    timeframes: List[str]
    residency: Dict[str, str]
    coverage_source: str
    first_date: Optional[str]
    last_date: Optional[str]
    silver_available: bool
    price_mode: str
    adjustment_revision: Optional[int]


async def get_instrument(services: LakeServices, symbol: str, asset_class: str) -> InstrumentDetail:
    """One instrument, with the timeframes that exist on disk. Five exact-file probes
    per tree, never a listing: the catalog measures no equity intraday."""
    spec = spec_or_raise(asset_class)
    provider = services.require_provider()
    silver_daily = False
    adjustment_revision = None
    if spec.supports_adjusted and provider.silver_root is not None:
        try:
            pinned = await asyncio.to_thread(provider.pin_snapshot)
            silver_daily = (
                await asyncio.to_thread(pinned.silver_artifact_path, symbol, "daily") is not None
            )
            if silver_daily and pinned.snapshot is not None:
                adjustment_revision = pinned.snapshot.revision
        except AdjustedDataUnavailable as exc:
            raise LakeError(
                "adjusted_unavailable", str(exc), symbol=symbol, asset_class=spec.name
            ) from exc
    residency: Dict[str, str] = {}
    timeframes = []
    for tf in spec.timeframes:
        live = parquet_path(provider.bronze_root, symbol, tf, spec.name).exists()
        archived = delisted_artifact_exists(provider, symbol, tf, spec.name)
        # Silver can outlive its Bronze source, so a Bronze-only probe would omit "1d"
        # from a symbol that /bars serves in adjusted mode.
        if live or (tf == "1d" and silver_daily):
            timeframes.append(tf)
        if live or archived:
            residency[tf] = "dual" if live and archived else "live" if live else "archive"
    if not timeframes:
        raise LakeError(
            "unknown_symbol",
            f"no artifact for {symbol} under {spec.partition}",
            symbol=symbol,
            asset_class=spec.name,
        )
    dates = None
    coverage_source = "not_configured" if services.catalog is None else "livewire_coverage_snapshot"
    if services.catalog is not None:
        try:
            dates = await asyncio.to_thread(services.catalog.get_instrument, symbol, spec.name)
        except CoverageUnavailable as exc:
            # Timeframes came from disk and are still correct; say why dates are null.
            logger.warning("coverage catalog unreadable, serving %s without dates: %s", symbol, exc)
            dates, coverage_source = None, "unavailable"
    return InstrumentDetail(
        symbol=symbol,
        asset_class=spec.name,
        timeframes=timeframes,
        residency=residency,
        coverage_source=coverage_source,
        first_date=dates.first_date if dates else None,
        last_date=dates.last_date if dates else None,
        silver_available=silver_daily,
        price_mode=provider.effective_price_mode(spec.name),
        adjustment_revision=adjustment_revision,
    )


@dataclass(frozen=True)
class CoverageResult:
    catalog: CatalogIdentity
    page: Page[CoverageRow]


async def coverage(
    services: LakeServices,
    *,
    symbol: Optional[str] = None,
    asset_class: Optional[str] = None,
    include_silver: bool = True,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> CoverageResult:
    """Raw catalog rows. The catalog is a daily snapshot: pages are not one snapshot
    unless ``catalog`` identity is unchanged between calls."""
    size, skip = check_page(limit, offset)
    if asset_class is not None:
        spec_or_raise(asset_class)
    catalog = services.require_catalog()
    try:
        identity = await asyncio.to_thread(catalog.identity)
        rows, truncated = await asyncio.to_thread(
            catalog.list_coverage,
            symbol=symbol,
            asset_class=asset_class,
            include_silver=include_silver,
            limit=size,
            offset=skip,
        )
    except CoverageUnavailable as exc:
        raise LakeError("provider_not_configured", str(exc)) from exc
    return CoverageResult(identity, Page(rows, size, skip, truncated))


@dataclass(frozen=True)
class FuturesContract:
    symbol: str
    contract_id: Optional[int]
    root_symbol: Optional[str]
    expiry_date: Optional[str]
    first_date: Optional[str]
    last_date: Optional[str]
    rows: int
    in_catalog: bool


async def futures_contracts(
    services: LakeServices,
    root: str,
    *,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> Page[FuturesContract]:
    """Contracts under one root, ordered by symbol (``ROOT_YYYYMM``)."""
    size, skip = check_page(limit, offset)
    root = root.strip().upper()
    if not _FUTURES_ROOT_RE.match(root):
        raise LakeError("invalid_parameter", f"invalid futures root {root!r}")
    provider = services.require_provider()
    partition = provider.bronze_root / "asset_class=futures"

    def listing() -> List[str]:
        try:
            names = [entry.name for entry in partition.iterdir()]
        except FileNotFoundError:
            return []
        prefix = f"symbol={root}_"
        return sorted(n[len("symbol=") :] for n in names if n.startswith(prefix))

    symbols = await asyncio.to_thread(listing)
    if not symbols:
        raise LakeError(
            "unknown_symbol",
            f"no futures contracts under root {root}",
            asset_class="futures",
        )
    window = symbols[skip : skip + size + 1]
    catalogued: set[str] = set()
    if services.catalog is not None:
        try:
            rows, _ = await asyncio.to_thread(
                services.catalog.list_coverage,
                asset_class="futures",
                symbol_prefix=f"{root}_",
                limit=2000,
            )
            catalogued = {row.symbol for row in rows}
        except CoverageUnavailable as exc:
            logger.warning("coverage catalog unreadable for futures %s: %s", root, exc)
            catalogued = set()
    contracts = []
    for symbol in window[:size]:
        facts = await asyncio.to_thread(provider.fetch_futures_contract, symbol)
        contracts.append(FuturesContract(symbol=symbol, in_catalog=symbol in catalogued, **facts))
    return Page(contracts, size, skip, len(window) > size)


async def lake_status(services: LakeServices) -> Dict[str, Any]:
    """Which sources are configured and readable, and how fresh; never paths or keys."""
    provider = services.provider
    status: Dict[str, Any] = {}
    status["bronze"] = {
        "configured": provider is not None,
        "available": provider is not None and await asyncio.to_thread(provider.bronze_root.is_dir),
    }
    delisted = getattr(provider, "delisted_root", None)
    status["delisted"] = {
        "configured": delisted is not None,
        "available": delisted is not None and await asyncio.to_thread(delisted.is_dir),
    }
    status["silver"] = await _silver_status(services)
    status["pit"] = await _pit_status(services)
    status["catalog"] = await _catalog_status(services)
    membership = services.membership
    status["membership"] = {"configured": membership is not None}
    if membership is not None:
        try:
            status["membership"]["indices"] = await asyncio.to_thread(membership.list_indices)
            status["membership"]["available"] = True
        except MembershipDataError as exc:
            logger.warning("membership status unavailable: %s", exc)
            status["membership"].update(available=False, error=type(exc).__name__)
    status["repairs"] = await asyncio.to_thread(services.repairs.status)
    return status


async def _silver_status(services: LakeServices) -> Dict[str, Any]:
    if services.silver is None:
        return {"configured": False, "available": False}
    try:
        retained = await asyncio.to_thread(services.silver.list_revisions)
        current = await asyncio.to_thread(services.silver.current_revision_number)
    except RevisionManifestError as exc:
        return {"configured": True, "available": False, "error": str(exc)[:200]}
    return {
        "configured": True,
        "available": True,
        "current_revision": current,
        "retained_revisions": len(retained),
    }


async def _pit_status(services: LakeServices) -> Dict[str, Any]:
    if services.pit is None:
        return {"configured": False, "available": False}
    try:
        summaries = await asyncio.to_thread(services.pit.list_revisions)
    except PitUnavailable as exc:
        return {"configured": True, "available": False, "error": str(exc)[:200]}
    latest: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:  # newest first, so the first per index is the latest
        latest.setdefault(
            summary.index_id,
            {"revision": summary.revision, "publisher_status": summary.status},
        )
    return {
        "configured": True,
        "available": bool(summaries),
        "revisions": len(summaries),
        "latest_per_index": latest,
    }


async def _catalog_status(services: LakeServices) -> Dict[str, Any]:
    if services.catalog is None:
        return {"configured": False, "available": False}
    try:
        identity = await asyncio.to_thread(services.catalog.identity)
    except CoverageUnavailable as exc:
        logger.warning("coverage catalog unavailable in status: %s", exc)
        return {"configured": True, "available": False}
    return {
        "configured": True,
        "available": True,
        "modified_at": identity.modified_at,
        "size_bytes": identity.size_bytes,
    }
