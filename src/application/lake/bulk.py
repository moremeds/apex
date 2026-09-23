"""Bulk equity bars: many symbols on ONE adjustment basis.

A symbol that cannot be served lands in ``missing`` with its reason -- one bad ticker
in 200 must not cost the other 199 -- while request-level faults (bad listing, bad
pin, budget) fail the whole request. Split out of ``bars.py`` on the single- vs
many-series seam.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from src.application.lake.bars import (
    OutputPolicy,
    check_pins,
    check_price_mode,
    pin_silver,
    tail_for,
    trim_tail,
)
from src.application.lake.errors import LakeError
from src.application.lake.guards import (
    DEFAULT_BARS,
    artifact_exists,
    check_listing,
    check_timeframe,
    resolve_window,
    silver_revision,
    spec_or_raise,
)
from src.application.lake.services import LakeServices
from src.domain.events.domain_events import BarData
from src.infrastructure.adapters.livewire.ohlc_provider import AdjustedDataUnavailable
from src.infrastructure.adapters.livewire.parquet_reads import QueryTimeout

BULK_MAX_SYMBOLS = 200
BULK_BOUNDED_DEFAULT = 50
BULK_BOUNDED_MAX = 2000
BULK_ROW_BUDGET = 10000


@dataclass(frozen=True)
class BulkSeries:
    listing_status: str
    bars: List[BarData]
    truncated: bool


@dataclass(frozen=True)
class BulkResult:
    price_mode: str
    timeframe: str
    adjustment_revision: Optional[int]
    pinned_silver_revision: Optional[int]
    window_start: datetime
    window_end: datetime
    series: Dict[str, BulkSeries] = field(default_factory=dict)
    missing: Dict[str, str] = field(default_factory=dict)


def normalize_symbols(raw: Sequence[str]) -> List[str]:
    """Upper-cased, de-duplicated, order-preserving; 1..200 symbols."""
    ordered: Dict[str, None] = {}
    for part in raw:
        symbol = part.strip().upper()
        if symbol:
            ordered[symbol] = None
    symbols = list(ordered)
    if not symbols:
        raise LakeError("invalid_parameter", "symbols is required and must be non-empty")
    if len(symbols) > BULK_MAX_SYMBOLS:
        raise LakeError(
            "invalid_parameter",
            f"{len(symbols)} symbols requested; at most {BULK_MAX_SYMBOLS} per call",
        )
    return symbols


async def query_bulk_bars(
    services: LakeServices,
    *,
    symbols: Sequence[str],
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
    price_mode: Optional[str] = None,
    listing: str = "listed",
    silver_revision_pin: Optional[int] = None,
    policy: OutputPolicy = "legacy",
) -> BulkResult:
    """Equity bars for many symbols on ONE pinned Silver revision.

    A symbol that cannot be served lands in ``missing`` with its reason: one bad
    ticker in 200 must not cost the other 199.
    """
    requested = normalize_symbols(symbols)
    spec = spec_or_raise("equity")
    check_timeframe(spec, timeframe)
    check_price_mode(price_mode, spec, requested[0])
    if listing not in ("listed", "delisted", "any"):
        # Validated once, up front: check_listing raises per symbol, and the loop below
        # files per-symbol errors under `missing`, which would turn a malformed request
        # into a 200 with an empty result map.
        raise LakeError(
            "invalid_parameter",
            f"unknown listing filter {listing!r} (have listed, delisted, any)",
        )
    pin = check_pins(spec, requested[0], timeframe, listing, price_mode, silver_revision_pin, None)
    provider = services.require_provider()
    tail, window_limit = tail_for(
        policy, limit, BULK_BOUNDED_DEFAULT, BULK_BOUNDED_MAX, DEFAULT_BARS
    )
    if policy == "bounded" and tail is not None and tail * len(requested) > BULK_ROW_BUDGET:
        raise LakeError(
            "invalid_parameter",
            f"{len(requested)} symbols x limit {tail} exceeds the {BULK_ROW_BUDGET}-row budget",
        )
    effective = "adjusted" if pin else (price_mode or provider.effective_price_mode(spec.name))
    try:
        if pin == "silver":
            provider = await pin_silver(services, provider, silver_revision_pin)
        elif effective == "adjusted":
            # One Silver revision for the whole table: a revision landing mid-request
            # would adjust some symbols on one corporate-action set, the rest on another.
            provider = await asyncio.to_thread(provider.pin_snapshot)
    except AdjustedDataUnavailable as exc:
        raise LakeError("adjusted_unavailable", str(exc)) from exc
    # The window is resolved once for the table; any request that may touch the
    # archived tier reads from the epoch.
    window_start, window_end, legacy_tail = resolve_window(
        timeframe, start, end, window_limit, from_epoch=listing != "listed"
    )
    if policy == "legacy":
        tail = legacy_tail
    result = BulkResult(
        price_mode=effective,
        timeframe=timeframe,
        adjustment_revision=silver_revision(provider) if effective == "adjusted" else None,
        pinned_silver_revision=silver_revision_pin,
        window_start=window_start,
        window_end=window_end,
    )
    for symbol in requested:
        try:
            status = check_listing(provider, listing, symbol, spec.name, timeframe, effective)
            bars = await provider.fetch_bars(
                symbol,
                timeframe,
                window_start,
                window_end,
                asset_class=spec.name,
                price_mode=effective,
                listing=status,
                tail=None if tail is None else tail + 1,
            )
        except LakeError as exc:
            result.missing[symbol] = exc.message
            continue
        except (AdjustedDataUnavailable, QueryTimeout) as exc:
            result.missing[symbol] = str(exc)
            continue
        if not bars and not artifact_exists(provider, symbol, timeframe, spec, effective, status):
            result.missing[symbol] = f"no artifact for {symbol} under {spec.partition}"
            continue
        trimmed, truncated = trim_tail(bars, tail)
        result.series[symbol] = BulkSeries(status, trimmed, truncated)
    return result
