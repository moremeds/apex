"""Discovery tools: asset classes, instruments, coverage, gaps, lake status, futures."""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.api.payload.lake import (
    coverage_payload,
    futures_payload,
    gaps_payload,
    instrument_payload,
    instruments_payload,
)
from src.application.lake import catalog, gaps
from src.application.lake.services import LakeServices, check_page
from src.mcp_server._common import Result, lake_tool, parse_day

Limit = Annotated[Optional[int], Field(description="page size (1..2000, default 100)")]
Offset = Annotated[Optional[int], Field(description="page offset (default 0)")]


class Page(Result):
    limit: int
    offset: int
    returned: int
    truncated: bool
    next_offset: Optional[int] = None


class AssetClasses(Result):
    asset_classes: List[Dict[str, Any]]


class Instruments(Result):
    instruments: List[Dict[str, Any]]
    count: int
    source: str


class Instrument(Result):
    symbol: str
    asset_class: str
    timeframes: Any
    residency: Any


class Coverage(Page):
    catalog: Dict[str, Any]
    rows: List[Dict[str, Any]]


class Gaps(Result):
    symbol: str
    asset_class: str
    timeframe: str
    status: str
    window: Dict[str, Any]
    gaps: List[Dict[str, Any]]
    gaps_total: int
    truncated: bool


class LakeStatus(Result):
    sources: Dict[str, Any]


class FuturesContracts(Page):
    root: str
    contracts: List[Dict[str, Any]]


def register(server: MCPServer, services: LakeServices) -> None:
    tool = lake_tool(server)

    @tool
    async def list_asset_classes() -> AssetClasses:
        """Registered asset classes: payload kind, timeframes, adjusted support."""
        return AssetClasses(asset_classes=catalog.asset_classes())

    @tool
    async def search_instruments(
        q: Annotated[Optional[str], Field(description="symbol prefix")] = None,
        asset_class: Optional[str] = None,
        limit: Limit = None,
    ) -> Instruments:
        """Live instruments from the coverage catalog (dates are its daily snapshot)."""
        rows = await catalog.search_instruments(
            services, q=q, asset_class=asset_class, limit=check_page(limit, None)[0]
        )
        return Instruments(**instruments_payload(rows))

    @tool
    async def get_instrument(symbol: str, asset_class: str = "equity") -> Instrument:
        """One instrument: catalog row plus the timeframes that exist on disk and where."""
        return Instrument(
            **instrument_payload(await catalog.get_instrument(services, symbol, asset_class))
        )

    @tool
    async def get_coverage(
        symbol: Annotated[Optional[str], Field(description="exact symbol")] = None,
        asset_class: Optional[str] = None,
        include_silver: bool = True,
        limit: Limit = None,
        offset: Offset = None,
    ) -> Coverage:
        """Raw coverage-catalog rows, paged; echoes the catalog identity."""
        result = await catalog.coverage(
            services,
            symbol=symbol,
            asset_class=asset_class,
            include_silver=include_silver,
            limit=limit,
            offset=offset,
        )
        return Coverage(**coverage_payload(result))

    @tool
    async def find_gaps(
        symbol: str,
        asset_class: str = "equity",
        timeframe: str = "1d",
        start: Annotated[
            Optional[str], Field(description="first session, YYYY-MM-DD (default end - 365d)")
        ] = None,
        end: Annotated[
            Optional[str], Field(description="last session, YYYY-MM-DD (default today UTC)")
        ] = None,
        listing: Annotated[str, Field(description="listed | delisted | any")] = "listed",
        max_gaps: Annotated[int, Field(description="1..2000")] = 100,
        calendar: Annotated[str, Field(description="auto | xnys | weekdays")] = "auto",
    ) -> Gaps:
        """Session-presence gaps against an explicit calendar, with repair evidence.
        Presence of a session is not proof of intraday completeness."""
        first, last = gaps.gap_window_default(parse_day(end, "end"), parse_day(start, "start"))
        result = await gaps.find_gaps(
            services,
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            start=first,
            end=last,
            listing=listing,
            max_gaps=max_gaps,
            calendar=calendar,  # type: ignore[arg-type]  # validated in find_gaps
        )
        return Gaps(**gaps_payload(result))

    @tool
    async def get_lake_status() -> LakeStatus:
        """Per-source configured / available / freshness. No paths or secrets."""
        return LakeStatus(sources=await catalog.lake_status(services))

    @tool
    async def list_futures_contracts(
        root: str, limit: Limit = None, offset: Offset = None
    ) -> FuturesContracts:
        """Contracts under one futures root, each with identity and catalog presence."""
        page = await catalog.futures_contracts(services, root, limit=limit, offset=offset)
        return FuturesContracts(**futures_payload(root, page))
