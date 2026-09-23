"""Series tools: get_bars, get_bulk_bars, get_rate_series.

Same shared queries as REST under the bounded output policy (design §3.2), rendered
by the REST payload builders, with each series turned into columns + rows.
"""

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.api.payload.chart import (
    bars_payload_from,
    bulk_bars_payload_from,
    rates_payload_from,
)
from src.application.lake.bars import query_bars, query_rates
from src.application.lake.bulk import query_bulk_bars
from src.application.lake.services import LakeServices
from src.mcp_server._common import Result, columnar, lake_tool

Start = Annotated[
    Optional[datetime],
    Field(description="window start, ISO-8601 with timezone; omit for the default window"),
]
End = Annotated[
    Optional[datetime],
    Field(description="window end, ISO-8601 with timezone; default now"),
]
PriceMode = Annotated[
    Optional[Literal["raw", "adjusted"]],
    Field(description="omit for the server's configured mode"),
]
Listing = Annotated[
    Literal["listed", "delisted", "any"],
    Field(description="listed = live tree, delisted = archive, any = both (live wins a date)"),
]
SilverPin = Annotated[
    Optional[int],
    Field(description="pin a retained numbered Silver revision (equity, 1d, listed)"),
]


class Series(Result):
    columns: List[str]
    rows: List[List[Any]]
    truncated: bool
    window: Dict[str, Any]


class Bars(Series):
    symbol: str
    asset_class: str
    timeframe: str
    price_mode: str
    listing_status: Optional[str] = None
    provenance: Dict[str, Any]


class BulkBars(Result):
    timeframe: str
    price_mode: str
    window: Dict[str, Any]
    symbols: Dict[str, Dict[str, Any]]
    missing: Dict[str, Any]


class Rates(Series):
    symbol: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_columns(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {**{k: v for k, v in payload.items() if k != key}, **columnar(payload[key])}


def register(server: MCPServer, services: LakeServices) -> None:
    tool = lake_tool(server)

    @tool
    async def get_bars(
        symbol: str,
        asset_class: str = "equity",
        timeframe: str = "1d",
        start: Start = None,
        end: End = None,
        limit: Annotated[
            Optional[int],
            Field(description="last N rows of the window (1..5000, default 250)"),
        ] = None,
        price_mode: PriceMode = None,
        listing: Listing = "listed",
        silver_revision: SilverPin = None,
        pit_revision: Annotated[
            Optional[int],
            Field(description="serve through one published PIT revision (equity, 1d, listed)"),
        ] = None,
    ) -> Bars:
        """OHLCV bars for one symbol, oldest first; `truncated` says older rows exist."""
        result = await query_bars(
            services,
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            price_mode=price_mode,
            listing=listing,
            silver_revision_pin=silver_revision,
            pit_revision=pit_revision,
            policy="bounded",
        )
        return Bars(**_as_columns(bars_payload_from(result, generated_at=_now()), "bars"))

    @tool
    async def get_bulk_bars(
        symbols: Annotated[List[str], Field(description="1..200 equity tickers")],
        timeframe: str = "1d",
        start: Start = None,
        end: End = None,
        limit: Annotated[
            Optional[int],
            Field(description="rows per symbol (1..2000, default 50; symbols x limit <= 10000)"),
        ] = None,
        price_mode: PriceMode = None,
        listing: Listing = "listed",
        silver_revision: SilverPin = None,
    ) -> BulkBars:
        """Equity bars for many symbols on one Silver revision; absent ones are in `missing`."""
        result = await query_bulk_bars(
            services,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            price_mode=price_mode,
            listing=listing,
            silver_revision_pin=silver_revision,
            policy="bounded",
        )
        payload = bulk_bars_payload_from(result, generated_at=_now())
        payload["symbols"] = {
            symbol: _as_columns(series, "bars") for symbol, series in payload["symbols"].items()
        }
        return BulkBars(**payload)

    @tool
    async def get_rate_series(
        symbol: str,
        start: Start = None,
        end: End = None,
        limit: Annotated[
            Optional[int], Field(description="last N points (1..5000, default 500)")
        ] = None,
    ) -> Rates:
        """A Treasury yield series (percent), oldest first."""
        result = await query_rates(
            services, symbol=symbol, start=start, end=end, limit=limit, policy="bounded"
        )
        payload = rates_payload_from(result, generated_at=_now(), bounded=True)
        return Rates(**_as_columns(payload, "points"))
