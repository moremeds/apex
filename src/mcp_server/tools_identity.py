"""Identity tools: corporate actions, delisting intervals, security resolution, membership.

Lists always page here (design §3.2); REST keeps its unpaged envelopes unless asked.
"""

from datetime import date
from typing import Annotated, Any, Dict, List, Literal, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.api.payload.lake import (
    actions_payload,
    delisting_payload,
    history_payload,
    indices_payload,
    members_payload,
    security_payload,
)
from src.application.lake import identity
from src.application.lake.services import LakeServices, check_page, page_of
from src.mcp_server._common import Result, lake_tool
from src.mcp_server.tools_discovery import Limit, Offset, Page

AsOf = Annotated[Optional[date], Field(description="YYYY-MM-DD (default today UTC)")]
KnownAt = Annotated[
    Optional[date], Field(description="only facts known by this date (point-in-time)")
]


class Actions(Page):
    symbol: str
    identity: str
    actions: List[Dict[str, Any]]


class Delisting(Page):
    symbol: str
    identity: str
    delisting_reason_available: bool
    intervals: List[Dict[str, Any]]


class Security(Result):
    symbol: str
    security_id: Optional[str] = None


class Indices(Page):
    indices: List[str]


class Members(Page):
    index_id: str
    as_of: str
    members: List[Dict[str, Any]]
    unresolved_count: int
    total: int


class History(Page):
    symbol: str
    security_id: Optional[str] = None
    events: List[Dict[str, Any]]


def register(server: MCPServer, services: LakeServices) -> None:
    tool = lake_tool(server)

    @tool
    async def get_corporate_actions(
        symbol: str,
        action_type: Optional[Literal["split", "cash_dividend"]] = None,
        start: Annotated[Optional[date], Field(description="earliest ex_date")] = None,
        end: Annotated[Optional[date], Field(description="latest ex_date")] = None,
        limit: Limit = None,
        offset: Offset = None,
    ) -> Actions:
        """Active corporate actions for a TICKER. A reused ticker's actions may belong to
        another issuer; check get_delisting / resolve_security first."""
        result = await identity.corporate_actions(
            services,
            symbol,
            action_type=action_type,
            start=start,
            end=end,
            limit=limit,
            offset=offset,
            paged=True,
        )
        return Actions(**actions_payload(result, paged=True))

    @tool
    async def get_delisting(symbol: str, limit: Limit = None, offset: Offset = None) -> Delisting:
        """Identity intervals for a ticker. No delisting reason or consideration exists
        in the source; a closed interval says only that the ticker stopped resolving."""
        page = await identity.delisting(services, symbol, limit=limit, offset=offset, paged=True)
        return Delisting(**delisting_payload(symbol, page, paged=True))

    @tool
    async def resolve_security(
        symbol: str, as_of: AsOf = None, known_at: KnownAt = None
    ) -> Security:
        """Ticker -> one security_id at a date; ambiguity is an error, never a guess."""
        resolution = await identity.resolve_security(
            services, symbol, as_of or identity.today_utc(), known_at
        )
        return Security(**security_payload(resolution))

    @tool
    async def list_indices(limit: Limit = None, offset: Offset = None) -> Indices:
        """Index ids present in the lake."""
        indices = await identity.list_indices(services)
        return Indices(**indices_payload(indices, page_of(indices, *check_page(limit, offset))))

    @tool
    async def get_index_members(
        index_id: str,
        as_of: AsOf = None,
        known_at: KnownAt = None,
        include_candidates: Annotated[
            bool, Field(description="every non-rejected event, not just verified ones")
        ] = False,
        limit: Limit = None,
        offset: Offset = None,
    ) -> Members:
        """Members of an index at a date, replayed from its event log."""
        result = await identity.index_members(
            services,
            index_id,
            as_of or identity.today_utc(),
            known_at=known_at,
            include_candidates=include_candidates,
            limit=limit,
            offset=offset,
            paged=True,
        )
        return Members(**members_payload(result, paged=True))

    @tool
    async def get_membership_history(
        symbol: str,
        as_of: Annotated[
            Optional[date], Field(description="date used to resolve the ticker")
        ] = None,
        index_id: Optional[str] = None,
        limit: Limit = None,
        offset: Offset = None,
    ) -> History:
        """A security's effective index-membership timeline."""
        result = await identity.membership_history(
            services,
            symbol,
            as_of or identity.today_utc(),
            index_id=index_id,
            limit=limit,
            offset=offset,
            paged=True,
        )
        return History(**history_payload(result, paged=True))
