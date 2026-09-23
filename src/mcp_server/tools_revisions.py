"""Revision tools: retained Silver revisions and published PIT manifests.

There is no PIT "current": Livewire's PIT current pointer is shared by every index,
so a PIT revision is always named explicitly (design §3.4).
"""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.api.payload.lake import (
    pit_detail_payload,
    pit_list_payload,
    silver_detail_payload,
    silver_list_payload,
)
from src.application.lake import revisions
from src.application.lake.services import LakeServices
from src.mcp_server._common import lake_tool
from src.mcp_server.tools_discovery import Limit, Offset, Page


class SilverRevisions(Page):
    current: Optional[int] = None
    revisions: List[Dict[str, Any]]


class SilverRevision(Page):
    revision: int
    is_current: bool
    affected: List[Dict[str, Any]]


class PitRevisions(Page):
    revisions: List[Dict[str, Any]]


class PitRevision(Page):
    revision: int
    index_id: str
    publisher_status: str
    daily_bar_cutoff: str
    members: List[Dict[str, Any]]


def register(server: MCPServer, services: LakeServices) -> None:
    tool = lake_tool(server)

    @tool
    async def list_silver_revisions(limit: Limit = None, offset: Offset = None) -> SilverRevisions:
        """Retained numbered Silver revisions, newest first, current flagged."""
        result = await revisions.list_silver_revisions(services, limit=limit, offset=offset)
        return SilverRevisions(**silver_list_payload(result))

    @tool
    async def get_silver_revision(
        revision: Annotated[
            Optional[int], Field(description="omit for the current revision")
        ] = None,
        limit: Limit = None,
        offset: Offset = None,
    ) -> SilverRevision:
        """One Silver revision: summary and paged affected symbols (never the full manifest)."""
        detail = await revisions.silver_revision_detail(
            services, revision, limit=limit, offset=offset
        )
        return SilverRevision(**silver_detail_payload(detail))

    @tool
    async def list_pit_revisions(
        index_id: Optional[str] = None, limit: Limit = None, offset: Offset = None
    ) -> PitRevisions:
        """Published PIT revisions with per-index status and the latest per index."""
        result = await revisions.list_pit_revisions(
            services, index_id=index_id, limit=limit, offset=offset
        )
        return PitRevisions(**pit_list_payload(result))

    @tool
    async def get_pit_revision(
        revision: int, limit: Limit = None, offset: Offset = None
    ) -> PitRevision:
        """One PIT manifest: summary including publisher_status, and paged member scopes
        (`members`: security_id, symbol, session_from/session_to)."""
        detail = await revisions.pit_revision_detail(services, revision, limit=limit, offset=offset)
        return PitRevision(**pit_detail_payload(detail))
