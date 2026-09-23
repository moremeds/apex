"""The lake sources a query may touch, gathered in one container.

REST builds one per request from ``app.state`` (so tests keep injecting fakes there);
the MCP server builds one at startup. Each field is optional: an unset source disables
only the queries that need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Sequence, TypeVar

from src.application.lake.errors import LakeError
from src.infrastructure.adapters.livewire.coverage import CoverageCatalog
from src.infrastructure.adapters.livewire.membership import MembershipReader
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.reference import LivewireReferenceReader
from src.infrastructure.adapters.livewire.repairs import RepairsReader
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader

T = TypeVar("T")

PAGE_DEFAULT = 100
PAGE_MAX = 2000


@dataclass(frozen=True)
class LakeServices:
    # Any, not LivewireOhlcProvider: route tests inject duck-typed fakes.
    provider: Any = None
    catalog: Optional[CoverageCatalog] = None
    membership: Optional[MembershipReader] = None
    reference: LivewireReferenceReader = field(
        default_factory=lambda: LivewireReferenceReader(None, None)
    )
    silver: Optional[RevisionManifestReader] = None
    pit: Optional[PitRevisionReader] = None
    repairs: RepairsReader = field(default_factory=lambda: RepairsReader(None))

    def require_provider(self) -> Any:
        if self.provider is None:
            raise LakeError("provider_not_configured", "bar provider not configured")
        return self.provider

    def require_catalog(self) -> CoverageCatalog:
        if self.catalog is None:
            raise LakeError(
                "provider_not_configured",
                "coverage catalog not configured (set APEX_LIVEWIRE_COVERAGE_DB)",
            )
        return self.catalog


@dataclass(frozen=True)
class Page(Generic[T]):
    """One page of a deterministic list. ``truncated`` comes from reading one item
    past the page, never from a guess."""

    items: List[T]
    limit: int
    offset: int
    truncated: bool

    @property
    def next_offset(self) -> Optional[int]:
        return self.offset + len(self.items) if self.truncated else None


def check_page(limit: Optional[int], offset: Optional[int]) -> tuple[int, int]:
    resolved = PAGE_DEFAULT if limit is None else limit
    start = 0 if offset is None else offset
    if not 1 <= resolved <= PAGE_MAX:
        raise LakeError("invalid_parameter", f"limit must be between 1 and {PAGE_MAX}")
    if start < 0:
        raise LakeError("invalid_parameter", "offset must be >= 0")
    return resolved, start


def page_of(items: Sequence[T], limit: int, offset: int) -> Page[T]:
    """Slice an already-ordered in-memory list."""
    window = list(items[offset : offset + limit + 1])
    return Page(window[:limit], limit, offset, len(window) > limit)
