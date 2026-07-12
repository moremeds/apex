"""Continuously apply atomically published Livewire Silver revisions."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Protocol

from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader, SilverRevision

logger = logging.getLogger(__name__)


class _RevisionManager(Protocol):
    async def refresh_revision(self, revision: SilverRevision) -> Any: ...


class RevisionWatcher:
    """Poll a Docker-mounted current manifest and reseed active subscriptions."""

    def __init__(
        self,
        reader: RevisionManifestReader,
        manager: _RevisionManager,
        poll_seconds: float = 30.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("revision poll interval must be positive")
        self._reader = reader
        self._manager = manager
        self._poll_seconds = poll_seconds
        self._task: asyncio.Task[None] | None = None
        self.observed_revision = 0
        self.last_fully_applied_revision = 0
        self.per_symbol_revision: dict[str, int] = {}
        self.failed: dict[str, str] = {}
        self.last_error: str | None = None
        self.last_success_at: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._task = asyncio.create_task(self._run(), name="silver-revision-watcher")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            await self.poll_once()
            await asyncio.sleep(self._poll_seconds)

    async def poll_once(self) -> None:
        """Read and apply one current revision; errors become health state."""
        try:
            revision = await asyncio.to_thread(self._reader.read_current)
            if revision.revision < self.observed_revision:
                raise ValueError(
                    f"Silver revision regressed from {self.observed_revision} to {revision.revision}"
                )
            if revision.revision == self.observed_revision and not self.failed:
                return

            self.observed_revision = revision.revision
            result = await self._manager.refresh_revision(revision)
            self.per_symbol_revision.update(result.applied)
            self.failed = dict(result.failed)
            if self.failed:
                self.last_error = "; ".join(
                    f"{symbol}: {error}" for symbol, error in sorted(self.failed.items())
                )
                return

            self.last_fully_applied_revision = revision.revision
            self.last_success_at = datetime.now(timezone.utc)
            self.last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep long-running watcher alive
            self.last_error = str(exc)
            logger.warning("Silver revision poll failed: %s", exc)

    def health(self) -> dict[str, Any]:
        age_seconds = None
        if self.last_success_at is not None:
            age_seconds = round(
                (datetime.now(timezone.utc) - self.last_success_at).total_seconds(), 1
            )
        return {
            "enabled": True,
            "running": self.is_running,
            "observed_revision": self.observed_revision,
            "last_fully_applied_revision": self.last_fully_applied_revision,
            "per_symbol_revision": dict(self.per_symbol_revision),
            "failed": dict(self.failed),
            "last_error": self.last_error,
            "last_success_age_seconds": age_seconds,
        }
