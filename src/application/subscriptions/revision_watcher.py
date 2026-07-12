"""Continuously apply atomically published Livewire Silver revisions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol

from src.infrastructure.adapters.livewire.revisions import (
    AffectedSymbol,
    RevisionManifestReader,
    SilverRevision,
)

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
        max_backoff_multiplier: int = 8,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("revision poll interval must be positive")
        if max_backoff_multiplier < 1:
            raise ValueError("max backoff multiplier must be >= 1")
        self._reader = reader
        self._manager = manager
        self._poll_seconds = poll_seconds
        self._max_backoff_multiplier = max_backoff_multiplier
        self._task: asyncio.Task[None] | None = None
        self.observed_revision = 0
        self.last_fully_applied_revision = 0
        self.per_symbol_revision: dict[str, int] = {}
        self.failed: dict[str, str] = {}
        # Symbols still needing a successful reseed, carried across polls so a
        # later revision that does not re-list them cannot let
        # last_fully_applied_revision advance past their stale state.
        self._pending: dict[str, AffectedSymbol] = {}
        self._consecutive_failures = 0
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
            await asyncio.sleep(self._next_delay())

    def _next_delay(self) -> float:
        """Fixed interval on success; exponential (capped) backoff on failure."""
        if not self._consecutive_failures:
            return self._poll_seconds
        # int() pins the type: typeshed types ``2 ** n`` as Any for a variable
        # exponent (negative powers yield float), which would leak into the return.
        factor: int = min(int(2**self._consecutive_failures), self._max_backoff_multiplier)
        return self._poll_seconds * factor

    async def poll_once(self) -> None:
        """Read and apply one current revision; errors become health state."""
        try:
            revision = await asyncio.to_thread(self._reader.read_current)
            if revision.revision < self.observed_revision:
                raise ValueError(
                    f"Silver revision regressed from {self.observed_revision} to {revision.revision}"
                )

            is_new = revision.revision > self.observed_revision
            if not is_new and not self._pending:
                return

            # Attempt = symbols this revision changed, merged with symbols still
            # owed a successful reseed from an earlier revision.
            attempt: dict[str, AffectedSymbol] = dict(self._pending)
            if is_new:
                for item in revision.affected:
                    attempt[item.symbol] = item
            self.observed_revision = revision.revision

            if not attempt:
                # A new revision that changed no currently-owed symbol is
                # trivially fully applied.
                self._mark_fully_applied(revision.revision)
                return

            merged = replace(revision, affected=tuple(attempt.values()))
            result = await self._manager.refresh_revision(merged)

            for symbol, applied_revision in result.applied.items():
                self.per_symbol_revision[symbol] = applied_revision
                self._pending.pop(symbol, None)
            # Symbols we asked to reseed that came back neither applied nor
            # failed are no longer active (unsubscribed) — stop owing them.
            for symbol in attempt:
                if symbol not in result.applied and symbol not in result.failed:
                    self._pending.pop(symbol, None)
            for symbol, error in result.failed.items():
                self._pending[symbol] = attempt[symbol]
            self.failed = dict(result.failed)

            if self._pending:
                self._consecutive_failures += 1
                self.last_error = "; ".join(
                    f"{symbol}: {error}" for symbol, error in sorted(result.failed.items())
                ) or f"pending reseed: {', '.join(sorted(self._pending))}"
                return

            self._mark_fully_applied(revision.revision)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep long-running watcher alive
            self._consecutive_failures += 1
            self.last_error = str(exc)
            logger.warning("Silver revision poll failed: %s", exc)

    def _mark_fully_applied(self, revision: int) -> None:
        self.failed = {}
        self._consecutive_failures = 0
        self.last_fully_applied_revision = revision
        self.last_success_at = datetime.now(timezone.utc)
        self.last_error = None

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
            "pending": sorted(self._pending),
            "consecutive_failures": self._consecutive_failures,
            "last_error": self.last_error,
            "last_success_age_seconds": age_seconds,
        }
