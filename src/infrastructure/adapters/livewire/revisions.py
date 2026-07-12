"""Read and validate Livewire Silver revision manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .paths import SUPPORTED_TIMEFRAMES

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RevisionManifestError(ValueError):
    """A Silver revision manifest is malformed or references invalid artifacts."""


@dataclass(frozen=True)
class AffectedSymbol:
    """One symbol whose adjusted history changed in a revision."""

    symbol: str
    earliest_date: date
    timeframes: tuple[str, ...]


@dataclass(frozen=True)
class SilverRevision:
    """Validated current Silver revision."""

    schema_version: int
    revision: int
    generation_id: str
    published_at: datetime
    corporate_actions_as_of: datetime
    affected: tuple[AffectedSymbol, ...]


class RevisionManifestReader:
    """Load ``revisions/current.json`` and verify every referenced artifact."""

    def __init__(self, silver_root: Path) -> None:
        self._root = Path(silver_root).resolve()

    def read_current(self) -> SilverRevision:
        manifest_path = self._root / "revisions" / "current.json"
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RevisionManifestError(f"cannot read Silver revision manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise RevisionManifestError("Silver revision manifest must be a JSON object")

        schema_version = payload.get("schema_version")
        if schema_version != 1:
            raise RevisionManifestError(f"unsupported schema_version: {schema_version!r}")
        revision = payload.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise RevisionManifestError("revision must be a positive integer")

        generation_id = payload.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise RevisionManifestError("generation_id must be a non-empty string")

        published_at = self._utc_timestamp(payload.get("published_at"), "published_at")
        actions_as_of = self._utc_timestamp(
            payload.get("corporate_actions_as_of"), "corporate_actions_as_of"
        )
        affected = self._parse_affected(payload.get("affected"))
        self._verify_artifacts(payload.get("artifacts"))
        return SilverRevision(
            schema_version=1,
            revision=revision,
            generation_id=generation_id,
            published_at=published_at,
            corporate_actions_as_of=actions_as_of,
            affected=affected,
        )

    @staticmethod
    def _utc_timestamp(value: Any, field: str) -> datetime:
        if not isinstance(value, str):
            raise RevisionManifestError(f"{field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RevisionManifestError(f"{field} must be an ISO timestamp") from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise RevisionManifestError(f"{field} must be UTC")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_affected(value: Any) -> tuple[AffectedSymbol, ...]:
        if not isinstance(value, list):
            raise RevisionManifestError("affected must be a list")
        parsed: list[AffectedSymbol] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                raise RevisionManifestError("affected entries must be objects")
            symbol = item.get("symbol")
            if not isinstance(symbol, str) or not symbol.strip():
                raise RevisionManifestError("affected symbol must be non-empty")
            if symbol in seen:
                raise RevisionManifestError(f"duplicate affected symbol: {symbol}")
            seen.add(symbol)
            try:
                earliest = date.fromisoformat(item["earliest_date"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RevisionManifestError(f"invalid earliest_date for {symbol}") from exc
            raw_timeframes = item.get("timeframes")
            if not isinstance(raw_timeframes, list) or not raw_timeframes:
                raise RevisionManifestError(f"timeframes must be a non-empty list for {symbol}")
            if len(raw_timeframes) != len(set(raw_timeframes)):
                raise RevisionManifestError(f"duplicate timeframe for {symbol}")
            unsupported = [tf for tf in raw_timeframes if tf not in SUPPORTED_TIMEFRAMES]
            if unsupported:
                raise RevisionManifestError(
                    f"unsupported timeframe for {symbol}: {unsupported[0]!r}"
                )
            parsed.append(AffectedSymbol(symbol, earliest, tuple(raw_timeframes)))
        return tuple(parsed)

    def _verify_artifacts(self, value: Any) -> None:
        if not isinstance(value, list):
            raise RevisionManifestError("artifacts must be a list")
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, dict):
                raise RevisionManifestError("artifact entries must be objects")
            raw_path = item.get("path")
            digest = item.get("sha256")
            if not isinstance(raw_path, str) or not raw_path:
                raise RevisionManifestError("artifact path must be non-empty")
            if raw_path in seen:
                raise RevisionManifestError(f"duplicate artifact path: {raw_path}")
            seen.add(raw_path)
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise RevisionManifestError(f"invalid sha256 for artifact {raw_path}")
            candidate = Path(raw_path)
            if candidate.is_absolute():
                raise RevisionManifestError(f"artifact outside Silver root: {raw_path}")
            resolved = (self._root / candidate).resolve()
            if not resolved.is_relative_to(self._root):
                raise RevisionManifestError(f"artifact outside Silver root: {raw_path}")
            try:
                actual = self._sha256(resolved)
            except OSError as exc:
                raise RevisionManifestError(f"cannot read artifact {raw_path}: {exc}") from exc
            if actual != digest:
                raise RevisionManifestError(f"checksum mismatch for artifact {raw_path}")

    @staticmethod
    def _sha256(path: Path) -> str:
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()
