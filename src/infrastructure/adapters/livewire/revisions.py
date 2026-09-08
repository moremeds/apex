"""Read and validate Livewire Silver revision manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping
from urllib.parse import unquote

from .paths import SUPPORTED_TIMEFRAMES, encode_symbol

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RevisionManifestError(ValueError):
    """A Silver revision manifest is malformed or references invalid artifacts."""


@dataclass(frozen=True)
class AffectedSymbol:
    """One symbol whose adjusted history changed in a revision."""

    symbol: str
    earliest_date: date
    timeframes: tuple[str, ...]


ArtifactKind = Literal["daily", "factors"]
_ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("daily", "factors")


@dataclass(frozen=True)
class SilverArtifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class SilverRevision:
    """One committed manifest; verify selected artifacts when they are read."""

    schema_version: int
    revision: int
    generation_id: str
    published_at: datetime
    corporate_actions_as_of: datetime
    affected: tuple[AffectedSymbol, ...]
    artifacts: Mapping[tuple[str, ArtifactKind], SilverArtifact] = field(
        default_factory=lambda: MappingProxyType({})
    )
    root: Path | None = None

    def artifact_path(self, symbol: str, kind: ArtifactKind, *, verify: bool = True) -> Path | None:
        artifact = self.artifacts.get((symbol, kind))
        if artifact is None:
            return None
        if self.root is None:
            raise RevisionManifestError("Silver snapshot has no root")
        path = (self.root / artifact.path).resolve()
        if not path.is_relative_to(self.root):
            raise RevisionManifestError(f"artifact outside Silver root: {artifact.path}")
        if verify:
            try:
                actual = RevisionManifestReader._sha256(path)
            except OSError as exc:
                raise RevisionManifestError(f"cannot read artifact {artifact.path}: {exc}") from exc
            if actual != artifact.sha256:
                raise RevisionManifestError(f"checksum mismatch for artifact {artifact.path}")
        return path

    def verify_artifacts(self) -> None:
        """Explicit full verification for audits, never implicit per chart request."""
        for symbol, kind in self.artifacts:
            self.artifact_path(symbol, kind)


class RevisionManifestReader:
    """Pin one current manifest, validating its structure and immutable commit record."""

    def __init__(self, silver_root: Path) -> None:
        self._root = Path(silver_root).resolve()

    def read_current(self) -> SilverRevision:
        manifest_path = self._root / "revisions" / "current.json"
        try:
            current_bytes = manifest_path.read_bytes()
            payload = json.loads(current_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RevisionManifestError(f"cannot read Silver revision manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise RevisionManifestError("Silver revision manifest must be a JSON object")

        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise RevisionManifestError(f"unsupported schema_version: {schema_version!r}")
        revision = payload.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise RevisionManifestError("revision must be a positive integer")
        immutable = self._root / "revisions" / f"revision={revision}.json"
        try:
            if immutable.read_bytes() != current_bytes:
                raise RevisionManifestError(
                    "current Silver pointer does not match immutable manifest"
                )
        except OSError as exc:
            raise RevisionManifestError(f"cannot read immutable Silver manifest: {exc}") from exc

        generation_id = payload.get("generation_id")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise RevisionManifestError("generation_id must be a non-empty string")

        published_at = self._utc_timestamp(payload.get("published_at"), "published_at")
        actions_as_of = self._utc_timestamp(
            payload.get("corporate_actions_as_of"), "corporate_actions_as_of"
        )
        affected = self._parse_affected(payload.get("affected"))
        artifacts = self._parse_artifacts(payload.get("artifacts"))
        expected_artifacts = {(item.symbol, kind) for item in affected for kind in _ARTIFACT_KINDS}
        if set(artifacts) != expected_artifacts:
            raise RevisionManifestError(
                "artifacts must contain exactly one daily and one factors entry "
                "for every affected symbol"
            )
        return SilverRevision(
            schema_version=1,
            revision=revision,
            generation_id=generation_id,
            published_at=published_at,
            corporate_actions_as_of=actions_as_of,
            affected=affected,
            artifacts=MappingProxyType(artifacts),
            root=self._root,
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
        if not isinstance(value, list) or not value:
            raise RevisionManifestError("affected must be a non-empty list")
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
            if any(not isinstance(tf, str) for tf in raw_timeframes):
                raise RevisionManifestError(f"timeframes must contain strings for {symbol}")
            if len(raw_timeframes) != len(set(raw_timeframes)):
                raise RevisionManifestError(f"duplicate timeframe for {symbol}")
            unsupported = [tf for tf in raw_timeframes if tf not in SUPPORTED_TIMEFRAMES]
            if unsupported:
                raise RevisionManifestError(
                    f"unsupported timeframe for {symbol}: {unsupported[0]!r}"
                )
            parsed.append(AffectedSymbol(symbol, earliest, tuple(raw_timeframes)))
        return tuple(parsed)

    def _parse_artifacts(self, value: Any) -> dict[tuple[str, ArtifactKind], SilverArtifact]:
        if not isinstance(value, list):
            raise RevisionManifestError("artifacts must be a list")
        seen: set[str] = set()
        parsed: dict[tuple[str, ArtifactKind], SilverArtifact] = {}
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
            candidate = PurePosixPath(raw_path)
            if candidate.is_absolute() or ".." in candidate.parts or "\\" in raw_path:
                raise RevisionManifestError(f"artifact outside Silver root: {raw_path}")
            parts = candidate.parts
            if (
                len(parts) < 3
                or parts[-3] != "asset_class=equity"
                or not parts[-2].startswith("symbol=")
            ):
                raise RevisionManifestError(f"invalid Silver artifact path: {raw_path}")
            encoded = parts[-2][len("symbol=") :]
            symbol = unquote(encoded)
            if not symbol or encode_symbol(symbol) != encoded:
                raise RevisionManifestError(f"invalid Silver symbol encoding: {raw_path}")
            kind: ArtifactKind
            if parts[-1] == "1d.parquet":
                kind = "daily"
            elif parts[-1] == "factors.parquet" and len(parts) >= 4 and parts[-4] == "adjustments":
                kind = "factors"
            else:
                raise RevisionManifestError(f"invalid Silver artifact path: {raw_path}")
            key = (symbol, kind)
            if key in parsed:
                raise RevisionManifestError(f"duplicate Silver artifact for {symbol}/{kind}")
            parsed[key] = SilverArtifact(raw_path, digest)
        return parsed

    @staticmethod
    def _sha256(path: Path) -> str:
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()
