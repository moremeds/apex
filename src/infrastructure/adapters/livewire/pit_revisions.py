"""Read Livewire point-in-time (PIT) Silver revision manifests.

Contract: Livewire ``clients/pit_silver_revision.py`` (``policy_version=pit-silver-v1``),
checked at Livewire 1fad842 on 2026-09-23 and against the first production publish
(revisions 1-2). Manifests live at ``<silver>/pit-revisions/revision={n}.json``.

What apex does, and deliberately does not do (design §3.4, user decision 2026-09-23):

- Parse a manifest by EXPLICIT revision. ``current.json`` is one pointer shared by
  every index (a byte copy of whichever index published last), so it is never used
  to resolve an index's PIT revision.
- Serve a symbol's daily artifact from ``inputs.silver_artifacts`` after comparing
  that entry's sha256 with the file bytes -- the same byte check ``revisions.py``
  applies to Silver.
- Echo Livewire's ``status`` (PROVEN / PARTIAL) unchanged.
- NOT replay lineage: ``input_hash``, the corporate-action receipt, the membership and
  security-master prefix hashes and the source evidence are Livewire's own ``verify``
  to check. The lake-root references are carried as metadata and never opened.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping
from urllib.parse import unquote

from .manifest_cache import ManifestCache
from .paths import encode_symbol
from .revisions import list_revision_numbers

POLICY_VERSION = "pit-silver-v1"
_STATUSES = ("PROVEN", "PARTIAL")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_KEYS = (
    "schema_version",
    "revision",
    "policy_version",
    "index_id",
    "status",
    "as_of",
    "published_at",
    "daily_bar_cutoff",
    "membership_revision",
    "silver_revision",
    "members",
    "inputs",
)

PublisherStatus = Literal["PROVEN", "PARTIAL"]


class PitRevisionNotFound(LookupError):
    """No numbered PIT manifest exists for the requested revision."""


class PitUnavailable(RuntimeError):
    """A PIT manifest or the artifact it names cannot be served as published.

    Covers a malformed manifest, a missing or evicted artifact, a hash mismatch and a
    member with no daily artifact entry. Never answered with another revision or
    with raw data.
    """


@dataclass(frozen=True)
class PitScope:
    """One member entry: the sessions ``symbol`` belongs to the index under this id.

    ``session_to`` is exclusive and ``None`` means still a member at the manifest's
    ``as_of`` (Livewire ``pit_silver_revision.py:316-318``).
    """

    symbol: str
    security_id: str
    session_from: date
    session_to: date | None
    effective_from: str | None
    effective_to: str | None
    membership_event_id: str | None
    identity_event_id: str | None

    def contains(self, trade_date: date) -> bool:
        return self.session_from <= trade_date and (
            self.session_to is None or trade_date < self.session_to
        )

    def intersects(self, start: date, end: date) -> bool:
        return self.session_from <= end and (self.session_to is None or start < self.session_to)


# (Silver-relative path, sha256): a plain tuple so ~14k references per cached manifest
# stay untracked by the GC (see revisions.SilverArtifact).
PitArtifact = tuple[str, str]


@dataclass(frozen=True)
class PitRevisionSummary:
    """What discovery lists per manifest, without members or artifacts."""

    revision: int
    index_id: str
    status: PublisherStatus
    as_of: datetime
    published_at: datetime
    daily_bar_cutoff: date
    silver_revision: int
    membership_revision: int
    member_count: int


@dataclass(frozen=True)
class PitRevision:
    """One parsed PIT manifest. Artifact bytes are checked only when served."""

    summary: PitRevisionSummary
    policy_version: str
    session_policy: str | None
    corporate_actions_as_of: str | None
    generation_id: str | None
    input_hash: str | None
    members: tuple[PitScope, ...]
    # inputs without silver_artifacts: silver_manifest, security_master, membership,
    # corporate-action receipt references -- metadata, never opened by apex.
    input_references: Mapping[str, Any]
    daily_artifacts: Mapping[str, PitArtifact]
    silver_root: Path

    @property
    def revision(self) -> int:
        return self.summary.revision

    def scopes_for(self, symbol: str) -> tuple[PitScope, ...]:
        return tuple(scope for scope in self.members if scope.symbol == symbol)

    def daily_artifact_path(self, symbol: str) -> Path:
        """The verified daily artifact for ``symbol``, or ``PitUnavailable``."""
        artifact = self.daily_artifacts.get(symbol)
        if artifact is None:
            raise PitUnavailable(
                f"PIT revision {self.revision} has no daily artifact entry for {symbol}"
            )
        relative, digest = artifact
        path = (self.silver_root / relative).resolve()
        if not path.is_relative_to(self.silver_root):
            raise PitUnavailable(f"PIT artifact escapes the Silver root: {relative}")
        try:
            actual = _sha256(path)
        except OSError as exc:
            raise PitUnavailable(
                f"PIT revision {self.revision} artifact for {symbol} is unreadable "
                f"(evicted or missing): {exc.strerror or exc}"
            ) from exc
        if actual != digest:
            raise PitUnavailable(
                f"PIT revision {self.revision} artifact for {symbol} does not match its hash"
            )
        return path


# Parsed PIT manifests by content hash; ~5 MB each parsed, so the LRU stays small.
_PARSED: "ManifestCache[PitRevision]" = ManifestCache(max_entries=8)


class PitRevisionReader:
    """Read-only access to ``<silver>/pit-revisions``."""

    def __init__(self, silver_root: Path) -> None:
        self._silver_root = Path(silver_root).resolve()

    @property
    def directory(self) -> Path:
        return self._silver_root / "pit-revisions"

    def available(self) -> bool:
        return bool(list_revision_numbers(self.directory))

    def list_revisions(self) -> list[PitRevisionSummary]:
        """Every retained manifest, newest first. A malformed one raises ``PitUnavailable``
        rather than silently shrinking the list. Each file is re-read and re-hashed;
        only the parse is reused, so a replaced file is never served stale."""
        return [self.read(revision).summary for revision in list_revision_numbers(self.directory)]

    def read(self, revision: int) -> PitRevision:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise PitRevisionNotFound(f"invalid PIT revision {revision!r}")
        path = self._path(revision)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise PitRevisionNotFound(f"PIT revision {revision} does not exist") from exc
        except OSError as exc:
            raise PitUnavailable(
                f"cannot read PIT revision {revision}: {exc.strerror or type(exc).__name__}"
            ) from exc

        def parse() -> PitRevision:
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PitUnavailable(f"PIT revision {revision} is not valid JSON") from exc
            return _parse(payload, revision, self._silver_root)

        manifest = _PARSED.get_or_parse(self._silver_root, raw, parse)
        if manifest.revision != revision:  # the same bytes cached under another name
            raise PitUnavailable(f"PIT revision {revision} is malformed: names another revision")
        return manifest

    def _path(self, revision: int) -> Path:
        return self.directory / f"revision={revision}.json"


def _parse(payload: Any, revision: int, silver_root: Path) -> PitRevision:
    """Every malformation -- missing key, wrong type, bad date -- is PitUnavailable,
    never an uncaught KeyError/ValueError (which would surface as a 500)."""
    try:
        return _parse_checked(payload, revision, silver_root)
    except PitUnavailable:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PitUnavailable(
            f"PIT revision {revision} is malformed: {type(exc).__name__}: {exc}"
        ) from exc


def _parse_checked(payload: Any, revision: int, silver_root: Path) -> PitRevision:
    def bad(reason: str) -> PitUnavailable:
        return PitUnavailable(f"PIT revision {revision} is malformed: {reason}")

    if not isinstance(payload, dict):
        raise bad("not a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise bad(f"missing {missing}")
    if payload["schema_version"] != 1:
        raise bad(f"unsupported schema_version {payload['schema_version']!r}")
    if payload["policy_version"] != POLICY_VERSION:
        raise bad(f"unsupported policy_version {payload['policy_version']!r}")
    if _positive_int(payload["revision"]) != revision:
        raise bad("filename and payload revision disagree")
    index_id = payload["index_id"]
    if not isinstance(index_id, str) or not index_id:
        raise bad("index_id must be a non-empty string")
    status = payload["status"]
    if status not in _STATUSES:
        raise bad(f"unknown status {status!r}")
    inputs = payload["inputs"]
    if not isinstance(inputs, dict):
        raise bad("inputs must be an object")
    try:
        summary_members = _parse_members(payload["members"])
        summary = PitRevisionSummary(
            revision=revision,
            index_id=index_id,
            status=status,
            as_of=_aware(payload["as_of"]),
            published_at=_aware(payload["published_at"]),
            daily_bar_cutoff=date.fromisoformat(str(payload["daily_bar_cutoff"])),
            silver_revision=_positive_int(payload["silver_revision"]),
            membership_revision=_non_negative_int(payload["membership_revision"]),
            member_count=len(summary_members),
        )
        daily = _parse_daily_artifacts(inputs.get("silver_artifacts"))
    except (TypeError, ValueError) as exc:
        raise bad(str(exc)) from exc
    return PitRevision(
        summary=summary,
        policy_version=POLICY_VERSION,
        session_policy=_optional_str(payload.get("session_policy")),
        corporate_actions_as_of=_optional_str(payload.get("corporate_actions_as_of")),
        generation_id=_optional_str(payload.get("generation_id")),
        input_hash=_optional_str(payload.get("input_hash")),
        members=summary_members,
        input_references=MappingProxyType(
            {key: value for key, value in inputs.items() if key != "silver_artifacts"}
        ),
        daily_artifacts=MappingProxyType(daily),
        silver_root=silver_root,
    )


def _parse_members(value: Any) -> tuple[PitScope, ...]:
    if not isinstance(value, list):
        raise ValueError("members must be a list")
    scopes = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("member entries must be objects")
        symbol, security_id = entry.get("symbol"), entry.get("security_id")
        if not isinstance(symbol, str) or not symbol or not isinstance(security_id, str):
            raise ValueError("member entries need symbol and security_id")
        session_to = entry.get("session_to")
        scope = PitScope(
            symbol=symbol,
            security_id=security_id,
            session_from=date.fromisoformat(str(entry["session_from"])),
            session_to=None if session_to is None else date.fromisoformat(str(session_to)),
            effective_from=_optional_str(entry.get("effective_from")),
            effective_to=_optional_str(entry.get("effective_to")),
            membership_event_id=_optional_str(entry.get("membership_event_id")),
            identity_event_id=_optional_str(entry.get("identity_event_id")),
        )
        if scope.session_to is not None and scope.session_to <= scope.session_from:
            raise ValueError(f"empty member scope for {symbol}")
        scopes.append(scope)
    return tuple(scopes)


def _parse_daily_artifacts(value: Any) -> dict[str, PitArtifact]:
    """Index the daily entries by symbol. Paths are relative to the Silver root
    (Livewire ``pit_silver_revision.py:408-425``); factors entries are not served."""
    if not isinstance(value, list):
        raise ValueError("inputs.silver_artifacts must be a list")
    daily: dict[str, PitArtifact] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("silver_artifacts entries must be objects")
        raw_path, digest = item.get("path"), item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise ValueError("silver_artifacts entries need path and sha256")
        parts = PurePosixPath(raw_path).parts
        if parts[-1:] != ("1d.parquet",) or len(parts) < 3 or parts[-3] != "asset_class=equity":
            continue
        if (
            PurePosixPath(raw_path).is_absolute()
            or ".." in parts
            or "\\" in raw_path
            or any(part.startswith("._") for part in parts)
            or _SHA256_RE.fullmatch(digest) is None
            or not parts[-2].startswith("symbol=")
        ):
            raise ValueError(f"invalid Silver artifact reference {raw_path!r}")
        encoded = parts[-2][len("symbol=") :]
        symbol = unquote(encoded)
        if not symbol or encode_symbol(symbol) != encoded:
            raise ValueError(f"invalid Silver symbol encoding {raw_path!r}")
        if symbol in daily:
            raise ValueError(f"duplicate daily artifact for {symbol}")
        daily[symbol] = (raw_path, digest)
    return daily


def _aware(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {value!r} has no offset")
    return parsed


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return int(value)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"expected a non-negative integer, got {value!r}")
    return int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()
