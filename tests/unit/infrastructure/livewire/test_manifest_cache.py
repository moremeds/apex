"""The parsed-manifest cache keys on content, never on file metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.infrastructure.adapters.livewire.manifest_cache import ManifestCache
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from tests.support.pit_manifest import pit_payload, publish_pit


def test_hit_reuses_the_parse_and_eviction_is_bounded() -> None:
    cache: ManifestCache[str] = ManifestCache(max_entries=2)
    calls: list[bytes] = []

    def parse(raw: bytes):
        return lambda: calls.append(raw) or raw.decode()

    for raw in (b"a", b"a", b"b", b"c", b"a"):
        cache.get_or_parse("ns", raw, parse(raw))
    assert calls == [b"a", b"b", b"c", b"a"] and len(cache) == 2


def test_failed_parse_is_not_cached() -> None:
    cache: ManifestCache[str] = ManifestCache(max_entries=2)

    def boom() -> str:
        raise ValueError("corrupt")

    for _ in range(2):
        try:
            cache.get_or_parse("ns", b"x", boom)
        except ValueError:
            pass
    assert len(cache) == 0


def test_same_size_same_mtime_replacement_is_reparsed(tmp_path: Path) -> None:
    """Atomic replacement that keeps size and mtime: the status byte flips, the
    metadata does not, and the reader still sees the new content."""
    path = publish_pit(tmp_path, pit_payload(tmp_path, 1, status="PARTIAL"))
    reader = PitRevisionReader(tmp_path)
    assert reader.read(1).summary.status == "PARTIAL"
    stat = path.stat()

    payload = json.loads(path.read_bytes())
    # PARTIAL -> PROVEN drops one byte; one extra byte in generation_id restores the size.
    payload["status"] = "PROVEN"
    payload["generation_id"] += "x"
    replacement = json.dumps(payload, sort_keys=True).encode()
    assert len(replacement) == stat.st_size
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(replacement)
    os.utime(temporary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(temporary, path)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)

    assert reader.read(1).summary.status == "PROVEN"
    assert reader.list_revisions()[0].status == "PROVEN"
