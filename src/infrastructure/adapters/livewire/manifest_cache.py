"""Parsed-manifest cache keyed by the SHA-256 of the manifest's exact bytes.

Measured on the mini lake (2026-09-23, P1.6): parsing Silver revision 77's 6.8 MB
manifest costs ~113 ms and ``json.loads`` another ~14 ms, on every adjusted or pinned
read, while reading the bytes costs ~1 ms warm. Hashing the bytes (~5 ms) and reusing
the parse turns that into a lookup.

The key is content, never file metadata: a manifest replaced atomically with the same
size and mtime hashes differently and misses. Artifact bytes are still SHA-256 checked
on every read that serves them; only the manifest parse is reused.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


class ManifestCache(Generic[T]):
    """A small LRU of parsed manifests. ``max_entries`` bounds memory: one parsed
    Silver revision (27k artifact references) is on the order of 10 MB."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._entries: "OrderedDict[tuple[Hashable, str], T]" = OrderedDict()
        self._lock = threading.Lock()

    def get_or_parse(self, namespace: Hashable, raw: bytes, parse: Callable[[], T]) -> T:
        """Return the parse of ``raw``; ``parse`` runs only on a miss. A parse that
        raises is not cached, so a corrupt manifest is re-examined next time."""
        key = (namespace, hashlib.sha256(raw).hexdigest())
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
        value = parse()
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        return len(self._entries)
