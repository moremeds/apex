"""
OPT-014: Read-Copy-Update (RCU) pattern for lock-free reads.

This module provides RCU-based data structures that enable:
- Completely lock-free reads (no contention)
- Copy-on-write updates (atomic reference swap)
- Thread-safe concurrent access

The key insight is that Python's reference assignment is atomic, so readers
always see a consistent snapshot without needing locks. Writers create a
new copy and atomically swap the reference.

Performance characteristics:
- Reads: O(1), no locks, no contention
- Writes: O(n) copy + O(1) atomic swap
- Memory: 2x during write (briefly)

Usage:
    store = RCUDict[str, MarketData]()
    store.set("AAPL", MarketData(...))

    # Lock-free read
    data = store.get("AAPL")

    # Batch update (single copy)
    store.update({"AAPL": md1, "GOOG": md2})
"""

from __future__ import annotations

import threading
from typing import TypeVar, Generic, Dict, Optional, List, Callable, Iterator

K = TypeVar('K')
V = TypeVar('V')


class RCUDict(Generic[K, V]):
    """
    Read-Copy-Update dictionary for lock-free reads.

    Reads are completely lock-free - they see a consistent snapshot.
    Writes use copy-on-write with atomic reference swap.

    Thread-safe for concurrent reads and writes.

    Example:
        >>> store = RCUDict[str, int]()
        >>> store.set("x", 1)
        >>> store.get("x")
        1
        >>> store.update({"y": 2, "z": 3})
        >>> len(store)
        3
    """

    __slots__ = ('_data', '_write_lock', '_version')

    def __init__(self, initial: Optional[Dict[K, V]] = None):
        """
        Initialize RCU dictionary.

        Args:
            initial: Optional initial data (will be copied).
        """
        self._data: Dict[K, V] = dict(initial) if initial else {}
        self._write_lock = threading.Lock()
        self._version = 0

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Lock-free read of a single key.

        Args:
            key: Key to look up.
            default: Value to return if key not found.

        Returns:
            Value for key, or default if not found.
        """
        return self._data.get(key, default)

    def get_all(self) -> Dict[K, V]:
        """
        Lock-free snapshot of all data.

        Returns:
            Reference to internal dict (do not modify!).

        Note:
            The returned dict should be treated as immutable.
            For a safe copy, use dict(store.get_all()).
        """
        return self._data

    def get_many(self, keys: List[K]) -> Dict[K, V]:
        """
        Lock-free read of multiple keys.

        Args:
            keys: List of keys to look up.

        Returns:
            Dict of found keys and their values.
        """
        data = self._data  # Single atomic read
        return {k: data[k] for k in keys if k in data}

    def set(self, key: K, value: V) -> None:
        """
        Copy-on-write update of a single key.

        Args:
            key: Key to set.
            value: Value to set.
        """
        with self._write_lock:
            new_data = dict(self._data)
            new_data[key] = value
            self._data = new_data  # Atomic reference swap
            self._version += 1

    def update(self, updates: Dict[K, V]) -> None:
        """
        Batch copy-on-write update.

        More efficient than multiple set() calls as it creates
        only one copy.

        Args:
            updates: Dict of key-value pairs to update.
        """
        if not updates:
            return
        with self._write_lock:
            new_data = dict(self._data)
            new_data.update(updates)
            self._data = new_data
            self._version += 1

    def delete(self, key: K) -> bool:
        """
        Copy-on-write delete of a key.

        Args:
            key: Key to delete.

        Returns:
            True if key was deleted, False if not found.
        """
        with self._write_lock:
            if key not in self._data:
                return False
            new_data = dict(self._data)
            del new_data[key]
            self._data = new_data
            self._version += 1
            return True

    def batch_delete(self, keys: List[K]) -> int:
        """
        HIGH-008: Copy-on-write delete of multiple keys in single copy.

        This is O(n) vs O(n*k) for k individual delete() calls.

        Args:
            keys: List of keys to delete.

        Returns:
            Number of keys actually deleted.
        """
        if not keys:
            return 0
        with self._write_lock:
            # Find which keys actually exist
            to_delete = [k for k in keys if k in self._data]
            if not to_delete:
                return 0
            # HIGH-008 FIX: Create set once before comprehension (was O(n²) recreating set each iteration)
            delete_set = set(to_delete)
            # Single copy with all deletions - O(n)
            new_data = {k: v for k, v in self._data.items() if k not in delete_set}
            self._data = new_data
            self._version += 1
            return len(to_delete)

    def clear(self) -> None:
        """Clear all data (atomic)."""
        with self._write_lock:
            self._data = {}
            self._version += 1

    def compute_if_absent(self, key: K, factory: Callable[[], V]) -> V:
        """
        Get value or create it atomically if not present.

        Args:
            key: Key to look up.
            factory: Function to create value if key not found.

        Returns:
            Existing or newly created value.
        """
        # Fast path: check without lock
        value = self._data.get(key)
        if value is not None:
            return value

        # Slow path with lock
        with self._write_lock:
            # Double-check after acquiring lock
            if key in self._data:
                return self._data[key]

            value = factory()
            new_data = dict(self._data)
            new_data[key] = value
            self._data = new_data
            self._version += 1
            return value

    def __len__(self) -> int:
        """Return number of items."""
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        """Check if key exists (lock-free)."""
        return key in self._data

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys (snapshot)."""
        return iter(self._data)

    def keys(self) -> List[K]:
        """Return list of keys (snapshot)."""
        return list(self._data.keys())

    def values(self) -> List[V]:
        """Return list of values (snapshot)."""
        return list(self._data.values())

    def items(self) -> List[tuple]:
        """Return list of (key, value) pairs (snapshot)."""
        return list(self._data.items())

    @property
    def version(self) -> int:
        """
        Get current version number.

        Useful for change detection without comparing data.
        """
        return self._version


class RCUList(Generic[V]):
    """
    Read-Copy-Update list for lock-free reads.

    Similar to RCUDict but for ordered sequences.
    """

    __slots__ = ('_data', '_write_lock', '_version')

    def __init__(self, initial: Optional[List[V]] = None):
        """
        Initialize RCU list.

        Args:
            initial: Optional initial data (will be copied).
        """
        self._data: List[V] = list(initial) if initial else []
        self._write_lock = threading.Lock()
        self._version = 0

    def get_all(self) -> List[V]:
        """
        Lock-free snapshot of all data.

        Returns:
            Reference to internal list (do not modify!).
        """
        return self._data

    def set_all(self, items: List[V]) -> None:
        """
        Replace all data atomically.

        Args:
            items: New list of items (will be copied).
        """
        with self._write_lock:
            self._data = list(items)
            self._version += 1

    def append(self, item: V) -> None:
        """Append item (copy-on-write)."""
        with self._write_lock:
            new_data = list(self._data)
            new_data.append(item)
            self._data = new_data
            self._version += 1

    def clear(self) -> None:
        """Clear all data (atomic)."""
        with self._write_lock:
            self._data = []
            self._version += 1

    def __len__(self) -> int:
        """Return number of items."""
        return len(self._data)

    def __iter__(self) -> Iterator[V]:
        """Iterate over items (snapshot)."""
        return iter(self._data)

    @property
    def version(self) -> int:
        """Get current version number."""
        return self._version
