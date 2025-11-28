"""Thread-safe in-memory account store."""

from __future__ import annotations
from typing import Optional
from threading import RLock
from ...models.account import AccountInfo


class AccountStore:
    """Thread-safe in-memory account store (single account)."""

    def __init__(self) -> None:
        self._account: Optional[AccountInfo] = None
        self._lock = RLock()

    def update(self, account: AccountInfo) -> None:
        """
        Update account information.

        Args:
            account: AccountInfo object.
        """
        with self._lock:
            self._account = account

    def get(self) -> Optional[AccountInfo]:
        """Get latest account information."""
        with self._lock:
            return self._account

    def get_latest(self) -> Optional[AccountInfo]:
        """Alias for get() - Get latest account information."""
        return self.get()

    def clear(self) -> None:
        """Clear account information."""
        with self._lock:
            self._account = None
