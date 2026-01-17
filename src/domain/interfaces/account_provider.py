"""Account provider protocol for account data."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, runtime_checkable

from ..events.domain_events import AccountSnapshot


@runtime_checkable
class AccountProvider(Protocol):
    """
    Protocol for account data providers.

    Implementations:
    - IbLiveAdapter
    - FutuAdapter
    - BrokerManager (aggregates multiple accounts)

    Usage:
        provider: AccountProvider = IbLiveAdapter(...)
        account = await provider.fetch_account()
    """

    async def connect(self) -> None:
        """
        Connect to the account source.

        Raises:
            ConnectionError: If unable to connect.
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect from the account source."""
        ...

    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        ...

    async def fetch_account(self) -> AccountSnapshot:
        """
        Fetch current account state.

        Returns:
            AccountSnapshot with balances and margin info.

        Raises:
            ConnectionError: If not connected.
        """
        ...

    async def fetch_accounts(self) -> Dict[str, AccountSnapshot]:
        """
        Fetch all accounts (for multi-account providers).

        Returns:
            Dict mapping account_id to AccountSnapshot.
        """
        ...

    async def subscribe_account(self) -> None:
        """
        Subscribe to account updates.

        Account changes are delivered via set_account_callback.
        """
        ...

    def unsubscribe_account(self) -> None:
        """Unsubscribe from account updates."""
        ...

    def set_account_callback(self, callback: Optional[Callable[[AccountSnapshot], None]]) -> None:
        """
        Set callback for account updates.

        Args:
            callback: Function to call with updated account.
                     Set to None to disable.
        """
        ...

    def get_cached_account(self) -> Optional[AccountSnapshot]:
        """
        Get cached account without fetching.

        Returns:
            Cached AccountSnapshot (may be stale) or None.
        """
        ...

    def get_account_id(self) -> Optional[str]:
        """
        Get the primary account ID.

        Returns:
            Account ID string or None if not connected.
        """
        ...

    def get_buying_power(self) -> float:
        """
        Get current buying power.

        Returns:
            Buying power value (0 if not available).
        """
        ...

    def get_margin_utilization(self) -> float:
        """
        Get margin utilization percentage.

        Returns:
            Margin utilization as percentage (0-100).
        """
        ...
