"""Thread-safe in-memory account store with event subscription."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from threading import RLock
import logging

from ...models.account import AccountInfo

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus

logger = logging.getLogger(__name__)


class AccountStore:
    """Thread-safe in-memory account store (single account)."""

    def __init__(self) -> None:
        self._account: Optional[AccountInfo] = None
        self._lock = RLock()
        # Track accounts from multiple sources for aggregation
        self._ib_account: Optional[AccountInfo] = None
        self._futu_account: Optional[AccountInfo] = None

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
            self._ib_account = None
            self._futu_account = None

    def subscribe_to_events(self, event_bus: "EventBus") -> None:
        """
        Subscribe to account-related events.

        Args:
            event_bus: Event bus to subscribe to.
        """
        from ...domain.interfaces.event_bus import EventType

        event_bus.subscribe(EventType.ACCOUNT_UPDATED, self._on_account_updated)
        logger.debug("AccountStore subscribed to events")

    def _on_account_updated(self, payload: dict) -> None:
        """
        Handle account update event.

        Args:
            payload: Event payload with 'account_info', 'ib_account', 'futu_account'.
        """
        # Handle aggregated account info from orchestrator
        account_info = payload.get("account_info")
        ib_account = payload.get("ib_account")
        futu_account = payload.get("futu_account")

        with self._lock:
            # Store source-specific accounts
            if ib_account:
                self._ib_account = ib_account
            if futu_account:
                self._futu_account = futu_account

            # Use pre-aggregated account or aggregate from sources
            if account_info:
                self._account = account_info
            else:
                self._account = self._aggregate_accounts()

        logger.debug("AccountStore updated from event")

    def _aggregate_accounts(self) -> AccountInfo:
        """
        Aggregate account info from multiple sources.

        Returns:
            Aggregated AccountInfo combining IB and Futu.
        """
        from datetime import datetime

        # If only one source, return it directly
        if self._ib_account and not self._futu_account:
            return self._ib_account
        if self._futu_account and not self._ib_account:
            return self._futu_account
        if not self._ib_account and not self._futu_account:
            # Return empty account with all required fields
            return AccountInfo(
                net_liquidation=0.0,
                total_cash=0.0,
                buying_power=0.0,
                margin_used=0.0,
                margin_available=0.0,
                maintenance_margin=0.0,
                init_margin_req=0.0,
                excess_liquidity=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                timestamp=datetime.now(),
                account_id="empty",
            )

        # Aggregate from both sources
        ib = self._ib_account
        futu = self._futu_account

        return AccountInfo(
            net_liquidation=(ib.net_liquidation or 0) + (futu.net_liquidation or 0),
            total_cash=(ib.total_cash or 0) + (futu.total_cash or 0),
            buying_power=(ib.buying_power or 0) + (futu.buying_power or 0),
            margin_used=(ib.margin_used or 0) + (futu.margin_used or 0),
            margin_available=(ib.margin_available or 0) + (futu.margin_available or 0),
            maintenance_margin=(ib.maintenance_margin or 0) + (futu.maintenance_margin or 0),
            init_margin_req=(ib.init_margin_req or 0) + (futu.init_margin_req or 0),
            excess_liquidity=(ib.excess_liquidity or 0) + (futu.excess_liquidity or 0),
            realized_pnl=(ib.realized_pnl or 0) + (futu.realized_pnl or 0),
            unrealized_pnl=(ib.unrealized_pnl or 0) + (futu.unrealized_pnl or 0),
            timestamp=datetime.now(),
            account_id="aggregated",
        )

    def get_ib_account(self) -> Optional[AccountInfo]:
        """Get IB-specific account info."""
        with self._lock:
            return self._ib_account

    def get_futu_account(self) -> Optional[AccountInfo]:
        """Get Futu-specific account info."""
        with self._lock:
            return self._futu_account
