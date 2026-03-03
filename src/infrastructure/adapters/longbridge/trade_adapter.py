"""Longbridge TradeAdapter — implements BrokerAdapter for positions and account info."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.models.account import AccountInfo
from src.models.order import Order, Trade
from src.models.position import AssetType, Position, PositionSource

logger = logging.getLogger(__name__)


class LongbridgeTradeAdapter:
    """
    Broker adapter using Longbridge (LongPort) TradeContext SDK.

    Provides position and account info only. Order placement is NOT supported
    (raises NotImplementedError).

    Credentials: LONGPORT_APP_KEY, LONGPORT_APP_SECRET, LONGPORT_ACCESS_TOKEN env vars.
    """

    def __init__(self, default_market: str = "US") -> None:
        self._ctx: Any = None
        self._connected = False
        self._lock = threading.Lock()
        self._default_market = default_market

    # ── BrokerAdapter protocol ─────────────────────────────────

    async def connect(self) -> None:
        """Connect to Longbridge using env var credentials."""
        from longport.openapi import Config, TradeContext

        config = Config.from_env()
        self._ctx = await asyncio.to_thread(TradeContext, config)
        self._connected = True
        logger.info("Longbridge TradeContext connected")

    async def disconnect(self) -> None:
        """Disconnect from Longbridge."""
        self._connected = False
        self._ctx = None
        logger.info("Longbridge TradeContext disconnected")

    def is_connected(self) -> bool:
        return self._connected

    async def fetch_positions(self) -> List[Position]:
        """Fetch current positions from Longbridge."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge TradeContext")

        raw = await asyncio.to_thread(self._ctx.stock_positions)
        positions: List[Position] = []

        for channel in raw:
            for pos in channel.stock_info:
                try:
                    converted = self._convert_position(pos)
                    if converted is not None:
                        positions.append(converted)
                except Exception:
                    logger.exception("Error converting LB position: %s", pos)

        logger.debug("Fetched %d positions from Longbridge", len(positions))
        return positions

    async def fetch_account_info(self) -> AccountInfo:
        """Fetch account balance from Longbridge (cash-only, no margin data)."""
        if not self._connected or self._ctx is None:
            raise ConnectionError("Not connected to Longbridge TradeContext")

        balances = await asyncio.to_thread(self._ctx.account_balance)

        total_cash = 0.0
        available_cash = 0.0
        account_id: Optional[str] = None

        for bal in balances:
            # Sum across currencies, preferring USD
            total_cash += float(bal.total_cash)
            available_cash += (
                float(bal.max_finance_amount)
                if hasattr(bal, "max_finance_amount")
                else float(bal.total_cash)
            )
            if account_id is None and hasattr(bal, "currency"):
                account_id = str(getattr(bal, "account_id", "longbridge"))

        return AccountInfo(
            net_liquidation=total_cash,  # Cash-only: best approximation
            total_cash=total_cash,
            buying_power=available_cash,
            margin_used=0.0,
            margin_available=0.0,
            maintenance_margin=0.0,
            init_margin_req=0.0,
            excess_liquidity=available_cash,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(timezone.utc),
            account_id=account_id or "longbridge",
        )

    async def fetch_orders(
        self,
        include_open: bool = True,
        include_completed: bool = True,
        days_back: int = 30,
    ) -> List[Order]:
        """Not implemented — Longbridge trade adapter is read-only."""
        raise NotImplementedError("LongBridge trade adapter: order fetching not implemented")

    async def fetch_trades(self, days_back: int = 30) -> List[Trade]:
        """Not implemented — Longbridge trade adapter is read-only."""
        raise NotImplementedError("LongBridge trade adapter: trade fetching not implemented")

    # ── Conversion helpers ─────────────────────────────────────

    def _convert_position(self, pos: Any) -> Optional[Position]:
        """Convert SDK StockPositionInfo to domain Position."""
        symbol_raw = getattr(pos, "symbol", "")
        internal_symbol = self._to_internal_symbol(symbol_raw)

        quantity = float(getattr(pos, "quantity", 0))
        if quantity == 0:
            return None

        avg_price = float(getattr(pos, "cost_price", 0.0))

        return Position(
            symbol=internal_symbol,
            underlying=internal_symbol,
            asset_type=AssetType.STOCK,
            quantity=quantity,
            avg_price=avg_price,
            source=PositionSource.LONGBRIDGE,
            account_id="longbridge",
            last_updated=datetime.now(timezone.utc),
        )

    # ── Symbol mapping ─────────────────────────────────────────

    def _to_lb_symbol(self, symbol: str) -> str:
        """Internal symbol → Longbridge format. AAPL → AAPL.US"""
        if "." in symbol:
            return symbol
        return f"{symbol}.{self._default_market}"

    def _to_internal_symbol(self, lb_symbol: str) -> str:
        """Longbridge format → internal symbol. AAPL.US → AAPL"""
        if "." in lb_symbol:
            return lb_symbol.rsplit(".", 1)[0]
        return lb_symbol
