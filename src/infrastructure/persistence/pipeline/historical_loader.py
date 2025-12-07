"""
Historical data loader for backfilling trade data.

Orchestrates the fetching, raw persistence, and strategy classification
of historical order/trade data from Futu and IB.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
import asyncio

from ..duckdb_adapter import DuckDBAdapter
from ..repositories.raw_data_repo import RawDataRepository
from ..repositories.order_repository import OrderRepository
from ..classify.strategy_classifier import (
    StrategyClassifierV1,
    LegInfo,
    group_trades_by_strategy,
)

logger = logging.getLogger(__name__)


class HistoricalLoader:
    """
    Historical data loader with smart backfill.

    Coordinates:
    1. Fetching historical data from brokers (Futu API, IB Flex)
    2. Persisting raw payloads to audit tables
    3. Normalizing data to standard format
    4. Running strategy classification

    Supports:
    - Full reload (first-time or force refresh)
    - Incremental load (since last sync)
    - Smart skip (if data already exists)
    """

    def __init__(
        self,
        db: DuckDBAdapter,
        futu_adapter=None,  # Optional FutuAdapter
        ib_adapter=None,    # Optional IbAdapter
        ib_flex_token: Optional[str] = None,
        ib_flex_query_id: Optional[str] = None,
        flex_refresh_hours: int = 24,
    ):
        """
        Initialize historical loader.

        Args:
            db: DuckDB adapter
            futu_adapter: Optional FutuAdapter instance
            ib_adapter: Optional IbAdapter instance
            ib_flex_token: IB Flex Web Service token
            ib_flex_query_id: IB Flex Query ID
            flex_refresh_hours: Hours between Flex report refreshes (default: 24)
        """
        self.db = db
        self.futu_adapter = futu_adapter
        self.ib_adapter = ib_adapter
        self.ib_flex_token = ib_flex_token
        self.ib_flex_query_id = ib_flex_query_id
        self.flex_refresh_hours = flex_refresh_hours
        self._last_flex_refresh: Optional[datetime] = None

        # Initialize repositories
        self.raw_repo = RawDataRepository(db)
        self.order_repo = OrderRepository(db)
        self.classifier = StrategyClassifierV1()

        # Stats tracking
        self._stats = {
            "futu_orders_loaded": 0,
            "futu_trades_loaded": 0,
            "futu_fees_loaded": 0,
            "ib_orders_loaded": 0,
            "ib_trades_loaded": 0,
            "strategies_classified": 0,
            "last_sync_time": None,
        }

    async def full_reload(
        self,
        days_back: int = 365,
        force: bool = False,
        include_futu: bool = True,
        include_ib: bool = True,
    ) -> Dict[str, Any]:
        """
        Full historical backfill with smart skip.

        Args:
            days_back: Number of days to look back
            force: Force reload even if data exists
            include_futu: Include Futu data
            include_ib: Include IB data

        Returns:
            Dict with load statistics
        """
        logger.info(f"Starting full reload (days_back={days_back}, force={force})")

        # Smart skip: check if raw data already exists
        if not force:
            futu_count = self.raw_repo.count_futu_orders_raw()
            ib_count = self.raw_repo.count_ib_orders_raw()

            if futu_count > 0 or ib_count > 0:
                logger.info(
                    f"Found existing raw data (futu={futu_count}, ib={ib_count}), "
                    "running incremental instead"
                )
                return await self.incremental_load()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        results = {
            "mode": "full_reload",
            "days_back": days_back,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "futu": {},
            "ib": {},
            "classification": {},
        }

        # Load Futu data
        if include_futu and self.futu_adapter:
            try:
                futu_result = await self._load_futu_history(days_back)
                results["futu"] = futu_result
            except Exception as e:
                logger.error(f"Failed to load Futu history: {e}")
                results["futu"]["error"] = str(e)

        # Load IB data (Flex report)
        if include_ib and self.ib_flex_token and self.ib_flex_query_id:
            try:
                ib_result = await self._load_ib_flex_history()
                results["ib"] = ib_result
            except Exception as e:
                logger.error(f"Failed to load IB Flex history: {e}")
                results["ib"]["error"] = str(e)

        # Run strategy classification
        try:
            class_result = await self._classify_all()
            results["classification"] = class_result
        except Exception as e:
            logger.error(f"Failed to classify strategies: {e}")
            results["classification"]["error"] = str(e)

        self._stats["last_sync_time"] = datetime.now()
        logger.info(f"Full reload complete: {results}")
        return results

    async def incremental_load(
        self,
        overlap_hours: int = 1,
    ) -> Dict[str, Any]:
        """
        Load data since last sync.

        Args:
            overlap_hours: Hours of overlap to catch missed records

        Returns:
            Dict with load statistics
        """
        logger.info(f"Starting incremental load (overlap_hours={overlap_hours})")

        # Get data boundaries to determine last sync
        futu_boundaries = self.raw_repo.get_data_boundaries('FUTU')
        ib_boundaries = self.raw_repo.get_data_boundaries('IB')

        # Calculate start time from latest data
        latest_times = [
            futu_boundaries.get("trades_max"),
            ib_boundaries.get("trades_max"),
        ]
        latest_times = [t for t in latest_times if t is not None]

        if latest_times:
            latest = max(latest_times)
            start_time = latest - timedelta(hours=overlap_hours)
            days_back = (datetime.now() - start_time).days + 1
        else:
            # No existing data, do full reload
            logger.info("No existing data found, performing full reload")
            return await self.full_reload(force=True)

        results = {
            "mode": "incremental",
            "start_time": start_time.isoformat(),
            "overlap_hours": overlap_hours,
            "futu": {},
            "ib": {},
            "classification": {},
        }

        # Load Futu data (incremental)
        if self.futu_adapter:
            try:
                futu_result = await self._load_futu_history(days_back)
                results["futu"] = futu_result
            except Exception as e:
                logger.error(f"Failed to load Futu incremental: {e}")
                results["futu"]["error"] = str(e)

        # Load IB real-time data (always current session only)
        if self.ib_adapter:
            try:
                ib_result = await self._load_ib_realtime()
                results["ib"] = ib_result
            except Exception as e:
                logger.error(f"Failed to load IB realtime: {e}")
                results["ib"]["error"] = str(e)

        # Check if Flex refresh is due (daily historical backfill for IB)
        if self._should_refresh_flex():
            try:
                flex_result = await self._load_ib_flex_history()
                results["ib_flex"] = flex_result
                self._last_flex_refresh = datetime.now()
                logger.info("IB Flex report refreshed successfully")
            except Exception as e:
                logger.warning(f"Flex refresh failed (non-critical): {e}")
                results["ib_flex"] = {"error": str(e)}

        # Run strategy classification (only new orders)
        try:
            class_result = await self._classify_new()
            results["classification"] = class_result
        except Exception as e:
            logger.error(f"Failed to classify strategies: {e}")
            results["classification"]["error"] = str(e)

        self._stats["last_sync_time"] = datetime.now()
        logger.info(f"Incremental load complete: {results}")
        return results

    async def _load_futu_history(self, days_back: int) -> Dict[str, Any]:
        """Load historical data from Futu."""
        if not self.futu_adapter:
            return {"error": "No Futu adapter configured"}

        result = {
            "orders_raw": {"inserted": 0, "updated": 0},
            "deals_raw": {"inserted": 0, "updated": 0},
            "fees_raw": {"inserted": 0, "updated": 0},
        }

        try:
            # Get account ID
            acc_id = getattr(self.futu_adapter, '_acc_id', None)
            if not acc_id:
                return {"error": "Futu adapter not connected (no acc_id)"}

            # Fetch raw orders (original API payloads)
            logger.info(f"Fetching Futu raw orders (days_back={days_back})")
            raw_orders = await self.futu_adapter.fetch_orders_raw(
                include_open=True,
                include_completed=True,
                days_back=days_back,
            )

            # Persist raw payloads
            if raw_orders:
                raw_result = self.raw_repo.batch_persist_futu_orders_raw(acc_id, raw_orders)
                result["orders_raw"] = raw_result
                self._stats["futu_orders_loaded"] += len(raw_orders)

            # Fetch raw deals (original API payloads)
            logger.info(f"Fetching Futu raw deals (days_back={days_back})")
            raw_deals = await self.futu_adapter.fetch_deals_raw(days_back=days_back)

            if raw_deals:
                raw_result = self.raw_repo.batch_persist_futu_deals_raw(acc_id, raw_deals)
                result["deals_raw"] = raw_result
                self._stats["futu_trades_loaded"] += len(raw_deals)

            # Fetch and persist fees for filled orders (with rate limiting)
            filled_order_ids = [
                str(o.get('order_id', ''))
                for o in raw_orders
                if str(o.get('order_status', '')).upper() in ('FILLED_ALL', 'FILLED_PART')
            ]
            if filled_order_ids:
                fee_result = await self._fetch_and_persist_futu_fees(acc_id, filled_order_ids)
                result["fees_raw"] = fee_result

            # Also fetch via standard adapters for normalized tables
            orders = await self.futu_adapter.fetch_orders(
                include_open=True,
                include_completed=True,
                days_back=days_back,
            )
            trades = await self.futu_adapter.fetch_trades(days_back=days_back)

            # Normalize to orders/trades tables
            if orders:
                self.order_repo.upsert_orders(orders)
            if trades:
                self.order_repo.upsert_trades(trades)

            logger.info(f"Loaded Futu history: {result}")
            return result

        except Exception as e:
            logger.error(f"Error loading Futu history: {e}")
            return {"error": str(e)}

    async def _fetch_and_persist_futu_fees(
        self,
        acc_id: int,
        order_ids: List[str],
    ) -> Dict[str, int]:
        """
        Fetch and persist Futu fees with 10 req/30s throttle.

        Args:
            acc_id: Futu account ID
            order_ids: List of filled order IDs to fetch fees for

        Returns:
            Dict with inserted/updated counts
        """
        result = {"inserted": 0, "updated": 0}

        if not order_ids:
            return result

        logger.info(f"Fetching fees for {len(order_ids)} filled orders")

        # Batch by 400, with 3-second delay between batches (10 req/30s = 1 req/3s)
        for i in range(0, len(order_ids), 400):
            if i > 0:
                logger.debug(f"Rate limit pause before fee batch {i // 400 + 1}")
                await asyncio.sleep(3)  # Rate limit: 10 req/30s

            batch_ids = order_ids[i:i + 400]
            try:
                fees = await self.futu_adapter.fetch_order_fees(batch_ids)

                for fee in fees:
                    order_id = str(fee.get('order_id', ''))
                    if not order_id:
                        continue

                    is_new = self.raw_repo.persist_futu_fee_raw(
                        acc_id=acc_id,
                        order_id=order_id,
                        fee_amount=float(fee.get('fee_amount', 0) or 0),
                        fee_details=fee.get('fee_list'),
                        payload=fee,  # Original API response
                    )
                    if is_new:
                        result["inserted"] += 1
                    else:
                        result["updated"] += 1

                self._stats["futu_fees_loaded"] += len(fees)

            except Exception as e:
                logger.warning(f"Failed to fetch fees for batch starting at {i}: {e}")
                continue

        logger.info(f"Persisted Futu fees: {result}")
        return result

    async def _load_ib_flex_history(self) -> Dict[str, Any]:
        """Load historical data from IB Flex report."""
        if not self.ib_flex_token or not self.ib_flex_query_id:
            return {"error": "No IB Flex credentials configured"}

        result = {
            "orders_raw": {"inserted": 0, "updated": 0},
            "trades_raw": {"inserted": 0, "updated": 0},
        }

        try:
            from ..adapters.ib.flex_parser import (
                FlexParser,
                flex_trade_to_dict,
                flex_order_to_dict,
            )

            parser = FlexParser(self.ib_flex_token, self.ib_flex_query_id)
            flex_data = parser.fetch_and_parse()

            trades = flex_data.get("trades", [])
            orders = flex_data.get("orders", [])

            # Persist raw IB data
            for trade in trades:
                trade_dict = flex_trade_to_dict(trade)
                is_new = self.raw_repo.persist_ib_execution_raw(
                    account=trade.account,
                    exec_id=trade.trade_id,
                    payload=trade.raw_data or trade_dict,
                    perm_id=trade.perm_id,
                    trade_time_raw=trade.trade_time.isoformat() if trade.trade_time else None,
                    source='FLEX',
                )
                if is_new:
                    result["trades_raw"]["inserted"] += 1
                else:
                    result["trades_raw"]["updated"] += 1

                # Also persist commission
                if trade.commission > 0:
                    self.raw_repo.persist_ib_commission_raw(
                        account=trade.account,
                        exec_id=trade.trade_id,
                        commission=trade.commission,
                        currency=trade.currency,
                        source='FLEX',
                    )

            for order in orders:
                order_dict = flex_order_to_dict(order)
                is_new = self.raw_repo.persist_ib_order_raw(
                    account=order.account,
                    perm_id=order.perm_id or 0,
                    payload=order.raw_data or order_dict,
                    create_time_raw=order.create_time.isoformat() if order.create_time else None,
                    source='FLEX',
                )
                if is_new:
                    result["orders_raw"]["inserted"] += 1
                else:
                    result["orders_raw"]["updated"] += 1

            self._stats["ib_orders_loaded"] += len(orders)
            self._stats["ib_trades_loaded"] += len(trades)

            logger.info(f"Loaded IB Flex history: {result}")
            return result

        except ImportError:
            return {"error": "ibflex library not installed"}
        except Exception as e:
            logger.error(f"Error loading IB Flex history: {e}")
            return {"error": str(e)}

    async def _load_ib_realtime(self) -> Dict[str, Any]:
        """Load current session data from IB API."""
        if not self.ib_adapter:
            return {"error": "No IB adapter configured"}

        result = {
            "orders": 0,
            "trades": 0,
        }

        try:
            # Fetch current orders
            orders = await self.ib_adapter.fetch_orders()
            if orders:
                self.order_repo.upsert_orders(orders)
                result["orders"] = len(orders)

            # Fetch current trades
            trades = await self.ib_adapter.fetch_trades()
            if trades:
                self.order_repo.upsert_trades(trades)
                result["trades"] = len(trades)

                # Persist raw executions
                for trade in trades:
                    trade_dict = self._trade_to_dict(trade)
                    self.raw_repo.persist_ib_execution_raw(
                        account=trade.account_id,
                        exec_id=trade.trade_id,
                        payload=trade_dict,
                        perm_id=getattr(trade, 'perm_id', None),
                        trade_time_raw=trade.trade_time.isoformat() if trade.trade_time else None,
                        source='API',
                    )

            logger.info(f"Loaded IB realtime: {result}")
            return result

        except Exception as e:
            logger.error(f"Error loading IB realtime: {e}")
            return {"error": str(e)}

    async def _classify_all(self) -> Dict[str, Any]:
        """Run strategy classification on all unclassified orders."""
        return await self._run_classification(limit=10000)

    async def _classify_new(self) -> Dict[str, Any]:
        """Run strategy classification on recently added orders."""
        return await self._run_classification(limit=1000)

    async def _run_classification(self, limit: int = 1000) -> Dict[str, Any]:
        """Run strategy classification on unclassified orders."""
        result = {
            "orders_processed": 0,
            "strategies_found": 0,
            "by_type": {},
        }

        try:
            # Get unclassified orders
            unclassified = self.raw_repo.get_unclassified_orders(limit=limit)
            if not unclassified:
                logger.info("No unclassified orders found")
                return result

            # Convert to LegInfo objects
            legs = [LegInfo.from_order_dict(order) for order in unclassified]

            # Group by time window
            groups = group_trades_by_strategy(legs, window_seconds=5)

            # Classify each group
            for group in groups:
                if not group:
                    continue

                strategy_result = self.classifier.classify(group)
                result["strategies_found"] += 1

                # Count by type
                stype = strategy_result.strategy_type.value
                result["by_type"][stype] = result["by_type"].get(stype, 0) + 1

                # Persist strategy mapping for each leg
                for i, leg in enumerate(strategy_result.legs):
                    # Extract broker and account from order_uid
                    parts = leg.order_uid.split('_')
                    if len(parts) >= 3:
                        broker = parts[0]
                        account_id = parts[1]
                    else:
                        broker = "UNKNOWN"
                        account_id = "UNKNOWN"

                    self.raw_repo.upsert_strategy_mapping(
                        broker=broker,
                        account_id=account_id,
                        order_uid=leg.order_uid,
                        strategy_id=strategy_result.strategy_id,
                        strategy_type=strategy_result.strategy_type.value,
                        strategy_name=strategy_result.name,
                        confidence=strategy_result.confidence,
                        leg_index=i,
                        legs=strategy_result.to_dict()["legs"],
                    )

                result["orders_processed"] += len(group)

            self._stats["strategies_classified"] += result["strategies_found"]
            logger.info(f"Classification complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {"error": str(e)}

    def _order_to_dict(self, order) -> dict:
        """Convert Order object to dict for raw storage."""
        return {
            "order_id": order.order_id,
            "source": order.source.value if hasattr(order.source, 'value') else str(order.source),
            "account_id": order.account_id,
            "symbol": order.symbol,
            "underlying": order.underlying,
            "asset_type": order.asset_type,
            "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
            "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "filled_quantity": order.filled_quantity,
            "avg_fill_price": order.avg_fill_price,
            "commission": order.commission,
            "create_time": order.created_time.isoformat() if order.created_time else None,
            "update_time": order.updated_time.isoformat() if order.updated_time else None,
            "filled_time": order.filled_time.isoformat() if order.filled_time else None,
            "expiry": order.expiry,
            "strike": order.strike,
            "option_right": order.right,
        }

    def _trade_to_dict(self, trade) -> dict:
        """Convert Trade object to dict for raw storage."""
        return {
            "trade_id": trade.trade_id,
            "deal_id": trade.trade_id,  # Alias for Futu
            "order_id": trade.order_id,
            "source": trade.source.value if hasattr(trade.source, 'value') else str(trade.source),
            "account_id": trade.account_id,
            "symbol": trade.symbol,
            "underlying": trade.underlying,
            "asset_type": trade.asset_type,
            "side": trade.side.value if hasattr(trade.side, 'value') else str(trade.side),
            "quantity": trade.quantity,
            "price": trade.price,
            "commission": trade.commission,
            "create_time": trade.trade_time.isoformat() if trade.trade_time else None,
            "expiry": trade.expiry,
            "strike": trade.strike,
            "option_right": trade.right,
            "exchange": trade.exchange,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            **self._stats,
            "raw_data": {
                "futu_orders": self.raw_repo.count_futu_orders_raw(),
                "futu_deals": self.raw_repo.count_futu_deals_raw(),
                "ib_orders": self.raw_repo.count_ib_orders_raw(),
            },
            "classification": self.raw_repo.get_strategy_stats(),
            "flex_last_refresh": self._last_flex_refresh.isoformat() if self._last_flex_refresh else None,
            "flex_refresh_hours": self.flex_refresh_hours,
        }

    def _should_refresh_flex(self) -> bool:
        """
        Check if Flex report should be refreshed.

        Flex refresh is due if:
        - IB Flex credentials are configured
        - Last refresh was more than flex_refresh_hours ago
        - Or we've never refreshed before

        Returns:
            True if Flex should be refreshed
        """
        if not self.ib_flex_token or not self.ib_flex_query_id:
            return False

        if self._last_flex_refresh is None:
            # Check DB for last Flex ingest timestamp
            try:
                result = self.db.fetch_one("""
                    SELECT MAX(ingest_ts) as last_ts
                    FROM orders_raw_ib
                    WHERE source = 'FLEX'
                """)
                if result and result.get("last_ts"):
                    self._last_flex_refresh = result["last_ts"]
                    logger.debug(f"Found last Flex refresh: {self._last_flex_refresh}")
                else:
                    logger.info("No previous Flex refresh found - will refresh")
                    return True
            except Exception as e:
                logger.warning(f"Error checking Flex refresh time: {e}")
                return True

        # Check if enough time has passed
        hours_since = (datetime.now() - self._last_flex_refresh).total_seconds() / 3600
        should_refresh = hours_since >= self.flex_refresh_hours

        if should_refresh:
            logger.info(f"Flex refresh due: {hours_since:.1f}h since last refresh")

        return should_refresh
