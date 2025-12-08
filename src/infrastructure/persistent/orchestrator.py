"""
Persistence orchestrator for coordinating data loading and processing.

Handles:
- Full reload (backfill all historical data)
- Incremental load (only new data since last sync)
- Normalization, classification, and reconciliation
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import yaml

from .store import PostgresStore
from .normalizers.futu import FutuNormalizer
from .normalizers.ib import IbNormalizer
from .classify.strategy_classifier import StrategyClassifier
from .reconciler import Reconciler

logger = logging.getLogger(__name__)


class PersistenceOrchestrator:
    """
    Coordinates the persistence pipeline:
    Extract → Raw Persist → Transform/Normalize → Classify → Load → Reconcile
    """

    def __init__(
        self,
        config: Dict[str, Any],
        futu_adapter=None,
        ib_adapter=None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Configuration dict (from persistent.yaml).
            futu_adapter: Optional FutuAdapter instance.
            ib_adapter: Optional IbAdapter instance.
        """
        self.config = config

        # Storage configuration
        storage_cfg = config.get("storage", {})
        self.store = PostgresStore(
            dsn=storage_cfg.get("dsn", "postgresql://localhost:5432/apex_risk"),
            pool_min=storage_cfg.get("pool_min", 2),
            pool_max=storage_cfg.get("pool_max", 10),
        )

        # Broker adapters (can be injected or created later)
        self.futu_adapter = futu_adapter
        self.ib_adapter = ib_adapter

        # Normalizers
        self.futu_normalizer = FutuNormalizer()
        self.ib_normalizer = IbNormalizer()

        # Strategy classifier
        self.classifier = StrategyClassifier()

        # Reconciler (initialized after store connects)
        self.reconciler = None

        # Default settings
        self.lookback_days = config.get("reload", {}).get("lookback_days_default", 365)

    async def connect(self) -> None:
        """Connect to PostgreSQL and initialize schema."""
        await self.store.connect()
        # Ensure schema exists (creates tables if needed)
        await self.store.ensure_schema()
        self.reconciler = Reconciler(self.store)
        logger.info("Persistence orchestrator connected")

    async def close(self) -> None:
        """Close database connection."""
        await self.store.close()
        logger.info("Persistence orchestrator closed")

    async def run(
        self,
        full_reload: bool = False,
        days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        skip_classify: bool = False,
        skip_reconcile: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point for running the persistence pipeline.

        Args:
            full_reload: If True, truncate normalized tables and reload all data.
            days: Override default lookback days.
            start_date: Explicit start date (overrides days).
            skip_classify: Skip strategy classification.
            skip_reconcile: Skip reconciliation checks.

        Returns:
            Summary dict with counts and any errors.
        """
        await self.connect()

        try:
            end_date = datetime.now()

            # Use explicit start_date if provided, otherwise calculate from days
            if start_date:
                pass  # Use provided start_date
            elif days:
                start_date = end_date - timedelta(days=days)
            else:
                start_date = end_date - timedelta(days=self.lookback_days)

            logger.info(
                f"Starting persistence run: "
                f"{'FULL RELOAD' if full_reload else 'INCREMENTAL'}, "
                f"date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            )

            summary = {
                "mode": "full_reload" if full_reload else "incremental",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "futu": {"orders": 0, "trades": 0, "fees": 0},
                "ib": {"orders": 0, "trades": 0, "fees": 0},
                "normalized": {"orders": 0, "trades": 0, "fees": 0},
                "strategies": 0,
                "anomalies": 0,
                "errors": [],
            }

            if full_reload:
                await self.store.truncate_normalized_tables()

            # Fetch and persist Futu data
            if self.futu_adapter:
                try:
                    futu_counts = await self._process_futu(start_date, end_date)
                    summary["futu"] = futu_counts
                except Exception as e:
                    logger.error(f"Futu processing failed: {e}")
                    summary["errors"].append(f"Futu: {str(e)}")

            # Fetch and persist IB data
            if self.ib_adapter:
                try:
                    ib_counts = await self._process_ib(start_date, end_date)
                    summary["ib"] = ib_counts
                except Exception as e:
                    logger.error(f"IB processing failed: {e}")
                    summary["errors"].append(f"IB: {str(e)}")

            # Classify strategies
            if not skip_classify:
                try:
                    summary["strategies"] = await self._classify_strategies()
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    summary["errors"].append(f"Classification: {str(e)}")

            # Run reconciliation
            if not skip_reconcile and self.reconciler:
                try:
                    anomalies = await self.reconciler.run()
                    summary["anomalies"] = len(anomalies)
                except Exception as e:
                    logger.error(f"Reconciliation failed: {e}")
                    summary["errors"].append(f"Reconciliation: {str(e)}")

            # Get final counts
            summary["normalized"]["orders"] = await self.store.get_order_count()
            summary["normalized"]["trades"] = await self.store.get_trade_count()

            logger.info(
                f"Persistence run complete: "
                f"{summary['normalized']['orders']} orders, "
                f"{summary['normalized']['trades']} trades, "
                f"{summary['strategies']} strategies, "
                f"{summary['anomalies']} anomalies"
            )

            return summary

        finally:
            await self.close()

    async def _process_futu(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, int]:
        """Process Futu historical data month by month."""
        if not self.futu_adapter:
            return {"orders": 0, "trades": 0, "fees": 0}

        logger.info("Processing Futu data...")
        counts = {"orders": 0, "trades": 0, "fees": 0}

        # Get account ID from adapter (keep as int for database BIGINT column)
        acc_id = int(self.futu_adapter._acc_id)

        # Generate monthly date ranges
        monthly_ranges = self._generate_monthly_ranges(start_date, end_date)
        logger.info(f"Fetching Futu data in {len(monthly_ranges)} monthly chunks")

        all_raw_orders = []
        all_raw_deals = []

        for i, (month_start, month_end) in enumerate(monthly_ranges):
            month_label = month_start.strftime("%Y-%m")
            logger.info(f"Fetching Futu data for {month_label} ({i+1}/{len(monthly_ranges)})")

            # Calculate days for this month chunk
            days_back = (month_end - month_start).days + 1

            # Fetch raw orders for this month
            try:
                raw_orders = await self.futu_adapter.fetch_orders_raw(
                    days_back=days_back,
                    include_open=(i == len(monthly_ranges) - 1),  # Only include open orders in the last (most recent) chunk
                    include_completed=True,
                    start_date=month_start,
                    end_date=month_end,
                )
                logger.info(f"  {month_label}: fetched {len(raw_orders)} orders from API")
            except Exception as e:
                logger.warning(f"  {month_label}: failed to fetch orders: {e}")
                raw_orders = []

            # Add acc_id and persist orders
            if raw_orders:
                for order in raw_orders:
                    order["acc_id"] = acc_id
                try:
                    await self.store.upsert_orders_raw_futu(raw_orders)
                    logger.info(f"  {month_label}: persisted {len(raw_orders)} orders to DB")
                    all_raw_orders.extend(raw_orders)
                except Exception as e:
                    logger.error(f"  {month_label}: failed to persist orders: {e}")

            # Sleep between API calls
            await asyncio.sleep(1)

            # Fetch raw deals for this month
            try:
                raw_deals = await self.futu_adapter.fetch_deals_raw(
                    days_back=days_back,
                    start_date=month_start,
                    end_date=month_end,
                )
                logger.info(f"  {month_label}: fetched {len(raw_deals)} deals from API")
            except Exception as e:
                logger.warning(f"  {month_label}: failed to fetch deals: {e}")
                raw_deals = []

            # Add acc_id and persist deals
            if raw_deals:
                for deal in raw_deals:
                    deal["acc_id"] = acc_id
                try:
                    await self.store.upsert_trades_raw_futu(raw_deals)
                    logger.info(f"  {month_label}: persisted {len(raw_deals)} deals to DB")
                    all_raw_deals.extend(raw_deals)
                except Exception as e:
                    logger.error(f"  {month_label}: failed to persist deals: {e}")

            # Sleep 1 second between months
            if i < len(monthly_ranges) - 1:
                await asyncio.sleep(1)

        counts["orders"] = len(all_raw_orders)
        counts["trades"] = len(all_raw_deals)
        logger.info(f"Fetched {counts['orders']} total orders, {counts['trades']} total deals from Futu")

        # Fetch fees for filled orders
        filled_order_ids = [
            str(o.get("order_id"))
            for o in all_raw_orders
            if str(o.get("order_status", "")).upper() in ("FILLED_ALL", "FILLED")
        ]

        all_fees = []
        if filled_order_ids:
            logger.info(f"Fetching fees for {len(filled_order_ids)} filled orders...")
            # Batch fetch fees (max 400 per request)
            for i in range(0, len(filled_order_ids), 400):
                batch = filled_order_ids[i : i + 400]
                fees = await self.futu_adapter.fetch_order_fees(batch)
                for fee in fees:
                    fee["acc_id"] = acc_id
                all_fees.extend(fees)
                # Rate limiting
                if i + 400 < len(filled_order_ids):
                    await asyncio.sleep(3)

            await self.store.upsert_fees_raw_futu(all_fees)
            counts["fees"] = len(all_fees)
            logger.info(f"Fetched {len(all_fees)} fee records")

        # Normalize data (normalizers expect string account_id for TEXT column)
        await self._normalize_futu(all_raw_orders, all_raw_deals, all_fees, str(acc_id))

        return counts

    def _generate_monthly_ranges(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[tuple]:
        """Generate list of (month_start, month_end) tuples."""
        from dateutil.relativedelta import relativedelta

        ranges = []
        current = start_date.replace(day=1)  # Start from first of month

        while current <= end_date:
            month_start = max(current, start_date)
            # End of month or end_date, whichever is earlier
            next_month = current + relativedelta(months=1)
            month_end = min(next_month - timedelta(days=1), end_date)

            ranges.append((month_start, month_end))
            current = next_month

        return ranges

    async def _normalize_futu(
        self,
        raw_orders: List[Dict],
        raw_deals: List[Dict],
        raw_fees: List[Dict],
        acc_id: str,
    ) -> None:
        """Normalize Futu data and persist."""
        # Normalize orders
        norm_orders = []
        for raw in raw_orders:
            norm = self.futu_normalizer.normalize_order(raw, acc_id)
            if norm:
                norm_orders.append(norm)

        if norm_orders:
            await self.store.upsert_orders_norm(norm_orders)
            logger.debug(f"Normalized {len(norm_orders)} Futu orders")

        # Normalize trades
        norm_trades = []
        for raw in raw_deals:
            norm = self.futu_normalizer.normalize_trade(raw, acc_id)
            if norm:
                norm_trades.append(norm)

        if norm_trades:
            await self.store.upsert_trades_norm(norm_trades)
            logger.debug(f"Normalized {len(norm_trades)} Futu trades")

        # Normalize fees
        norm_fees = []
        for raw in raw_fees:
            fees = self.futu_normalizer.normalize_fee(raw, acc_id)
            norm_fees.extend(fees)

        if norm_fees:
            await self.store.upsert_fees_norm(norm_fees)
            logger.debug(f"Normalized {len(norm_fees)} Futu fees")

            # Update total_fee in apex_order from apex_fees
            await self.store.update_order_fees_from_apex_fees()
            logger.debug("Updated order total_fee from fees")

    async def _process_ib(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, int]:
        """Process IB historical data via Flex reports."""
        if not self.ib_adapter:
            return {"orders": 0, "trades": 0, "fees": 0}

        logger.info("Processing IB data...")
        counts = {"orders": 0, "trades": 0, "fees": 0}

        # Get IB Flex configuration
        ib_config = self.config.get("ib", {})
        flex_token = ib_config.get("flex_token")
        flex_query_id = ib_config.get("flex_query_id")

        # Validate credentials - check for missing or placeholder values
        def is_valid_credential(val: str) -> bool:
            if not val:
                return False
            # Check for placeholder patterns like ${VAR} or $VAR
            if val.startswith("${") or val.startswith("$"):
                return False
            return True

        if not is_valid_credential(flex_token) or not is_valid_credential(flex_query_id):
            logger.warning(
                "IB Flex credentials not configured (set IB_FLEX_TOKEN and IB_FLEX_QUERY_ID env vars), "
                "skipping IB Flex processing"
            )
            return counts

        try:
            from ..adapters.ib.flex_parser import FlexParser, flex_trade_to_dict, flex_order_to_dict

            parser = FlexParser(flex_token, flex_query_id)
            result = parser.fetch_and_parse()

            # Process orders
            for flex_order in result.get("orders", []):
                account_id = flex_order.account
                raw_dict = flex_order_to_dict(flex_order)
                raw_dict["source"] = "FLEX"
                raw_dict["account"] = account_id

                await self.store.upsert_orders_raw_ib([raw_dict])

                norm = self.ib_normalizer.normalize_flex_order(flex_order, account_id)
                if norm:
                    await self.store.upsert_orders_norm([norm])

            counts["orders"] = len(result.get("orders", []))

            # Process trades
            for flex_trade in result.get("trades", []):
                account_id = flex_trade.account
                raw_dict = flex_trade_to_dict(flex_trade)
                raw_dict["source"] = "FLEX"
                raw_dict["account"] = account_id
                raw_dict["exec_id"] = flex_trade.trade_id

                await self.store.upsert_trades_raw_ib([raw_dict])

                # Normalize trade
                norm = self.ib_normalizer.normalize_flex_trade(flex_trade, account_id)
                if norm:
                    await self.store.upsert_trades_norm([norm])

                # Normalize fees
                fees = self.ib_normalizer.normalize_flex_fee(flex_trade, account_id)
                if fees:
                    await self.store.upsert_fees_norm(fees)
                    counts["fees"] += len(fees)

            counts["trades"] = len(result.get("trades", []))

            # Update total_fee in apex_order from apex_fees
            if counts["fees"] > 0:
                await self.store.update_order_fees_from_apex_fees()
                logger.debug("Updated order total_fee from IB fees")

            logger.info(
                f"Processed IB Flex: {counts['orders']} orders, "
                f"{counts['trades']} trades, {counts['fees']} fees"
            )

        except ImportError:
            logger.warning("ibflex not installed, skipping IB Flex processing")
        except Exception as e:
            logger.error(f"IB Flex processing failed: {e}")
            # Don't raise - let orchestrator continue with other brokers

        return counts

    async def _classify_strategies(self) -> int:
        """Run strategy classification on unclassified trades."""
        logger.info("Classifying strategies...")

        # Get unclassified trades
        trades = await self.store.get_unclassified_trades(limit=5000)

        if not trades:
            logger.debug("No unclassified trades found")
            return 0

        # Classify in batches
        results = self.classifier.classify_batch(trades)

        # Convert to mappings and persist
        total_mappings = 0
        for result in results:
            mappings = self.classifier.result_to_mappings(result)
            if mappings:
                await self.store.upsert_strategy_mappings(mappings)
                total_mappings += len(mappings)

        logger.info(f"Classified {len(results)} strategies ({total_mappings} order mappings)")
        return len(results)

    async def save_position_snapshot(
        self,
        positions: List[Dict[str, Any]],
        account_id: str,
        snapshot_id: Optional[str] = None,
    ) -> int:
        """
        Save a position snapshot.

        Args:
            positions: List of position dicts.
            account_id: Account identifier.
            snapshot_id: Optional snapshot ID (default: timestamp-based).

        Returns:
            Number of positions saved.
        """
        if not snapshot_id:
            snapshot_id = f"snap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        return await self.store.save_position_snapshot(
            snapshot_id=snapshot_id,
            account_id=account_id,
            positions=positions,
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("persistent", config)


async def run_backfill(
    config_path: str = "config/persistent.yaml",
    full_reload: bool = False,
    days: Optional[int] = None,
    futu_adapter=None,
    ib_adapter=None,
) -> Dict[str, Any]:
    """
    Convenience function to run backfill.

    Args:
        config_path: Path to configuration file.
        full_reload: If True, truncate and reload all data.
        days: Override lookback days.
        futu_adapter: Optional FutuAdapter instance.
        ib_adapter: Optional IbAdapter instance.

    Returns:
        Summary dict.
    """
    config = load_config(config_path)

    orchestrator = PersistenceOrchestrator(
        config=config,
        futu_adapter=futu_adapter,
        ib_adapter=ib_adapter,
    )

    return await orchestrator.run(full_reload=full_reload, days=days)
