"""
Integration test for multi-broker setup (IBKR + Futu).

Tests:
- BrokerManager with multiple adapters
- Position aggregation across brokers
- Account info aggregation
- Health monitoring for all brokers
- Market data from IBKR (primary source)

Run with: python tests/integration/test_multi_broker.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.domain.services.pos_reconciler import Reconciler
from src.infrastructure.adapters import BrokerManager, FutuAdapter, IbCompositeAdapter
from src.infrastructure.adapters.ib import ConnectionPoolConfig
from src.infrastructure.monitoring import HealthMonitor, HealthStatus
from src.models.position import PositionSource


def print_health_status(health_monitor: HealthMonitor):
    """Print health status for all components."""
    print("\n  Health Status:")
    for health in health_monitor.get_all_health():
        status_icon = {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.DEGRADED: "🟡",
            HealthStatus.UNHEALTHY: "🔴",
            HealthStatus.UNKNOWN: "⚪",
        }.get(health.status, "⚪")
        print(
            f"    {status_icon} {health.component_name}: {health.status.value} - {health.message}"
        )
        if health.metadata:
            for key, value in health.metadata.items():
                if value is not None:
                    print(f"       └─ {key}: {value}")


async def test_multi_broker():
    """Test multi-broker position and account aggregation."""

    print("\n" + "=" * 70)
    print("Multi-Broker Integration Test (IBKR + Futu)")
    print("=" * 70)

    # Create health monitor
    health_monitor = HealthMonitor()

    # Create broker manager with health monitor
    broker_manager = BrokerManager(health_monitor=health_monitor)

    # Create and register adapters
    pool_config = ConnectionPoolConfig(host="127.0.0.1", port=4001)
    ib_adapter = IbCompositeAdapter(pool_config=pool_config)

    futu_adapter = FutuAdapter(
        host="127.0.0.1",
        port=11111,
        security_firm="FUTUSECURITIES",
        trd_env="REAL",
        filter_trading_market="US",
    )

    broker_manager.register_adapter("ibkr", ib_adapter)
    broker_manager.register_adapter("futu", futu_adapter)

    try:
        # Connect to all brokers
        print("\n" + "-" * 70)
        print("Connecting to brokers...")
        print("-" * 70)
        await broker_manager.connect()

        # Check status
        for name, status in broker_manager.get_all_status().items():
            status_icon = "✓" if status.connected else "✗"
            print(
                f"  {status_icon} {name}: {'Connected' if status.connected else status.last_error or 'Not connected'}"
            )

        # Show health status after connection
        print_health_status(health_monitor)

        if not broker_manager.is_connected():
            print("\n✗ No brokers connected. Exiting.")
            return False

        # Fetch positions from all brokers
        print("\n" + "-" * 70)
        print("Fetching positions from all brokers...")
        print("-" * 70)

        positions_by_broker = await broker_manager.fetch_positions_by_broker()

        for broker_name, positions in positions_by_broker.items():
            print(f"\n  [{broker_name.upper()}] {len(positions)} positions:")
            for pos in positions[:5]:  # Show first 5
                print(f"    {pos.asset_type.value:8} | {pos.symbol:20} | Qty: {pos.quantity:>8.0f}")
            if len(positions) > 5:
                print(f"    ... and {len(positions) - 5} more")

        # Merge positions using Reconciler
        print("\n" + "-" * 70)
        print("Merging positions across brokers...")
        print("-" * 70)

        reconciler = Reconciler()
        merged_positions = reconciler.merge_all_positions(
            {
                "ib": positions_by_broker.get("ibkr", []),
                "futu": positions_by_broker.get("futu", []),
                "manual": [],
            }
        )

        print(f"\n  Merged: {len(merged_positions)} unique positions")

        # Group by source for summary
        by_source = {}
        for pos in merged_positions:
            source = pos.source.value
            by_source[source] = by_source.get(source, 0) + 1

        print(f"  By source: {by_source}")

        # Show merged positions
        print("\n  Merged positions:")
        for pos in merged_positions[:10]:
            print(
                f"    {pos.source.value:6} | {pos.asset_type.value:8} | {pos.symbol:20} | Qty: {pos.quantity:>8.0f}"
            )
        if len(merged_positions) > 10:
            print(f"    ... and {len(merged_positions) - 10} more")

        # Fetch account info from all brokers
        print("\n" + "-" * 70)
        print("Fetching account info from all brokers...")
        print("-" * 70)

        accounts_by_broker = await broker_manager.fetch_account_info_by_broker()

        for broker_name, account in accounts_by_broker.items():
            print(f"\n  [{broker_name.upper()}] Account Summary:")
            print(f"    Net Liquidation:  ${account.net_liquidation:>15,.2f}")
            print(f"    Total Cash:       ${account.total_cash:>15,.2f}")
            print(f"    Buying Power:     ${account.buying_power:>15,.2f}")
            print(f"    Unrealized P&L:   ${account.unrealized_pnl:>15,.2f}")

        # Fetch aggregated account info
        print("\n" + "-" * 70)
        print("Aggregated Account Summary:")
        print("-" * 70)

        aggregated_account = await broker_manager.fetch_account_info()
        print(f"\n  [TOTAL across all brokers]")
        print(f"    Net Liquidation:  ${aggregated_account.net_liquidation:>15,.2f}")
        print(f"    Total Cash:       ${aggregated_account.total_cash:>15,.2f}")
        print(f"    Buying Power:     ${aggregated_account.buying_power:>15,.2f}")
        print(f"    Margin Used:      ${aggregated_account.margin_used:>15,.2f}")
        print(f"    Unrealized P&L:   ${aggregated_account.unrealized_pnl:>15,.2f}")
        print(f"    Realized P&L:     ${aggregated_account.realized_pnl:>15,.2f}")

        # Final health check
        print("\n" + "-" * 70)
        print("Final Health Check:")
        print("-" * 70)
        await broker_manager.check_all_health()
        print_health_status(health_monitor)

        # Summary
        summary = health_monitor.summary()
        print(
            f"\n  Summary: {summary['healthy']} healthy, {summary['degraded']} degraded, {summary['unhealthy']} unhealthy"
        )

        print("\n" + "=" * 70)
        print("Multi-broker test completed successfully!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        await broker_manager.disconnect()
        print("\nDisconnected from all brokers")


if __name__ == "__main__":
    success = asyncio.run(test_multi_broker())
    sys.exit(0 if success else 1)
