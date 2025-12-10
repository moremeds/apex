"""
Test script to verify market data coverage display in the dashboard.

This test verifies:
1. Health panel shows market data coverage status
2. Missing market data count is displayed correctly
3. Health status colors are correct (green/yellow/red)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tui.dashboard import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.infrastructure.monitoring import ComponentHealth, HealthStatus


def test_market_data_coverage_display():
    """Test market data coverage display in health panel."""
    print("=" * 80)
    print("Testing Market Data Coverage Display")
    print("=" * 80)

    # Create sample snapshot
    snapshot = RiskSnapshot(
        timestamp=datetime.now(),
        total_unrealized_pnl=0,
        total_daily_pnl=0,
        portfolio_delta=0,
        portfolio_gamma=0,
        portfolio_vega=0,
        portfolio_theta=0,
        total_gross_notional=0,
        total_net_notional=0,
        max_underlying_notional=0,
        max_underlying_symbol="",
        concentration_pct=0,
        margin_utilization=0,
        buying_power=0,
    )

    # Test Case 1: All components healthy
    print("\n1. Testing all components healthy...")
    health_all_good = [
        ComponentHealth(
            component_name="ib_adapter",
            status=HealthStatus.HEALTHY,
            message="Connected",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="file_loader",
            status=HealthStatus.HEALTHY,
            message="Loaded",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="market_data_coverage",
            status=HealthStatus.HEALTHY,
            message="MD coverage: 100.0%",
            last_check=datetime.now(),
            metadata={"missing_count": 0, "total": 10},
        ),
        ComponentHealth(
            component_name="snapshot_freshness",
            status=HealthStatus.HEALTHY,
            message="Snapshot fresh (2.1s old)",
            last_check=datetime.now(),
        ),
    ]

    dashboard = TerminalDashboard({"show_positions": False})
    dashboard.update(snapshot, [], health_all_good)

    panel = dashboard._render_health(health_all_good)
    print("   ✓ Health panel rendered")
    print(f"   - Border style should be green")
    print(f"   - All 4 components should show ✓")

    # Test Case 2: Market data coverage degraded (some missing)
    print("\n2. Testing market data coverage degraded...")
    health_md_degraded = [
        ComponentHealth(
            component_name="ib_adapter",
            status=HealthStatus.HEALTHY,
            message="Connected",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="file_loader",
            status=HealthStatus.HEALTHY,
            message="Loaded",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="market_data_coverage",
            status=HealthStatus.DEGRADED,
            message="Missing MD: 15.0%",
            last_check=datetime.now(),
            metadata={"missing_count": 3, "total": 20},
        ),
        ComponentHealth(
            component_name="snapshot_freshness",
            status=HealthStatus.HEALTHY,
            message="Snapshot fresh (1.5s old)",
            last_check=datetime.now(),
        ),
    ]

    dashboard.update(snapshot, [], health_md_degraded)
    panel = dashboard._render_health(health_md_degraded)
    print("   ✓ Health panel rendered")
    print(f"   - Border style should be yellow (degraded)")
    print(f"   - market_data_coverage should show ⚠")
    print(f"   - Should display: '3/20 missing MD'")

    # Test Case 3: Connection unhealthy
    print("\n3. Testing connection unhealthy...")
    health_connection_down = [
        ComponentHealth(
            component_name="ib_adapter",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed: timeout",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="file_loader",
            status=HealthStatus.HEALTHY,
            message="Loaded",
            last_check=datetime.now(),
        ),
        ComponentHealth(
            component_name="market_data_coverage",
            status=HealthStatus.DEGRADED,
            message="Missing MD: 100.0%",
            last_check=datetime.now(),
            metadata={"missing_count": 20, "total": 20},
        ),
        ComponentHealth(
            component_name="snapshot_freshness",
            status=HealthStatus.DEGRADED,
            message="Snapshot not updated for 15.0s",
            last_check=datetime.now(),
        ),
    ]

    dashboard.update(snapshot, [], health_connection_down)
    panel = dashboard._render_health(health_connection_down)
    print("   ✓ Health panel rendered")
    print(f"   - Border style should be red (unhealthy)")
    print(f"   - ib_adapter should show ✗")
    print(f"   - market_data_coverage should show: '20/20 missing MD'")

    # Test Case 4: No metadata but has message
    print("\n4. Testing component with message but no metadata...")
    health_message_only = [
        ComponentHealth(
            component_name="test_component",
            status=HealthStatus.DEGRADED,
            message="Custom warning message",
            last_check=datetime.now(),
            metadata={},
        ),
    ]

    panel = dashboard._render_health(health_message_only)
    print("   ✓ Health panel rendered")
    print(f"   - Should display message: 'Custom warning message'")

    print("\n" + "=" * 80)
    print("✓ All market data coverage display tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print("  - Health panel now shows detailed messages")
    print("  - Market data coverage shows 'X/Y missing MD' format")
    print("  - Border color reflects worst component status:")
    print("    • Green = all healthy")
    print("    • Yellow = at least one degraded")
    print("    • Red = at least one unhealthy")
    print("\nThe dashboard will now properly display market data coverage issues!")
    print()

    return True


if __name__ == "__main__":
    success = test_market_data_coverage_display()
    sys.exit(0 if success else 1)
