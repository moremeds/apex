"""
Test to verify all 4 health components are properly registered and displayed.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.application.simple_event_bus import SimpleEventBus
from src.infrastructure.monitoring import HealthMonitor, HealthStatus, Watchdog
from src.tui.dashboard import TerminalDashboard


def test_all_health_components():
    """Test that all 4 health components are created and displayed."""
    print("=" * 80)
    print("Testing All Health Components Registration and Display")
    print("=" * 80)

    # Create health monitor
    print("\n1. Creating health monitor...")
    health_monitor = HealthMonitor()
    print(f"   ✓ Health monitor created")

    # Create watchdog (this initializes 2 components)
    print("\n2. Creating watchdog (initializes market_data_coverage, snapshot_freshness)...")
    event_bus = SimpleEventBus()
    watchdog = Watchdog(
        health_monitor=health_monitor,
        event_bus=event_bus,
        config={
            "snapshot_stale_sec": 10,
            "max_missing_md_ratio": 0.2,
            "reconnect_backoff_sec": {"initial": 1, "max": 60, "factor": 2},
        },
    )
    print(f"   ✓ Watchdog created")

    # Manually register ib_adapter and file_loader (simulating orchestrator)
    print("\n3. Simulating orchestrator connection phase...")
    health_monitor.update_component_health(
        "ib_adapter", HealthStatus.UNHEALTHY, "Connection failed: ib_async library not installed"
    )
    health_monitor.update_component_health("file_loader", HealthStatus.HEALTHY, "Loaded")
    print(f"   ✓ IB adapter and file_loader health registered")

    # Update market data coverage with some data
    print("\n4. Updating market data coverage...")
    watchdog.check_missing_market_data(positions_count=6, missing_md_count=0)
    print(f"   ✓ Market data coverage updated")

    # Get all health components
    print("\n5. Retrieving all health components...")
    health_list = health_monitor.get_all_health()
    print(f"   ✓ Retrieved {len(health_list)} components")

    # Display details
    print("\n6. Health component details:")
    for i, h in enumerate(health_list, 1):
        icon = (
            "✓"
            if h.status == HealthStatus.HEALTHY
            else (
                "⚠"
                if h.status == HealthStatus.DEGRADED
                else "✗" if h.status == HealthStatus.UNHEALTHY else "○"
            )
        )
        print(f"   [{i}] {icon} {h.component_name}")
        print(f"       Status: {h.status.value}")
        print(f"       Message: {h.message}")
        if h.metadata:
            print(f"       Metadata: {h.metadata}")

    # Test dashboard rendering
    print("\n7. Testing dashboard rendering...")
    dashboard = TerminalDashboard({"show_positions": False})
    panel = dashboard._render_health(health_list)

    from rich.console import Console

    console = Console()
    print("\n8. Health panel as it would appear in dashboard:")
    console.print(panel)

    # Verification
    print("\n9. Verification:")
    expected_components = {
        "ib_adapter",
        "file_loader",
        "market_data_coverage",
        "snapshot_freshness",
    }
    actual_components = {h.component_name for h in health_list}

    print(f"   Expected: {sorted(expected_components)}")
    print(f"   Actual:   {sorted(actual_components)}")

    missing = expected_components - actual_components
    extra = actual_components - expected_components

    if missing:
        print(f"   ✗ Missing components: {missing}")
    if extra:
        print(f"   ⚠ Extra components: {extra}")

    if len(health_list) == 4 and not missing:
        print(f"   ✓ All 4 expected components present!")
        success = True
    else:
        print(f"   ✗ Component count mismatch: {len(health_list)}/4")
        success = False

    print("\n" + "=" * 80)
    if success:
        print("✓ All health components test PASSED!")
    else:
        print("✗ All health components test FAILED!")
    print("=" * 80)
    print()

    return success


if __name__ == "__main__":
    success = test_all_health_components()
    sys.exit(0 if success else 1)
