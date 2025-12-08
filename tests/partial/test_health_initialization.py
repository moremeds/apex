"""
Test that health components are properly initialized and displayed.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tui.dashboard import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.infrastructure.monitoring import ComponentHealth, HealthStatus, HealthMonitor, Watchdog
from src.application.simple_event_bus import SimpleEventBus


def test_health_initialization():
    """Test that health components are initialized properly."""
    print("=" * 80)
    print("Testing Health Component Initialization")
    print("=" * 80)

    # Create health monitor and watchdog
    print("\n1. Creating health monitor and watchdog...")
    health_monitor = HealthMonitor()
    event_bus = SimpleEventBus()

    watchdog_config = {
        "snapshot_stale_sec": 10,
        "max_missing_md_ratio": 0.2,
        "reconnect_backoff_sec": {
            "initial": 1,
            "max": 60,
            "factor": 2,
        }
    }

    watchdog = Watchdog(
        health_monitor=health_monitor,
        event_bus=event_bus,
        config=watchdog_config
    )
    print("   ✓ Watchdog created")

    # Simulate connection health updates
    print("\n2. Simulating provider connections...")
    health_monitor.update_component_health(
        "ib_adapter",
        HealthStatus.HEALTHY,
        "Connected"
    )
    health_monitor.update_component_health(
        "file_loader",
        HealthStatus.HEALTHY,
        "Loaded"
    )
    print("   ✓ Provider health updated")

    # Get all health components
    print("\n3. Getting all health components...")
    health_list = health_monitor.get_all_health()
    print(f"   ✓ Found {len(health_list)} health components:")

    for h in health_list:
        icon = "✓" if h.status == HealthStatus.HEALTHY else "⚠" if h.status == HealthStatus.DEGRADED else "✗" if h.status == HealthStatus.UNHEALTHY else "○"
        print(f"     {icon} {h.component_name}: {h.message}")

    # Create dashboard and render
    print("\n4. Rendering dashboard with initialized health...")
    dashboard = TerminalDashboard({"show_positions": False})

    # Empty snapshot (simulating startup before first cycle)
    empty_snapshot = RiskSnapshot()

    dashboard.update(empty_snapshot, [], health_list, [], {})

    # Render health panel
    from rich.console import Console
    console = Console()

    print("\n5. Health panel:")
    panel = dashboard._render_health(health_list)
    console.print(panel)

    # Verify all expected components are present
    print("\n6. Verification:")
    expected_components = ["ib_adapter", "file_loader", "market_data_coverage", "snapshot_freshness"]
    component_names = [h.component_name for h in health_list]

    all_present = all(name in component_names for name in expected_components)
    if all_present:
        print(f"   ✓ All {len(expected_components)} expected components present")
    else:
        missing = [name for name in expected_components if name not in component_names]
        print(f"   ✗ Missing components: {missing}")
        return False

    print("\n" + "=" * 80)
    print("✓ Health initialization test passed!")
    print("=" * 80)
    print("\nSummary:")
    print("  - Watchdog initializes market_data_coverage and snapshot_freshness")
    print("  - Connection health is updated when providers connect")
    print("  - Dashboard displays all health components immediately")
    print("  - UNKNOWN status shows as ○ (dim)")
    print("\nThe dashboard will now show health status from startup!")
    print()

    return True


if __name__ == "__main__":
    success = test_health_initialization()
    sys.exit(0 if success else 1)
