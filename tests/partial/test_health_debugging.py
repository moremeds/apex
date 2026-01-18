"""
Debug test to check if health components are being created and displayed.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.monitoring import HealthMonitor, HealthStatus
from src.models.risk_snapshot import RiskSnapshot
from src.tui.dashboard import TerminalDashboard


def test_health_creation():
    """Test if health components display correctly."""
    print("=" * 80)
    print("Debugging Health Component Display")
    print("=" * 80)

    # Create health monitor and add components
    print("\n1. Creating health monitor and components...")
    health_monitor = HealthMonitor()

    # Add components
    health_monitor.update_component_health("ib_adapter", HealthStatus.HEALTHY, "Connected")

    health_monitor.update_component_health("file_loader", HealthStatus.HEALTHY, "Loaded")

    health_monitor.update_component_health(
        "market_data_coverage",
        HealthStatus.DEGRADED,
        "Missing MD: 15.0%",
        {"missing_count": 3, "total": 20},
    )

    health_monitor.update_component_health(
        "snapshot_freshness",
        HealthStatus.HEALTHY,
        "Fresh (2.1s)",
    )

    print("   ✓ Added 4 health components")

    # Get all health components
    print("\n2. Retrieving health components...")
    health_list = health_monitor.get_all_health()
    print(f"   ✓ Retrieved {len(health_list)} components")

    # Print details of each component
    print("\n3. Health component details:")
    for h in health_list:
        print(f"   - {h.component_name}:")
        print(f"     Status: {h.status}")
        print(f"     Message: '{h.message}'")
        print(f"     Metadata: {h.metadata}")

    # Create dashboard and test rendering
    print("\n4. Testing dashboard rendering...")
    dashboard = TerminalDashboard({"show_positions": False})

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
    )

    # Update dashboard
    dashboard.update(snapshot, [], health_list)

    # Render health panel directly
    print("\n5. Rendering health panel...")
    panel = dashboard._render_health(health_list)
    print(f"   ✓ Panel created with title: '{panel.title}'")
    print(f"   ✓ Panel border style: '{panel.border_style}'")

    # Try to print the panel
    print("\n6. Health panel content:")
    from rich.console import Console

    console = Console()
    console.print(panel)

    print("\n" + "=" * 80)
    print("✓ Health debugging complete")
    print("=" * 80)
    print("\nIf you see the health panel above with all details, the code is working!")
    print("If not, there may be an issue with the terminal rendering or the dashboard update loop.")
    print()

    return True


if __name__ == "__main__":
    success = test_health_creation()
    sys.exit(0 if success else 1)
