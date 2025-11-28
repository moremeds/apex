"""
Example: How to use Market Alerts in the dashboard.

This example demonstrates:
1. Using the MarketAlertDetector service
2. Fetching market-wide data (VIX, SPY, etc.)
3. Displaying alerts in the dashboard
"""

import asyncio
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.market_alert_detector import MarketAlertDetector
from src.presentation.dashboard import TerminalDashboard
from src.models.risk_snapshot import RiskSnapshot
from src.infrastructure.monitoring import ComponentHealth, HealthStatus


def example_market_data_fetch():
    """
    Example: Fetch market-wide data.

    In a real implementation, you would:
    1. Use your IB adapter to fetch VIX, SPY, QQQ data
    2. Calculate daily % changes
    3. Calculate realized volatility

    For now, we'll simulate this data.
    """
    # Simulated market data
    return {
        "vix": 28.5,  # Current VIX level
        "spy_change_pct": -2.3,  # SPY down 2.3% today
        "qqq_change_pct": -1.8,  # QQQ down 1.8% today
        "spy_realized_vol": 35.2,  # SPY 20-day realized vol
        "timestamp": datetime.now(),
    }


def example_1_basic_alerts():
    """Example 1: Basic market alert detection."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Market Alert Detection")
    print("="*80 + "\n")

    # Initialize detector with custom thresholds
    config = {
        "vix_warning_threshold": 25.0,
        "vix_critical_threshold": 35.0,
        "vix_spike_pct": 15.0,
        "market_drop_warning": -2.0,
        "market_drop_critical": -3.0,
    }

    detector = MarketAlertDetector(config)

    # Fetch market data
    market_data = example_market_data_fetch()
    print(f"Market Data:")
    print(f"  VIX: {market_data['vix']}")
    print(f"  SPY Change: {market_data['spy_change_pct']:.1f}%")
    print(f"  QQQ Change: {market_data['qqq_change_pct']:.1f}%")
    print(f"  SPY Realized Vol: {market_data['spy_realized_vol']:.1f}%")

    # Detect alerts
    alerts = detector.detect_alerts(market_data)

    print(f"\nDetected {len(alerts)} alerts:")
    for alert in alerts:
        severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🔴"}
        icon = severity_icon.get(alert['severity'], "")
        print(f"  {icon} [{alert['severity']}] {alert['type']}: {alert['message']}")

    return alerts


def example_2_dashboard_integration():
    """Example 2: Display alerts in dashboard."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Dashboard Integration")
    print("="*80 + "\n")

    # Detect alerts
    detector = MarketAlertDetector()
    market_data = example_market_data_fetch()
    alerts = detector.detect_alerts(market_data)

    # Create dashboard
    config = {"show_positions": True}
    dashboard = TerminalDashboard(config)

    # Create dummy snapshot
    snapshot = RiskSnapshot(
        timestamp=datetime.now(),
        total_unrealized_pnl=5000,
        total_daily_pnl=-1200,
        portfolio_delta=250,
        portfolio_gamma=-15,
        portfolio_vega=-500,
        portfolio_theta=80,
        total_gross_notional=500000,
        total_net_notional=250000,
        max_underlying_notional=150000,
        max_underlying_symbol="SPY",
        concentration_pct=0.30,
        margin_utilization=0.42,
        buying_power=75000,
    )

    health = [
        ComponentHealth("ib_adapter", HealthStatus.HEALTHY, datetime.now(), "Connected"),
        ComponentHealth("market_data_feed", HealthStatus.HEALTHY, datetime.now(), "Live"),
    ]

    # Update dashboard with alerts
    dashboard.update(
        snapshot=snapshot,
        breaches=[],
        health=health,
        positions=[],
        market_data={},
        market_alerts=alerts,  # Pass the alerts here
    )

    print("✓ Dashboard updated with market alerts")
    print(f"✓ {len(alerts)} alerts displayed in Market Alerts section")


def example_3_integration_in_main():
    """Example 3: How to integrate into main.py."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Integration into main.py")
    print("="*80 + "\n")

    print("To integrate market alerts into main.py:")
    print()
    print("1. Initialize MarketAlertDetector in main.py:")
    print("   ```python")
    print("   # In main_async() function")
    print("   alert_config = config.raw.get('market_alerts', {})")
    print("   alert_detector = MarketAlertDetector(alert_config)")
    print("   ```")
    print()
    print("2. Fetch market-wide data in the update loop:")
    print("   ```python")
    print("   # In the while True loop")
    print("   # Fetch VIX, SPY, QQQ data using IB adapter")
    print("   vix_data = await ib_adapter.fetch_market_data_for_symbol('VIX')")
    print("   spy_data = await ib_adapter.fetch_market_data_for_symbol('SPY')")
    print("   ")
    print("   market_data = {")
    print("       'vix': vix_data.last,")
    print("       'spy_change_pct': calculate_change_pct(spy_data),")
    print("       'timestamp': datetime.now(),")
    print("   }")
    print("   ```")
    print()
    print("3. Detect alerts and pass to dashboard:")
    print("   ```python")
    print("   # Detect market alerts")
    print("   market_alerts = alert_detector.detect_alerts(market_data)")
    print("   ")
    print("   # Update dashboard")
    print("   dashboard.update(")
    print("       snapshot, breaches, health,")
    print("       positions, market_data,")
    print("       market_alerts  # Pass alerts here")
    print("   )")
    print("   ```")
    print()
    print("4. Add configuration to risk_config.yaml:")
    print("   ```yaml")
    print("   # Market alert thresholds")
    print("   market_alerts:")
    print("     vix_warning_threshold: 25.0")
    print("     vix_critical_threshold: 35.0")
    print("     vix_spike_pct: 15.0")
    print("     market_drop_warning: -2.0")
    print("     market_drop_critical: -3.0")
    print("   ```")


if __name__ == "__main__":
    # Run examples
    example_1_basic_alerts()
    example_2_dashboard_integration()
    example_3_integration_in_main()

    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80 + "\n")
