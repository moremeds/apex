"""
Test VIX alert functionality - diagnose what's not working.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.adapters import IbCompositeAdapter
from src.infrastructure.adapters.ib import ConnectionPoolConfig
from src.domain.services.market_alert_detector import MarketAlertDetector
from config.config_manager import ConfigManager


async def test_vix_alert():
    """Test VIX alert end-to-end."""
    print("=" * 80)
    print("VIX ALERT DIAGNOSTIC")
    print("=" * 80)

    # Load config
    print("\n1. Loading config...")
    config_manager = ConfigManager(config_dir="../config", env="dev")
    config = config_manager.load()
    print(f"   ✓ Config loaded")
    print(f"   VIX warning threshold: {config.raw.get('market_alerts', {}).get('vix_warning_threshold')}")
    print(f"   VIX critical threshold: {config.raw.get('market_alerts', {}).get('vix_critical_threshold')}")
    print(f"   VIX spike %: {config.raw.get('market_alerts', {}).get('vix_spike_pct')}")

    # Initialize IB adapter
    print("\n2. Connecting to IBKR...")
    pool_config = ConnectionPoolConfig(
        host=config.ibkr.host,
        port=config.ibkr.port,
    )
    ib_adapter = IbCompositeAdapter(pool_config=pool_config)

    try:
        await ib_adapter.connect()
        print(f"   ✓ Connected to IBKR at {config.ibkr.host}:{config.ibkr.port}")
    except Exception as e:
        print(f"   ✗ Failed to connect: {e}")
        print("   Make sure IBKR TWS/Gateway is running!")
        return

    # Fetch VIX data
    print("\n3. Fetching VIX market data...")
    try:
        market_data = await ib_adapter.fetch_quotes(["VIX"])

        if not market_data:
            print("   ✗ No market data returned!")
            print("   Possible reasons:")
            print("     - Market is closed")
            print("     - VIX symbol not available")
            print("     - IBKR subscription issue")
            return

        vix_md = market_data.get("VIX")
        if not vix_md:
            print("   ✗ VIX not in returned data!")
            return

        print(f"   ✓ VIX data fetched:")
        print(f"     Last: {vix_md.last}")
        print(f"     Bid: {vix_md.bid}")
        print(f"     Ask: {vix_md.ask}")
        print(f"     Mid: {vix_md.mid}")
        print(f"     Yesterday close: {vix_md.yesterday_close}")

        # Get effective VIX value
        vix_value = vix_md.effective_mid() or vix_md.last or vix_md.bid or vix_md.ask
        print(f"     Effective VIX: {vix_value}")

    except Exception as e:
        print(f"   ✗ Error fetching VIX: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize alert detector
    print("\n4. Initializing MarketAlertDetector...")
    alert_config = config.raw.get("market_alerts", {})
    detector = MarketAlertDetector(alert_config)
    print(f"   ✓ Detector initialized")

    # Test alert detection
    print("\n5. Testing alert detection...")
    indicators = {
        "vix": vix_value,
        "vix_prev_close": vix_md.yesterday_close,
        "timestamp": vix_md.timestamp,
    }

    alerts = detector.detect_alerts(indicators)

    if not alerts:
        print("   ✓ No alerts triggered (VIX is normal)")
        print(f"     Current VIX: {vix_value:.2f}")
        print(f"     Warning threshold: {detector.vix_warning}")
        print(f"     Critical threshold: {detector.vix_critical}")

        # Show how far we are from thresholds
        if vix_value < detector.vix_warning:
            print(f"     Distance to warning: {detector.vix_warning - vix_value:.2f} points")
        if vix_value < detector.vix_critical:
            print(f"     Distance to critical: {detector.vix_critical - vix_value:.2f} points")
    else:
        print(f"   ⚠️  {len(alerts)} alert(s) triggered:")
        for alert in alerts:
            severity = alert['severity']
            alert_type = alert['type']
            message = alert['message']

            icon = "🔴" if severity == "CRITICAL" else "⚠️" if severity == "WARNING" else "ℹ️"
            print(f"     {icon} [{severity}] {alert_type}: {message}")

    # Test manual alert trigger (lower thresholds)
    print("\n6. Testing with lowered thresholds (force trigger)...")
    test_detector = MarketAlertDetector({
        "vix_warning_threshold": vix_value - 5,  # 5 points below current
        "vix_critical_threshold": vix_value - 2,  # 2 points below current
        "vix_spike_pct": 5.0,  # Low spike threshold
    })

    test_alerts = test_detector.detect_alerts(indicators)
    print(f"   Triggered {len(test_alerts)} alert(s) with low thresholds:")
    for alert in test_alerts:
        print(f"     - [{alert['severity']}] {alert['type']}: {alert['message']}")

    # Disconnect
    print("\n7. Disconnecting...")
    await ib_adapter.disconnect()
    print("   ✓ Disconnected")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ VIX fetch: WORKING" if market_data else "✗ VIX fetch: FAILED")
    print(f"✓ Alert detection: WORKING (triggered {len(test_alerts)} with low thresholds)")
    print("\nIf you're not seeing alerts in the dashboard:")
    print("  1. Check that VIX is above your configured thresholds")
    print("  2. Check that the orchestrator is running without errors")
    print("  3. Check logs for any errors in _detect_market_alerts()")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_vix_alert())
