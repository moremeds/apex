"""
Live test to check health component registration in real application flow.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_manager import ConfigManager
from src.infrastructure.adapters import IbAdapter, FileLoader
from src.infrastructure.stores import PositionStore, MarketDataStore, AccountStore
from src.infrastructure.monitoring import HealthMonitor, Watchdog
from src.domain.services.risk.risk_engine import RiskEngine
from src.domain.services.pos_reconciler import Reconciler
from src.domain.services.mdqc import MDQC
from src.domain.services.risk.rule_engine import RuleEngine
from src.application import Orchestrator, SimpleEventBus


async def test_health_registration():
    """Test health component registration in real flow."""
    print("=" * 80)
    print("Testing Health Component Registration (Live)")
    print("=" * 80)

    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config_manager = ConfigManager(config_dir="../..", env="dev")
        config = config_manager.load()
        print(f"   ✓ Configuration loaded")

        # Initialize event bus
        event_bus = SimpleEventBus()

        # Initialize data stores
        position_store = PositionStore()
        market_data_store = MarketDataStore()
        account_store = AccountStore()

        # Initialize adapters
        ib_adapter = IbAdapter(
            host=config.ibkr.host,
            port=config.ibkr.port,
            client_id=config.ibkr.client_id,
        )

        file_loader = FileLoader(
            file_path=config.manual_positions.file,
            reload_interval_sec=config.manual_positions.reload_interval_sec,
        )

        # Initialize domain services
        risk_engine = RiskEngine(config=config.raw)
        reconciler = Reconciler(stale_threshold_seconds=300)
        mdqc = MDQC(
            stale_seconds=config.mdqc.stale_seconds,
            ignore_zero_quotes=config.mdqc.ignore_zero_quotes,
            enforce_bid_ask_sanity=config.mdqc.enforce_bid_ask_sanity,
        )
        rule_engine = RuleEngine(
            risk_limits=config.raw.get("risk_limits", {}),
            soft_threshold=config.risk_limits.soft_breach_threshold,
        )

        # Initialize monitoring
        print("\n2. Creating health monitor...")
        health_monitor = HealthMonitor()
        print(f"   ✓ Health monitor created")
        health = health_monitor.get_all_health()
        print(f"   - Components: {len(health)}")

        print("\n3. Creating watchdog...")
        watchdog = Watchdog(
            health_monitor=health_monitor,
            event_bus=event_bus,
            config=config.raw.get("watchdog", {}),
        )
        print(f"   ✓ Watchdog created")
        health = health_monitor.get_all_health()
        print(f"   - Components after watchdog: {len(health)}")
        for h in health:
            print(f"     • {h.component_name}: {h.status.value}")

        # Initialize orchestrator
        print("\n4. Creating orchestrator...")
        orchestrator = Orchestrator(
            ib_adapter=ib_adapter,
            file_loader=file_loader,
            position_store=position_store,
            market_data_store=market_data_store,
            account_store=account_store,
            risk_engine=risk_engine,
            reconciler=reconciler,
            mdqc=mdqc,
            rule_engine=rule_engine,
            health_monitor=health_monitor,
            watchdog=watchdog,
            event_bus=event_bus,
            config=config.raw,
        )
        print(f"   ✓ Orchestrator created")

        # Start orchestrator
        print("\n5. Starting orchestrator...")
        await orchestrator.start()
        print(f"   ✓ Orchestrator started")

        # Wait a bit for initialization
        await asyncio.sleep(1)

        # Check health components
        print("\n6. Checking health components after start...")
        health = health_monitor.get_all_health()
        print(f"   ✓ Total components: {len(health)}")

        if len(health) < 4:
            print(f"   ✗ ISSUE: Expected 4 components, got {len(health)}")
        else:
            print(f"   ✓ All 4 components registered!")

        print("\n7. Component details:")
        for i, h in enumerate(health, 1):
            icon = "✓" if h.status.value == "HEALTHY" else "⚠" if h.status.value == "DEGRADED" else "✗" if h.status.value == "UNHEALTHY" else "○"
            print(f"   [{i}] {icon} {h.component_name}: {h.status.value}")
            if h.message:
                print(f"       Message: {h.message}")

        # Stop orchestrator
        print("\n8. Stopping orchestrator...")
        await orchestrator.stop()
        print(f"   ✓ Orchestrator stopped")

        print("\n" + "=" * 80)
        print(f"✓ Test completed: {len(health)}/4 components registered")
        print("=" * 80)
        print()

        return len(health) == 4

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_health_registration())
    sys.exit(0 if success else 1)
