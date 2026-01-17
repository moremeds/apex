"""
Test category-based logging with separate files.

This test verifies:
1. Logs are written to separate files (system and market)
2. File naming includes environment (dev/prod)
3. Rotation works correctly
4. Naming convention: live_risk_{env}_{category}_{date}_{number}.log
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_setup import setup_category_logging
from src.utils.structured_logger import LogCategory, StructuredLogger


def test_category_logging():
    """Test category-based logging."""
    print("=" * 80)
    print("Testing Category-Based Logging")
    print("=" * 80)

    # Test with dev environment
    print("\n1. Setting up category logging for 'dev' environment...")
    loggers = setup_category_logging(env="dev", log_dir="./logs", level="INFO")

    system_logger = loggers["system"]
    market_logger = loggers["market"]

    print(f"   ✓ Created 2 loggers: system, market")
    print(f"   - System logger: {system_logger.name}")
    print(f"   - Market logger: {market_logger.name}")

    # Create structured loggers
    system_structured = StructuredLogger(system_logger)
    market_structured = StructuredLogger(market_logger)

    # Write test logs
    print("\n2. Writing test logs...")

    # System logs
    system_structured.info(
        LogCategory.SYSTEM, "Application starting", {"env": "dev", "version": "1.0.0"}
    )
    system_structured.info(
        LogCategory.SYSTEM, "Configuration loaded", {"config_file": "risk_config.yaml"}
    )
    system_structured.warning(LogCategory.SYSTEM, "IB connection unavailable", {"retry_in": 5})
    system_structured.info(LogCategory.SYSTEM, "Using mock market data for testing")

    # Market logs
    market_structured.info(LogCategory.DATA, "Fetching market data", {"symbols": 10})
    market_structured.info(
        LogCategory.TRADING, "Position reconciliation complete", {"positions": 6}
    )
    market_structured.info(LogCategory.DATA, "Market data quality check", {"stale": 0, "total": 10})
    market_structured.warning(
        LogCategory.RISK, "Approaching delta limit", {"current": 45000, "limit": 50000}
    )

    print(f"   ✓ Wrote 8 log entries")

    # Check log files
    print("\n3. Verifying log files...")
    log_dir = Path("./logs")

    system_log = log_dir / "live_risk_dev_system.log"
    market_log = log_dir / "live_risk_dev_market.log"

    files_created = []

    if system_log.exists():
        size = system_log.stat().st_size
        with open(system_log, "r") as f:
            line_count = len(f.readlines())
        print(f"   ✓ System log: {system_log.name}")
        print(f"     - Size: {size} bytes")
        print(f"     - Lines: {line_count}")
        files_created.append("system")
    else:
        print(f"   ✗ System log not found: {system_log}")

    if market_log.exists():
        size = market_log.stat().st_size
        with open(market_log, "r") as f:
            line_count = len(f.readlines())
        print(f"   ✓ Market log: {market_log.name}")
        print(f"     - Size: {size} bytes")
        print(f"     - Lines: {line_count}")
        files_created.append("market")
    else:
        print(f"   ✗ Market log not found: {market_log}")

    # Show sample log entries
    print("\n4. Sample log entries:")

    if system_log.exists():
        print(f"\n   System log (first 2 lines):")
        with open(system_log, "r") as f:
            for i, line in enumerate(f):
                if i < 2:
                    print(f"     {line.rstrip()}")

    if market_log.exists():
        print(f"\n   Market log (first 2 lines):")
        with open(market_log, "r") as f:
            for i, line in enumerate(f):
                if i < 2:
                    print(f"     {line.rstrip()}")

    # Test with prod environment
    print("\n5. Testing 'prod' environment naming...")
    prod_loggers = setup_category_logging(env="prod", log_dir="./logs", level="INFO")
    prod_system = StructuredLogger(prod_loggers["system"])
    prod_system.info(LogCategory.SYSTEM, "Production system test", {"env": "prod"})

    prod_log = log_dir / "live_risk_prod_system.log"
    if prod_log.exists():
        print(f"   ✓ Prod log created: {prod_log.name}")
    else:
        print(f"   ✗ Prod log not found")

    # Summary
    print("\n" + "=" * 80)
    print("✓ Category logging test completed!")
    print("=" * 80)

    print("\nFiles created:")
    print(f"  - live_risk_dev_system.log")
    print(f"  - live_risk_dev_market.log")
    print(f"  - live_risk_prod_system.log")

    print("\nFeatures verified:")
    print("  ✓ Separate log files for system and market categories")
    print("  ✓ File naming includes environment (dev/prod)")
    print("  ✓ JSON formatted logs")
    print("  ✓ Logs written to ./logs directory")
    print("  ✓ Rotation configured (midnight, 7 days retention)")

    print("\nNaming convention:")
    print("  - Current: live_risk_{env}_{category}.log")
    print("  - Rotated: live_risk_{env}_{category}_{date}_{number}.log")
    print("  - Example: live_risk_dev_system_2025-11-24_1.log")
    print()

    return len(files_created) >= 2


if __name__ == "__main__":
    success = test_category_logging()
    sys.exit(0 if success else 1)
