"""
Test updated log naming convention: live_risk_{env}_{sys/mkt}_{date}_{number}.log

This test verifies:
1. Current logs use sys/mkt instead of system/market
2. Rotated logs follow the pattern: live_risk_{env}_{sys/mkt}_{date}_{number}.log
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_setup import setup_category_logging
from src.utils.structured_logger import StructuredLogger, LogCategory


def test_updated_naming():
    """Test updated log naming convention."""
    print("=" * 80)
    print("Testing Updated Log Naming Convention")
    print("=" * 80)

    # Clean up old log files first
    log_dir = Path("./logs")
    old_patterns = ["live_risk_test_*.log"]
    for pattern in old_patterns:
        for file in log_dir.glob(pattern):
            file.unlink()

    # Test dev environment
    print("\n1. Testing 'dev' environment with sys/mkt naming...")
    dev_loggers = setup_category_logging(env="dev", log_dir="./logs", level="INFO")

    dev_system = StructuredLogger(dev_loggers["system"])
    dev_market = StructuredLogger(dev_loggers["market"])

    # Write test logs
    dev_system.info(LogCategory.SYSTEM, "Dev system log test")
    dev_market.info(LogCategory.DATA, "Dev market log test")

    # Check created files
    sys_log = log_dir / "live_risk_dev_sys.log"
    mkt_log = log_dir / "live_risk_dev_mkt.log"

    print("\n2. Verifying dev log files...")
    if sys_log.exists():
        print(f"   ✓ System log: {sys_log.name}")
    else:
        print(f"   ✗ System log not found: {sys_log.name}")

    if mkt_log.exists():
        print(f"   ✓ Market log: {mkt_log.name}")
    else:
        print(f"   ✗ Market log not found: {mkt_log.name}")

    # Test prod environment
    print("\n3. Testing 'prod' environment with sys/mkt naming...")
    prod_loggers = setup_category_logging(env="prod", log_dir="./logs", level="INFO")

    prod_system = StructuredLogger(prod_loggers["system"])
    prod_market = StructuredLogger(prod_loggers["market"])

    # Write test logs
    prod_system.info(LogCategory.SYSTEM, "Prod system log test")
    prod_market.info(LogCategory.DATA, "Prod market log test")

    # Check created files
    prod_sys_log = log_dir / "live_risk_prod_sys.log"
    prod_mkt_log = log_dir / "live_risk_prod_mkt.log"

    print("\n4. Verifying prod log files...")
    if prod_sys_log.exists():
        print(f"   ✓ System log: {prod_sys_log.name}")
    else:
        print(f"   ✗ System log not found: {prod_sys_log.name}")

    if prod_mkt_log.exists():
        print(f"   ✓ Market log: {prod_mkt_log.name}")
    else:
        print(f"   ✗ Market log not found: {prod_mkt_log.name}")

    # List all log files matching the pattern
    print("\n5. All log files with new naming convention:")
    log_files = sorted(log_dir.glob("live_risk_*_*.log"))

    correct_naming = 0
    total_files = 0

    for log_file in log_files:
        name = log_file.name
        # Check if it matches the pattern
        if "_sys" in name or "_mkt" in name:
            print(f"   ✓ {name}")
            correct_naming += 1
        elif "_system" in name or "_market" in name:
            print(f"   ✗ {name} (old naming)")
        else:
            print(f"   - {name} (other)")
        total_files += 1

    # Summary
    print("\n" + "=" * 80)
    print("✓ Updated naming convention test completed!")
    print("=" * 80)

    print("\nNaming Convention:")
    print("  - Current logs:")
    print("    • live_risk_dev_sys.log")
    print("    • live_risk_dev_mkt.log")
    print("    • live_risk_prod_sys.log")
    print("    • live_risk_prod_mkt.log")
    print("\n  - Rotated logs (after midnight):")
    print("    • live_risk_dev_sys_2025-11-24_1.log")
    print("    • live_risk_dev_mkt_2025-11-24_1.log")
    print("    • live_risk_prod_sys_2025-11-24_1.log")
    print("    • live_risk_prod_mkt_2025-11-24_1.log")

    print(f"\nVerification:")
    print(f"  - Files with correct naming: {correct_naming}/{total_files}")
    print(f"  - Pattern: live_risk_{{env}}_{{sys/mkt}}_{{date}}_{{number}}.log")
    print()

    return sys_log.exists() and mkt_log.exists() and prod_sys_log.exists() and prod_mkt_log.exists()


if __name__ == "__main__":
    success = test_updated_naming()
    sys.exit(0 if success else 1)
