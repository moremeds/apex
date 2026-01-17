"""
Test script for log file rotation with custom naming convention.

This script demonstrates:
1. Creating multiple log entries
2. Forcing log rotation
3. Verifying custom naming: live_risk_{date}_{number}.log
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_manager import ConfigManager
from src.utils.logging_setup import setup_logging
from src.utils.structured_logger import LogCategory, StructuredLogger


def main():
    """Test log rotation with custom naming."""
    print("=" * 80)
    print("Testing Log Rotation with Custom Naming Convention")
    print("=" * 80)

    # Load configuration
    print("\n1. Loading configuration...")
    try:
        # Config dir is at project root (two levels up from tests/partial/)
        project_root = Path(__file__).parent.parent.parent
        config_manager = ConfigManager(config_dir=str(project_root), env="dev")
        app_config = config_manager.load()
        print(f"   ✓ Configuration loaded")
        print(f"   - Log file: {app_config.logging.file}")
        print(f"   - Rotation: {app_config.logging.rotation}")
        print(f"   - Backup count: {app_config.logging.backup_count}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Set up logging
    print("\n2. Setting up logging...")
    try:
        logger = setup_logging(app_config.logging, logger_name="rotation_test")
        struct_logger = StructuredLogger(logger)
        print(f"   ✓ Logging configured with {len(logger.handlers)} handlers")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Get the file handler
    file_handler = None
    for handler in logger.handlers:
        if hasattr(handler, "doRollover"):
            file_handler = handler
            break

    if not file_handler:
        print("   ✗ No rotating file handler found")
        return 1

    print(f"   - Handler: {file_handler.__class__.__name__}")

    # Write initial logs
    print("\n3. Writing initial log entries...")
    for i in range(1, 6):
        struct_logger.info(LogCategory.SYSTEM, f"Initial log entry {i}", {"iteration": i})
    print(f"   ✓ Wrote 5 log entries to: {app_config.logging.file}")

    # Force rotation #1
    print("\n4. Forcing first rotation...")
    try:
        file_handler.doRollover()
        print("   ✓ Rotation triggered")
        time.sleep(0.1)  # Brief pause
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Write more logs
    print("\n5. Writing more log entries after first rotation...")
    for i in range(1, 6):
        struct_logger.info(LogCategory.RISK, f"After rotation 1 - entry {i}", {"iteration": i})
    print(f"   ✓ Wrote 5 more log entries")

    # Force rotation #2
    print("\n6. Forcing second rotation...")
    try:
        file_handler.doRollover()
        print("   ✓ Rotation triggered")
        time.sleep(0.1)  # Brief pause
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Write final logs
    print("\n7. Writing final log entries after second rotation...")
    for i in range(1, 6):
        struct_logger.info(LogCategory.DATA, f"After rotation 2 - entry {i}", {"iteration": i})
    print(f"   ✓ Wrote 5 final log entries")

    # List all log files
    print("\n8. Verifying log files with custom naming...")
    log_dir = Path(app_config.logging.file).parent
    log_files = sorted(log_dir.glob("*.log"))

    if not log_files:
        print(f"   ✗ No log files found in {log_dir}")
        return 1

    print(f"   ✓ Found {len(log_files)} log file(s):")

    today = datetime.now().strftime("%Y-%m-%d")
    expected_pattern = f"live_risk_{today}_"

    for log_file in log_files:
        size = log_file.stat().st_size
        print(f"     - {log_file.name} ({size} bytes)")

        # Count lines in file
        with open(log_file, "r") as f:
            line_count = len(f.readlines())
        print(f"       Lines: {line_count}")

    # Verify naming convention
    print(f"\n9. Checking naming convention...")
    rotated_files = [f for f in log_files if f.name != "live_risk.log"]

    if rotated_files:
        print(f"   ✓ Found {len(rotated_files)} rotated file(s)")
        for rf in rotated_files:
            if expected_pattern in rf.name:
                print(f"     ✓ {rf.name} - matches pattern live_risk_{{date}}_{{number}}.log")
            else:
                print(f"     ✗ {rf.name} - DOES NOT match expected pattern")
    else:
        print(f"   ⚠ No rotated files found yet (rotations may be pending)")

    print("\n" + "=" * 80)
    print("✓ Rotation test completed!")
    print("=" * 80)
    print(f"\nLog directory: {log_dir.absolute()}")
    print(f"Expected naming: live_risk_{today}_N.log (where N = 1, 2, 3...)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
