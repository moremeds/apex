"""
Test script for size-based log rotation with custom naming.

This script demonstrates size-based rotation with naming: live_risk_{date}_{number}.log
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.models import LoggingConfig
from src.utils.logging_setup import setup_logging
from src.utils.structured_logger import StructuredLogger, LogCategory


def main():
    """Test size-based rotation."""
    print("=" * 80)
    print("Testing Size-Based Rotation with Custom Naming")
    print("=" * 80)

    # Create custom config with size-based rotation
    print("\n1. Creating size-based logging config...")
    log_config = LoggingConfig(
        level="INFO",
        json=True,
        file="./logs/size_test.log",
        rotation="size",
        max_bytes=500,  # Small size to force rotation
        backup_count=5,
        when="midnight",
        interval=1,
    )
    print(f"   ✓ Config created")
    print(f"   - Max bytes: {log_config.max_bytes}")
    print(f"   - Backup count: {log_config.backup_count}")

    # Set up logging
    print("\n2. Setting up logging...")
    logger = setup_logging(log_config, logger_name="size_test")
    struct_logger = StructuredLogger(logger)
    print(f"   ✓ Logging configured")

    # Write logs to trigger rotation
    print("\n3. Writing logs to trigger size-based rotation...")
    for i in range(1, 21):
        struct_logger.info(
            LogCategory.SYSTEM,
            f"Size test log entry {i} with some extra text to increase size",
            {"iteration": i, "data": "x" * 50}
        )
    print(f"   ✓ Wrote 20 log entries")

    # List all log files
    print("\n4. Verifying rotated log files...")
    log_dir = Path(log_config.file).parent
    log_files = sorted(log_dir.glob("size_test*.log"))

    print(f"   ✓ Found {len(log_files)} log file(s):")
    for log_file in log_files:
        size = log_file.stat().st_size
        with open(log_file, 'r') as f:
            line_count = len(f.readlines())
        print(f"     - {log_file.name} ({size} bytes, {line_count} lines)")

    # Verify naming
    print("\n5. Checking naming convention...")
    rotated_files = [f for f in log_files if f.name != "size_test.log"]

    if rotated_files:
        print(f"   ✓ Found {len(rotated_files)} rotated file(s)")
        for rf in rotated_files:
            if "_" in rf.stem and rf.stem.count("_") >= 2:
                print(f"     ✓ {rf.name} - matches pattern size_test_{{date}}_{{number}}.log")
            else:
                print(f"     ✗ {rf.name} - DOES NOT match expected pattern")
    else:
        print("   ⚠ No rotation occurred (file size too small)")

    print("\n" + "=" * 80)
    print("✓ Size-based rotation test completed!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
