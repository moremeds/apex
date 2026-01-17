"""
Test script for logging setup with file rotation and retention.

This script demonstrates:
1. Loading logging configuration
2. Setting up file logging with rotation
3. Writing test log messages
4. Verifying log file creation
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_manager import ConfigManager
from src.utils.logging_setup import setup_logging
from src.utils.structured_logger import LogCategory, StructuredLogger


def main():
    """Test logging setup."""
    print("=" * 80)
    print("Testing Logging Setup with File Rotation and Retention")
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
        print(f"   - Log level: {app_config.logging.level}")
        print(f"   - JSON format: {app_config.logging.json}")
        print(f"   - Rotation: {app_config.logging.rotation}")
        print(f"   - Backup count: {app_config.logging.backup_count}")

        if app_config.logging.rotation == "size":
            print(
                f"   - Max bytes: {app_config.logging.max_bytes:,} bytes ({app_config.logging.max_bytes / 1024 / 1024:.1f} MB)"
            )
        elif app_config.logging.rotation == "time":
            print(f"   - When: {app_config.logging.when}")
            print(f"   - Interval: {app_config.logging.interval}")

    except Exception as e:
        print(f"   ✗ Failed to load configuration: {e}")
        return 1

    # Set up logging
    print("\n2. Setting up logging...")
    try:
        logger = setup_logging(app_config.logging, logger_name="test_logger")
        print(f"   ✓ Logging configured")
        print(f"   - Handlers: {len(logger.handlers)}")
        for i, handler in enumerate(logger.handlers):
            print(f"     [{i}] {handler.__class__.__name__}")

    except Exception as e:
        print(f"   ✗ Failed to set up logging: {e}")
        return 1

    # Create structured logger
    print("\n3. Creating structured logger...")
    try:
        struct_logger = StructuredLogger(logger)
        print(f"   ✓ Structured logger created")
    except Exception as e:
        print(f"   ✗ Failed to create structured logger: {e}")
        return 1

    # Write test messages
    print("\n4. Writing test log messages...")
    try:
        # Test different log levels
        struct_logger.debug(LogCategory.SYSTEM, "Debug message - system check", {"test": True})
        struct_logger.info(LogCategory.SYSTEM, "Application started", {"version": "1.0.0"})
        struct_logger.info(
            LogCategory.RISK,
            "Portfolio snapshot calculated",
            {"total_pnl": 12345.67, "delta": -1234.56, "positions": 42},
        )
        struct_logger.warning(
            LogCategory.RISK,
            "Soft breach detected",
            {"limit_type": "delta", "current": 45000, "limit": 50000, "utilization": 0.90},
        )
        struct_logger.error(
            LogCategory.DATA, "Market data stale", {"symbol": "AAPL", "age_seconds": 15}
        )
        struct_logger.critical(
            LogCategory.ALERT,
            "Hard limit breach",
            {"limit_type": "margin_utilization", "current": 0.75, "limit": 0.60},
        )

        # Test regular logger messages
        logger.info("Regular log message (non-JSON)")
        logger.warning("Warning: This is a test")

        print(f"   ✓ {8} log messages written")

    except Exception as e:
        print(f"   ✗ Failed to write log messages: {e}")
        return 1

    # Verify log file
    print("\n5. Verifying log file...")
    try:
        log_file = Path(app_config.logging.file)
        if log_file.exists():
            size = log_file.stat().st_size
            print(f"   ✓ Log file created: {log_file}")
            print(f"   - Size: {size} bytes")

            # Read and display last few lines
            with open(log_file, "r") as f:
                lines = f.readlines()
                print(f"   - Total lines: {len(lines)}")
                print(f"\n   Last 3 log entries:")
                for line in lines[-3:]:
                    print(f"     {line.rstrip()}")
        else:
            print(f"   ✗ Log file not found: {log_file}")
            return 1

    except Exception as e:
        print(f"   ✗ Failed to verify log file: {e}")
        return 1

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
    print(f"\nLog file location: {Path(app_config.logging.file).absolute()}")
    print(f"\nRetention policy:")
    print(f"  - Rotation method: {app_config.logging.rotation}")
    print(f"  - Backup count: {app_config.logging.backup_count}")
    print(
        f"  - Old logs will be automatically deleted after {app_config.logging.backup_count} rotations"
    )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
