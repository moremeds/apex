#!/usr/bin/env python3
"""
APEX System State Verification Script.

Verify the health of historical data, DuckDB coverage, and PostgreSQL signal storage.
Prints a colored pass/fail status report.

Usage:
    python scripts/verify_system_state.py
    python scripts/verify_system_state.py --verbose
    python scripts/verify_system_state.py --json  # Output as JSON for CI/CD
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from src.infrastructure.stores.duckdb_coverage_store import DuckDBCoverageStore


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def colorize(text: str, color: str, use_color: bool = True) -> str:
    """Apply ANSI color if terminal supports it."""
    if not use_color:
        return text
    return f"{color}{text}{Colors.RESET}"


@dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Complete verification report."""

    timestamp: datetime
    historical_data: List[CheckResult]
    duckdb_coverage: List[CheckResult]
    postgresql: List[CheckResult]

    @property
    def all_passed(self) -> bool:
        all_checks = self.historical_data + self.duckdb_coverage + self.postgresql
        return all(c.passed for c in all_checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "all_passed": self.all_passed,
            "historical_data": [
                {"name": c.name, "passed": c.passed, "message": c.message, "details": c.details}
                for c in self.historical_data
            ],
            "duckdb_coverage": [
                {"name": c.name, "passed": c.passed, "message": c.message, "details": c.details}
                for c in self.duckdb_coverage
            ],
            "postgresql": [
                {"name": c.name, "passed": c.passed, "message": c.message, "details": c.details}
                for c in self.postgresql
            ],
        }


def format_bytes(size: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_number(n: int) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def verify_historical_data(base_dir: Path, verbose: bool = False) -> List[CheckResult]:
    """Verify parquet files exist and are readable."""
    results = []

    if not base_dir.exists():
        results.append(CheckResult(
            name="Historical Data Directory",
            passed=False,
            message=f"Directory not found: {base_dir}",
        ))
        return results

    # Scan for parquet files
    parquet_files = list(base_dir.glob("*/*.parquet"))

    if not parquet_files:
        results.append(CheckResult(
            name="Parquet Files",
            passed=False,
            message="No parquet files found",
        ))
        return results

    # Group by symbol
    symbols: Dict[str, List[Path]] = {}
    total_size = 0

    for pf in parquet_files:
        symbol = pf.parent.name
        if symbol not in symbols:
            symbols[symbol] = []
        symbols[symbol].append(pf)
        total_size += pf.stat().st_size

    details = []
    for symbol in sorted(symbols.keys()):
        files = symbols[symbol]
        symbol_size = sum(f.stat().st_size for f in files)
        timeframes = [f.stem for f in files]
        details.append(f"{symbol}: {', '.join(timeframes)} ({format_bytes(symbol_size)})")

    results.append(CheckResult(
        name="Parquet Files",
        passed=True,
        message=f"{len(symbols)} symbols, {len(parquet_files)} files, {format_bytes(total_size)} total",
        details=details if verbose else details[:5] + [f"... and {len(details) - 5} more"] if len(details) > 5 else details,
    ))

    return results


def verify_duckdb_coverage(base_dir: Path, verbose: bool = False) -> List[CheckResult]:
    """Verify DuckDB coverage database."""
    results = []

    db_path = base_dir / "_metadata.duckdb"
    if not db_path.exists():
        results.append(CheckResult(
            name="DuckDB Metadata",
            passed=False,
            message=f"Database not found: {db_path}",
        ))
        return results

    try:
        store = DuckDBCoverageStore(db_path=db_path)
        symbols = store.get_all_symbols()

        if not symbols:
            results.append(CheckResult(
                name="DuckDB Coverage",
                passed=False,
                message="No coverage records found",
            ))
            store.close()
            return results

        details = []
        total_bars = 0
        timeframe_set = set()

        for symbol in sorted(symbols):
            summaries = store.get_coverage_summary(symbol)
            for s in summaries:
                timeframe_set.add(s["timeframe"])
                bars = s["total_bars"] or 0
                total_bars += bars
                if verbose:
                    earliest = s["earliest"].strftime("%Y-%m-%d") if s["earliest"] else "?"
                    latest = s["latest"].strftime("%Y-%m-%d") if s["latest"] else "?"
                    details.append(f"{symbol}/{s['timeframe']}: {earliest} to {latest} ({format_number(bars)} bars)")

        store.close()

        results.append(CheckResult(
            name="DuckDB Coverage",
            passed=True,
            message=f"{len(symbols)} symbols, {len(timeframe_set)} timeframes, {format_number(total_bars)} total bars",
            details=details[:10] + [f"... and {len(details) - 10} more"] if len(details) > 10 else details,
        ))

    except Exception as e:
        results.append(CheckResult(
            name="DuckDB Coverage",
            passed=False,
            message=f"Error reading DuckDB: {e}",
        ))

    return results


async def verify_postgresql(config: Optional[Any], verbose: bool = False) -> List[CheckResult]:
    """Verify PostgreSQL signal storage."""
    results = []

    if config is None or config.database is None:
        results.append(CheckResult(
            name="PostgreSQL Connection",
            passed=False,
            message="Database not configured",
        ))
        return results

    try:
        from src.infrastructure.persistence.database import Database
        from src.infrastructure.persistence.repositories.ta_signal_repository import TASignalRepository

        db = Database(config.database)
        await db.connect()

        # Check connection
        results.append(CheckResult(
            name="PostgreSQL Connection",
            passed=True,
            message=f"Connected to {config.database.host}:{config.database.port}/{config.database.database}",
        ))

        # Check tables exist and get counts
        tables = [
            ("ta_signals", "SELECT COUNT(*), MAX(time) FROM ta_signals"),
            ("indicator_values", "SELECT COUNT(*), MAX(time) FROM indicator_values"),
            ("confluence_scores", "SELECT COUNT(*), MAX(time) FROM confluence_scores"),
        ]

        details = []
        all_tables_ok = True

        for table_name, query in tables:
            try:
                row = await db.fetchrow(query)
                count = row[0] if row else 0
                last_update = row[1] if row else None

                if count > 0:
                    last_str = last_update.strftime("%Y-%m-%d %H:%M:%S") if last_update else "N/A"
                    details.append(f"{table_name}: {format_number(count)} records (last: {last_str})")
                else:
                    details.append(f"{table_name}: empty")
            except Exception as e:
                details.append(f"{table_name}: ERROR - {e}")
                all_tables_ok = False

        results.append(CheckResult(
            name="Signal Tables",
            passed=all_tables_ok,
            message="All tables accessible" if all_tables_ok else "Some tables have errors",
            details=details,
        ))

        # Check indicator summary (for Tab 7 verification)
        try:
            indicator_query = """
                SELECT indicator, COUNT(DISTINCT symbol) as symbols, MAX(time) as last_update
                FROM indicator_values
                GROUP BY indicator
                ORDER BY last_update DESC
            """
            rows = await db.fetch(indicator_query)

            if rows:
                indicator_details = []
                for r in rows[:5]:
                    last_str = r["last_update"].strftime("%H:%M:%S") if r["last_update"] else "N/A"
                    indicator_details.append(f"{r['indicator']}: {r['symbols']} symbols (last: {last_str})")

                results.append(CheckResult(
                    name="Indicator Activity",
                    passed=True,
                    message=f"{len(rows)} active indicators",
                    details=indicator_details,
                ))
            else:
                results.append(CheckResult(
                    name="Indicator Activity",
                    passed=True,
                    message="No indicator data yet (expected if market closed)",
                ))
        except Exception as e:
            results.append(CheckResult(
                name="Indicator Activity",
                passed=False,
                message=f"Error querying indicators: {e}",
            ))

        await db.close()

    except Exception as e:
        results.append(CheckResult(
            name="PostgreSQL Connection",
            passed=False,
            message=f"Connection failed: {e}",
        ))

    return results


def print_report(report: VerificationReport, use_color: bool = True) -> None:
    """Print verification report to console."""
    print()
    print(colorize("=" * 50, Colors.BOLD, use_color))
    print(colorize(" APEX System State Verification", Colors.BOLD, use_color))
    print(colorize("=" * 50, Colors.BOLD, use_color))
    print(f" Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    def print_section(title: str, checks: List[CheckResult]) -> None:
        print(colorize(f"[{title}]", Colors.CYAN, use_color))
        for check in checks:
            status = colorize("[OK]", Colors.GREEN, use_color) if check.passed else colorize("[FAIL]", Colors.RED, use_color)
            print(f"  {status} {check.name}: {check.message}")
            for detail in check.details:
                print(f"       {detail}")
        print()

    print_section("HISTORICAL DATA", report.historical_data)
    print_section("DUCKDB COVERAGE", report.duckdb_coverage)
    print_section("POSTGRESQL SIGNALS", report.postgresql)

    print(colorize("=" * 50, Colors.BOLD, use_color))
    if report.all_passed:
        print(colorize(" Status: ALL CHECKS PASSED", Colors.GREEN, use_color))
    else:
        print(colorize(" Status: SOME CHECKS FAILED", Colors.RED, use_color))
    print(colorize("=" * 50, Colors.BOLD, use_color))
    print()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify APEX system state")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--data-dir", default="data/historical", help="Historical data directory")
    args = parser.parse_args()

    # Load config
    try:
        config_manager = ConfigManager()
        config = config_manager.load()
    except Exception:
        config = None

    base_dir = Path(args.data_dir)

    # Run all verifications
    historical_results = verify_historical_data(base_dir, args.verbose)
    duckdb_results = verify_duckdb_coverage(base_dir, args.verbose)
    postgresql_results = await verify_postgresql(config, args.verbose)

    report = VerificationReport(
        timestamp=datetime.now(),
        historical_data=historical_results,
        duckdb_coverage=duckdb_results,
        postgresql=postgresql_results,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        use_color = not args.no_color and sys.stdout.isatty()
        print_report(report, use_color)

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
