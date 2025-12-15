"""
Database migration runner for schema versioning.

Tracks applied migrations in a schema_migrations table and applies
pending migrations in order. Supports both SQL files and Python migrations.

Usage:
    from migrations.runner import MigrationRunner
    from src.infrastructure.persistence.database import Database

    db = Database(config)
    await db.connect()

    runner = MigrationRunner(db, migrations_dir="migrations")
    await runner.run()  # Apply all pending migrations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.infrastructure.persistence.database import Database

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a single migration file."""

    version: str
    name: str
    filename: str
    path: Path
    applied_at: Optional[datetime] = None

    @property
    def is_applied(self) -> bool:
        return self.applied_at is not None


class MigrationError(Exception):
    """Migration execution failed."""

    pass


class MigrationRunner:
    """
    Runs database migrations with version tracking.

    Migrations are SQL files named with pattern: NNN_description.sql
    where NNN is a zero-padded version number (e.g., 001, 002, 010).

    The runner:
    1. Creates schema_migrations table if it doesn't exist
    2. Scans migrations directory for .sql files
    3. Applies pending migrations in version order
    4. Records applied migrations in schema_migrations table
    """

    # Pattern to match migration files: NNN_description.sql
    MIGRATION_PATTERN = re.compile(r"^(\d{3})_(.+)\.sql$")

    def __init__(self, db: Database, migrations_dir: str = "migrations"):
        """
        Initialize migration runner.

        Args:
            db: Database connection manager.
            migrations_dir: Directory containing migration files.
        """
        self._db = db
        self._migrations_dir = Path(migrations_dir)

    async def run(self, target_version: Optional[str] = None) -> List[Migration]:
        """
        Run all pending migrations up to target version.

        Args:
            target_version: Optional version to migrate to. If None, applies all.

        Returns:
            List of migrations that were applied.

        Raises:
            MigrationError: If migration fails.
        """
        # Ensure migrations table exists
        await self._ensure_migrations_table()

        # Get list of pending migrations
        pending = await self.get_pending_migrations()

        if not pending:
            logger.info("No pending migrations")
            return []

        # Filter to target version if specified
        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        applied = []
        for migration in pending:
            logger.info(f"Applying migration {migration.version}: {migration.name}")
            try:
                await self._apply_migration(migration)
                applied.append(migration)
                logger.info(f"Migration {migration.version} applied successfully")
            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                raise MigrationError(
                    f"Migration {migration.version} ({migration.name}) failed: {e}"
                ) from e

        logger.info(f"Applied {len(applied)} migration(s)")
        return applied

    async def get_pending_migrations(self) -> List[Migration]:
        """
        Get list of migrations that haven't been applied yet.

        Returns:
            List of pending Migration objects, sorted by version.
        """
        all_migrations = self._discover_migrations()
        applied_versions = await self._get_applied_versions()

        pending = [m for m in all_migrations if m.version not in applied_versions]
        return sorted(pending, key=lambda m: m.version)

    async def get_applied_migrations(self) -> List[Migration]:
        """
        Get list of migrations that have been applied.

        Returns:
            List of applied Migration objects with applied_at timestamps.
        """
        await self._ensure_migrations_table()

        rows = await self._db.fetch(
            "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
        )

        return [
            Migration(
                version=row["version"],
                name=row["name"],
                filename=f"{row['version']}_{row['name']}.sql",
                path=self._migrations_dir / f"{row['version']}_{row['name']}.sql",
                applied_at=row["applied_at"],
            )
            for row in rows
        ]

    async def get_current_version(self) -> Optional[str]:
        """
        Get the current schema version.

        Returns:
            Version string of the latest applied migration, or None if no migrations.
        """
        await self._ensure_migrations_table()

        version = await self._db.fetchval(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        )
        return version

    async def reset(self) -> None:
        """
        Reset the migrations table (dangerous - for testing only).

        This does NOT roll back migrations, just clears the tracking table.
        """
        logger.warning("Resetting schema_migrations table")
        await self._db.execute("TRUNCATE TABLE schema_migrations")

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    async def _ensure_migrations_table(self) -> None:
        """Create the schema_migrations table if it doesn't exist."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                applied_at  TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

    def _discover_migrations(self) -> List[Migration]:
        """
        Discover all migration files in the migrations directory.

        Returns:
            List of Migration objects for all discovered files.
        """
        if not self._migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self._migrations_dir}")
            return []

        migrations = []
        for path in self._migrations_dir.glob("*.sql"):
            match = self.MIGRATION_PATTERN.match(path.name)
            if match:
                version, name = match.groups()
                migrations.append(
                    Migration(
                        version=version,
                        name=name,
                        filename=path.name,
                        path=path,
                    )
                )

        return sorted(migrations, key=lambda m: m.version)

    async def _get_applied_versions(self) -> set:
        """
        Get set of version numbers that have been applied.

        Returns:
            Set of version strings.
        """
        await self._ensure_migrations_table()

        rows = await self._db.fetch("SELECT version FROM schema_migrations")
        return {row["version"] for row in rows}

    async def _apply_migration(self, migration: Migration) -> None:
        """
        Apply a single migration.

        Args:
            migration: Migration to apply.

        Raises:
            MigrationError: If migration file cannot be read or executed.
        """
        # Read migration SQL
        try:
            sql = migration.path.read_text()
        except Exception as e:
            raise MigrationError(f"Cannot read migration file {migration.path}: {e}") from e

        # Execute migration in a transaction
        async with self._db.transaction() as conn:
            # Execute the migration SQL
            await conn.execute(sql)

            # Record the migration
            await conn.execute(
                """
                INSERT INTO schema_migrations (version, name)
                VALUES ($1, $2)
                ON CONFLICT (version) DO NOTHING
                """,
                migration.version,
                migration.name,
            )


async def run_migrations(db: Database, migrations_dir: str = "migrations") -> List[Migration]:
    """
    Convenience function to run all pending migrations.

    Args:
        db: Connected database instance.
        migrations_dir: Directory containing migration files.

    Returns:
        List of applied migrations.
    """
    runner = MigrationRunner(db, migrations_dir)
    return await runner.run()
