"""
File loader for manual YAML position loading.

Implements PositionProvider interface for loading positions from YAML files.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import List
from datetime import datetime, date
import logging
import yaml

from ...domain.interfaces.position_provider import PositionProvider
from ...models.position import Position, AssetType, PositionSource


logger = logging.getLogger(__name__)


class FileLoader(PositionProvider):
    """
    Manual position file loader (YAML format).

    Implements PositionProvider for loading positions from YAML files.
    Supports hot-reload with configurable interval.
    """

    def __init__(
        self,
        file_path: str | Path,
        reload_interval_sec: int = 60,
    ):
        """
        Initialize file loader.

        Args:
            file_path: Path to YAML position file.
            reload_interval_sec: Auto-reload interval in seconds.
        """
        self.file_path = Path(file_path)
        self.reload_interval_sec = reload_interval_sec
        self._positions: List[Position] = []
        self._last_loaded: datetime | None = None
        self._connected = False

    async def connect(self) -> None:
        """Initialize file loader (load positions)."""
        await self._load_positions()
        self._connected = True
        logger.info(f"FileLoader initialized from {self.file_path}")

    async def disconnect(self) -> None:
        """Disconnect file loader (no-op)."""
        self._connected = False
        logger.info("FileLoader disconnected")

    def is_connected(self) -> bool:
        """Check if file loader is ready."""
        return self._connected

    async def fetch_positions(self) -> List[Position]:
        """
        Fetch positions from YAML file.

        Returns:
            List of Position objects with source=MANUAL.

        Raises:
            ConnectionError: If file cannot be read.
            DataError: If YAML is malformed.
        """
        if not self.is_connected():
            raise ConnectionError("FileLoader not initialized")

        # Check if reload needed
        if self._should_reload():
            await self._load_positions()

        return list(self._positions)  # Return copy

    def _should_reload(self) -> bool:
        """Check if file should be reloaded based on interval."""
        if self._last_loaded is None:
            return True
        elapsed = (datetime.now() - self._last_loaded).total_seconds()
        return elapsed >= self.reload_interval_sec

    async def _load_positions(self) -> None:
        """
        Load positions from YAML file.

        Expected YAML format:
        ```yaml
        positions:
          - symbol: AAPL
            underlying: AAPL
            asset_type: STOCK
            quantity: 100
            avg_price: 150.0
            multiplier: 1
          - symbol: AAPL 20240315C160
            underlying: AAPL
            asset_type: OPTION
            quantity: 10
            avg_price: 5.5
            multiplier: 100
            expiry: 2024-03-15
            strike: 160.0
            right: C
            strategy_tag: bull_call_spread
        ```

        Raises:
            ConnectionError: If file cannot be read.
            DataError: If YAML is malformed.
        """
        try:
            with open(self.file_path, "r") as f:
                data = yaml.safe_load(f)

            if not data or "positions" not in data:
                logger.warning(f"No positions found in {self.file_path}")
                self._positions = []
                self._last_loaded = datetime.now()
                return

            positions = []
            for pos_dict in data["positions"]:
                position = self._parse_position(pos_dict)
                positions.append(position)

            self._positions = positions
            self._last_loaded = datetime.now()
            logger.info(f"Loaded {len(positions)} positions from {self.file_path}")

        except FileNotFoundError:
            raise ConnectionError(f"Position file not found: {self.file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading positions: {e}")

    def _parse_position(self, pos_dict: dict) -> Position:
        """
        Parse position from dict.

        Args:
            pos_dict: Position dictionary from YAML.

        Returns:
            Position object.

        Raises:
            ValueError: If required fields are missing.
        """
        # Required fields
        try:
            symbol = pos_dict["symbol"]
            underlying = pos_dict["underlying"]
            asset_type = AssetType[pos_dict["asset_type"].upper()]
            quantity = float(pos_dict["quantity"])  # Support fractional shares
            avg_price = float(pos_dict["avg_price"])
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")

        # Optional fields
        multiplier = int(pos_dict.get("multiplier", 1))
        strategy_tag = pos_dict.get("strategy_tag")

        # Option/Future fields
        # Convert all expiry formats to YYYYMMDD string format
        expiry = None
        if "expiry" in pos_dict:
            expiry_input = pos_dict["expiry"]
            if isinstance(expiry_input, str):
                # Handle both YYYY-MM-DD and YYYYMMDD string formats
                if "-" in expiry_input:
                    # Convert YYYY-MM-DD to YYYYMMDD
                    expiry_date = datetime.strptime(expiry_input, "%Y-%m-%d").date()
                    expiry = expiry_date.strftime("%Y%m%d")
                else:
                    # Already in YYYYMMDD format
                    expiry = expiry_input
            elif isinstance(expiry_input, date):
                # Convert date object to YYYYMMDD string
                expiry = expiry_input.strftime("%Y%m%d")

        strike = pos_dict.get("strike")
        if strike is not None:
            strike = float(strike)

        right = pos_dict.get("right")  # "C" or "P"

        return Position(
            symbol=symbol,
            underlying=underlying,
            asset_type=asset_type,
            quantity=quantity,
            avg_price=avg_price,
            multiplier=multiplier,
            expiry=expiry,
            strike=strike,
            right=right,
            source=PositionSource.MANUAL,
            strategy_tag=strategy_tag,
            last_updated=datetime.now(),
        )
