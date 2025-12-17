"""
Latency models for order/fill delay simulation.

Simulates delays in:
- Order acknowledgment
- Fill notifications
- Data feed latency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
import random


@dataclass
class LatencyResult:
    """Result of latency calculation."""
    delay_ms: float  # Delay in milliseconds
    delayed_time: datetime  # Time after applying delay


class LatencyModel(ABC):
    """
    Abstract base class for latency models.

    Latency models simulate execution delays.
    """

    @abstractmethod
    def calculate_order_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        """
        Calculate order submission latency.

        Args:
            symbol: Symbol being traded.
            timestamp: Current timestamp.
            venue: Trading venue (optional).

        Returns:
            LatencyResult with delay.
        """
        ...

    @abstractmethod
    def calculate_fill_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        """
        Calculate fill notification latency.

        Args:
            symbol: Symbol traded.
            timestamp: Fill timestamp.
            venue: Trading venue (optional).

        Returns:
            LatencyResult with delay.
        """
        ...


class ZeroLatencyModel(LatencyModel):
    """Zero latency - instant execution."""

    def calculate_order_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        return LatencyResult(delay_ms=0.0, delayed_time=timestamp)

    def calculate_fill_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        return LatencyResult(delay_ms=0.0, delayed_time=timestamp)


class ConstantLatencyModel(LatencyModel):
    """
    Constant latency model.

    Fixed delay for orders and fills.
    """

    def __init__(
        self,
        order_latency_ms: float = 50.0,
        fill_latency_ms: float = 100.0,
    ):
        """
        Initialize constant latency model.

        Args:
            order_latency_ms: Order submission delay in milliseconds.
            fill_latency_ms: Fill notification delay in milliseconds.
        """
        self.order_latency_ms = order_latency_ms
        self.fill_latency_ms = fill_latency_ms

    def calculate_order_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        delay = timedelta(milliseconds=self.order_latency_ms)
        return LatencyResult(
            delay_ms=self.order_latency_ms,
            delayed_time=timestamp + delay,
        )

    def calculate_fill_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        delay = timedelta(milliseconds=self.fill_latency_ms)
        return LatencyResult(
            delay_ms=self.fill_latency_ms,
            delayed_time=timestamp + delay,
        )


class RandomLatencyModel(LatencyModel):
    """
    Random latency within bounds.

    Adds realistic variance to execution delays.
    """

    def __init__(
        self,
        order_latency_min_ms: float = 10.0,
        order_latency_max_ms: float = 100.0,
        fill_latency_min_ms: float = 50.0,
        fill_latency_max_ms: float = 200.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize random latency model.

        Args:
            order_latency_min_ms: Minimum order latency.
            order_latency_max_ms: Maximum order latency.
            fill_latency_min_ms: Minimum fill latency.
            fill_latency_max_ms: Maximum fill latency.
            seed: Random seed for reproducibility.
        """
        self.order_latency_min_ms = order_latency_min_ms
        self.order_latency_max_ms = order_latency_max_ms
        self.fill_latency_min_ms = fill_latency_min_ms
        self.fill_latency_max_ms = fill_latency_max_ms
        self._random = random.Random(seed)

    def calculate_order_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        delay_ms = self._random.uniform(
            self.order_latency_min_ms,
            self.order_latency_max_ms,
        )
        delay = timedelta(milliseconds=delay_ms)
        return LatencyResult(
            delay_ms=delay_ms,
            delayed_time=timestamp + delay,
        )

    def calculate_fill_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        delay_ms = self._random.uniform(
            self.fill_latency_min_ms,
            self.fill_latency_max_ms,
        )
        delay = timedelta(milliseconds=delay_ms)
        return LatencyResult(
            delay_ms=delay_ms,
            delayed_time=timestamp + delay,
        )


class VenueLatencyModel(LatencyModel):
    """
    Venue-based latency model.

    Different latencies for different trading venues/exchanges.
    """

    # Default latencies by venue (in milliseconds)
    DEFAULT_VENUE_LATENCIES: Dict[str, tuple] = {
        # US Exchanges
        "NYSE": (20, 80),  # (order_latency, fill_latency)
        "NASDAQ": (15, 60),
        "ARCA": (25, 100),
        "BATS": (10, 50),
        "IEX": (30, 150),  # IEX has intentional delays
        # HK Exchange
        "HKEX": (50, 200),
        "SEHK": (50, 200),
        # Other
        "DEFAULT": (30, 100),
    }

    def __init__(
        self,
        venue_latencies: Optional[Dict[str, tuple]] = None,
        add_variance: bool = True,
        variance_pct: float = 0.2,
        seed: Optional[int] = None,
    ):
        """
        Initialize venue-based latency model.

        Args:
            venue_latencies: Dict of venue -> (order_latency_ms, fill_latency_ms).
            add_variance: Whether to add random variance.
            variance_pct: Variance as percentage of base latency.
            seed: Random seed.
        """
        self.venue_latencies = venue_latencies or self.DEFAULT_VENUE_LATENCIES
        self.add_variance = add_variance
        self.variance_pct = variance_pct
        self._random = random.Random(seed)

    def calculate_order_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        base_latency = self._get_order_latency(venue)
        delay_ms = self._apply_variance(base_latency)
        delay = timedelta(milliseconds=delay_ms)
        return LatencyResult(
            delay_ms=delay_ms,
            delayed_time=timestamp + delay,
        )

    def calculate_fill_latency(
        self,
        symbol: str,
        timestamp: datetime,
        venue: Optional[str] = None,
    ) -> LatencyResult:
        base_latency = self._get_fill_latency(venue)
        delay_ms = self._apply_variance(base_latency)
        delay = timedelta(milliseconds=delay_ms)
        return LatencyResult(
            delay_ms=delay_ms,
            delayed_time=timestamp + delay,
        )

    def _get_order_latency(self, venue: Optional[str]) -> float:
        """Get base order latency for venue."""
        if venue and venue.upper() in self.venue_latencies:
            return self.venue_latencies[venue.upper()][0]
        return self.venue_latencies.get("DEFAULT", (30, 100))[0]

    def _get_fill_latency(self, venue: Optional[str]) -> float:
        """Get base fill latency for venue."""
        if venue and venue.upper() in self.venue_latencies:
            return self.venue_latencies[venue.upper()][1]
        return self.venue_latencies.get("DEFAULT", (30, 100))[1]

    def _apply_variance(self, base: float) -> float:
        """Apply random variance to base latency."""
        if self.add_variance:
            variance = base * self.variance_pct
            return base + self._random.uniform(-variance, variance)
        return base

    def detect_venue(self, symbol: str) -> str:
        """
        Detect venue from symbol.

        Simple heuristic - override for more sophisticated detection.
        """
        if symbol.endswith(".HK") or symbol.isdigit():
            return "HKEX"
        # Default to NASDAQ for US symbols
        return "NASDAQ"
