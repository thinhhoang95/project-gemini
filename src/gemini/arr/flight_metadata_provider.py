"""Helper for retrieving flight-level metadata and jitter distributions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from gemini.arrivals.flight_list_gemini import FlightListGemini
from gemini.arrivals.ground_jitter_config import GroundJitterConfig, GroundJitterDistribution


@dataclass(frozen=True)
class FlightEntryContext:
    """Metadata required to evaluate FR entry probabilities for one flight."""

    flight_id: str
    takeoff_time: datetime
    minute_of_day: float
    origin: Optional[str]
    distribution: GroundJitterDistribution


class FlightMetadataProvider:
    """Caches per-flight metadata and the associated jitter distribution."""

    def __init__(self, flights: FlightListGemini, jitter_config: GroundJitterConfig):
        self._flights = flights
        self._jitter_config = jitter_config
        self._cache: Dict[str, FlightEntryContext] = {}

    def get_entry_context(self, flight_id: str) -> FlightEntryContext:
        """Return takeoff metadata + jitter distribution for ``flight_id``."""
        flight_key = str(flight_id)
        cached = self._cache.get(flight_key)
        if cached is not None:
            return cached
        metadata = self._flights.get_flight_metadata(flight_key)
        if not metadata:
            raise KeyError(f"Missing flight metadata for {flight_key}")
        takeoff_dt = metadata.get("takeoff_time")
        if not isinstance(takeoff_dt, datetime):
            raise ValueError(f"Flight {flight_key} missing takeoff_time")
        origin = metadata.get("origin")
        minute_of_day = _minutes_since_midnight(takeoff_dt)
        distribution = self._jitter_config.get_distribution(str(origin) if origin else None, int(minute_of_day))
        context = FlightEntryContext(
            flight_id=flight_key,
            takeoff_time=takeoff_dt,
            minute_of_day=minute_of_day,
            origin=str(origin) if origin else None,
            distribution=distribution,
        )
        self._cache[flight_key] = context
        return context


def _minutes_since_midnight(dt: datetime) -> float:
    """Return fractional minutes since local midnight for ``dt``."""
    return dt.hour * 60.0 + dt.minute + dt.second / 60.0 + dt.microsecond / 60_000_000.0
