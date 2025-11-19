"""Helper for retrieving flight-level metadata and jitter distributions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Mapping, Optional

from gemini.arrivals.delay_assignment_gemini import DelayAssignmentGemini
from gemini.arrivals.flight_list_gemini import FlightListGemini
from gemini.arrivals.ground_jitter_config import GroundJitterConfig, GroundJitterDistribution


@dataclass(frozen=True)
class FlightEntryContext:
    """Metadata required to evaluate FR entry probabilities for one flight."""

    flight_id: str
    takeoff_time: datetime
    effective_takeoff_time: datetime
    ground_hold_delay_min: int
    minute_of_day: float
    origin: Optional[str]
    distribution: GroundJitterDistribution


class FlightMetadataProvider:
    """Caches per-flight metadata and the associated jitter distribution."""

    def __init__(
        self,
        flights: FlightListGemini,
        jitter_config: GroundJitterConfig,
        ground_hold_delays: Mapping[str, int] | None = None,
    ):
        self._flights = flights
        self._jitter_config = jitter_config
        if isinstance(ground_hold_delays, DelayAssignmentGemini):
            self._ground_hold_delays = ground_hold_delays.as_dict()
        elif ground_hold_delays:
            self._ground_hold_delays = DelayAssignmentGemini(ground_hold_delays).as_dict()
        else:
            self._ground_hold_delays = {}
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
        delay_min = self._ground_hold_delays.get(flight_key, 0)
        effective_takeoff = takeoff_dt + timedelta(minutes=delay_min)
        minute_of_day = _minutes_since_midnight(effective_takeoff)
        distribution = self._jitter_config.get_distribution(
            str(origin) if origin else None, int(minute_of_day)
        )
        context = FlightEntryContext(
            flight_id=flight_key,
            takeoff_time=takeoff_dt,
            effective_takeoff_time=effective_takeoff,
            ground_hold_delay_min=delay_min,
            minute_of_day=minute_of_day,
            origin=str(origin) if origin else None,
            distribution=distribution,
        )
        self._cache[flight_key] = context
        return context


def _minutes_since_midnight(dt: datetime) -> float:
    """Return fractional minutes since local midnight for ``dt``."""
    return dt.hour * 60.0 + dt.minute + dt.second / 60.0 + dt.microsecond / 60_000_000.0
