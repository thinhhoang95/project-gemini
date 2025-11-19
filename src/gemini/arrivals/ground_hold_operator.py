"""
Ground Hold Operator: Deterministic FCFS Ground Hold Algorithm

This module implements a deterministic ground hold algorithm that applies delays to flights
based on rate-limited time windows (ground hold windows) using a First Come First Served
(FCFS) queue discipline.

Algorithm Overview:
-------------------
The algorithm processes flights grouped by origin airport. For each airport, it applies
ground hold windows that specify:
- A time window (start, end)
- A maximum rate (flights per hour, fph)
- An optional regulation identifier

Flights scheduled within a window are delayed to ensure they don't exceed the rate limit.
The algorithm uses FCFS ordering: flights are processed in order of their scheduled
takeoff time, and each flight is released at the earliest time that maintains the rate limit.

Delay Calculation:
------------------
For a window with rate R flights/hour:
1. Compute the minimum time interval between releases: tau = 60 / R minutes
2. For the i-th flight (1-indexed) in the window:
   - Calculate candidate release time: window_end + (i-1) * tau
   - Actual release time = max(scheduled_time, candidate_release_time)
   - Delay = release_time - scheduled_time

Example:
--------
Consider a ground hold window:
- Airport: KJFK
- Start: 2023-07-17 10:00:00
- End: 2023-07-17 12:00:00
- Rate: 30 flights/hour (tau = 2 minutes)

Flights scheduled:
- Flight A: 10:15:00
- Flight B: 10:30:00
- Flight C: 10:45:00

Processing:
1. Flight A (index 1):
   - Candidate release: 12:00 + (1-1)*2min = 12:00
   - Release time: max(10:15, 12:00) = 12:00
   - Delay: 105 minutes

2. Flight B (index 2):
   - Candidate release: 12:00 + (2-1)*2min = 12:02
   - Release time: max(10:30, 12:02) = 12:02
   - Delay: 92 minutes

3. Flight C (index 3):
   - Candidate release: 12:00 + (3-1)*2min = 12:04
   - Release time: max(10:45, 12:04) = 12:04
   - Delay: 79 minutes

Input/Output Examples:
----------------------
Input - FlightListGemini:
    flight_list = FlightListGemini("path/to/flights.csv")
    # Contains flights with metadata:
    # - flight_id: "ABC123"
    # - takeoff_time: datetime(2023, 7, 17, 10, 15, 0)
    # - origin: "KJFK"

Input - GroundHoldConfig:
    config = GroundHoldConfig.from_yaml("path/to/config.yaml")
    # Contains windows_by_airport:
    # {
    #   "KJFK": [
    #     GroundHoldWindow(
    #       start=datetime(2023, 7, 17, 10, 0, 0),
    #       end=datetime(2023, 7, 17, 12, 0, 0),
    #       rate_fph=30.0,
    #       airport="KJFK",
    #       regulation_id="GH001"
    #     )
    #   ]
    # }

Output - DelayAssignmentGemini:
    operator = GroundHoldOperator(flight_list, config)
    assignment = operator.compute_flight_delays()
    # Returns a mapping-like object:
    # assignment["ABC123"] = 105  # delay in minutes
    # assignment["DEF456"] = 92
    # assignment.get("GHI789", 0)  # returns 0 if no delay

Notes:
------
- Flights are sorted by scheduled takeoff time, then by flight ID for deterministic ordering
- Only flights with valid takeoff_time and origin metadata are processed
- Flights scheduled before a window start are skipped
- Flights scheduled at or after a window end are not included in that window
- Multiple windows for the same airport are processed independently
- Delays from multiple windows for the same flight are accumulated
- Timezone awareness must be consistent between flight times and window times
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

from .delay_assignment_gemini import DelayAssignmentGemini
from .flight_list_gemini import FlightListGemini
from .ground_hold_config import GroundHoldConfig, GroundHoldWindow

logger = logging.getLogger(__name__)

# Type alias for a flight entry: (flight_id, scheduled_takeoff_datetime)
FlightEntry = Tuple[str, datetime]


class GroundHoldOperator:
    """
    Apply deterministic ground holds to a flight list using FCFS queues.
    
    This operator processes flights grouped by origin airport and applies delays
    based on rate-limited ground hold windows. Flights are processed in FCFS order
    (by scheduled takeoff time) within each window.
    """

    def __init__(self, flight_list: FlightListGemini, config: GroundHoldConfig):
        """
        Initialize the ground hold operator.
        
        Args:
            flight_list: Container with flight metadata (takeoff_time, origin)
            config: Ground hold configuration with windows by airport
        
        The constructor immediately groups flights by airport and sorts them by
        scheduled takeoff time for efficient processing.
        """
        self.flight_list = flight_list
        self.config = config
        # Pre-compute flight groupings by airport for efficient lookup
        self._flights_by_airport = self._group_flights_by_airport()

    def compute_flight_delays(self, regulation_id: str = "ground_hold") -> DelayAssignmentGemini:
        """
        Compute ground hold delays for all flights based on configured windows.
        
        This is the main entry point that processes all airports and windows,
        accumulating delays for flights that fall within rate-limited windows.
        
        Args:
            regulation_id: Identifier for the regulation type (default: "ground_hold")
        
        Returns:
            DelayAssignmentGemini mapping flight_id -> delay_minutes
        
        Processing flow:
        1. Iterate over each airport with configured windows
        2. For each window, select flights that fall within the window time range
        3. Compute delays using FCFS algorithm
        4. Accumulate delays (flights may be affected by multiple windows)
        """
        # Initialize empty delay assignment
        assignment = DelayAssignmentGemini({}, regulation_id=regulation_id)
        
        # Process each airport that has ground hold windows configured
        for airport, windows in self.config.windows_by_airport.items():
            # Get all flights for this airport (already sorted by takeoff time)
            airport_flights = self._flights_by_airport.get(airport, [])
            if not airport_flights:
                continue  # Skip airports with no flights
            
            # Process each window for this airport
            for window in windows:
                # Select flights that fall within this window's time range
                flights = self._select_flights_in_window(airport_flights, window)
                if not flights:
                    continue  # Skip empty windows
                
                # Compute delays for flights in this window using FCFS
                delays = self._compute_window_delays(flights, window)
                
                # Accumulate delays (flights may be in multiple windows)
                for flight_id, delay in delays:
                    if delay <= 0:
                        continue  # Skip zero or negative delays
                    # Add delay to existing assignment (may already have delay from another window)
                    assignment[flight_id] = assignment.get(flight_id, 0) + delay
        
        return assignment

    def _compute_window_delays(
        self, flights: Sequence[FlightEntry], window: GroundHoldWindow
    ) -> Iterable[Tuple[str, int]]:
        """
        Compute delays for flights in a window using FCFS algorithm.
        
        The algorithm ensures flights are released at intervals that maintain
        the rate limit. Flights are processed in order (already sorted by
        scheduled takeoff time), and each flight is released at the earliest
        time that maintains the rate constraint.
        
        Args:
            flights: Sequence of (flight_id, scheduled_takeoff) tuples, sorted by takeoff time
            window: Ground hold window with rate limit and time bounds
        
        Yields:
            (flight_id, delay_minutes) tuples for each flight
        
        Algorithm:
        - Compute minimum interval between releases: tau = 60 / rate_fph minutes
        - For flight at position i (1-indexed):
          - Candidate release = window_end + (i-1) * tau
          - Actual release = max(scheduled_time, candidate_release)
          - Delay = release - scheduled_time (rounded up to nearest minute)
        """
        # Calculate minimum time interval between flight releases to maintain rate
        # Example: 30 flights/hour = 2 minutes between releases
        tau_minutes = 60.0 / window.rate_fph
        
        # Process flights in order (FCFS: First Come First Served)
        # enumerate with start=1 gives 1-indexed position for delay calculation
        for index, (flight_id, scheduled_dt) in enumerate(flights, start=1):
            # Calculate when this flight could be released based on its position in queue
            # First flight (index=1) releases at window.end, subsequent flights are spaced
            # by tau_minutes intervals
            release_candidate = window.end + timedelta(minutes=(index - 1) * tau_minutes)
            
            # Flight cannot be released before its scheduled time
            # (ensures we don't "advance" flights, only delay them)
            release_dt = max(scheduled_dt, release_candidate)
            
            # Calculate delay in minutes (ensure non-negative)
            delay_minutes = max((release_dt - scheduled_dt).total_seconds() / 60.0, 0.0)
            
            # Round up to nearest integer minute (conservative approach)
            delay_int = int(math.ceil(delay_minutes))
            
            # Log the delay assignment for debugging
            logger.debug(
                "Ground hold %s: flight %s scheduled %s delayed to %s (%d min)",
                window.regulation_id or window.airport,
                flight_id,
                scheduled_dt,
                release_dt,
                delay_int,
            )
            yield flight_id, delay_int

    def _select_flights_in_window(
        self, flights: Sequence[FlightEntry], window: GroundHoldWindow
    ) -> List[FlightEntry]:
        """
        Select flights that fall within a ground hold window's time range.
        
        Flights are selected if their scheduled takeoff time is:
        - >= window.start (inclusive)
        - < window.end (exclusive)
        
        Since flights are pre-sorted by takeoff time, we can break early
        when we encounter a flight scheduled at or after the window end.
        
        Args:
            flights: Sequence of (flight_id, takeoff_datetime) tuples, sorted by takeoff time
            window: Ground hold window with start and end times
        
        Returns:
            List of flight entries within the window, sorted by takeoff time
        
        Note:
            Validates timezone consistency between flight times and window times.
        """
        selected: List[FlightEntry] = []
        
        for flight_id, takeoff_dt in flights:
            # Validate timezone consistency (raises ValueError if mismatched)
            takeoff_dt = _ensure_comparable_datetimes(
                takeoff_dt, window.start, window.airport, flight_id
            )
            takeoff_dt = _ensure_comparable_datetimes(
                takeoff_dt, window.end, window.airport, flight_id
            )
            
            # Skip flights scheduled before the window starts
            if takeoff_dt < window.start:
                continue
            
            # Since flights are sorted, once we hit a flight at/after window end,
            # all subsequent flights are also outside the window
            if takeoff_dt >= window.end:
                break
            
            # Flight is within window: [start, end)
            selected.append((flight_id, takeoff_dt))
        
        return selected

    def _group_flights_by_airport(self) -> Dict[str, List[FlightEntry]]:
        """
        Group flights by origin airport and sort by scheduled takeoff time.
        
        This method processes all flights in the flight list and:
        1. Extracts takeoff_time and origin from flight metadata
        2. Groups flights by airport code (normalized to uppercase)
        3. Sorts flights within each airport by takeoff time, then by flight ID
           (for deterministic ordering when takeoff times are equal)
        
        Returns:
            Dictionary mapping airport code -> list of (flight_id, takeoff_datetime) tuples
        
        Note:
            Only flights with valid takeoff_time (datetime) and origin (non-empty)
            are included. Flights missing these fields are silently skipped.
        """
        grouped: Dict[str, List[FlightEntry]] = {}
        
        # Process each flight in the flight list
        for flight_id in self.flight_list.flight_ids:
            # Retrieve flight metadata
            metadata = self.flight_list.get_flight_metadata(flight_id)
            takeoff = metadata.get("takeoff_time")
            origin = metadata.get("origin")
            
            # Skip flights without valid takeoff time or origin
            if not isinstance(takeoff, datetime) or not origin:
                continue
            
            # Normalize airport code to uppercase (e.g., "kjfk" -> "KJFK")
            airport = str(origin).upper()
            
            # Add flight to airport's list
            grouped.setdefault(airport, []).append((str(flight_id), takeoff))
        
        # Sort flights within each airport for deterministic FCFS processing
        # Primary sort: takeoff time (earliest first)
        # Secondary sort: flight ID (for deterministic tie-breaking)
        for airport, flights in grouped.items():
            flights.sort(key=lambda item: (item[1], item[0]))
            logger.debug("Indexed %d flights for airport %s", len(flights), airport)
        
        return grouped


def _ensure_comparable_datetimes(
    flight_dt: datetime, reference: datetime, airport: str, flight_id: str
) -> datetime:
    """
    Validate that two datetime objects have compatible timezone awareness.
    
    Both datetimes must either both be timezone-aware or both be timezone-naive.
    Mixing timezone-aware and timezone-naive datetimes can lead to incorrect
    comparisons and calculations.
    
    Args:
        flight_dt: Flight's scheduled datetime
        reference: Window's reference datetime (start or end)
        airport: Airport code for error message
        flight_id: Flight identifier for error message
    Returns:
        Possibly adjusted flight datetime that is comparable to ``reference``.
    """
    flight_tz = flight_dt.tzinfo
    reference_tz = reference.tzinfo
    
    # Check if timezone awareness is mismatched (one is None, other is not)
    if (flight_tz is None) != (reference_tz is None):
        if flight_tz is None and reference_tz is not None:
            logger.debug(
                "Flight %s has no timezone; assuming UTC to compare with %s window bounds",
                flight_id,
                airport,
            )
            return flight_dt.replace(tzinfo=timezone.utc)
        raise ValueError(
            f"Timezone mismatch between flight {flight_id} ({flight_dt}) and ground-hold window for {airport}"
        )
    return flight_dt


__all__ = ["GroundHoldOperator"]
