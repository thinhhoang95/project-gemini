"""Helpers for building traversal records directly from FR artifacts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence

from gemini.arr.fr_artifacts_loader import FRSegment
from gemini.arrivals.flight_list_gemini import FlightListGemini

from .domain_types import EdgeId, TraversalRecord
from .tvtw_indexer import TVTWIndexer

logger = logging.getLogger(__name__)


@dataclass
class FRRouteGroup:
    """Container for the FR segments that describe a single (flight, route)."""

    flight_id: str
    route_index: int
    route_prob: float
    segments: List[FRSegment]
    group: str | None = None


@dataclass
class FRRouteTraversalResult:
    """Encapsulates traversal records plus bookkeeping for dropped entries."""

    traversals: List[TraversalRecord]
    dropped_departures: int
    censored_traversals: int


class FlightTakeoffLookup:
    """Caches minute-of-day metadata for each flight in the master list."""

    def __init__(self, flights: FlightListGemini):
        self._minutes_of_day: Dict[str, float] = {}
        self._load_minutes(flights)

    def _load_minutes(self, flights: FlightListGemini) -> None:
        missing = 0
        for flight_id, metadata in flights.flight_metadata.items():
            takeoff_time = metadata.get("takeoff_time")
            if isinstance(takeoff_time, datetime):
                self._minutes_of_day[str(flight_id)] = _minutes_since_midnight(takeoff_time)
            else:
                missing += 1
        if missing:
            logger.warning(
                "Missing takeoff timestamps for %s flights in %s",
                missing,
                flights.flights_csv_path,
            )

    def get_takeoff_minute_of_day(self, flight_id: str) -> float:
        key = str(flight_id)
        if key not in self._minutes_of_day:
            raise KeyError(f"Missing takeoff metadata for {key}")
        return self._minutes_of_day[key]

    def has_flight(self, flight_id: str) -> bool:
        return str(flight_id) in self._minutes_of_day

    def as_minutes_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the minute-of-day cache."""

        return dict(self._minutes_of_day)


def group_fr_segments_by_route(segments: Sequence[FRSegment]) -> List[FRRouteGroup]:
    """Group FR segments by (flight, route) while preserving order by entry offset."""

    groups: List[FRRouteGroup] = []
    current_segments: List[FRSegment] = []
    current_key: tuple[str, int] | None = None
    current_prob: float | None = None
    current_group: str | None = None

    for segment in segments:
        key = (segment.flight_id, segment.route_index)
        if key != current_key:
            if current_segments:
                groups.append(
                    FRRouteGroup(
                        flight_id=current_key[0],
                        route_index=current_key[1],
                        route_prob=float(current_prob or 0.0),
                        segments=current_segments,
                        group=current_group,
                    )
                )
            current_segments = [segment]
            current_key = key
            current_prob = segment.route_prob
            current_group = None
        else:
            current_segments.append(segment)
            if current_prob is not None and abs(segment.route_prob - current_prob) > 1e-9:
                logger.debug(
                    "Route probability drifted within flight %s route %s (%.6f vs %.6f).",
                    segment.flight_id,
                    segment.route_index,
                    current_prob,
                    segment.route_prob,
                )
    if current_segments and current_key is not None:
        groups.append(
            FRRouteGroup(
                flight_id=current_key[0],
                route_index=current_key[1],
                route_prob=float(current_prob or 0.0),
                segments=current_segments,
                group=current_group,
            )
        )
    return groups


def build_traversals_for_fr_route(
    *,
    route_group: FRRouteGroup,
    tvtw_indexer: TVTWIndexer,
    takeoff_minute_of_day: float,
    max_lag_bins: int,
    route_label: str | None = None,
) -> FRRouteTraversalResult:
    """Convert one FR route into traversal records consumable by Step K0."""

    max_minutes = tvtw_indexer.num_bins * tvtw_indexer.time_bin_minutes
    bins_per_hour = tvtw_indexer.bins_per_hour
    entries = [
        (segment.volume_id, takeoff_minute_of_day + float(segment.entry_offset_min))
        for segment in route_group.segments
    ]
    entries.sort(key=lambda entry: entry[1])

    traversals: List[TraversalRecord] = []
    dropped = 0
    censored = 0
    if len(entries) < 2:
        return FRRouteTraversalResult(traversals=traversals, dropped_departures=0, censored_traversals=0)

    label = route_label if route_label is not None else str(route_group.route_index)
    for idx in range(len(entries) - 1):
        upstream, dep_minutes = entries[idx]
        downstream, arr_minutes = entries[idx + 1]
        if upstream == downstream:
            continue
        if dep_minutes < 0 or dep_minutes >= max_minutes:
            dropped += 1
            continue

        dep_bin = tvtw_indexer.minutes_to_bin(dep_minutes)
        hour_index = dep_bin // bins_per_hour
        edge = EdgeId(upstream=upstream, downstream=downstream)

        arrival_observed = True
        censor_reason: str | None = None
        arr_bin = -1
        lag = 0

        if arr_minutes < 0 or arr_minutes >= max_minutes:
            arrival_observed = False
            censor_reason = "arrival_outside_horizon"
        else:
            arr_bin = tvtw_indexer.minutes_to_bin(arr_minutes)
            lag = arr_bin - dep_bin
            if lag <= 0:
                arrival_observed = False
                censor_reason = "nonpositive_lag"
            elif lag > max_lag_bins:
                arrival_observed = False
                censor_reason = "lag_exceeds_max"

        record = TraversalRecord(
            edge=edge,
            dep_bin=dep_bin,
            arr_bin=arr_bin,
            lag_bins=lag,
            hour_index=hour_index,
            dep_minutes=dep_minutes,
            arr_minutes=arr_minutes,
            flight_id=route_group.flight_id,
            route_label=label,
            group=route_group.group,
            arrival_observed=arrival_observed,
            censor_reason=censor_reason,
            weight=route_group.route_prob,
        )
        if not arrival_observed:
            censored += 1
        traversals.append(record)

    return FRRouteTraversalResult(
        traversals=traversals,
        dropped_departures=dropped,
        censored_traversals=censored,
    )


def _minutes_since_midnight(dt: datetime) -> float:
    """Return fractional minutes since midnight for ``dt``."""

    return dt.hour * 60.0 + dt.minute + dt.second / 60.0 + dt.microsecond / 60_000_000.0


def tag_route_groups(route_groups: Sequence[FRRouteGroup]) -> None:
    """Assign ORIGINAL/NONORIG tags per flight based on route probabilities."""

    flights: Dict[str, List[FRRouteGroup]] = {}
    for group in route_groups:
        flights.setdefault(group.flight_id, []).append(group)

    for flight_groups in flights.values():
        if not flight_groups:
            continue
        sorted_groups = sorted(flight_groups, key=lambda grp: grp.route_prob, reverse=True)
        sorted_groups[0].group = "ORIGINAL"
        for extra in sorted_groups[1:]:
            extra.group = "NONORIG"


__all__ = [
    "FRRouteGroup",
    "FRRouteTraversalResult",
    "FlightTakeoffLookup",
    "tag_route_groups",
    "group_fr_segments_by_route",
    "build_traversals_for_fr_route",
]
