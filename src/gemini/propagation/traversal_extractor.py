"""Convert 4D trajectories into edge-traversal records."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from shapely.geometry import LineString, Point

from .tvtw_indexer import TVTWIndexer
from .domain_types import EdgeId, TraversalRecord
from .volume_graph import VolumeLocator


@dataclass
class FlightRouteSegments:
    """Container for a flight's 4D segments under a specific route label."""

    flight_id: str
    route_label: str
    group: str
    segments: pd.DataFrame


@dataclass
class SamplePoint:
    timestamp: datetime
    point: Point
    altitude_ft: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate great-circle distance between two WGS84 points."""
    rad_lat1, rad_lon1 = math.radians(lat1), math.radians(lon1)
    rad_lat2, rad_lon2 = math.radians(lat2), math.radians(lon2)
    dlat = rad_lat2 - rad_lat1
    dlon = rad_lon2 - rad_lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    earth_radius_km = 6371.0
    return earth_radius_km * c


class TraversalExtractor:
    """Implements Step K0 from Section 6.2.1."""

    def __init__(
        self,
        tvtw_indexer: TVTWIndexer,
        volume_locator: VolumeLocator,
        planning_day: date,
        *,
        max_lag_bins: Optional[int] = None,
        sampling_distance_km: float = 10.0,
        sampling_time_seconds: float = 120.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.tvtw_indexer = tvtw_indexer
        self.volume_locator = volume_locator
        self.day_start = datetime.combine(planning_day, time.min)
        self.max_minutes = self.tvtw_indexer.num_bins * self.tvtw_indexer.time_bin_minutes
        self.max_lag_bins = max_lag_bins or self.tvtw_indexer.num_bins
        self.sampling_distance_km = sampling_distance_km
        self.sampling_time_seconds = sampling_time_seconds
        self.logger = logger or logging.getLogger(__name__)
        self.bins_per_hour = self.tvtw_indexer.bins_per_hour
        self.stats = {
            "segments_processed": 0,
            "sample_points": 0,
            "volume_entries": 0,
            "traversals_emitted": 0,
            "traversals_dropped": 0,
        }

    # ------------------------------------------------------------------ public API
    def extract_traversals(
        self, flight_segments: FlightRouteSegments
    ) -> Iterator[TraversalRecord]:
        """Yield causal traversals for a specific flight/route."""
        if flight_segments.segments is None or flight_segments.segments.empty:
            return

        samples = self._collect_samples(flight_segments.segments)
        self.stats["segments_processed"] += len(flight_segments.segments)
        self.stats["sample_points"] += len(samples)
        if len(samples) < 2:
            return

        entries = self._detect_volume_entries(samples)
        self.stats["volume_entries"] += len(entries)
        if len(entries) < 2:
            return

        for record in self._records_from_entries(entries, flight_segments):
            self.stats["traversals_emitted"] += 1
            yield record

    # ----------------------------------------------------------------- diagnostics
    def get_stats(self) -> dict:
        """Return cumulative extractor statistics."""
        return dict(self.stats)

    # ---------------------------------------------------------------- internals
    def _collect_samples(self, segments: pd.DataFrame) -> List[SamplePoint]:
        sort_cols = [col for col in ["time_begin_segment", "time_end_segment", "sequence"] if col in segments.columns]
        if not sort_cols:
            sort_cols = ["flight_identifier"]
        ordered = segments.sort_values(by=sort_cols, kind="mergesort")
        samples: List[SamplePoint] = []
        for record in ordered.to_dict("records"):
            samples.extend(self._sample_segment(record))
        samples.sort(key=lambda sample: sample.timestamp)
        return samples

    def _sample_segment(self, record: dict) -> List[SamplePoint]:
        parsed = self._segment_datetimes(record)
        if not parsed:
            return []
        start_dt, end_dt = parsed
        if end_dt <= start_dt:
            return []

        start_point = self._point_from_record(record, suffix="begin")
        end_point = self._point_from_record(record, suffix="end")
        if start_point is None or end_point is None:
            return []

        start_alt = self._altitude_ft(record, suffix="begin")
        end_alt = self._altitude_ft(record, suffix="end")

        segment_line = LineString([start_point, end_point])
        if segment_line.is_empty:
            return []

        distance_km = haversine_km(start_point.y, start_point.x, end_point.y, end_point.x)
        duration_seconds = (end_dt - start_dt).total_seconds()
        distance_samples = (
            int(distance_km / self.sampling_distance_km)
            if self.sampling_distance_km > 0
            else 0
        )
        time_samples = (
            int(duration_seconds / self.sampling_time_seconds)
            if self.sampling_time_seconds > 0
            else 0
        )
        num_samples = max(distance_samples, time_samples)

        points: List[SamplePoint] = [SamplePoint(start_dt, start_point, start_alt)]
        if num_samples > 0:
            for idx in range(1, num_samples + 1):
                fraction = idx / (num_samples + 1)
                interp_point = segment_line.interpolate(fraction, normalized=True)
                sample_time = start_dt + fraction * (end_dt - start_dt)
                sample_alt = start_alt + fraction * (end_alt - start_alt)
                points.append(SamplePoint(sample_time, interp_point, sample_alt))
        points.append(SamplePoint(end_dt, end_point, end_alt))
        return points

    def _segment_datetimes(self, record: dict) -> Optional[Tuple[datetime, datetime]]:
        start_dt = self._parse_datetime(
            record.get("_start_datetime"),
            record.get("date_begin_segment"),
            record.get("time_begin_segment"),
        )
        end_dt = self._parse_datetime(
            record.get("_end_datetime"),
            record.get("date_end_segment"),
            record.get("time_end_segment"),
        )
        if start_dt is None or end_dt is None:
            return None
        return start_dt, end_dt

    def _parse_datetime(
        self, iso_value: Optional[object], date_value: Optional[object], time_value: Optional[object]
    ) -> Optional[datetime]:
        if isinstance(iso_value, str) and iso_value.strip():
            try:
                dt_value = pd.to_datetime(iso_value)
                if isinstance(dt_value, pd.Timestamp):
                    return dt_value.to_pydatetime()
                if isinstance(dt_value, datetime):
                    return dt_value
            except (ValueError, TypeError):
                pass
        parsed_date = self._parse_date_value(date_value) or self.day_start.date()
        parsed_time = self._parse_time_value(time_value)
        if parsed_time is None:
            return None
        return datetime.combine(parsed_date, parsed_time)

    @staticmethod
    def _parse_date_value(value: Optional[object]) -> Optional[date]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
        text = str(value).strip()
        if not text:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        if len(digits) == 8:
            return datetime.strptime(digits, "%Y%m%d").date()
        if len(digits) == 6:  # YYMMDD
            return datetime.strptime(digits, "%y%m%d").date()
        return None

    @staticmethod
    def _parse_time_value(value: Optional[object]) -> Optional[time]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, datetime):
            return value.time()
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().time()
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        if len(digits) > 6:
            digits = digits[-6:]
        digits = digits.zfill(6)
        hour = int(digits[0:2])
        minute = int(digits[2:4])
        second = int(digits[4:6])
        if hour >= 24 or minute >= 60 or second >= 60:
            return None
        return time(hour, minute, second)

    @staticmethod
    def _point_from_record(record: dict, *, suffix: str) -> Optional[Point]:
        lat_key = f"latitude_{suffix}"
        lon_key = f"longitude_{suffix}"
        try:
            lat = float(record[lat_key])
            lon = float(record[lon_key])
        except (KeyError, TypeError, ValueError):
            return None
        if math.isnan(lat) or math.isnan(lon):
            return None
        return Point(lon, lat)

    @staticmethod
    def _altitude_ft(record: dict, *, suffix: str) -> float:
        level_key = f"flight_level_{suffix}"
        try:
            level = float(record.get(level_key))
        except (TypeError, ValueError):
            level = None
        if level is None or math.isnan(level):
            return 0.0
        return level * 100.0

    def _detect_volume_entries(self, samples: Sequence[SamplePoint]) -> List[Tuple[str, datetime]]:
        entries: List[Tuple[str, datetime]] = []
        active: set = set()
        for sample in samples:
            volumes = self.volume_locator.volumes_for_point(sample.point, sample.altitude_ft)
            exited = {tv for tv in active if tv not in volumes}
            if exited:
                active.difference_update(exited)
            newly_entered = volumes - active
            if newly_entered:
                for volume_id in sorted(newly_entered):
                    entries.append((volume_id, sample.timestamp))
                active.update(newly_entered)
        return entries

    def _records_from_entries(
        self,
        entries: Sequence[Tuple[str, datetime]],
        flight_segments: FlightRouteSegments,
    ) -> Iterator[TraversalRecord]:
        for idx in range(len(entries) - 1):
            upstream, dep_time = entries[idx]
            downstream, arr_time = entries[idx + 1]
            if upstream == downstream:
                continue
            dep_minutes = self._minutes_from_midnight(dep_time)
            arr_minutes = self._minutes_from_midnight(arr_time)
            if dep_minutes < 0 or arr_minutes < 0:
                continue
            if dep_minutes >= self.max_minutes or arr_minutes >= self.max_minutes:
                continue
            dep_bin = self.tvtw_indexer.minutes_to_bin(dep_minutes)
            arr_bin = self.tvtw_indexer.minutes_to_bin(arr_minutes)
            lag = arr_bin - dep_bin
            if lag <= 0 or lag > self.max_lag_bins:
                self.stats["traversals_dropped"] += 1
                continue
            hour_index = dep_bin // self.bins_per_hour
            edge = EdgeId(upstream=upstream, downstream=downstream)
            yield TraversalRecord(
                edge=edge,
                dep_bin=dep_bin,
                arr_bin=arr_bin,
                lag_bins=lag,
                hour_index=hour_index,
                dep_minutes=dep_minutes,
                arr_minutes=arr_minutes,
                flight_id=flight_segments.flight_id,
                route_label=flight_segments.route_label,
                group=flight_segments.group,
            )

    def _minutes_from_midnight(self, dt_value: datetime) -> float:
        return (dt_value - self.day_start).total_seconds() / 60.0
