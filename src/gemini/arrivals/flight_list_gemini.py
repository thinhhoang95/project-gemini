from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FlightListGemini:
    """Standalone flight list with minimal metadata required for Gemini workflows."""

    def __init__(self, flights_csv_path: str = "output/flights_20230717_0000-2359.csv"):
        self.flights_csv_path = Path(flights_csv_path)
        self.flight_metadata: Dict[str, Dict[str, object]] = {}
        self.flight_ids: List[str] = []
        self._load_flights()

    # -------------------------------------------------------------------------
    def get_flight_metadata(self, flight_id: str) -> Dict[str, object]:
        return self.flight_metadata.get(str(flight_id), {})

    # -------------------------------------------------------------------------
    def _load_flights(self) -> None:
        if not self.flights_csv_path.exists():
            raise FileNotFoundError(f"Flights CSV not found at {self.flights_csv_path}")
        segment_count = 0
        with self.flights_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Flights CSV missing header row")
            for row in reader:
                segment_count += 1
                flight_id = (row.get("flight_identifier") or "").strip()
                if not flight_id:
                    continue
                metadata = self.flight_metadata.get(flight_id)
                if metadata is None:
                    metadata = {
                        "takeoff_time": None,
                        "landing_time_filed": None,
                        "origin": None,
                        "destination": None,
                        "cruise_fl": 0,
                    }
                    self.flight_metadata[flight_id] = metadata
                    self.flight_ids.append(flight_id)
                takeoff_dt = self._parse_datetime(row.get("date_begin_segment"), row.get("time_begin_segment"))
                if metadata.get("takeoff_time") is None and takeoff_dt is not None:
                    metadata["takeoff_time"] = takeoff_dt
                origin = (row.get("origin_aerodrome") or "").strip()
                if origin and metadata.get("origin") is None:
                    metadata["origin"] = origin
                landing_dt = self._parse_datetime(row.get("date_end_segment"), row.get("time_end_segment"))
                if landing_dt is not None:
                    prev = metadata.get("landing_time_filed")
                    if prev is None or landing_dt >= prev:
                        metadata["landing_time_filed"] = landing_dt
                        destination = (row.get("destination_aerodrome") or "").strip()
                        if destination:
                            metadata["destination"] = destination
                self._update_cruise_fl(metadata, row)
        logger.info(
            "Loaded metadata for %d flights across %d segments from %s",
            len(self.flight_ids),
            segment_count,
            self.flights_csv_path,
        )

    @staticmethod
    def _parse_datetime(date_token: Optional[str], time_token: Optional[str]) -> Optional[datetime]:
        date_str = (date_token or "").strip()
        time_str = (time_token or "").strip()
        if not date_str or not time_str:
            return None
        normalized_time = time_str.zfill(6)
        try:
            return datetime.strptime(f"{date_str}{normalized_time}", "%y%m%d%H%M%S")
        except ValueError:
            return None

    @staticmethod
    def _parse_flight_level(value: Optional[str]) -> Optional[int]:
        token = (value or "").strip()
        if not token:
            return None
        try:
            return int(float(token))
        except Exception:
            return None

    def _update_cruise_fl(self, metadata: Dict[str, object], row: Dict[str, str]) -> None:
        levels = [
            self._parse_flight_level(row.get("flight_level_begin")),
            self._parse_flight_level(row.get("flight_level_end")),
        ]
        valid_levels = [lvl for lvl in levels if lvl is not None]
        if not valid_levels:
            return
        current = int(metadata.get("cruise_fl") or 0)
        metadata["cruise_fl"] = max(current, max(valid_levels))
