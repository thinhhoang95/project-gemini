from __future__ import annotations

import csv
import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


"""
This module provides a small in-memory cache of Type-1 demand records that are
emitted by the Project Gemini pipeline. Typical usage looks like:

    store = GeminiType1DemandStore(
        csv_gz_path="output/type1_demand.csv.gz",
        tvtw_indexer_path="output/tvtw_indexer.json",
    )
    for flight_id, records in store.iter_type1_records():
        # records is a sequence of Type1DemandRecord objects
        ...

The store lazily keeps all rows grouped by `flight_id`, which simplifies rapid
lookups during scenario planning and stateman workflows.
"""


@dataclass(frozen=True)
class Type1DemandRecord:
    """Per-flight demand statistics for a specific traffic volume/time bin."""

    # Identifiers describing what flight and traffic volume this row belongs to.
    flight_id: str
    volume_id: str

    # Indexes that map onto traffic volume/time-window grids.
    tv_index: int
    time_bin: int

    # Demand distribution parameters for that (flight, volume, bin) tuple.
    mu: float
    var: float

    # Metadata describing the resolution of the `time_bin` axis.
    time_bins_per_day: int

    @property
    def tvtw_index(self) -> int:
        """Dense index into the TV/TW grid, useful for array-based lookups."""
        return self.tv_index * self.time_bins_per_day + self.time_bin


class GeminiType1DemandStore:
    """Loads Type-1 demand records and TV/TW metadata for Project Gemini."""

    def __init__(self, csv_gz_path: str, tvtw_indexer_path: str = "output/tvtw_indexer.json"):
        # Normalize path arguments up front so downstream code can rely on Path APIs.
        self.csv_gz_path = Path(csv_gz_path)
        self.tvtw_indexer_path = Path(tvtw_indexer_path)

        # Mutable state created during the indexer/data load steps.
        self.time_bin_minutes: int = 0
        self._num_time_bins: int = 0
        self.tv_id_to_idx: Dict[str, int] = {}
        self.idx_to_tv_id: Dict[int, str] = {}
        self._type1_records_by_flight: Dict[str, List[Type1DemandRecord]] = {}
        self._type1_total_records: int = 0

        # Immediately hydrate the cache so the public API is ready for use.
        self._load_indexer()
        self._load_type1_demand()

    # --- public -----------------------------------------------------------------
    @property
    def num_time_bins(self) -> int:
        """Number of discretized time bins per day derived from the indexer."""
        return self._num_time_bins

    @property
    def num_traffic_volumes(self) -> int:
        """How many unique traffic volumes were observed in the indexer."""
        return len(self.tv_id_to_idx)

    @property
    def type1_record_count(self) -> int:
        """Total count of Type-1 demand rows retained in memory."""
        return self._type1_total_records

    @property
    def type1_flight_ids(self) -> Sequence[str]:
        """All flight identifiers with at least one associated Type-1 record."""
        return tuple(self._type1_records_by_flight.keys())

    def has_type1_data(self, flight_id: str) -> bool:
        """Quick membership test used by callers who may treat missing data specially."""
        return str(flight_id) in self._type1_records_by_flight

    def get_type1_records(self, flight_id: str) -> Sequence[Type1DemandRecord]:
        """Return all Type-1 records for a flight, or an empty tuple if unknown."""
        return self._type1_records_by_flight.get(str(flight_id), ())

    def iter_type1_records(
        self, flight_ids: Optional[Iterable[str]] = None
    ) -> Iterator[Tuple[str, Sequence[Type1DemandRecord]]]:
        """
        Yield `(flight_id, records)` pairs.

        If `flight_ids` is omitted, iterate over everything. Supplying a subset
        avoids materializing unrelated flights when a caller knows the exact
        slice it needs.
        """
        if flight_ids is None:
            for flight_id, records in self._type1_records_by_flight.items():
                yield flight_id, records
            return
        wanted = {str(fid) for fid in flight_ids}
        for flight_id in wanted:
            records = self._type1_records_by_flight.get(flight_id)
            if records:
                yield flight_id, records

    # --- loaders ----------------------------------------------------------------
    def _load_indexer(self) -> None:
        """Populate TV/TW metadata from the pre-generated indexer JSON."""
        if not self.tvtw_indexer_path.exists():
            raise FileNotFoundError(f"TVTW indexer not found at {self.tvtw_indexer_path}")

        # The JSON file contains metadata describing the time discretization and
        # a mapping from traffic volume IDs to integer indices.
        with self.tvtw_indexer_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if "time_bin_minutes" not in data:
            raise ValueError("TVTW indexer missing 'time_bin_minutes'")
        time_bin_minutes = int(data["time_bin_minutes"])
        if time_bin_minutes <= 0:
            raise ValueError("time_bin_minutes must be positive")
        self.time_bin_minutes = time_bin_minutes

        tv_map = data.get("tv_id_to_idx")
        if not isinstance(tv_map, Mapping):
            raise ValueError("TVTW indexer missing 'tv_id_to_idx' mapping")
        self.tv_id_to_idx = {str(tv_id): int(idx) for tv_id, idx in tv_map.items()}
        self.idx_to_tv_id = {idx: tv_id for tv_id, idx in self.tv_id_to_idx.items()}

        num_time_bins_val = data.get("num_time_bins")
        if num_time_bins_val is None:
            # Older indexers may omit this field; infer it using minutes per bin.
            minutes = max(self.time_bin_minutes, 1)
            num_time_bins_val = max(int(round(1440 / minutes)), 1)
            logger.debug(
                "TVTW indexer missing num_time_bins; inferring %d bins from %d-minute slices",
                num_time_bins_val,
                minutes,
            )

        self._num_time_bins = int(num_time_bins_val)
        if self._num_time_bins <= 0:
            raise ValueError("num_time_bins must be positive")

    def _load_type1_demand(self) -> None:
        """
        Read the Type-1 CSV (optionally gzipped) and group rows by `flight_id`.

        The logic is careful to skip rows referencing unknown traffic volumes
        or out-of-range time bins, logging counters for post-run diagnostics.
        """
        if not self.csv_gz_path.exists():
            raise FileNotFoundError(f"Type-1 demand file not found at {self.csv_gz_path}")

        # Support both plain CSV files and gzip-compressed equivalents.
        opener = gzip.open if self.csv_gz_path.suffix == ".gz" else open

        records: Dict[str, List[Type1DemandRecord]] = {}
        skipped_tv = 0
        skipped_bins = 0
        total_rows = 0
        num_time_bins = self._num_time_bins

        def _record(
            flight_id: str,
            volume_id: str,
            time_bin_val: int,
            mu_val: float,
            var_val: float,
        ) -> Optional[Type1DemandRecord]:
            """Validate raw values and construct a dataclass instance on success."""
            tv_index = self.tv_id_to_idx.get(volume_id)
            if tv_index is None:
                return None
            if time_bin_val < 0 or time_bin_val >= num_time_bins:
                return None
            return Type1DemandRecord(
                flight_id=flight_id,
                volume_id=volume_id,
                tv_index=int(tv_index),
                time_bin=int(time_bin_val),
                mu=float(mu_val),
                var=float(var_val),
                time_bins_per_day=num_time_bins,
            )

        with opener(self.csv_gz_path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                try:
                    # Pull tokens from the CSV row; keep them as strings for keys.
                    flight_token = row.get("flight_id")
                    volume_token = row.get("volume_id")
                    if not flight_token or not volume_token:
                        continue
                    flight_id = str(flight_token)
                    volume_id = str(volume_token)

                    # Coerce numeric fields; time_bin may be serialized as float.
                    time_bin = int(float(row.get("time_bin", 0)))
                    mu = float(row.get("mu", 0.0))
                    var = float(row.get("var", 0.0))
                except Exception:
                    logger.debug("Unable to parse Type-1 row: %s", row)
                    continue

                # Validate the row and append it to the flight bucket.
                record_obj = _record(flight_id, volume_id, time_bin, mu, var)
                if record_obj is None:
                    if self.tv_id_to_idx.get(volume_id) is None:
                        skipped_tv += 1
                    else:
                        skipped_bins += 1
                    continue
                records.setdefault(flight_id, []).append(record_obj)

        self._type1_records_by_flight = records
        self._type1_total_records = sum(len(v) for v in records.values())

        if skipped_tv or skipped_bins:
            logger.warning(
                "Skipped %d Type-1 rows due to unknown TV ids and %d due to invalid time bins",
                skipped_tv,
                skipped_bins,
            )
        logger.info(
            "Loaded %d Type-1 demand records covering %d flights from %s",
            self._type1_total_records,
            len(self._type1_records_by_flight),
            self.csv_gz_path,
        )


__all__ = ["GeminiType1DemandStore", "Type1DemandRecord"]
