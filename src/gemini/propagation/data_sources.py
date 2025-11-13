"""Streaming helpers for loading 4D segments."""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterator, Sequence

import pandas as pd

from .routes import RouteCatalog
from .traversal_extractor import FlightRouteSegments


logger = logging.getLogger(__name__)


ORIGINAL_COLUMNS: Sequence[str] = [
    "flight_identifier",
    "time_begin_segment",
    "time_end_segment",
    "date_begin_segment",
    "date_end_segment",
    "_start_datetime",
    "latitude_begin",
    "longitude_begin",
    "latitude_end",
    "longitude_end",
    "flight_level_begin",
    "flight_level_end",
    "sequence",
]

NONORIG_COLUMNS: Sequence[str] = [
    "flight_identifier",
    "time_begin_segment",
    "time_end_segment",
    "latitude_begin",
    "longitude_begin",
    "latitude_end",
    "longitude_end",
    "flight_level_begin",
    "flight_level_end",
    "route",
]


def iter_original_segments(
    master_csv_path: str,
    route_catalog: RouteCatalog,
    *,
    chunksize: int = 250_000,
) -> Iterator[FlightRouteSegments]:
    """Yield ORIGINAL flight trajectories in manageable batches."""
    flight_ids = {str(fid) for fid in route_catalog.original_flights}
    if not flight_ids:
        return
    reader = pd.read_csv(
        master_csv_path,
        usecols=list(dict.fromkeys(ORIGINAL_COLUMNS)),
        chunksize=chunksize,
    )
    chunk_idx = 0
    cumulative_matches = 0
    for chunk in reader:
        chunk_idx += 1
        chunk["flight_identifier"] = chunk["flight_identifier"].astype(str)
        filtered = chunk[chunk["flight_identifier"].isin(flight_ids)]
        match_count = 0
        if not filtered.empty:
            match_count = filtered["flight_identifier"].nunique()
        if logger.isEnabledFor(logging.DEBUG):
            if match_count:
                logger.debug(
                    "iter_original_segments chunk=%s matched %s flights (cumulative=%s)",
                    chunk_idx,
                    match_count,
                    cumulative_matches + match_count,
                )
            elif chunk_idx <= 5 or chunk_idx % 10 == 0:
                logger.debug(
                    "iter_original_segments chunk=%s matched 0 flights (rows=%s)",
                    chunk_idx,
                    len(chunk),
                )
        cumulative_matches += match_count
        if filtered.empty:
            continue
        for flight_id, flight_df in filtered.groupby("flight_identifier"):
            yield FlightRouteSegments(
                flight_id=flight_id,
                route_label="ORIGINAL",
                group="ORIGINAL",
                segments=flight_df.copy(),
            )


def iter_nonorig_segments(
    segments_dir: str,
    route_catalog: RouteCatalog,
    *,
    chunksize: int = 200_000,
) -> Iterator[FlightRouteSegments]:
    """Yield candidate non-ORIGINAL trajectories from partitioned CSV.GZ files."""
    route_keys = route_catalog.nonorig_route_keys
    if not route_keys:
        return
    if not os.path.isdir(segments_dir):
        raise FileNotFoundError(f"nonorig_4d_segments_dir does not exist: {segments_dir}")
    pattern = os.path.join(segments_dir, "*.csv*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CSV/CSV.GZ files were found under {segments_dir} matching '*.csv*'."
        )
    for file_path in files:
        reader = pd.read_csv(
            file_path,
            usecols=list(dict.fromkeys(NONORIG_COLUMNS)),
            chunksize=chunksize,
        )
        chunk_idx = 0
        cumulative_matches = 0
        for chunk in reader:
            chunk_idx += 1
            chunk["flight_identifier"] = chunk["flight_identifier"].astype(str)
            chunk["route"] = chunk["route"].astype(str).str.strip()
            chunk["__key"] = list(zip(chunk["flight_identifier"], chunk["route"]))
            filtered = chunk[chunk["__key"].isin(route_keys)]
            match_count = 0
            if not filtered.empty:
                match_count = filtered["__key"].nunique()
            if logger.isEnabledFor(logging.DEBUG):
                if match_count:
                    logger.debug(
                        "iter_nonorig_segments file=%s chunk=%s matched %s route keys (cumulative=%s)",
                        os.path.basename(file_path),
                        chunk_idx,
                        match_count,
                        cumulative_matches + match_count,
                    )
                elif chunk_idx <= 5 or chunk_idx % 10 == 0:
                    logger.debug(
                        "iter_nonorig_segments file=%s chunk=%s matched 0 route keys (rows=%s)",
                        os.path.basename(file_path),
                        chunk_idx,
                        len(chunk),
                    )
            cumulative_matches += match_count
            if filtered.empty:
                continue
            filtered = filtered.drop(columns="__key")
            for (flight_id, route_label), flight_df in filtered.groupby(
                ["flight_identifier", "route"]
            ):
                yield FlightRouteSegments(
                    flight_id=flight_id,
                    route_label=route_label,
                    group="NONORIG",
                    segments=flight_df.copy(),
                )
