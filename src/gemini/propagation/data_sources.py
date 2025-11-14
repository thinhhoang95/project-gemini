"""Streaming helpers for loading 4D segments."""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterator, Optional, Sequence

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
    "p_route",
]

NONORIG_OPTIONAL_COLUMNS = {"p_route"}


def _determine_nonorig_usecols(csv_path: str) -> list[str]:
    """Return the subset of NONORIG columns that exist inside the given CSV."""
    desired_cols = list(dict.fromkeys(NONORIG_COLUMNS))
    header_df = pd.read_csv(csv_path, nrows=0)
    available = set(header_df.columns)
    missing_required = [
        column
        for column in desired_cols
        if column not in available and column not in NONORIG_OPTIONAL_COLUMNS
    ]
    if missing_required:
        missing_list = ", ".join(missing_required)
        raise ValueError(f"{csv_path} is missing required columns: {missing_list}")
    missing_optional = [
        column for column in NONORIG_OPTIONAL_COLUMNS if column in desired_cols and column not in available
    ]
    if missing_optional and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Optional columns %s missing in %s; continuing without weights.",
            ", ".join(missing_optional),
            os.path.basename(csv_path),
        )
    return [column for column in desired_cols if column in available]


def _extract_route_weight(
    route_df: pd.DataFrame, *, flight_id: str, route_label: str
) -> Optional[float]:
    """Return the weight for a route, defaulting to 1.0 if p_route absent."""
    if "p_route" not in route_df.columns:
        return 1.0
    p_series = pd.to_numeric(route_df["p_route"], errors="coerce").dropna()
    if p_series.empty:
        logger.warning(
            "Skipping NON-ORIGINAL flight %s (%s) due to missing p_route values.",
            flight_id,
            route_label,
        )
        return None
    unique_values = pd.unique(p_series)
    route_weight = float(unique_values[0])
    if len(unique_values) > 1:
        logger.warning(
            "Found multiple p_route values for flight %s (%s); using %s.",
            flight_id,
            route_label,
            route_weight,
        )
    clamped_weight = min(max(route_weight, 0.0), 1.0)
    if clamped_weight != route_weight:
        logger.warning(
            "Clamped p_route for flight %s (%s) from %s to %s.",
            flight_id,
            route_label,
            route_weight,
            clamped_weight,
        )
    return clamped_weight

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
        chunk["flight_identifier"] = chunk["flight_identifier"].astype(str).str.strip()
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
                route_weight=1.0,
            )


def iter_nonorig_segments(
    segments_dir: str,
    route_catalog: RouteCatalog,
    *,
    chunksize: int = 200_000,
) -> Iterator[FlightRouteSegments]:
    """Yield candidate non-ORIGINAL trajectories from partitioned CSV.GZ files.

    Each CSV partitions data by flight identifier (not route), so chunks may contain
    interleaved segments for many candidate routes. We therefore ensure that all rows
    for a specific flight are assembled before we split them into per-route trajectories.
    This prevents returning the same (flight, route) pair multiple times and guarantees
    that every yielded FlightRouteSegments object contains the complete set of segments
    for that candidate route.
    """
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
    target_flights = set(route_catalog.nonorig_routes)
    desired_columns = list(dict.fromkeys(NONORIG_COLUMNS))
    carryover = pd.DataFrame(columns=desired_columns)
    for file_path in files:
        usecols = _determine_nonorig_usecols(file_path)
        reader = pd.read_csv(
            file_path,
            usecols=usecols,
            chunksize=chunksize,
        )
        chunk_idx = 0
        cumulative_matches = 0
        for chunk in reader:
            chunk_idx += 1
            chunk["flight_identifier"] = chunk["flight_identifier"].astype(str).str.strip()
            chunk["route"] = chunk["route"].astype(str).str.strip()
            if not carryover.empty:
                chunk = pd.concat([carryover, chunk], ignore_index=True)
                carryover = carryover.iloc[0:0]
            if chunk.empty:
                continue

            last_flight_id = chunk["flight_identifier"].iloc[-1]
            carryover = chunk[chunk["flight_identifier"] == last_flight_id].copy()
            process_df = chunk[chunk["flight_identifier"] != last_flight_id]

            if process_df.empty:
                if logger.isEnabledFor(logging.DEBUG) and (
                    chunk_idx <= 5 or chunk_idx % 10 == 0
                ):
                    logger.debug(
                        "iter_nonorig_segments file=%s chunk=%s matched 0 route keys (rows=%s)",
                        os.path.basename(file_path),
                        chunk_idx,
                        len(chunk),
                    )
                continue

            filtered = process_df[process_df["flight_identifier"].isin(target_flights)].copy()
            if filtered.empty:
                if logger.isEnabledFor(logging.DEBUG) and (
                    chunk_idx <= 5 or chunk_idx % 10 == 0
                ):
                    logger.debug(
                        "iter_nonorig_segments file=%s chunk=%s matched 0 route keys (rows=%s)",
                        os.path.basename(file_path),
                        chunk_idx,
                        len(process_df),
                    )
                continue

            filtered["__key"] = list(zip(filtered["flight_identifier"], filtered["route"]))
            filtered = filtered[filtered["__key"].isin(route_keys)]
            match_count = filtered["__key"].nunique() if not filtered.empty else 0

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
                        len(process_df),
                    )
            cumulative_matches += match_count
            if filtered.empty:
                continue

            filtered = filtered.drop(columns="__key")
            for flight_id, flight_df in filtered.groupby("flight_identifier", sort=False):
                routes = route_catalog.nonorig_routes.get(flight_id)
                if not routes:
                    continue
                for route_label, route_df in flight_df.groupby("route", sort=False):
                    if route_label not in routes:
                        continue
                    route_weight = _extract_route_weight(
                        route_df, flight_id=flight_id, route_label=route_label
                    )
                    if route_weight is None:
                        continue
                    yield FlightRouteSegments(
                        flight_id=flight_id,
                        route_label=route_label,
                        group="NONORIG",
                        segments=route_df.copy(),
                        route_weight=route_weight,
                    )

    if not carryover.empty:
        filtered = carryover[carryover["flight_identifier"].isin(target_flights)].copy()
        if not filtered.empty:
            filtered["route"] = filtered["route"].astype(str).str.strip()
            filtered["__key"] = list(zip(filtered["flight_identifier"], filtered["route"]))
            filtered = filtered[filtered["__key"].isin(route_keys)].drop(columns="__key")
            for flight_id, flight_df in filtered.groupby("flight_identifier", sort=False):
                routes = route_catalog.nonorig_routes.get(flight_id)
                if not routes:
                    continue
                for route_label, route_df in flight_df.groupby("route", sort=False):
                    if route_label not in routes:
                        continue
                    route_weight = _extract_route_weight(
                        route_df, flight_id=flight_id, route_label=route_label
                    )
                    if route_weight is None:
                        continue
                    yield FlightRouteSegments(
                        flight_id=flight_id,
                        route_label=route_label,
                        group="NONORIG",
                        segments=route_df.copy(),
                        route_weight=route_weight,
                    )
