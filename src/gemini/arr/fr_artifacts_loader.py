"""Utilities for reading FR demand / route catalogue artifacts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FRSegment:
    """Joined FR demand + route catalogue row for a (flight, route, volume) entry."""

    flight_id: str
    route_index: int
    volume_id: str
    route_prob: float
    entry_offset_min: float


def load_fr_segments(
    demand_path: str,
    route_catalog_path: str,
    *,
    min_probability: float = 1e-9,
) -> List[FRSegment]:
    """Return FR segments joined with route probabilities.

    Parameters
    ----------
    demand_path:
        Path to ``gem_artifacts_demand_all`` (CSV or CSV.GZ).
    route_catalog_path:
        Path to ``gem_artifacts_route_catalogue_all`` (CSV or CSV.GZ).
    min_probability:
        Optional numeric cutoff for discarding negligible route probabilities.

    Returns
    -------
    List[FRSegment]
        Rows are sorted by ``(flight_id, volume_id, route_index)`` to make it easy
        to stream them flight-by-flight during the Poisson-binomial accumulation.
    """

    demand_cols: Sequence[str] = [
        "flight_id",
        "route_index",
        "traffic_volume_name",
        "entry_time_from_takeoff_s",
    ]
    route_cols: Sequence[str] = [
        "flight_id",
        "route_index",
        "probability",
    ]

    demand_df = _read_csv_subset(demand_path, demand_cols)
    route_df = _read_csv_subset(route_catalog_path, route_cols)
    if demand_df.empty:
        logger.warning("FR demand table at %s is empty", demand_path)
        return []
    if route_df.empty:
        logger.warning("FR route catalogue at %s is empty", route_catalog_path)
        return []

    merged = pd.merge(
        demand_df,
        route_df,
        on=["flight_id", "route_index"],
        how="inner",
        copy=False,
    )
    if merged.empty:
        logger.warning("FR join produced zero rows – check source artifacts.")
        return []

    merged = merged[
        (merged["probability"].astype(float) > float(min_probability))
        & merged["traffic_volume_name"].notna()
        & merged["entry_time_from_takeoff_s"].notna()
    ].copy()

    if merged.empty:
        logger.warning("No FR rows survived filtering after join.")
        return []

    merged["entry_offset_min"] = merged["entry_time_from_takeoff_s"].astype(float) / 60.0
    merged["flight_id"] = merged["flight_id"].astype(str)
    merged["volume_id"] = merged["traffic_volume_name"].astype(str)
    merged["route_prob"] = merged["probability"].astype(float)
    merged["route_index"] = merged["route_index"].astype(int)

    merged.sort_values(
        ["flight_id", "volume_id", "route_index"],
        inplace=True,
        kind="mergesort",
    )

    segments: List[FRSegment] = []
    for row in merged.itertuples(index=False):
        offset = float(row.entry_offset_min)
        if not _is_finite_positive(offset):
            continue
        segments.append(
            FRSegment(
                flight_id=str(row.flight_id),
                route_index=int(row.route_index),
                volume_id=str(row.volume_id),
                route_prob=float(row.route_prob),
                entry_offset_min=offset,
            )
        )

    logger.info(
        "Loaded %d FR segments from %s and %s",
        len(segments),
        demand_path,
        route_catalog_path,
    )
    return segments


def _read_csv_subset(path: str, columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(columns))
    return df


def _is_finite_positive(value: float) -> bool:
    try:
        numeric = float(value)
    except Exception:
        return False
    return math.isfinite(numeric) and numeric >= 0.0
