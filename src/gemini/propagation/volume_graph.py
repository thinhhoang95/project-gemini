"""Helpers for loading traffic volumes and querying spatial membership."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set

import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from .domain_types import Volume


@dataclass
class VolumeGraph:
    """Light-weight registry of known traffic volumes."""

    volumes: Dict[str, Volume]
    geo_dataframe: pd.DataFrame
    locator: "VolumeLocator"

    @classmethod
    def from_geojson(cls, path: str) -> "VolumeGraph":
        """Load traffic volumes from a GeoJSON file."""
        geo_df = _load_geojson_dataframe(path)
        if "traffic_volume_id" not in geo_df.columns:
            raise ValueError(
                "GeoJSON file must have a 'traffic_volume_id' column/property."
            )

        volumes: Dict[str, Volume] = {}
        for row in geo_df.itertuples(index=False):
            volume_id = str(getattr(row, "traffic_volume_id"))
            name = getattr(row, "name", None)
            capacity = getattr(row, "capacity", None)
            metadata = {
                col: getattr(row, col)
                for col in geo_df.columns
                if col not in {"geometry", "traffic_volume_id"}
            }
            capacity_value: Optional[float] = None
            if capacity not in (None, ""):
                try:
                    capacity_value = float(capacity)
                except (TypeError, ValueError):
                    capacity_value = None

            volumes[volume_id] = Volume(
                id=volume_id,
                name=name if isinstance(name, str) else None,
                capacity=capacity_value,
                metadata=metadata,
            )

        locator = VolumeLocator(geo_df)
        return cls(volumes=volumes, geo_dataframe=geo_df, locator=locator)

    def get_volume(self, volume_id: str) -> Optional[Volume]:
        """Return the stored metadata for a traffic volume."""
        return self.volumes.get(volume_id)


def _load_geojson_dataframe(path: str) -> pd.DataFrame:
    """Read a GeoJSON file into a pandas DataFrame with shapely geometries."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    features = payload.get("features") or []
    rows = []
    for feature in features:
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry")
        geom = shape(geometry) if geometry else None
        rows.append({**properties, "geometry": geom})
    return pd.DataFrame(rows)


class VolumeLocator:
    """Spatial helper that matches 4D sample points to traffic volumes."""

    def __init__(self, geo_df: pd.DataFrame):
        self._geo_df = geo_df.copy()
        self._geo_df.reset_index(drop=True, inplace=True)
        self._geom_rows = []
        self._geom_indices = []
        for idx, geom in enumerate(self._geo_df["geometry"]):
            if geom is None or geom.is_empty:
                continue
            self._geom_rows.append(geom)
            self._geom_indices.append(idx)
        self._sindex = STRtree(self._geom_rows) if self._geom_rows else None

        self._min_alt_ft = []
        self._max_alt_ft = []
        for row in self._geo_df.itertuples():
            self._min_alt_ft.append(self._to_altitude_ft(getattr(row, "min_fl", None)))
            self._max_alt_ft.append(
                self._to_altitude_ft(getattr(row, "max_fl", None), default=60000.0)
            )

    @staticmethod
    def _to_altitude_ft(value: object, default: float = 0.0) -> float:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        try:
            return float(value) * 100.0
        except (TypeError, ValueError):
            return default

    def volumes_for_point(self, point: Point, altitude_ft: float) -> Set[str]:
        """Return the set of volume IDs covering the point at the given altitude."""
        if point.is_empty or self._sindex is None:
            return set()

        candidate_idx = list(self._sindex.query(point, predicate="intersects"))
        if not candidate_idx:
            return set()

        matches: Set[str] = set()
        for tree_idx in candidate_idx:
            idx = self._geom_indices[int(tree_idx)]
            if altitude_ft < self._min_alt_ft[idx] or altitude_ft > self._max_alt_ft[idx]:
                continue
            tv_row = self._geo_df.iloc[idx]
            geom = tv_row.geometry
            if geom is None or geom.is_empty:
                continue
            if not geom.covers(point):
                continue
            matches.add(str(tv_row["traffic_volume_id"]))
        return matches

    def batch_volumes_for_points(
        self, samples: Iterable[Point], altitudes_ft: Iterable[float]
    ) -> Iterable[Set[str]]:
        """Vectorised convenience wrapper used by some callers."""
        for point, altitude in zip(samples, altitudes_ft):
            yield self.volumes_for_point(point, altitude)
