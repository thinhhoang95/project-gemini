"""Route catalog loaders shared across the CLI and extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd


def _normalize_route(route: object) -> str:
    if route is None or (isinstance(route, float) and pd.isna(route)):
        return ""
    return str(route).strip()


def _normalize_flight_id(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


@dataclass
class RouteCatalog:
    """Partitions flights into ORIGINAL vs non-ORIGINAL groups."""

    original_flights: Set[str] = field(default_factory=set)
    nonorig_routes: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def nonorig_route_keys(self) -> Set[Tuple[str, str]]:
        keys: Set[Tuple[str, str]] = set()
        for flight_id, routes in self.nonorig_routes.items():
            for route in routes:
                keys.add((flight_id, route))
        return keys

    def group_for(self, flight_id: str, route_label: Optional[str]) -> str:
        normalized = _normalize_flight_id(flight_id)
        route_text = _normalize_route(route_label)
        if normalized in self.original_flights or route_text.upper() == "ORIGINAL":
            return "ORIGINAL"
        return "NONORIG"

    @classmethod
    def from_csv(cls, path: str) -> "RouteCatalog":
        df = pd.read_csv(path, usecols=["flight_identifier", "route"])
        original: Set[str] = set()
        nonorig: Dict[str, Set[str]] = {}
        for row in df.itertuples(index=False):
            flight_id = _normalize_flight_id(getattr(row, "flight_identifier"))
            if not flight_id:
                continue
            route = _normalize_route(getattr(row, "route"))
            if route.upper() == "ORIGINAL":
                original.add(flight_id)
            elif route:
                nonorig.setdefault(flight_id, set()).add(route)
        return cls(original_flights=original, nonorig_routes=nonorig)
