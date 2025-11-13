"""Core dataclasses shared across the propagation package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class Volume:
    """Metadata describing a regulated traffic volume."""

    id: str
    name: Optional[str] = None
    capacity: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeId:
    """Directed connection between two traffic volumes."""

    upstream: str
    downstream: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.upstream}->{self.downstream}"


@dataclass
class TraversalRecord:
    """Single traversal of an edge with the timing metadata required for K0–K3."""

    edge: EdgeId
    dep_bin: int
    arr_bin: int
    lag_bins: int
    hour_index: int
    dep_minutes: float
    arr_minutes: float
    flight_id: Optional[str] = None
    route_label: Optional[str] = None
    group: Optional[str] = None  # ORIGINAL vs NONORIG
