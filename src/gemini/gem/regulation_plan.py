"""Regulation plan loader that maps YAML specs to per-bin capacities."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from gemini.propagation.domain_types import Volume
from gemini.propagation.tvtw_indexer import TVTWIndexer


@dataclass(frozen=True)
class VolumeRegulation:
    """Single contiguous regulation for a volume expressed in bin indices."""

    volume_id: str
    start_bin: int  # inclusive
    end_bin: int  # exclusive
    capacity_per_bin: float

    def to_dict(self, time_bin_minutes: int) -> Dict[str, object]:
        """Serialize to YAML-friendly structure."""
        start_time = _format_time(self.start_bin, time_bin_minutes)
        end_time = _format_time(self.end_bin, time_bin_minutes)
        return {
            "volume_id": self.volume_id,
            "start_time": start_time,
            "end_time": end_time,
            "capacity_per_hour": self.capacity_per_bin * (60 / time_bin_minutes),
        }


@dataclass
class RegulationPlan:
    """Collection of capacity reductions expressed per traffic volume."""

    time_bin_minutes: int
    regulations: List[VolumeRegulation] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    # --------------------------------------------------------------------- I/O --
    @classmethod
    def load(cls, path: str, tvtw: TVTWIndexer) -> "RegulationPlan":
        """Load a plan from YAML and align it to the provided TVTW indexer."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        tbm = int(payload.get("time_bin_minutes", 0))
        if tbm <= 0:
            raise ValueError("Regulation plan must specify time_bin_minutes > 0.")
        if tbm != tvtw.time_bin_minutes:
            raise ValueError("time_bin_minutes mismatch between plan and TVTWIndexer.")

        bins_per_hour = tvtw.bins_per_hour
        regulations: List[VolumeRegulation] = []

        volumes_section = payload.get("volumes") or {}
        for volume_id, spec in volumes_section.items():
            regs_section = spec.get("regulations") or []
            for reg_spec in regs_section:
                start = _parse_time(reg_spec["start_time"])
                end = _parse_time(reg_spec["end_time"])
                start_bin = (start.hour * 60 + start.minute) // tbm
                end_bin = (end.hour * 60 + end.minute) // tbm
                if end_bin <= start_bin:
                    raise ValueError(
                        f"Regulation for {volume_id} must have end_time > start_time."
                    )
                cap_hour = float(reg_spec["capacity_per_hour"])
                cap_bin = cap_hour / bins_per_hour
                regulations.append(
                    VolumeRegulation(
                        volume_id=str(volume_id),
                        start_bin=int(start_bin),
                        end_bin=int(end_bin),
                        capacity_per_bin=float(cap_bin),
                    )
                )

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"time_bin_minutes", "volumes"}
        }
        return cls(time_bin_minutes=tbm, regulations=regulations, metadata=metadata)

    def save(self, path: str) -> None:
        """Persist the plan back to YAML."""
        bins_per_hour = 60 // self.time_bin_minutes
        volume_map: Dict[str, Dict[str, object]] = {}
        for reg in self.regulations:
            entry = volume_map.setdefault(reg.volume_id, {"regulations": []})
            entry["regulations"].append(
                {
                    "start_time": _format_time(reg.start_bin, self.time_bin_minutes),
                    "end_time": _format_time(reg.end_bin, self.time_bin_minutes),
                    "capacity_per_hour": reg.capacity_per_bin * bins_per_hour,
                }
            )

        payload: Dict[str, object] = {
            "time_bin_minutes": self.time_bin_minutes,
            **self.metadata,
        }
        if volume_map:
            payload["volumes"] = volume_map

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)

    # ---------------------------------------------------------------- utilities --
    def build_capacity_matrix(
        self, volumes: Dict[str, Volume], num_bins: int
    ) -> Dict[str, List[Optional[float]]]:
        """Return per-volume regulated capacities in flights/bin."""
        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")
        matrix: Dict[str, List[Optional[float]]] = {
            volume_id: [None] * num_bins for volume_id in volumes.keys()
        }
        for reg in self.regulations:
            if reg.volume_id not in matrix:
                matrix[reg.volume_id] = [None] * num_bins
            cap = float(reg.capacity_per_bin)
            start = max(reg.start_bin, 0)
            end = min(reg.end_bin, num_bins)
            if end <= start:
                continue
            slots = matrix[reg.volume_id]
            for idx in range(start, end):
                slots[idx] = cap
        return matrix


def _parse_time(value: str) -> dt.time:
    try:
        return dt.datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid time string '{value}', expected HH:MM.") from exc


def _format_time(bin_index: int, time_bin_minutes: int) -> str:
    minutes = bin_index * time_bin_minutes
    minutes = max(0, min(24 * 60, minutes))
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"
