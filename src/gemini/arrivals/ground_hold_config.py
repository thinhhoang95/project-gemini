from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping

import yaml

logger = logging.getLogger(__name__)


def _parse_iso_datetime(token: object) -> datetime:
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Ground-hold windows require non-empty ISO datetime strings")
    try:
        return datetime.fromisoformat(token.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime: {token!r}") from exc


@dataclass(frozen=True)
class GroundHoldWindow:
    start: datetime
    end: datetime
    rate_fph: float
    airport: str
    regulation_id: str | None = None

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("Ground-hold window start must be earlier than end")
        if (self.start.tzinfo is None) != (self.end.tzinfo is None):
            raise ValueError("Ground-hold window start and end must share timezone awareness")
        if self.rate_fph <= 0:
            raise ValueError("Ground-hold recovery rate must be positive")
        if not self.airport:
            raise ValueError("Ground-hold window must be associated with an airport")
        object.__setattr__(self, "airport", self.airport.upper())


@dataclass
class GroundHoldConfig:
    windows_by_airport: Dict[str, List[GroundHoldWindow]] = field(default_factory=dict)
    version: str | None = None
    default_rate_fph: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GroundHoldConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Ground-hold YAML not found at {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise TypeError("Ground-hold YAML must contain a mapping at the top level")
        version = data.get("version")
        default_rate = data.get("default_rate_fph")
        if default_rate is not None:
            default_rate = float(default_rate)
            if default_rate <= 0:
                raise ValueError("default_rate_fph must be positive when provided")
        airports = data.get("airports") or {}
        if not isinstance(airports, Mapping):
            raise TypeError("'airports' must be a mapping of ICAO/IATA codes to window lists")
        windows = cls._parse_airport_windows(airports, default_rate)
        return cls(windows_by_airport=windows, version=version, default_rate_fph=default_rate)

    def to_yaml(self, path: str | Path) -> None:
        output: Dict[str, object] = {}
        if self.version is not None:
            output["version"] = self.version
        if self.default_rate_fph is not None:
            output["default_rate_fph"] = float(self.default_rate_fph)
        airports: MutableMapping[str, List[Mapping[str, object]]] = {}
        for airport, windows in sorted(self.windows_by_airport.items()):
            airports[airport] = []
            for window in windows:
                entry: Dict[str, object] = {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "rate_fph": float(window.rate_fph),
                }
                if window.regulation_id:
                    entry["regulation_id"] = window.regulation_id
                airports[airport].append(entry)
        output["airports"] = airports
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(output, handle, sort_keys=True)

    def windows_for_airport(self, airport: str) -> List[GroundHoldWindow]:
        if not airport:
            return []
        return list(self.windows_by_airport.get(airport.upper(), []))

    @staticmethod
    def _parse_airport_windows(
        airports: Mapping[str, object], default_rate: float | None
    ) -> Dict[str, List[GroundHoldWindow]]:
        windows: Dict[str, List[GroundHoldWindow]] = {}
        for airport_code, window_defs in airports.items():
            airport = str(airport_code or "").strip().upper()
            if not airport:
                raise ValueError("Airport codes in ground-hold config cannot be empty")
            if not isinstance(window_defs, list):
                raise TypeError(f"Ground-hold windows for {airport} must be provided as a list")
            parsed: List[GroundHoldWindow] = []
            for raw in window_defs:
                if not isinstance(raw, Mapping):
                    raise TypeError(f"Ground-hold window entries for {airport} must be mappings")
                rate_val = raw.get("rate_fph", default_rate)
                if rate_val is None:
                    raise ValueError(
                        f"Ground-hold window for {airport} missing 'rate_fph' and no default specified"
                    )
                rate_fph = float(rate_val)
                start = _parse_iso_datetime(raw.get("start"))
                end = _parse_iso_datetime(raw.get("end"))
                regulation_id = raw.get("regulation_id")
                parsed.append(
                    GroundHoldWindow(
                        start=start,
                        end=end,
                        rate_fph=rate_fph,
                        airport=airport,
                        regulation_id=str(regulation_id) if regulation_id is not None else None,
                    )
                )
            parsed.sort(key=lambda window: window.start)
            for prev, curr in zip(parsed, parsed[1:]):
                if curr.start < prev.end:
                    logger.warning(
                        "Ground-hold windows for %s overlap (%s-%s vs %s-%s)",
                        airport,
                        prev.start,
                        prev.end,
                        curr.start,
                        curr.end,
                    )
            windows[airport] = parsed
        return windows


__all__ = ["GroundHoldConfig", "GroundHoldWindow"]
