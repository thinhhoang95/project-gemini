from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping

import yaml

logger = logging.getLogger(__name__)


def _parse_hhmm(token: object, label: str) -> int:
    """
    Parse an HH:MM string into minutes since midnight.
    
    Args:
        token: Raw value from the configuration.
        label: Human-readable label for error messages.
    Returns:
        Minutes since midnight (0-1440). 24:00 is accepted as end-of-day.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError(f"Ground-hold windows require non-empty {label} strings")
    text = token.strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Ground-hold window {label} must be in HH:MM format: {text!r}")
    hour_str, minute_str = parts
    if not hour_str.isdigit() or not minute_str.isdigit():
        raise ValueError(f"Ground-hold window {label} must be numeric HH:MM: {text!r}")
    hour = int(hour_str)
    minute = int(minute_str)
    if hour < 0 or hour > 24 or minute < 0 or minute > 59:
        raise ValueError(f"Ground-hold window {label} out of range: {text!r}")
    if hour == 24 and minute != 0:
        raise ValueError(f"Ground-hold window {label} of 24 must use minutes 00: {text!r}")
    minutes = hour * 60 + minute
    if minutes > 1440:
        raise ValueError(f"Ground-hold window {label} exceeds 24 hours: {text!r}")
    return minutes


def _format_hhmm(minutes: int) -> str:
    hours, mins = divmod(int(minutes), 60)
    return f"{hours:02d}:{mins:02d}"


@dataclass(frozen=True, init=False)
class GroundHoldWindow:
    start_minutes: int
    end_minutes: int
    rate_fph: float
    airport: str
    regulation_id: str | None = None

    def __init__(
        self,
        *,
        start_minutes: int | float | None = None,
        end_minutes: int | float | None = None,
        start: str | int | float | None = None,
        end: str | int | float | None = None,
        rate_fph: float | int,
        airport: str,
        regulation_id: str | None = None,
    ) -> None:
        start_val = self._coerce_minutes(start_minutes, start, "start")
        end_val = self._coerce_minutes(end_minutes, end, "end")
        rate_val = float(rate_fph)
        airport_code = str(airport or "").strip().upper()
        regulation = str(regulation_id) if regulation_id is not None else None

        object.__setattr__(self, "start_minutes", start_val)
        object.__setattr__(self, "end_minutes", end_val)
        object.__setattr__(self, "rate_fph", rate_val)
        object.__setattr__(self, "airport", airport_code)
        object.__setattr__(self, "regulation_id", regulation)
        self._validate()

    @staticmethod
    def _coerce_minutes(
        minutes_value: int | float | None, raw_value: str | int | float | None, label: str
    ) -> int:
        """Normalize provided minute values, allowing either *_minutes or HH:MM strings."""
        normalized: int | None = None
        if minutes_value is not None:
            if isinstance(minutes_value, (int, float)):
                normalized = int(minutes_value)
            else:
                raise TypeError(f"Ground-hold window {label}_minutes must be numeric")

        if raw_value is not None:
            parsed = (
                _parse_hhmm(raw_value, label)
                if isinstance(raw_value, str)
                else int(raw_value)
            )
            if normalized is not None and parsed != normalized:
                raise ValueError(f"Conflicting ground-hold window {label} values provided")
            normalized = parsed

        if normalized is None:
            raise TypeError(
                f"GroundHoldWindow requires either {label}_minutes or {label} (HH:MM)"
            )
        return normalized

    def _validate(self) -> None:
        if self.start_minutes < 0 or self.end_minutes < 0:
            raise ValueError("Ground-hold window bounds must be non-negative")
        if self.start_minutes >= self.end_minutes:
            raise ValueError("Ground-hold window start must be earlier than end")
        if self.end_minutes > 1440:
            raise ValueError("Ground-hold window end cannot exceed 24:00")
        if self.rate_fph <= 0:
            raise ValueError("Ground-hold recovery rate must be positive")
        if not self.airport:
            raise ValueError("Ground-hold window must be associated with an airport")

    @property
    def start_str(self) -> str:
        return _format_hhmm(self.start_minutes)

    @property
    def end_str(self) -> str:
        return _format_hhmm(self.end_minutes)


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
                    "start": window.start_str,
                    "end": window.end_str,
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
                start_minutes = _parse_hhmm(raw.get("start"), "start")
                end_minutes = _parse_hhmm(raw.get("end"), "end")
                regulation_id = raw.get("regulation_id")
                parsed.append(
                    GroundHoldWindow(
                        start_minutes=start_minutes,
                        end_minutes=end_minutes,
                        rate_fph=rate_fph,
                        airport=airport,
                        regulation_id=str(regulation_id) if regulation_id is not None else None,
                    )
                )
            parsed.sort(key=lambda window: window.start_minutes)
            for prev, curr in zip(parsed, parsed[1:]):
                if curr.start_minutes < prev.end_minutes:
                    logger.warning(
                        "Ground-hold windows for %s overlap (%s-%s vs %s-%s)",
                        airport,
                        prev.start_str,
                        prev.end_str,
                        curr.start_str,
                        curr.end_str,
                    )
            windows[airport] = parsed
        return windows


__all__ = ["GroundHoldConfig", "GroundHoldWindow"]
