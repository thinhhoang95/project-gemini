from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Mapping, Tuple


def _normalize_delay(value: object) -> int:
    try:
        delay_int = int(value)
    except Exception as exc:
        raise ValueError(f"Invalid delay value: {value!r}") from exc
    if delay_int < 0:
        raise ValueError("Delay minutes cannot be negative")
    return delay_int


@dataclass
class DelayAssignmentGemini:
    """Lightweight container for per-flight delay assignments."""

    delays: Mapping[str, int] | None = None
    regulation_id: str | None = None
    _data: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        incoming = dict(self.delays or {})
        self._data = {}
        for flight_id, delay in incoming.items():
            self._data[str(flight_id)] = _normalize_delay(delay)

    # --- mapping protocol -----------------------------------------------------
    def __getitem__(self, flight_id: str) -> int:
        return self._data[str(flight_id)]

    def __setitem__(self, flight_id: str, delay_minutes: int) -> None:
        self._data[str(flight_id)] = _normalize_delay(delay_minutes)

    def __contains__(self, flight_id: object) -> bool:
        return flight_id in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> Iterable[Tuple[str, int]]:
        return self._data.items()

    def get(self, flight_id: str, default: int = 0) -> int:
        return self._data.get(str(flight_id), default)

    # --- helpers --------------------------------------------------------------
    def nonzero_items(self) -> Iterable[Tuple[str, int]]:
        return ((flight_id, delay) for flight_id, delay in self._data.items() if delay > 0)

    @property
    def num_delayed_flights(self) -> int:
        return sum(1 for _ in self.nonzero_items())

    @property
    def total_delay_minutes(self) -> int:
        return sum(delay for _fid, delay in self.nonzero_items())

    def as_dict(self) -> Dict[str, int]:
        return dict(self._data)


__all__ = ["DelayAssignmentGemini"]
