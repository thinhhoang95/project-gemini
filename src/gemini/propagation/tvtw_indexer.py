"""Minimal TVTW indexer loader used for hourly-kernel extraction."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Dict, Optional, Tuple


class TVTWIndexer:
    """Maps (traffic volume, time-bin) pairs to compact indices."""

    def __init__(self, time_bin_minutes: int, tv_id_to_idx: Optional[Dict[str, int]] = None):
        if time_bin_minutes <= 0:
            raise ValueError("time_bin_minutes must be positive.")
        if 1440 % time_bin_minutes != 0:
            raise ValueError("time_bin_minutes must divide 1440.")
        self.time_bin_minutes = int(time_bin_minutes)
        self._num_bins = 1440 // self.time_bin_minutes
        self._tv_id_to_idx = {
            str(tv_id): int(idx) for tv_id, idx in (tv_id_to_idx or {}).items()
        }
        self._idx_to_tv_id = {idx: tv_id for tv_id, idx in self._tv_id_to_idx.items()}
        self._tvtw_to_idx: Dict[Tuple[str, int], int] = {}
        self._idx_to_tvtw: Dict[int, Tuple[str, int]] = {}
        self._populate_tvtw_mappings()

    # ------------------------------------------------------------------ builders
    def _populate_tvtw_mappings(self) -> None:
        self._tvtw_to_idx.clear()
        self._idx_to_tvtw.clear()
        for tv_id, tv_idx in self._tv_id_to_idx.items():
            for time_idx in range(self._num_bins):
                global_idx = tv_idx * self._num_bins + time_idx
                key = (tv_id, time_idx)
                self._tvtw_to_idx[key] = global_idx
                self._idx_to_tvtw[global_idx] = key

    # ---------------------------------------------------------------- properties
    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def bins_per_hour(self) -> int:
        if 60 % self.time_bin_minutes != 0:
            raise ValueError(
                "time_bin_minutes must divide 60 to define hour-level kernels."
            )
        return 60 // self.time_bin_minutes

    @property
    def hours_per_day(self) -> int:
        return self.num_bins // self.bins_per_hour

    # ----------------------------------------------------------------- indexing
    def get_tvtw_index(self, tv_id: str, time_bin: int) -> Optional[int]:
        return self._tvtw_to_idx.get((str(tv_id), int(time_bin)))

    def get_tvtw_from_index(self, index: int) -> Optional[Tuple[str, int]]:
        return self._idx_to_tvtw.get(int(index))

    # ------------------------------------------------------------------- timing
    def bin_of_datetime(self, dt: datetime) -> int:
        minute_of_day = dt.hour * 60 + dt.minute
        bin_idx = minute_of_day // self.time_bin_minutes
        return min(max(int(bin_idx), 0), self.num_bins - 1)

    def minutes_to_bin(self, minutes_from_midnight: float) -> int:
        if minutes_from_midnight < 0:
            raise ValueError("minutes_from_midnight must be non-negative.")
        bin_idx = int(minutes_from_midnight // self.time_bin_minutes)
        return min(bin_idx, self.num_bins - 1)

    def datetime_to_minutes(self, dt: datetime, day_start: datetime) -> float:
        delta = dt - day_start
        return delta.total_seconds() / 60.0

    # ------------------------------------------------------------------- IO
    def to_json_dict(self) -> Dict[str, object]:
        return {
            "time_bin_minutes": self.time_bin_minutes,
            "tv_id_to_idx": self._tv_id_to_idx,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_json_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "TVTWIndexer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        minutes = int(payload.get("time_bin_minutes", 0))
        tv_map = payload.get("tv_id_to_idx") or {}
        return cls(time_bin_minutes=minutes, tv_id_to_idx=tv_map)
