from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class StochasticTrafficCount:
    """Dense representation of stochastic traffic counts across all TVTWs."""

    mean_vector: np.ndarray
    variance_vector: np.ndarray
    tv_id_by_index: Dict[int, str]
    num_time_bins: int
    time_bin_minutes: int
    _tv_idx_by_id: Dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mean_vector.shape != self.variance_vector.shape:
            raise ValueError("Mean and variance vectors must share the same shape")
        if self.mean_vector.ndim != 1:
            raise ValueError("Mean and variance vectors must be 1-D arrays")
        self._tv_idx_by_id = {tv_id: idx for idx, tv_id in self.tv_id_by_index.items()}

    @property
    def num_entries(self) -> int:
        return int(self.mean_vector.size)

    @property
    def num_traffic_volumes(self) -> int:
        return len(self.tv_id_by_index)

    def total_mean(self) -> float:
        return float(self.mean_vector.sum())

    def total_variance(self) -> float:
        return float(self.variance_vector.sum())

    def slice_index(self, tv_index: int, time_bin: int) -> int:
        return tv_index * self.num_time_bins + time_bin

    def mean_for(self, tv_id: str, time_bin: int) -> float:
        idx = self._lookup_tv_index(tv_id)
        return float(self.mean_vector[self.slice_index(idx, time_bin)])

    def variance_for(self, tv_id: str, time_bin: int) -> float:
        idx = self._lookup_tv_index(tv_id)
        return float(self.variance_vector[self.slice_index(idx, time_bin)])

    def as_volume_dict(self, drop_zeros: bool = True) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Return nested mapping: tv_id -> time_bin -> (mean, variance)."""
        result: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for tv_index, tv_id in self.tv_id_by_index.items():
            base = tv_index * self.num_time_bins
            tv_slice = {}
            for time_bin in range(self.num_time_bins):
                mu = float(self.mean_vector[base + time_bin])
                var = float(self.variance_vector[base + time_bin])
                if drop_zeros and abs(mu) < 1e-12 and abs(var) < 1e-12:
                    continue
                tv_slice[time_bin] = (mu, var)
            if tv_slice:
                result[tv_id] = tv_slice
        return result

    def top_slices(self, top_n: int = 10) -> List[Tuple[str, int, float, float]]:
        """Return the top-N (tv, time_bin) slices ranked by mean count."""
        entries: List[Tuple[str, int, float, float]] = []
        for tv_index, tv_id in self.tv_id_by_index.items():
            base = tv_index * self.num_time_bins
            for time_bin in range(self.num_time_bins):
                mu = float(self.mean_vector[base + time_bin])
                if mu <= 0:
                    continue
                var = float(self.variance_vector[base + time_bin])
                entries.append((tv_id, time_bin, mu, var))
        entries.sort(key=lambda item: item[2], reverse=True)
        return entries[:top_n]

    def _lookup_tv_index(self, tv_id: str) -> int:
        try:
            return self._tv_idx_by_id[tv_id]
        except KeyError as exc:
            raise KeyError(f"Unknown traffic volume id '{tv_id}'") from exc

