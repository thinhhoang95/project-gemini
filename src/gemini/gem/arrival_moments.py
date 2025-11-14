"""Container for exogenous arrival moments supplied by upstream models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple


Key = Tuple[str, int]


@dataclass
class ArrivalMoments:
    """Sparse mapping from (volume, bin) to exogenous arrival statistics."""

    lambda_ext: Dict[Key, float] = field(default_factory=dict)
    nu_ext: Dict[Key, float] = field(default_factory=dict)
    gamma_ext: Dict[Key, float] = field(default_factory=dict)

    def mean(self, volume_id: str, bin_index: int) -> float:
        """Return λ^{ext}_{v,t}."""
        return self.lambda_ext.get((volume_id, bin_index), 0.0)

    def variance(self, volume_id: str, bin_index: int) -> float:
        """Return ν^{ext}_{v,t}."""
        return self.nu_ext.get((volume_id, bin_index), 0.0)

    def covariance_lag1(self, volume_id: str, bin_index: int) -> float:
        """Return γ^{ext}_{v,t,t+1}."""
        return self.gamma_ext.get((volume_id, bin_index), 0.0)

    @classmethod
    def zeros(cls, volume_ids: Iterable[str], num_bins: int) -> "ArrivalMoments":
        """Convenience builder that pre-initialises zero-valued dictionaries."""
        lam: Dict[Key, float] = {}
        nu: Dict[Key, float] = {}
        gamma: Dict[Key, float] = {}
        for volume_id in volume_ids:
            for t in range(num_bins):
                key = (volume_id, t)
                lam[key] = 0.0
                nu[key] = 0.0
                if t < num_bins - 1:
                    gamma[key] = 0.0
        return cls(lambda_ext=lam, nu_ext=nu, gamma_ext=gamma)
