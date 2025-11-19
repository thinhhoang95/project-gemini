"""Library API for building FR arrival moments without invoking the CLI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from gemini.arrivals.flight_list_gemini import FlightListGemini
from gemini.arrivals.ground_hold_config import GroundHoldConfig
from gemini.arrivals.ground_hold_operator import GroundHoldOperator
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from gemini.gem.arrival_moments import ArrivalMoments
from gemini.propagation.tvtw_indexer import TVTWIndexer

from .flight_metadata_provider import FlightMetadataProvider
from .fr_arrival_moments_builder import build_arrival_moments_from_fr
from .fr_artifacts_loader import FRSegment, load_fr_segments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FRArrivalArtifacts:
    """Filesystem configuration for the FR-derived arrival pipeline."""

    fr_demand_path: str | Path
    fr_route_catalogue_path: str | Path
    flights_csv_path: str | Path
    tvtw_indexer_path: str | Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "fr_demand_path", Path(self.fr_demand_path))
        object.__setattr__(self, "fr_route_catalogue_path", Path(self.fr_route_catalogue_path))
        object.__setattr__(self, "flights_csv_path", Path(self.flights_csv_path))
        object.__setattr__(self, "tvtw_indexer_path", Path(self.tvtw_indexer_path))


class FRArrivalMomentsService:
    """Reusable builder that caches FR artifacts and exposes a DataFrame API."""

    def __init__(self, artifacts: FRArrivalArtifacts):
        self._artifacts = artifacts
        self._segments: List[FRSegment] | None = None
        self._volume_ids: List[str] | None = None
        self._tvtw: TVTWIndexer | None = None

    # ------------------------------------------------------------------ loaders
    def load_segments(self) -> List[FRSegment]:
        """Return cached FR segments (loading them once per service instance)."""
        segments = self._ensure_segments()
        if not segments:
            raise ValueError("No FR segments available; cannot build arrival moments.")
        return segments

    def get_volume_ids(self) -> List[str]:
        """Return a deterministic list of traffic volumes seen in the FR artifacts."""
        if self._volume_ids is None:
            segments = self._ensure_segments()
            self._volume_ids = _collect_volume_ids(segments)
        return list(self._volume_ids)

    # --------------------------------------------------------------------- API
    def get_arrival_moments(
        self,
        jitter_config: GroundJitterConfig,
        *,
        ground_hold_config: GroundHoldConfig | None = None,
        tail_tolerance: float = 1e-6,
        volume_ids: Sequence[str] | None = None,
        segments: Iterable[FRSegment] | None = None,
    ) -> pd.DataFrame:
        """Return arrival moments as a pandas DataFrame.

        Parameters
        ----------
        jitter_config:
            Precomputed ground jitter configuration object.
        ground_hold_config:
            Optional deterministic ground hold configuration.
        tail_tolerance:
            Probability cutoff when integrating jitter distributions.
        volume_ids:
            Optional subset of traffic volume IDs. Defaults to all volumes observed
            in the cached FR artifacts.
        segments:
            Optional iterable of :class:`FRSegment` rows. Useful for plugging in
            custom progress instrumentation – otherwise cached segments are used.
        """

        if segments is None:
            segments_to_iterate: Iterable[FRSegment] = self._ensure_segments()
            if not segments_to_iterate:
                raise ValueError("No FR segments available; cannot build arrival moments.")
        else:
            segments_to_iterate = segments

        flights = FlightListGemini(str(self._artifacts.flights_csv_path))
        ground_hold_delays = None
        if ground_hold_config is not None:
            hold_operator = GroundHoldOperator(flights, ground_hold_config)
            ground_hold_delays = hold_operator.compute_flight_delays()
            logger.info(
                "Computed deterministic ground holds for %d flights using supplied config",
                ground_hold_delays.num_delayed_flights,
            )

        metadata_provider = FlightMetadataProvider(
            flights,
            jitter_config,
            ground_hold_delays=ground_hold_delays,
        )
        tvtw = self._ensure_tvtw()
        volumes = list(volume_ids) if volume_ids is not None else self.get_volume_ids()
        if not volumes:
            volumes = None
        moments = build_arrival_moments_from_fr(
            segments_to_iterate,
            metadata_provider,
            tvtw,
            volume_ids=volumes,
            tail_tolerance=tail_tolerance,
        )
        return arrival_moments_to_dataframe(moments)

    def get_per_flight_ground_hold_delay(self, ground_hold_config: GroundHoldConfig) -> pd.DataFrame:
        """Return deterministic ground-hold delays per flight as a DataFrame."""
        if ground_hold_config is None:
            raise ValueError("Ground hold configuration is required to compute delays.")

        flights = FlightListGemini(str(self._artifacts.flights_csv_path))
        hold_operator = GroundHoldOperator(flights, ground_hold_config)
        ground_hold_delays = hold_operator.compute_flight_delays()
        logger.info(
            "Computed deterministic ground holds for %d flights using supplied config",
            ground_hold_delays.num_delayed_flights,
        )

        columns = ["flight_id", "ground_hold_delay_min"]
        if not flights.flight_ids:
            return pd.DataFrame(columns=columns)

        rows = [
            {"flight_id": flight_id, "ground_hold_delay_min": ground_hold_delays.get(flight_id, 0)}
            for flight_id in flights.flight_ids
        ]
        return pd.DataFrame(rows, columns=columns)

    # ----------------------------------------------------------------- helpers
    def _ensure_segments(self) -> List[FRSegment]:
        if self._segments is None:
            self._segments = load_fr_segments(
                str(self._artifacts.fr_demand_path),
                str(self._artifacts.fr_route_catalogue_path),
            )
            self._volume_ids = _collect_volume_ids(self._segments)
        return self._segments

    def _ensure_tvtw(self) -> TVTWIndexer:
        if self._tvtw is None:
            self._tvtw = TVTWIndexer.load(str(self._artifacts.tvtw_indexer_path))
        return self._tvtw


def arrival_moments_to_dataframe(moments: ArrivalMoments) -> pd.DataFrame:
    """Convert sparse ArrivalMoments dictionaries into a tidy DataFrame."""
    columns = ["volume_id", "time_bin", "lambda_mean", "lambda_var", "gamma_lag1"]
    keys = set(moments.lambda_ext.keys())
    keys.update(moments.nu_ext.keys())
    keys.update(moments.gamma_ext.keys())
    if not keys:
        return pd.DataFrame(columns=columns)
    sorted_keys = sorted(keys, key=lambda item: (item[0], item[1]))
    rows = []
    for volume_id, time_bin in sorted_keys:
        rows.append(
            {
                "volume_id": volume_id,
                "time_bin": int(time_bin),
                "lambda_mean": moments.lambda_ext.get((volume_id, time_bin), 0.0),
                "lambda_var": moments.nu_ext.get((volume_id, time_bin), 0.0),
                "gamma_lag1": moments.gamma_ext.get((volume_id, time_bin), 0.0),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _collect_volume_ids(segments: Iterable[FRSegment]) -> List[str]:
    seen = {}
    for segment in segments:
        if segment.volume_id not in seen:
            seen[segment.volume_id] = None
    return list(seen.keys())


__all__ = [
    "FRArrivalArtifacts",
    "FRArrivalMomentsService",
    "arrival_moments_to_dataframe",
]
