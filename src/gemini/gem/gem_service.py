"""High-level API that ties FR arrivals to the ATFM network model.

This module provides the :class:`GemService` class, which orchestrates the end-to-end
GEM (Ground Entry Model) pipeline. It combines FR (FlightRadar) arrival data with
hourly kernel tables and an ATFM (Air Traffic Flow Management) network model to
simulate traffic propagation through the airspace network.

The service takes flight arrival data, applies ground jitter (entry-time noise) and
optional ground holds, then propagates traffic through the network using precomputed
kernels. The result includes per-volume statistics for queues, departures, and delays
across all time bins.

Inputs
------
The service requires several input components:

1. **FRArrivalArtifacts**: Configuration object specifying paths to:
   - FR demand data (gem_artifacts_demand_all)
   - FR route catalogue (gem_artifacts_route_catalogue_all)
   - Flight metadata CSV (flights CSV)
   - TVTW indexer JSON (time-volume-time-window indexer)

2. **HourlyKernelTable**: Precomputed kernel table loaded from CSV that defines
   traffic propagation probabilities between volumes across time.

3. **GroundJitterConfig**: Required configuration for entry-time noise modeling,
   typically loaded from JSON.

4. **Optional inputs**:
   - GroundHoldConfig: Deterministic ground holds to apply before entry
   - RegulationPlan: Per-volume capacity restrictions (YAML format)
   - volume_ids: Subset of traffic volumes to process
   - tail_tolerance: Probability cutoff for jitter distribution integration

Outputs
-------
The service returns a :class:`GemRunResult` containing:

1. **atfm_result**: :class:`ATFMRunResult` with network propagation results including:
   - by_volume: Dictionary mapping volume IDs to :class:`VolumeTimeSeries` objects
   - total_delay_mean: Aggregate mean delay across all volumes
   - total_delay_var: Aggregate delay variance

2. **arrival_moments**: :class:`ArrivalMoments` object with sparse arrival statistics
   (lambda_mean, lambda_var, gamma_lag1) keyed by (volume_id, time_bin).

3. **arrival_dataframe**: pandas DataFrame with columns:
   - volume_id: Traffic volume identifier
   - time_bin: Time bin index
   - lambda_mean: Mean arrival rate
   - lambda_var: Arrival rate variance
   - gamma_lag1: Lag-1 autocovariance (if available)

4. **volume_series**: pandas DataFrame with per-volume, per-time-bin statistics:
   - volume_id: Traffic volume identifier
   - time_bin: Time bin index
   - lambda_mean: Mean arrival rate at this volume/time
   - lambda_var: Arrival rate variance
   - queue_mean: Mean queue length
   - queue_var: Queue length variance
   - queue_cov_lag1: Queue lag-1 autocovariance
   - departure_mean: Mean departure rate
   - departure_var: Departure rate variance
   - departure_cov_lag1: Departure lag-1 autocovariance
   - queue_reflection_slope: Queue reflection slope parameter
   - delay_minutes_bin: Delay in minutes for this bin

5. **tvtw**: The :class:`TVTWIndexer` used for time binning.

6. **volume_graph**: The :class:`VolumeGraph` representing the network topology.

Example Usage
-------------
Basic usage without regulations:

.. code-block:: python

    from gemini.arr.fr_arrivals_service import FRArrivalArtifacts
    from gemini.gem.gem_service import GemService
    from gemini.gem.hourly_kernel_table import HourlyKernelTable
    from gemini.arrivals.ground_jitter_config import GroundJitterConfig

    # 1. Configure artifact paths
    artifacts = FRArrivalArtifacts(
        fr_demand_path="data/fr/gem_artifacts_demand_all",
        fr_route_catalogue_path="data/fr/gem_artifacts_route_catalogue_all",
        flights_csv_path="data/flights_20230717_0000-2359.csv",
        tvtw_indexer_path="data/tvtw_indexer.json",
    )

    # 2. Load hourly kernels
    kernels = HourlyKernelTable.from_csv("data/hourly_kernels.csv")

    # 3. Load ground jitter configuration
    jitter_config = GroundJitterConfig.from_json("data/fr/ground_jitter_default.json")

    # 4. Create and run the service
    service = GemService(artifacts, kernels=kernels)
    result = service.run(jitter_config)

    # 5. Access results
    print(result.volume_series.head())  # Per-volume queue/departure stats
    print(f"Total delay mean: {result.atfm_result.total_delay_mean}")
    print(f"Total delay variance: {result.atfm_result.total_delay_var}")

With ground holds and regulations:

.. code-block:: python

    from gemini.arrivals.ground_hold_config import GroundHoldConfig
    from gemini.gem.regulation_plan import RegulationPlan
    from gemini.propagation.tvtw_indexer import TVTWIndexer

    # Load additional configurations
    ground_hold_config = GroundHoldConfig.from_yaml("data/ground_hold_config.yaml")
    tvtw = TVTWIndexer.load("data/tvtw_indexer.json")
    regulation_plan = RegulationPlan.load("data/regulation_plan.yaml", tvtw)

    # Run with all options
    service = GemService(artifacts, kernels=kernels, tvtw=tvtw)
    result = service.run(
        jitter_config,
        ground_hold_config=ground_hold_config,
        regulation_plan=regulation_plan,
        volume_ids=["KPVD_1", "KJFK_3"],  # Optional: subset of volumes
        tail_tolerance=1e-6,
    )

    # Analyze specific volume statistics
    volume_stats = result.volume_series[
        result.volume_series["volume_id"] == "KPVD_1"
    ]
    print(volume_stats[["time_bin", "queue_mean", "departure_mean", "delay_minutes_bin"]])

Notes
-----
- The TVTW indexer's time_bin_minutes must match the hourly kernels' delta_minutes.
- If a regulation plan is provided, its time_bin_minutes must also match the TVTW indexer.
- The service automatically builds a volume graph from volumes present in kernels,
  arrivals, and the regulation plan.
- All volumes implied by kernels and regulations are included in network propagation,
  even if volume_ids restricts the arrival data request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from gemini.arr.fr_arrivals_service import FRArrivalArtifacts, FRArrivalMomentsService
from gemini.arrivals.ground_hold_config import GroundHoldConfig
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from gemini.gem.arrival_moments import ArrivalMoments
from gemini.gem.atfm_network import ATFMNetworkModel, ATFMRunResult
from gemini.gem.hourly_kernel_table import HourlyKernelTable
from gemini.gem.regulation_plan import RegulationPlan
from gemini.propagation.domain_types import Volume
from gemini.propagation.tvtw_indexer import TVTWIndexer
from gemini.propagation.volume_graph import VolumeGraph, VolumeLocator


@dataclass(frozen=True)
class GemRunResult:
    """Structured payload returned by :class:`GemService.run`."""

    atfm_result: ATFMRunResult
    arrival_moments: ArrivalMoments
    arrival_dataframe: pd.DataFrame
    volume_series: pd.DataFrame
    tvtw: TVTWIndexer
    volume_graph: VolumeGraph


class GemService:
    """Orchestrates FR arrivals, hourly kernels, and the ATFM network model.

    Example
    -------
    >>> artifacts = FRArrivalArtifacts(...paths...)
    >>> kernels = HourlyKernelTable.from_csv("hourly_kernels.csv")
    >>> jitter = GroundJitterConfig.from_json("ground_jitter.json")
    >>> plan = RegulationPlan.load("plan.yaml", TVTWIndexer.load("tvtw.json"))
    >>> service = GemService(artifacts, kernels=kernels)
    >>> result = service.run(jitter, regulation_plan=plan)
    >>> result.volume_series.head()  # tidy per-volume queue/departure stats
    """

    def __init__(
        self,
        artifacts: FRArrivalArtifacts,
        *,
        kernels: HourlyKernelTable,
        tvtw: TVTWIndexer | None = None,
        arrival_service: FRArrivalMomentsService | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._kernels = kernels
        self._arrival_service = arrival_service or FRArrivalMomentsService(artifacts)
        self._tvtw = tvtw or TVTWIndexer.load(str(self._artifacts.tvtw_indexer_path))

        if self._tvtw.time_bin_minutes != self._kernels.delta_minutes:
            raise ValueError("Hourly kernels must share the TVTW time_bin_minutes.")

    def run(
        self,
        jitter_config: GroundJitterConfig,
        *,
        ground_hold_config: GroundHoldConfig | None = None,
        regulation_plan: RegulationPlan | None = None,
        volume_ids: Sequence[str] | None = None,
        tail_tolerance: float = 1e-6,
    ) -> GemRunResult:
        """Execute the end-to-end GEM pipeline.

        Parameters
        ----------
        jitter_config:
            Precomputed :class:`GroundJitterConfig` describing entry-time noise.
        ground_hold_config:
            Optional deterministic holds to apply before entry.
        regulation_plan:
            Optional per-volume capacity plan object; passed straight to
            :meth:`ATFMNetworkModel.run`.
        volume_ids:
            Optional subset of traffic volumes to request from the FR arrival
            builder. All volumes implied by kernels and the regulation plan will
            still be included in the network propagation.
        tail_tolerance:
            Probability cutoff when integrating jitter distributions.
        """

        if regulation_plan is not None and regulation_plan.time_bin_minutes != self._tvtw.time_bin_minutes:
            raise ValueError("Regulation plan time_bin_minutes must match the TVTW indexer.")

        arrival_df = self._arrival_service.get_arrival_moments(
            jitter_config,
            ground_hold_config=ground_hold_config,
            tail_tolerance=tail_tolerance,
            volume_ids=volume_ids,
        )
        arrival_moments = _dataframe_to_arrival_moments(arrival_df, self._tvtw.num_bins)
        volume_graph = self._build_volume_graph(arrival_moments, regulation_plan)

        model = ATFMNetworkModel(
            tvtw=self._tvtw,
            volume_graph=volume_graph,
            kernels=self._kernels,
            arrivals=arrival_moments,
        )
        atfm_result = model.run(regulation_plan)
        volume_series = _build_volume_series_dataframe(
            atfm_result,
            model.num_bins,
            model.delta_minutes,
        )

        return GemRunResult(
            atfm_result=atfm_result,
            arrival_moments=arrival_moments,
            arrival_dataframe=arrival_df,
            volume_series=volume_series,
            tvtw=self._tvtw,
            volume_graph=volume_graph,
        )

    # ----------------------------------------------------------------- helpers
    def _build_volume_graph(
        self, arrivals: ArrivalMoments, plan: RegulationPlan | None
    ) -> VolumeGraph:
        volume_ids = _collect_volume_ids(self._kernels, arrivals, plan)
        if not volume_ids:
            raise ValueError("No traffic volumes detected in kernels, arrivals, or regulation plan.")

        volumes = {vid: Volume(id=vid) for vid in volume_ids}
        geo_df = pd.DataFrame(
            {
                "traffic_volume_id": volume_ids,
                "geometry": [None] * len(volume_ids),
            }
        )
        locator = VolumeLocator(geo_df)
        return VolumeGraph(volumes=volumes, geo_dataframe=geo_df, locator=locator)


def _dataframe_to_arrival_moments(df: pd.DataFrame, num_bins: int) -> ArrivalMoments:
    """Convert an arrival DataFrame into :class:`ArrivalMoments`."""
    lambda_ext: dict[tuple[str, int], float] = {}
    nu_ext: dict[tuple[str, int], float] = {}
    gamma_ext: dict[tuple[str, int], float] = {}

    if df is None or df.empty:
        return ArrivalMoments(lambda_ext=lambda_ext, nu_ext=nu_ext, gamma_ext=gamma_ext)

    required_cols = {"volume_id", "time_bin", "lambda_mean", "lambda_var"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Arrival DataFrame missing columns: {', '.join(sorted(missing))}")

    gamma_col = "gamma_lag1" if "gamma_lag1" in df.columns else None

    for row in df.itertuples(index=False):
        volume_id = str(row.volume_id)
        time_bin = int(row.time_bin)
        if time_bin < 0 or time_bin >= num_bins:
            continue
        lambda_ext[(volume_id, time_bin)] = float(row.lambda_mean)
        nu_ext[(volume_id, time_bin)] = float(row.lambda_var)
        if gamma_col is not None:
            gamma_value = getattr(row, gamma_col)
            if isinstance(gamma_value, (int, float)) and not pd.isna(gamma_value):
                gamma_ext[(volume_id, time_bin)] = float(gamma_value)

    return ArrivalMoments(lambda_ext=lambda_ext, nu_ext=nu_ext, gamma_ext=gamma_ext)


def _collect_volume_ids(
    kernels: HourlyKernelTable, arrivals: ArrivalMoments, plan: RegulationPlan | None
) -> List[str]:
    volume_ids = set()
    for edge in kernels.edges.keys():
        volume_ids.add(edge.upstream)
        volume_ids.add(edge.downstream)
    for volume_id, _ in arrivals.lambda_ext.keys():
        volume_ids.add(volume_id)
    for volume_id, _ in arrivals.nu_ext.keys():
        volume_ids.add(volume_id)
    for volume_id, _ in arrivals.gamma_ext.keys():
        volume_ids.add(volume_id)
    if plan is not None:
        for reg in plan.regulations:
            volume_ids.add(reg.volume_id)
    return sorted(volume_ids)


def _build_volume_series_dataframe(
    result: ATFMRunResult, num_bins: int, delta_minutes: int
) -> pd.DataFrame:
    rows = []
    for volume_id, series in result.by_volume.items():
        for t in range(num_bins):
            rows.append(
                {
                    "volume_id": volume_id,
                    "time_bin": t,
                    "lambda_mean": series.lambda_mean[t],
                    "lambda_var": series.lambda_var[t],
                    "queue_mean": series.queue_mean[t],
                    "queue_var": series.queue_var[t],
                    "queue_cov_lag1": series.queue_cov_lag1[t] if t < len(series.queue_cov_lag1) else 0.0,
                    "departure_mean": series.departure_mean[t],
                    "departure_var": series.departure_var[t],
                    "departure_cov_lag1": series.departure_cov_lag1[t],
                    "queue_reflection_slope": series.queue_reflection_slope[t],
                    "delay_minutes_bin": delta_minutes * series.queue_mean[t],
                }
            )
    return pd.DataFrame(rows)


__all__ = ["GemRunResult", "GemService"]
