from __future__ import annotations

import pandas as pd
import pytest

from gemini.arr.fr_arrivals_service import FRArrivalArtifacts
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from gemini.gem.gem_service import GemService
from gemini.gem.hourly_kernel_table import EdgeKernel, HourlyKernelTable
from gemini.gem.regulation_plan import RegulationPlan, VolumeRegulation
from gemini.propagation.domain_types import EdgeId
from gemini.propagation.tvtw_indexer import TVTWIndexer


class StubArrivalService:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.calls: list[dict[str, object]] = []

    def get_arrival_moments(
        self,
        jitter_config,
        *,
        ground_hold_config=None,
        tail_tolerance: float = 1e-6,
        volume_ids=None,
    ):
        self.calls.append(
            {
                "jitter_config": jitter_config,
                "ground_hold_config": ground_hold_config,
                "tail_tolerance": tail_tolerance,
                "volume_ids": list(volume_ids) if volume_ids is not None else None,
            }
        )
        return self.dataframe


def _make_kernels(delta_minutes: int = 60, edges: dict[EdgeId, EdgeKernel] | None = None) -> HourlyKernelTable:
    edges = edges or {}
    incoming = {}
    outgoing = {}
    for edge in edges.keys():
        incoming.setdefault(edge.downstream, []).append(edge)
        outgoing.setdefault(edge.upstream, []).append(edge)
    return HourlyKernelTable(
        delta_minutes=delta_minutes,
        edges=edges,
        incoming_edges=incoming,
        outgoing_edges=outgoing,
    )


def _make_jitter() -> GroundJitterConfig:
    return GroundJitterConfig.from_mapping(
        {"default": {"00:00-24:00": {"p_hurdle": 0.0, "mean": 1.0, "std": 1.0}}}
    )


def _make_artifacts() -> FRArrivalArtifacts:
    return FRArrivalArtifacts(
        fr_demand_path="demand.csv",
        fr_route_catalogue_path="route.csv",
        flights_csv_path="flights.csv",
        tvtw_indexer_path="tvtw.json",
    )


def test_gem_service_returns_volume_series():
    arrival_df = pd.DataFrame(
        [
            {"volume_id": "V1", "time_bin": 0, "lambda_mean": 2.0, "lambda_var": 1.0, "gamma_lag1": 0.0},
        ]
    )
    service = GemService(
        _make_artifacts(),
        kernels=_make_kernels(),
        tvtw=TVTWIndexer(time_bin_minutes=60),
        arrival_service=StubArrivalService(arrival_df),
    )

    result = service.run(_make_jitter())

    assert not result.volume_series.empty
    assert len(result.volume_series) == result.tvtw.num_bins  # one volume, all bins
    first_bin = result.volume_series[result.volume_series["time_bin"] == 0].iloc[0]
    assert first_bin["lambda_mean"] == pytest.approx(2.0)
    assert first_bin["departure_mean"] == pytest.approx(2.0)
    assert result.atfm_result.total_delay_mean == pytest.approx(0.0)


def test_gem_service_applies_regulation_plan():
    arrival_df = pd.DataFrame(
        [
            {"volume_id": "V1", "time_bin": 0, "lambda_mean": 3.0, "lambda_var": 1.0},
        ]
    )
    plan = RegulationPlan(
        time_bin_minutes=60,
        regulations=[
            VolumeRegulation(volume_id="V1", start_bin=0, end_bin=1, capacity_per_bin=0.5),
        ],
    )
    service = GemService(
        _make_artifacts(),
        kernels=_make_kernels(),
        tvtw=TVTWIndexer(time_bin_minutes=60),
        arrival_service=StubArrivalService(arrival_df),
    )

    result = service.run(_make_jitter(), regulation_plan=plan)
    series = result.atfm_result.by_volume["V1"]

    assert series.queue_mean[1] > 0.0  # backlog should accumulate
    assert series.departure_mean[0] < 3.0  # capacity should bind
    assert result.atfm_result.total_delay_mean > 0.0


def test_volume_filter_forwarded_and_kernel_volumes_preserved():
    edge = EdgeId("U", "D")
    edge_kernel = EdgeKernel(edge=edge, max_lag_bins=1, kernels={0: {1: 0.25}})
    arrival_df = pd.DataFrame(
        [
            {"volume_id": "D", "time_bin": 0, "lambda_mean": 1.0, "lambda_var": 0.5},
        ]
    )
    arrival_service = StubArrivalService(arrival_df)
    service = GemService(
        _make_artifacts(),
        kernels=_make_kernels(edges={edge: edge_kernel}),
        tvtw=TVTWIndexer(time_bin_minutes=60),
        arrival_service=arrival_service,
    )

    result = service.run(_make_jitter(), volume_ids=["D"])

    assert arrival_service.calls[-1]["volume_ids"] == ["D"]
    assert set(result.volume_graph.volumes.keys()) == {"U", "D"}
    assert len(result.volume_series) == result.tvtw.num_bins * 2  # both kernel volumes present
