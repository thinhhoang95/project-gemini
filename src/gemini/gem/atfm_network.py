"""ATFM network propagation implementing Sections 5 and 6 of the master guide."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from gemini.propagation.domain_types import EdgeId, Volume
from gemini.propagation.tvtw_indexer import TVTWIndexer
from gemini.propagation.volume_graph import VolumeGraph, VolumeLocator

from .arrival_moments import ArrivalMoments
from .hourly_kernel_table import HourlyKernelTable
from .regulation_plan import RegulationPlan


@dataclass
class VolumeTimeSeries:
    """All per-bin statistics tracked for a single volume."""

    lambda_mean: List[float]
    lambda_var: List[float]
    queue_mean: List[float]
    queue_var: List[float]
    departure_mean: List[float]
    departure_var: List[float]


@dataclass
class ATFMRunResult:
    """Results returned by the network propagation."""

    by_volume: Dict[str, VolumeTimeSeries]
    total_delay_mean: float
    total_delay_var: float


class ATFMNetworkModel:
    """Chronological propagation with Level 0/1 queueing approximations."""

    def __init__(
        self,
        *,
        tvtw: TVTWIndexer,
        volume_graph: VolumeGraph,
        kernels: HourlyKernelTable,
        arrivals: ArrivalMoments,
    ) -> None:
        if kernels.delta_minutes != tvtw.time_bin_minutes:
            raise ValueError("Kernel delta_minutes must match TVTWIndexer bin size.")
        self.tvtw = tvtw
        self.volume_graph = volume_graph
        self.kernels = kernels
        self.arrivals = arrivals
        self.num_bins = tvtw.num_bins
        self.delta_minutes = tvtw.time_bin_minutes
        self.bins_per_hour = tvtw.bins_per_hour
        self.volume_ids: List[str] = list(volume_graph.volumes.keys())

    # ---------------------------------------------------------------------- API --
    def run(self, plan: Optional[RegulationPlan] = None) -> ATFMRunResult:
        """Execute the chronological propagation with optional regulation plan."""
        if plan is None:
            capacities = self._build_unregulated_capacities()
        else:
            capacities = plan.build_capacity_matrix(
                self.volume_graph.volumes, self.num_bins
            )
        return self._run_with_capacities(capacities)

    # ----------------------------------------------------------------- internal --
    def _build_unregulated_capacities(self) -> Dict[str, List[Optional[float]]]:
        return {
            volume_id: [None] * self.num_bins for volume_id in self.volume_ids
        }

    def _run_with_capacities(
        self, capacities: Dict[str, List[Optional[float]]]
    ) -> ATFMRunResult:
        series = self._init_series()
        prev_pair_weight: Dict[str, Optional[float]] = {v: None for v in self.volume_ids}
        w_bin: Dict[Tuple[str, int], float] = {}

        for t in range(self.num_bins):
            # Step 1: assemble arrival moments
            for volume_id in self.volume_ids:
                lam, nu = self._compute_arrival_mean_var(volume_id, t, series)
                series[volume_id].lambda_mean[t] = lam
                series[volume_id].lambda_var[t] = nu

            # Step 1b: compute F1 pair/per-bin weights
            for volume_id in self.volume_ids:
                nu_t = series[volume_id].lambda_var[t]
                w_val: float
                if t < self.num_bins - 1:
                    gamma = self._compute_arrival_cov_lag1(volume_id, t, series)
                    nu_next = self._predict_next_variance(volume_id, t, series)
                    denom = max(nu_t + nu_next, 1e-6)
                    pair = 1.0 + 2.0 * gamma / denom
                    pair = _clip_weight(pair)
                    prev_pair = prev_pair_weight.get(volume_id)
                    if prev_pair is None:
                        w_val = pair
                    else:
                        w_val = 0.5 * (prev_pair + pair)
                    prev_pair_weight[volume_id] = pair
                else:
                    prev_pair = prev_pair_weight.get(volume_id)
                    w_val = prev_pair if prev_pair is not None else 1.0
                w_bin[(volume_id, t)] = _clip_weight(w_val)

            # Step 2: queue update per volume using deflated variance
            for volume_id in self.volume_ids:
                self._queue_step(volume_id, t, capacities, w_bin, series)

        total_mean, total_var = self._aggregate_delay(series)
        return ATFMRunResult(
            by_volume=series,
            total_delay_mean=total_mean,
            total_delay_var=total_var,
        )

    def _init_series(self) -> Dict[str, VolumeTimeSeries]:
        T = self.num_bins
        return {
            v: VolumeTimeSeries(
                lambda_mean=[0.0] * T,
                lambda_var=[0.0] * T,
                queue_mean=[0.0] * (T + 1),
                queue_var=[0.0] * (T + 1),
                departure_mean=[0.0] * T,
                departure_var=[0.0] * T,
            )
            for v in self.volume_ids
        }

    # ---------------------------------------------------------- arrival moments --
    def _compute_arrival_mean_var(
        self, volume_id: str, t: int, series: Dict[str, VolumeTimeSeries]
    ) -> Tuple[float, float]:
        lam = self.arrivals.mean(volume_id, t)
        nu = self.arrivals.variance(volume_id, t)

        incoming_edges = self.kernels.get_incoming(volume_id)
        if not incoming_edges:
            return lam, nu

        for edge in incoming_edges:
            edge_kernel = self.kernels.edges[edge]
            max_lag = min(edge_kernel.max_lag_bins, t)
            if max_lag <= 0:
                continue
            upstream_series = series[edge.upstream]
            for lag in range(1, max_lag + 1):
                departure_bin = t - lag
                hour_index = departure_bin // self.bins_per_hour
                kernel_val = edge_kernel.get(hour_index, lag)
                if kernel_val == 0.0:
                    continue
                dep_mean = upstream_series.departure_mean[departure_bin]
                dep_var = upstream_series.departure_var[departure_bin]
                lam += kernel_val * dep_mean
                nu += kernel_val * (1.0 - kernel_val) * dep_mean + (kernel_val**2) * dep_var
        return lam, nu

    def _compute_arrival_cov_lag1(
        self, volume_id: str, t: int, series: Dict[str, VolumeTimeSeries]
    ) -> float:
        if t >= self.num_bins - 1:
            return 0.0
        gamma = self.arrivals.covariance_lag1(volume_id, t)
        incoming_edges = self.kernels.get_incoming(volume_id)
        if not incoming_edges:
            return gamma

        for edge in incoming_edges:
            edge_kernel = self.kernels.edges[edge]
            max_lag = min(edge_kernel.max_lag_bins - 1, t)
            if max_lag <= 0:
                continue
            upstream_series = series[edge.upstream]
            for lag in range(1, max_lag + 1):
                departure_bin = t - lag
                hour_index = departure_bin // self.bins_per_hour
                k_val = edge_kernel.get(hour_index, lag)
                k_next = edge_kernel.get(hour_index, lag + 1)
                if k_val == 0.0 or k_next == 0.0:
                    continue
                dep_mean = upstream_series.departure_mean[departure_bin]
                dep_var = upstream_series.departure_var[departure_bin]
                gamma += (-dep_mean + dep_var) * k_val * k_next
        return gamma

    def _predict_next_variance(
        self, volume_id: str, t: int, series: Dict[str, VolumeTimeSeries]
    ) -> float:
        """Online proxy for ν_{t+1} used in the F1 weight."""
        next_bin = t + 1
        if next_bin >= self.num_bins:
            return series[volume_id].lambda_var[t]
        # Default to exogenous variance if provided, otherwise reuse current.
        nu_next = self.arrivals.variance(volume_id, next_bin)
        if nu_next == 0.0:
            nu_next = series[volume_id].lambda_var[t]
        return nu_next

    # ------------------------------------------------------------- queue update --
    def _queue_step(
        self,
        volume_id: str,
        t: int,
        capacities: Dict[str, List[Optional[float]]],
        w_bin: Dict[Tuple[str, int], float],
        series: Dict[str, VolumeTimeSeries],
    ) -> None:
        vol_series = series[volume_id]
        lam = vol_series.lambda_mean[t]
        nu = vol_series.lambda_var[t]
        q_mean = vol_series.queue_mean[t]
        q_var = vol_series.queue_var[t]
        w_t = w_bin.get((volume_id, t), 1.0)
        nu_deflated = w_t * nu

        capacity_list = capacities.get(volume_id)
        cap = None
        if capacity_list is not None and t < len(capacity_list):
            cap = capacity_list[t]

        if cap is None:
            # Entire backlog is released immediately when there is no cap.
            vol_series.queue_mean[t + 1] = 0.0
            vol_series.queue_var[t + 1] = 0.0
            vol_series.departure_mean[t] = lam + q_mean
            vol_series.departure_var[t] = max(nu_deflated + q_var, 0.0)
            return

        delta = lam - cap
        mu = q_mean + delta
        sigma2 = max(q_var + nu_deflated, 0.0)
        sigma = math.sqrt(max(sigma2, 1e-12))
        a = mu / sigma if sigma > 0 else 0.0
        phi = _std_normal_pdf(a)
        Phi = _std_normal_cdf(a)

        E_Q = sigma * phi + mu * Phi
        EQ2 = (mu * mu + sigma2) * Phi + mu * sigma * phi
        Var_Q = max(EQ2 - E_Q * E_Q, 0.0)

        vol_series.queue_mean[t + 1] = E_Q
        vol_series.queue_var[t + 1] = Var_Q
        D_mean = lam + q_mean - E_Q
        vol_series.departure_mean[t] = D_mean
        cov_x_y = EQ2 - mu * E_Q  # Covariance of unconstrained queue and its positive part
        D_var = sigma2 + Var_Q - 2.0 * cov_x_y
        vol_series.departure_var[t] = max(D_var, 0.0)

    # ---------------------------------------------------------- aggregation -----
    def _aggregate_delay(self, series: Dict[str, VolumeTimeSeries]) -> Tuple[float, float]:
        total_mean = 0.0
        total_var = 0.0
        dt = self.delta_minutes
        for volume_series in series.values():
            total_mean += dt * sum(volume_series.queue_mean[:-1])
            total_var += (dt * dt) * sum(volume_series.queue_var[:-1])
        return total_mean, total_var


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clip_weight(value: float) -> float:
    return max(0.6, min(1.0, value))


# ========================================================================= CLI ==
def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ATFM network propagation using hourly kernels."
    )
    parser.add_argument("--kernels-csv", required=True, help="Path to hourly kernels CSV.")
    parser.add_argument(
        "--arrivals-csv",
        required=True,
        help="CSV containing exogenous arrival moments (volume_id,time_bin,...).",
    )
    parser.add_argument(
        "--regulation-plan",
        help="Optional regulation plan YAML describing per-volume capacity reductions.",
    )
    parser.add_argument(
        "--tvtw-indexer",
        help="Optional TVTW indexer JSON. If omitted, binning is inferred from kernels.",
    )
    parser.add_argument(
        "--output-arrivals-csv",
        help="Optional path to write per-volume time series (lambda, queue, departures).",
    )
    parser.add_argument(
        "--top-volumes",
        type=int,
        default=5,
        help="Number of top-delay volumes to print in the summary (default: 5).",
    )
    return parser.parse_args(argv)


def _load_arrival_moments_csv(path: str, num_bins: int) -> ArrivalMoments:
    import pandas as pd

    df = pd.read_csv(path)
    required_cols = {"volume_id", "time_bin", "lambda_mean", "lambda_var"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Arrival CSV missing columns: {', '.join(sorted(missing))}")

    lambda_ext: Dict[Tuple[str, int], float] = {}
    nu_ext: Dict[Tuple[str, int], float] = {}
    gamma_ext: Dict[Tuple[str, int], float] = {}
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
            if isinstance(gamma_value, (int, float)):
                gamma_ext[(volume_id, time_bin)] = float(gamma_value)
    return ArrivalMoments(lambda_ext=lambda_ext, nu_ext=nu_ext, gamma_ext=gamma_ext)


def _collect_volume_ids(
    kernels: HourlyKernelTable,
    arrivals: ArrivalMoments,
    plan: Optional[RegulationPlan],
) -> List[str]:
    volume_ids = set()
    for edge in kernels.edges.keys():
        volume_ids.add(edge.upstream)
        volume_ids.add(edge.downstream)
    for volume_id, _ in arrivals.lambda_ext.keys():
        volume_ids.add(volume_id)
    if plan is not None:
        for reg in plan.regulations:
            volume_ids.add(reg.volume_id)
    return sorted(volume_ids)


def _build_volume_graph(volume_ids: Iterable[str]) -> VolumeGraph:
    import pandas as pd

    ids = sorted(set(volume_ids))
    volumes = {vid: Volume(id=vid) for vid in ids}
    geo_df = pd.DataFrame(
        {
            "traffic_volume_id": ids,
            "geometry": [None] * len(ids),
        }
    )
    locator = VolumeLocator(geo_df)
    return VolumeGraph(volumes=volumes, geo_dataframe=geo_df, locator=locator)


def _write_volume_series_csv(
    path: str, result: ATFMRunResult, num_bins: int, delta_minutes: int
) -> None:
    import pandas as pd

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
                    "departure_mean": series.departure_mean[t],
                    "departure_var": series.departure_var[t],
                    "delay_minutes_bin": delta_minutes * series.queue_mean[t],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _print_summary(result: ATFMRunResult, num_bins: int, delta_minutes: int, top_k: int) -> None:
    total_mean = result.total_delay_mean
    total_var = result.total_delay_var
    total_hours = total_mean / 60.0 if total_mean else 0.0
    print("=== ATFM Network Summary ===")
    print(f"Total Type-3 delay mean : {total_mean:,.2f} flight-minutes ({total_hours:,.2f} hours)")
    print(f"Total Type-3 delay var  : {total_var:,.2f} (flight-minutes)^2")
    delays = []
    for volume_id, series in result.by_volume.items():
        delay = delta_minutes * sum(series.queue_mean[:-1])
        if delay > 0:
            delays.append((delay, volume_id))
    if not delays:
        print("No regulated queues detected (all delays zero).")
        return
    delays.sort(reverse=True)
    print(f"Top {min(top_k, len(delays))} volumes by delay contribution:")
    for delay, volume_id in delays[:top_k]:
        print(f"  {volume_id}: {delay:,.2f} flight-minutes")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_cli_args(argv)
    kernels = HourlyKernelTable.from_csv(args.kernels_csv)

    if args.tvtw_indexer:
        tvtw = TVTWIndexer.load(args.tvtw_indexer)
        if tvtw.time_bin_minutes != kernels.delta_minutes:
            raise ValueError("tvtw_indexer bin size must match kernels delta_minutes.")
    else:
        tvtw = TVTWIndexer(time_bin_minutes=kernels.delta_minutes)

    arrivals = _load_arrival_moments_csv(args.arrivals_csv, tvtw.num_bins)
    plan = RegulationPlan.load(args.regulation_plan, tvtw) if args.regulation_plan else None
    volume_ids = _collect_volume_ids(kernels, arrivals, plan)
    volume_graph = _build_volume_graph(volume_ids)

    model = ATFMNetworkModel(
        tvtw=tvtw,
        volume_graph=volume_graph,
        kernels=kernels,
        arrivals=arrivals,
    )
    result = model.run(plan)
    _print_summary(result, model.num_bins, model.delta_minutes, args.top_volumes)
    if args.output_arrivals_csv:
        _write_volume_series_csv(
            args.output_arrivals_csv, result, model.num_bins, model.delta_minutes
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
