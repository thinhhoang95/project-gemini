"""Hourly kernel estimator implementing Steps K1–K3."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .domain_types import EdgeId, TraversalRecord


class HourlyKernelEstimator:
    """Accumulates traversal counts and produces shrunken hourly kernels."""

    def __init__(
        self,
        *,
        delta_minutes: int,
        num_bins: int,
        bins_per_hour: int,
        max_lag_bins: int,
        shrinkage_M: float = 75.0,
        min_traversals_per_edge: int = 1,
        emit_empty_hours: bool = True,
    ) -> None:
        if bins_per_hour <= 0:
            raise ValueError("bins_per_hour must be positive.")
        self.delta_minutes = int(delta_minutes)
        self.num_bins = int(num_bins)
        self.bins_per_hour = int(bins_per_hour)
        self.max_lag_bins = int(max_lag_bins)
        self.shrinkage_M = float(shrinkage_M)
        self.min_traversals_per_edge = int(min_traversals_per_edge)
        self.emit_empty_hours = emit_empty_hours
        self.hours_per_day = self.num_bins // self.bins_per_hour

        self.edge_hour_counts: Dict[Tuple[EdgeId, int], float] = defaultdict(float)
        self.edge_hour_lag_counts: Dict[Tuple[EdgeId, int], Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.edge_hour_lost_counts: Dict[Tuple[EdgeId, int], float] = defaultdict(float)
        self.edge_totals: Dict[EdgeId, float] = defaultdict(float)
        self.edge_lag_totals: Dict[EdgeId, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.edge_lost_totals: Dict[EdgeId, float] = defaultdict(float)
        self.total_records = 0.0
        self.total_departures = 0.0
        self.total_censored = 0.0
        self.dropped_records = 0

    # ------------------------------------------------------------------ ingestion
    def add_traversal(self, record: TraversalRecord) -> None:
        """Consume a traversal if it satisfies the lag constraints."""
        arrival_observed = getattr(record, "arrival_observed", True)
        if record.dep_bin < 0 or record.dep_bin >= self.num_bins:
            self.dropped_records += 1
            return
        if record.hour_index < 0:
            self.dropped_records += 1
            return

        if arrival_observed:
            if record.arr_bin < 0 or record.arr_bin >= self.num_bins:
                self.dropped_records += 1
                return
            if record.lag_bins <= 0 or record.lag_bins > self.max_lag_bins:
                self.dropped_records += 1
                return

        weight = float(getattr(record, "weight", 1.0))
        if weight <= 0.0:
            return

        key = (record.edge, record.hour_index)
        self.edge_hour_counts[key] += weight
        self.edge_totals[record.edge] += weight
        self.total_departures += weight

        if arrival_observed:
            self.edge_hour_lag_counts[key][record.lag_bins] += weight
            self.edge_lag_totals[record.edge][record.lag_bins] += weight
            self.total_records += weight
        else:
            self.edge_hour_lost_counts[key] += weight
            self.edge_lost_totals[record.edge] += weight
            self.total_censored += weight

    # ---------------------------------------------------------------- computation
    def finalize_kernels(self) -> List[Dict[str, float]]:
        """Return a table-like list with per-edge hourly kernels."""
        rows: List[Dict[str, float]] = []
        if not self.edge_totals:
            return rows

        for edge, total_traversals in self.edge_totals.items():
            if total_traversals < self.min_traversals_per_edge:
                continue
            global_lag_counts = self.edge_lag_totals[edge]
            lost_count_edge = self.edge_lost_totals.get(edge, 0.0)
            hours: Iterable[int]
            if self.emit_empty_hours:
                hours = range(self.hours_per_day)
            else:
                hours = sorted(
                    hour for (edge_key, hour) in self.edge_hour_counts.keys() if edge_key == edge
                )

            for hour in hours:
                N_eh = self.edge_hour_counts.get((edge, hour), 0.0)
                if not self.emit_empty_hours and N_eh == 0:
                    continue
                alpha = 0.0
                if N_eh > 0:
                    alpha = N_eh / (N_eh + self.shrinkage_M)
                lag_counts = self.edge_hour_lag_counts.get((edge, hour), {})
                lost_count_hour = self.edge_hour_lost_counts.get((edge, hour), 0.0)
                lost_fraction_hour = (lost_count_hour / N_eh) if N_eh > 0 else 0.0

                for lag in range(1, self.max_lag_bins + 1):
                    local_prob = (lag_counts.get(lag, 0.0) / N_eh) if N_eh > 0 else 0.0
                    global_prob = (
                        global_lag_counts.get(lag, 0.0) / total_traversals
                        if total_traversals > 0
                        else 0.0
                    )
                    kernel_value = alpha * local_prob + (1.0 - alpha) * global_prob
                    if not self.emit_empty_hours and N_eh == 0 and kernel_value == 0.0:
                        continue
                    rows.append(
                        {
                            "edge_u": edge.upstream,
                            "edge_v": edge.downstream,
                            "edge_id": str(edge),
                            "hour_index": hour,
                            "lag_bins": lag,
                            "lag_minutes": lag * self.delta_minutes,
                            "kernel_value": kernel_value,
                            "traversal_count_hour": N_eh,
                            "lag_count_hour": lag_counts.get(lag, 0.0),
                            "traversal_count_edge": total_traversals,
                            "lost_count_hour": lost_count_hour,
                            "lost_fraction_hour": lost_fraction_hour,
                            "lost_count_edge": lost_count_edge,
                            "alpha": alpha,
                            "delta_minutes": self.delta_minutes,
                        }
                    )
        return rows
