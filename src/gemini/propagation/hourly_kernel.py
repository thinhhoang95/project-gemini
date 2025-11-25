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

        # 1. Identify retained edges first to ensure we only normalize over valid routes
        retained_edges = {
            edge
            for edge, total in self.edge_totals.items()
            if total >= self.min_traversals_per_edge
        }

        # 2. Pre-compute upstream totals for routing probability normalization
        # We calculate totals only over retained edges so that probability mass
        # isn't lost to dropped "noise" edges.
        upstream_totals: Dict[str, float] = defaultdict(float)
        upstream_hour_totals: Dict[Tuple[str, int], float] = defaultdict(float)

        for edge in retained_edges:
            upstream_totals[edge.upstream] += self.edge_totals[edge]

        for (edge, hour), count in self.edge_hour_counts.items():
            if edge in retained_edges:
                upstream_hour_totals[(edge.upstream, hour)] += count

        for edge in retained_edges:
            total_traversals = self.edge_totals[edge]
            global_lag_counts = self.edge_lag_totals[edge]
            lost_count_edge = self.edge_lost_totals.get(edge, 0.0)
            
            hours: Iterable[int]
            if self.emit_empty_hours:
                hours = range(self.hours_per_day)
            else:
                hours = sorted(
                    hour for (edge_key, hour) in self.edge_hour_counts.keys() if edge_key == edge
                )

            # Global routing probability P(v | u)
            total_u = upstream_totals.get(edge.upstream, 0.0)
            global_routing_prob = (total_traversals / total_u) if total_u > 0 else 0.0

            for hour in hours:
                N_eh = self.edge_hour_counts.get((edge, hour), 0.0)
                if not self.emit_empty_hours and N_eh == 0:
                    continue

                # --- Routing Probability P(v | u, h) ---
                # Shrink hourly routing choice towards global routing choice.
                N_uh = upstream_hour_totals.get((edge.upstream, hour), 0.0)
                
                alpha_route = 0.0
                if N_uh > 0:
                    alpha_route = N_uh / (N_uh + self.shrinkage_M)
                
                local_routing_prob = (N_eh / N_uh) if N_uh > 0 else 0.0
                routing_prob = alpha_route * local_routing_prob + (1.0 - alpha_route) * global_routing_prob

                # --- Lag Probability P(lag | u->v, h) ---
                # Shrink hourly lag distribution towards global lag distribution.
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
                    
                    # Conditional kernel: P(lag | u->v, h)
                    kernel_cond = alpha * local_prob + (1.0 - alpha) * global_prob

                    # Joint kernel: P(v, lag | u, h) = P(lag | u->v, h) * P(v | u, h)
                    kernel_value = kernel_cond * routing_prob

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
