"""Loader for hourly kernel CSV tables used during propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from gemini.propagation.domain_types import EdgeId


@dataclass
class EdgeKernel:
    """Kernel lookup for a single directed edge."""

    edge: EdgeId
    max_lag_bins: int
    kernels: Dict[int, Dict[int, float]] = field(default_factory=dict)

    def get(self, hour_index: int, lag_bins: int) -> float:
        """Return K_{edge,h}(lag) or 0.0 if missing."""
        return self.kernels.get(int(hour_index), {}).get(int(lag_bins), 0.0)


@dataclass
class HourlyKernelTable:
    """Convenience wrapper to query hourly kernels and edge adjacency."""

    delta_minutes: int
    edges: Dict[EdgeId, EdgeKernel]
    incoming_edges: Dict[str, List[EdgeId]]
    outgoing_edges: Dict[str, List[EdgeId]]

    @classmethod
    def from_csv(cls, path: str) -> "HourlyKernelTable":
        """Load a kernel table exported by compute_hourly_kernels_cli."""
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Hourly kernel CSV is empty.")

        if "delta_minutes" not in df.columns:
            raise ValueError("Hourly kernel CSV must include 'delta_minutes'.")
        delta_values = df["delta_minutes"].unique()
        if len(delta_values) != 1:
            raise ValueError("All rows in kernel CSV must share the same delta_minutes.")
        delta_minutes = int(delta_values[0])

        edges: Dict[EdgeId, EdgeKernel] = {}
        incoming: Dict[str, List[EdgeId]] = {}
        outgoing: Dict[str, List[EdgeId]] = {}

        grouped = df.groupby(["edge_u", "edge_v"])
        for (upstream, downstream), group in grouped:
            edge = EdgeId(upstream=str(upstream), downstream=str(downstream))
            max_lag = int(group["lag_bins"].max())
            kernel = EdgeKernel(edge=edge, max_lag_bins=max_lag)
            for row in group.itertuples(index=False):
                hour = int(row.hour_index)
                lag = int(row.lag_bins)
                value = float(row.kernel_value)
                kernel.kernels.setdefault(hour, {})[lag] = value
            edges[edge] = kernel
            incoming.setdefault(edge.downstream, []).append(edge)
            outgoing.setdefault(edge.upstream, []).append(edge)

        return cls(
            delta_minutes=delta_minutes,
            edges=edges,
            incoming_edges=incoming,
            outgoing_edges=outgoing,
        )

    # ---------------------------- helpers for callers that need adjacency -----
    def get_incoming(self, volume_id: str) -> List[EdgeId]:
        return self.incoming_edges.get(volume_id, [])

    def get_outgoing(self, volume_id: str) -> List[EdgeId]:
        return self.outgoing_edges.get(volume_id, [])
