"""Compute per-bin entry probabilities for a flight/route segment."""

from __future__ import annotations

import math
from typing import Iterable, Iterator, Tuple

from gemini.arrivals.ground_jitter_config import GroundJitterDistribution
from gemini.propagation.tvtw_indexer import TVTWIndexer


def iter_segment_bin_probabilities(
    baseline_entry_minutes: float,
    distribution: GroundJitterDistribution,
    tvtw: TVTWIndexer,
    *,
    tail_tolerance: float = 1e-6,
    max_bins: int | None = None,
) -> Iterator[Tuple[int, float]]:
    """Yield ``(time_bin, probability)`` for one (flight, route, volume) segment.

    Parameters
    ----------
    baseline_entry_minutes:
        Deterministic entry time ``T^0 = s_f + tau_{f,r->v}`` relative to the day
        origin used by ``TVTWIndexer``.
    distribution:
        Ground-jitter distribution providing ``interval_probability``.
    tvtw:
        Defines bin size ``Δ`` and the finite horizon ``T``.
    tail_tolerance:
        Optional tolerance for trimming negligible tail mass.
    max_bins:
        Optional explicit limit on how many bins to consider for a single
        segment. ``None`` (default) means keep propagating until either the tail
        probability or ``tail_tolerance`` stops the loop.
    """

    delta = float(tvtw.time_bin_minutes)
    if delta <= 0:
        raise ValueError("TVTWIndexer must have positive time_bin_minutes.")
    num_bins = tvtw.num_bins
    if baseline_entry_minutes < 0:
        return
    t_base = int(math.floor(baseline_entry_minutes / delta))
    if t_base >= num_bins:
        return
    offset = baseline_entry_minutes - t_base * delta
    remaining_mass = 1.0
    k = 0
    while t_base + k < num_bins:
        if max_bins is not None and k >= max_bins:
            break
        start = k * delta - offset
        end = (k + 1) * delta - offset
        start = max(start, 0.0)
        end = max(end, 0.0)
        if end <= start:
            k += 1
            continue
        prob = distribution.interval_probability(start, end)
        if prob <= 0.0:
            tail = distribution.tail_probability(end)
            if tail <= tail_tolerance:
                break
        else:
            bin_idx = t_base + k
            if bin_idx >= num_bins:
                break
            yield bin_idx, prob
            remaining_mass -= prob
            if remaining_mass <= tail_tolerance:
                break
        k += 1
