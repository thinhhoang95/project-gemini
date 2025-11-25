"""Aggregate FR segments into ArrivalMoments compatible with ATFMNetworkModel."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from gemini.gem.arrival_moments import ArrivalMoments
from gemini.propagation.tvtw_indexer import TVTWIndexer

from .flight_metadata_provider import FlightMetadataProvider
from .fr_artifacts_loader import FRSegment
from .fr_entry_probability import iter_segment_bin_probabilities

logger = logging.getLogger(__name__)


def build_arrival_moments_from_fr(
    segments: Iterable[FRSegment],
    metadata_provider: FlightMetadataProvider,
    tvtw: TVTWIndexer,
    *,
    volume_ids: Optional[Sequence[str]] = None,
    tail_tolerance: float = 1e-6,
) -> ArrivalMoments:
    """Return Poisson-binomial arrival moments derived from FR segments."""

    num_bins = tvtw.num_bins
    lambda_map: Dict[str, np.ndarray] = {}
    nu_map: Dict[str, np.ndarray] = {}
    gamma_map: Dict[str, np.ndarray] = {}

    def ensure_volume(volume_id: str) -> None:
        if volume_id in lambda_map:
            return
        lambda_map[volume_id] = np.zeros(num_bins, dtype=np.float64)
        nu_map[volume_id] = np.zeros(num_bins, dtype=np.float64)
        gamma_map[volume_id] = np.zeros(num_bins, dtype=np.float64)

    if volume_ids:
        for volume_id in volume_ids:
            ensure_volume(str(volume_id))

    current_key: Optional[Tuple[str, str]] = None
    current_segments: List[FRSegment] = []
    seen_keys: set[Tuple[str, str]] = set()

    def flush_group() -> None:
        if not current_segments:
            return
        _process_segment_group(
            current_segments,
            metadata_provider,
            tvtw,
            ensure_volume,
            lambda_map,
            nu_map,
            gamma_map,
            tail_tolerance=tail_tolerance,
        )

    for segment in segments:
        key = (segment.flight_id, segment.volume_id)
        if current_key is None:
            current_key = key
        if key != current_key:
            seen_keys.add(current_key)
            if key in seen_keys:
                raise ValueError(
                    "Segments iterable must be grouped by (flight_id, volume_id). "
                    "Detected non-contiguous key order."
                )
            flush_group()
            current_segments = []
            current_key = key
        current_segments.append(segment)

    flush_group()

    return _build_arrival_moments(lambda_map, nu_map, gamma_map, num_bins)


def _process_segment_group(
    segments: Sequence[FRSegment],
    metadata_provider: FlightMetadataProvider,
    tvtw: TVTWIndexer,
    ensure_volume,
    lambda_map: MutableMapping[str, np.ndarray],
    nu_map: MutableMapping[str, np.ndarray],
    gamma_map: MutableMapping[str, np.ndarray],
    *,
    tail_tolerance: float,
) -> None:
    if not segments:
        return
    flight_id = segments[0].flight_id
    volume_id = segments[0].volume_id
    ensure_volume(volume_id)
    try:
        context = metadata_provider.get_entry_context(flight_id)
    except Exception as exc:
        logger.warning("Skipping flight %s due to metadata error: %s", flight_id, exc)
        return

    p_bins: Dict[int, float] = {}
    for seg in segments:
        entry_offset = float(seg.entry_offset_min)
        if entry_offset < 0:
            continue
        baseline_entry = context.minute_of_day + entry_offset
        for bin_idx, prob in iter_segment_bin_probabilities(
            baseline_entry,
            context.distribution,
            tvtw,
            tail_tolerance=tail_tolerance,
        ):
            contribution = seg.route_prob * prob
            if contribution <= 0.0:
                continue
            p_bins[bin_idx] = p_bins.get(bin_idx, 0.0) + contribution

    if not p_bins:
        return

    lam = lambda_map[volume_id]
    nu = nu_map[volume_id]
    gamma = gamma_map[volume_id]

    prev_bin: Optional[int] = None
    prev_prob: Optional[float] = None

    for bin_idx in sorted(p_bins.keys()):
        prob = p_bins[bin_idx]
        if prob <= 0.0:
            continue
        lam[bin_idx] += prob
        nu[bin_idx] += prob * (1.0 - prob)
        if prev_bin is not None and prev_prob is not None and bin_idx == prev_bin + 1:
            gamma[prev_bin] += -prev_prob * prob
        prev_bin = bin_idx
        prev_prob = prob


def _build_arrival_moments(
    lambda_map: Mapping[str, np.ndarray],
    nu_map: Mapping[str, np.ndarray],
    gamma_map: Mapping[str, np.ndarray],
    num_bins: int,
) -> ArrivalMoments:
    lambda_ext: Dict[Tuple[str, int], float] = {}
    nu_ext: Dict[Tuple[str, int], float] = {}
    gamma_ext: Dict[Tuple[str, int], float] = {}

    for volume_id, lam in lambda_map.items():
        nu = nu_map[volume_id]
        gamma = gamma_map[volume_id]
        for t in range(num_bins):
            key = (volume_id, t)
            mean_val = float(lam[t])
            var_val = float(nu[t])
            if mean_val != 0.0:
                lambda_ext[key] = mean_val
            if var_val != 0.0:
                nu_ext[key] = var_val
            if t < num_bins - 1:
                cov_val = float(gamma[t])
                if cov_val != 0.0:
                    gamma_ext[key] = cov_val

    return ArrivalMoments(lambda_ext=lambda_ext, nu_ext=nu_ext, gamma_ext=gamma_ext)
