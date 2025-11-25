from __future__ import annotations

"""
Ground jitter operator
======================

This module applies the Project Gemini hurdle–bulk–tail (HBT) ground jitter
model to per-flight stochastic demand records.  The model is described in
detail in `prompts/gemini/pre_plan_jitters.md` and combines:

* **Hurdle (occurrence) component** – the probability a flight experiences an
  additional ground delay.
* **Bulk (shifted lognormal) component** – the magnitude of routine delays once
  the hurdle is cleared.
* **Tail (Generalised Pareto) component** – the extreme-delay behaviour beyond
  a configurable threshold.

`GroundJitterOperator` converts those abstract distributions into either:

* aggregated stochastic counts (means/variances per traffic-volume/time-bin), or
* concrete samples of additional departure delays that can be applied to a
  `FlightListGemini`.

Example
-------

Input example (abbreviated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

```python
config = GroundJitterConfig.load_from_json("/path/to/hbt_config.json")
flight_list = FlightListGemini.from_records(
    [
        {
            "flight_id": "AAL123",
            "origin": "LFPG",
            "takeoff_time": datetime.fromisoformat("2025-03-01T06:05:00"),
        },
        {
            "flight_id": "BAW456",
            "origin": "EGLL",
            "takeoff_time": datetime.fromisoformat("2025-03-01T07:40:00"),
        },
    ],
    tvtw_indexer,
)
store = GeminiType1DemandStore.from_records(
    [
        # tv_index, time_bin, mu, var correspond to a single traffic volume & time bin
        Type1DemandRecord(tv_index=17, time_bin=32, mu=0.15, var=0.11),
        Type1DemandRecord(tv_index=17, time_bin=33, mu=0.11, var=0.08),
        Type1DemandRecord(tv_index=9, time_bin=15, mu=0.06, var=0.05),
    ],
    num_time_bins=48,
    time_bin_minutes=15,
    tv_id_by_index={17: "LFPGZ1", 9: "EGLLR1"},
)
```

Usage and output examples
~~~~~~~~~~~~~~~~~~~~~~~~~

```python
operator = GroundJitterOperator(flight_list, store, config)

# Aggregated counts (alias: run())
counts = operator.run_aggregated_counts()
# counts.mean_vector -> np.ndarray aligned with tv/time-bin index
# counts.variance_vector -> np.ndarray capturing added dispersion

# Per-flight counts showing the contribution of each flight
per_flight = operator.run_per_flight_counts()
# {
#   "AAL123": {"LFPGZ1": {32: (0.12, 0.09), 33: (0.08, 0.06)}},
#   "BAW456": {"EGLLR1": {15: (0.05, 0.04)}},
# }

# Concrete sampled delays (alias: sample_ground_jitter())
sample = operator.sample_flight_delays(seed=17)
# {
#   "AAL123": 27,  # minutes added to departure
#   "BAW456": 0,   # omitted because delay ≤ 0 minutes
# }

# Access a single sampled delay
delay_minutes = sample["AAL123"]  # 27 minutes of additional taxi/gate time
```
"""

import logging
import math
from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .delay_assignment_gemini import DelayAssignmentGemini
from .flight_list_gemini import FlightListGemini
from .ground_jitter_config import GroundJitterConfig, GroundJitterDistribution
from .stochastic_traffic_count import StochasticTrafficCount
from .type1_demand_store import GeminiType1DemandStore, Type1DemandRecord

logger = logging.getLogger(__name__)


def _minutes_since_midnight(value: Optional[datetime]) -> int:
    """Return the integer minute-of-day (0..1439) for a datetime."""
    if not isinstance(value, datetime):
        return 0
    midnight = value.replace(hour=0, minute=0, second=0, microsecond=0)
    delta = value - midnight
    minutes = int(delta.total_seconds() // 60)
    return minutes % 1440


class GroundJitterOperator:
    """
    Applies a `GroundJitterConfig` to demand records produced by `FlightListGemini`.

    The configuration supplies the hurdle–bulk–splice parameters as a function
    of airport and time-of-day.  Each call resolves those parameters to a
    concrete `GroundJitterDistribution` (providing:
    `interval_probability`, `tail_probability` and `sample_minutes` methods).

    Two complementary integrations plus a sampler are supported:

    * :meth:`run_aggregated_counts` – integrate the continuous HBT distribution
      into per-volume means/variances, preserving stochastic information for
      later Monte Carlo stages.  The legacy :meth:`run` alias remains for
      backwards compatibility.
    * :meth:`run_per_flight_counts` – apply the same shift-and-smear logic per
      flight and report the resulting contributions keyed by traffic volume and
      time bin.
    * :meth:`sample_flight_delays` – draw a single Monte Carlo realisation that
      can be applied to the flight list (e.g. to stress-test downstream
      schedules).  The :meth:`sample_ground_jitter` alias matches the previous
      API.
    """

    def __init__(
        self,
        flight_list: FlightListGemini,
        demand_store: GeminiType1DemandStore,
        config: GroundJitterConfig,
        *,
        cdf_tolerance: float = 1e-6,
        min_interval_prob: float = 1e-5,
        max_shift_bins: Optional[int] = None,
    ):
        self.flight_list = flight_list
        self.demand_store = demand_store
        self.config = config
        self.cdf_tolerance = max(float(cdf_tolerance), 1e-9)
        self.min_interval_prob = max(float(min_interval_prob), 0.0)
        self.max_shift_bins = max_shift_bins
        if demand_store.type1_record_count == 0:
            logger.warning("GeminiType1DemandStore does not contain any Type-1 demand records")

    def run_aggregated_counts(self, flight_ids: Optional[Iterable[str]] = None) -> StochasticTrafficCount:
        """
        Aggregate HBT-induced stochastic counts across all (or selected) flights.

        For each Type-1 demand record we smear its mean/variance across future
        time bins according to the cumulative probability implied by the HBT
        distribution.  This preserves both the expected load (mean) and the
        extra dispersion contributed by ground jitter (variance) at each
        traffic volume and time bin.  Use :meth:`run` if you rely on the legacy
        Project Gemini API name.
        """
        num_entries = self.demand_store.num_traffic_volumes * self.demand_store.num_time_bins
        mean_vector = np.zeros(num_entries, dtype=np.float64)
        variance_vector = np.zeros(num_entries, dtype=np.float64)

        processed_flights = 0
        for flight_id, records in self.demand_store.iter_type1_records(flight_ids):
            if not records:
                continue
            metadata = self.flight_list.get_flight_metadata(flight_id)
            takeoff_dt = metadata.get("takeoff_time")
            origin = metadata.get("origin")
            minute = _minutes_since_midnight(takeoff_dt)
            distribution = self.config.get_distribution(origin, minute)
            self._accumulate(records, distribution, mean_vector, variance_vector)
            processed_flights += 1

        logger.info("Ground jitter aggregated over %d flights", processed_flights)
        return StochasticTrafficCount(
            mean_vector=mean_vector,
            variance_vector=variance_vector,
            tv_id_by_index=dict(self.demand_store.idx_to_tv_id),
            num_time_bins=int(self.demand_store.num_time_bins),
            time_bin_minutes=int(self.demand_store.time_bin_minutes),
        )

    run = run_aggregated_counts
    run.__doc__ = (
        "Deprecated alias for :meth:`GroundJitterOperator.run_aggregated_counts`. "
        "Prefer :meth:`run_aggregated_counts` for clarity."
    )

    def run_per_flight_counts(
        self, flight_ids: Optional[Iterable[str]] = None
    ) -> Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]:
        """
        Apply the HBT model independently per flight and report TV/TW slices.

        Returns
        -------
        Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]
            Nested mapping: ``flight_id -> tv_id -> time_bin -> (mean, variance)``.
            Only entries with positive probability mass are emitted.  ``time_bin``
            is the zero-based index within :attr:`demand_store.num_time_bins`
            (each bin spans :attr:`demand_store.time_bin_minutes` minutes).

        Example
        -------
        ``{"AAL123": {"LFPGZ1": {32: (0.12, 0.09)}}, "BAW456": {"EGLLR1": {15: (0.05, 0.04)}}}``

        Notes
        -----
        Per-flight delay outcomes are independent conditional on the airport and
        time-of-day parameters supplied by :class:`GroundJitterConfig`. Multiple
        Type-1 records for the same flight/traffic-volume/time-bin contribute
        additively to the returned mean/variance.
        """
        tv_id_by_idx = self.demand_store.idx_to_tv_id
        per_flight: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = {}
        processed_flights = 0

        for flight_id, records in self.demand_store.iter_type1_records(flight_ids):
            if not records:
                continue
            metadata = self.flight_list.get_flight_metadata(flight_id) or {}
            takeoff_dt = metadata.get("takeoff_time")
            origin = metadata.get("origin")
            minute = _minutes_since_midnight(takeoff_dt)
            distribution = self.config.get_distribution(origin, minute)
            flight_counts: Dict[str, Dict[int, Tuple[float, float]]] = {}
            self._accumulate_per_flight_dict(records, distribution, flight_counts, tv_id_by_idx)
            if flight_counts:
                per_flight[flight_id] = flight_counts
                processed_flights += 1

        logger.info("Ground jitter per-flight counts computed for %d flights", processed_flights)
        return per_flight

    def sample_flight_delays(
        self,
        flight_ids: Optional[Iterable[str]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        regulation_id: Optional[str] = "ground_jitter",
    ) -> DelayAssignmentGemini:
        """
        Sample concrete ground jitter delays for the requested flights.

        The sampler draws from the hurdle–bulk–tail distribution supplied by
        :class:`GroundJitterConfig`.  Conceptually:

        1. Flip the hurdle coin (`p_hurdle(time-of-day)`) to decide whether a
           delay occurs.
        2. If delayed, draw a shifted lognormal magnitude, and if the draw
           exceeds the configured threshold, model the exceedance with a
           Generalised Pareto tail (capturing rare, extreme delays).
        3. Round up to full minutes—consistent with downstream traffic-volume
           binning—and assign the delay to the flight.

        Parameters
        ----------
        flight_ids:
            Optional subset of flight ids to sample.  When omitted we cover
            every flight that has Type-1 demand data in the associated demand
            store.
        rng / seed:
            Supply either an explicit :class:`numpy.random.Generator` or a
            simple seed for reproducible Monte Carlo draws.
        regulation_id:
            Identifier used to tag the resulting `DelayAssignmentGemini`.

        Returns
        -------
        DelayAssignmentGemini
            Mapping from flight id to sampled positive delay (in minutes).
            Flights with zero sampled delay are omitted.

        Notes
        -----
        The legacy :meth:`sample_ground_jitter` wrapper is retained for
        backwards compatibility.
        """
        if rng is not None and seed is not None:
            raise ValueError("Provide either 'rng' or 'seed', not both")
        generator = rng or np.random.default_rng(seed)
        assignment = DelayAssignmentGemini({}, regulation_id=regulation_id)

        total_flights = 0
        delayed_flights = 0
        skipped_missing = 0
        for flight_id in self._iter_target_flights(flight_ids):
            metadata = self.flight_list.get_flight_metadata(flight_id)
            if not metadata:
                skipped_missing += 1
                continue
            takeoff_dt = metadata.get("takeoff_time")
            if not isinstance(takeoff_dt, datetime):
                skipped_missing += 1
                continue
            total_flights += 1
            origin = metadata.get("origin")
            minute = _minutes_since_midnight(takeoff_dt)
            distribution = self.config.get_distribution(origin, minute)
            delay_minutes = int(math.ceil(distribution.sample_minutes(generator)))
            if delay_minutes <= 0:
                continue
            assignment[flight_id] = delay_minutes
            delayed_flights += 1

        label = assignment.regulation_id or "ground_jitter"
        logger.info(
            "Sampled ground jitter for %d flights (%d delayed) using label '%s'",
            total_flights,
            delayed_flights,
            label,
        )
        if skipped_missing:
            logger.debug("Ground jitter sampling skipped %d flights missing metadata", skipped_missing)
        return assignment

    def sample_ground_jitter(
        self,
        flight_ids: Optional[Iterable[str]] = None,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        regulation_id: Optional[str] = "ground_jitter",
    ) -> DelayAssignmentGemini:
        """Deprecated alias for :meth:`sample_flight_delays`."""
        return self.sample_flight_delays(
            flight_ids,
            rng=rng,
            seed=seed,
            regulation_id=regulation_id,
        )

    # --- internal helpers ------------------------------------------------------
    def _accumulate(
        self,
        records: Sequence[Type1DemandRecord],
        distribution: GroundJitterDistribution,
        mean_vector: np.ndarray,
        variance_vector: np.ndarray,
    ) -> None:
        bin_minutes = float(self.demand_store.time_bin_minutes)
        num_time_bins = int(self.demand_store.num_time_bins)
        max_bin_index = num_time_bins - 1
        tail_tol = self.cdf_tolerance
        min_prob = self.min_interval_prob
        global_shift_limit = self.max_shift_bins

        for record in records:
            remaining_prob = 1.0
            # `local_shift_limit` bounds how far into the future we smear the
            # probability mass.  It is capped by the distribution tail (CDF
            # tolerance) and optionally by `max_shift_bins`.
            local_shift_limit = max_bin_index - record.time_bin
            if global_shift_limit is not None:
                local_shift_limit = min(local_shift_limit, max(global_shift_limit, 0))

            for shift in range(local_shift_limit + 1):
                interval_start = shift * bin_minutes
                interval_end = (shift + 1) * bin_minutes
                prob = distribution.interval_probability(interval_start, interval_end)
                if prob < min_prob and distribution.tail_probability(interval_end) < tail_tol:
                    break
                prob = min(prob, remaining_prob)
                if prob <= 0:
                    continue
                dst_bin = min(record.time_bin + shift, max_bin_index)
                idx = record.tv_index * num_time_bins + dst_bin
                mean_vector[idx] += record.mu * prob
                variance_vector[idx] += record.var * prob + (record.mu ** 2) * prob * (1.0 - prob)
                remaining_prob -= prob
                if remaining_prob <= tail_tol:
                    break
                if dst_bin == max_bin_index:
                    # No later bin exists; collapse any residual mass below.
                    break

            if remaining_prob > tail_tol:
                # Deposit any small residual probability mass into the final bin.
                final_bin = min(record.time_bin + local_shift_limit, max_bin_index)
                idx = record.tv_index * num_time_bins + final_bin
                mean_vector[idx] += record.mu * remaining_prob
                variance_vector[idx] += record.var * remaining_prob + (record.mu ** 2) * remaining_prob * (1.0 - remaining_prob)

    def _accumulate_per_flight_dict(
        self,
        records: Sequence[Type1DemandRecord],
        distribution: GroundJitterDistribution,
        dst: Dict[str, Dict[int, Tuple[float, float]]],
        tv_id_by_idx: Mapping[int, str],
    ) -> None:
        bin_minutes = float(self.demand_store.time_bin_minutes)
        num_time_bins = int(self.demand_store.num_time_bins)
        max_bin_index = num_time_bins - 1
        tail_tol = self.cdf_tolerance
        min_prob = self.min_interval_prob
        global_shift_limit = self.max_shift_bins

        for record in records:
            tv_id = tv_id_by_idx.get(record.tv_index)
            if tv_id is None:
                continue
            remaining_prob = 1.0
            local_shift_limit = max_bin_index - record.time_bin
            if global_shift_limit is not None:
                local_shift_limit = min(local_shift_limit, max(global_shift_limit, 0))
            tv_bins = dst.get(tv_id)

            for shift in range(local_shift_limit + 1):
                interval_start = shift * bin_minutes
                interval_end = (shift + 1) * bin_minutes
                prob = distribution.interval_probability(interval_start, interval_end)
                if prob < min_prob and distribution.tail_probability(interval_end) < tail_tol:
                    break
                prob = min(prob, remaining_prob)
                if prob <= 0:
                    continue
                if tv_bins is None:
                    tv_bins = {}
                    dst[tv_id] = tv_bins
                dst_bin = min(record.time_bin + shift, max_bin_index)
                mu_contrib = record.mu * prob
                var_contrib = record.var * prob + (record.mu ** 2) * prob * (1.0 - prob)
                prev_mu, prev_var = tv_bins.get(dst_bin, (0.0, 0.0))
                tv_bins[dst_bin] = (prev_mu + mu_contrib, prev_var + var_contrib)
                remaining_prob -= prob
                if remaining_prob <= tail_tol:
                    break
                if dst_bin == max_bin_index:
                    break

            if remaining_prob > tail_tol:
                final_bin = min(record.time_bin + local_shift_limit, max_bin_index)
                if tv_bins is None:
                    tv_bins = {}
                    dst[tv_id] = tv_bins
                mu_contrib = record.mu * remaining_prob
                var_contrib = record.var * remaining_prob + (record.mu ** 2) * remaining_prob * (1.0 - remaining_prob)
                prev_mu, prev_var = tv_bins.get(final_bin, (0.0, 0.0))
                tv_bins[final_bin] = (prev_mu + mu_contrib, prev_var + var_contrib)

    def _iter_target_flights(self, flight_ids: Optional[Iterable[str]]) -> Iterable[str]:
        """Yield flight ids that have Type-1 demand data."""
        if flight_ids is None:
            yield from self.demand_store.type1_flight_ids
            return
        for flight_id in flight_ids:
            fid = str(flight_id)
            if self.demand_store.has_type1_data(fid):
                yield fid
