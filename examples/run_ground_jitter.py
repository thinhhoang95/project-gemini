from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, Optional

from project_gemini.classes import (
    FlightListGemini,
    GeminiType1DemandStore,
    GroundJitterConfig,
    GroundJitterOperator,
)


SYNTH_GROUND_JITTER: Dict[str, Dict[str, Dict[str, float]]] = {
    "default": {
        "00:00-06:00": {
            "p_hurdle": 0.08,
            "mu": 6.0,
            "sigma": 2.5,
            "threshold": 35.0,
            "tail_shape": 0.1,
            "tail_scale": 15.0,
        },
        "06:00-18:00": {
            "p_hurdle": 0.18,
            "mu": 10.0,
            "sigma": 4.0,
            "threshold": 45.0,
            "tail_shape": 0.25,
            "tail_scale": 25.0,
        },
        "18:00-24:00": {
            "p_hurdle": 0.12,
            "mu": 8.0,
            "sigma": 3.5,
            "threshold": 40.0,
            "tail_shape": 0.2,
            "tail_scale": 20.0,
        },
    },
    "LFPG": {
        "00:00-24:00": {
            "p_hurdle": 0.22,
            "mu": 12.0,
            "sigma": 5.0,
            "threshold": 50.0,
            "tail_shape": 0.3,
            "tail_scale": 30.0,
        }
    },
}


def _load_config(path: Optional[str]) -> GroundJitterConfig:
    if not path:
        logging.info("Using built-in synthetic ground jitter configuration")
        return GroundJitterConfig.from_mapping(SYNTH_GROUND_JITTER)
    with open(path, "r") as handle:
        data = json.load(handle)
    return GroundJitterConfig.from_mapping(data)


def _bin_to_clock(bin_index: int, bin_minutes: int) -> str:
    total_minutes = (bin_index * bin_minutes) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}Z"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Project Gemini ground jitter smoke test.")
    parser.add_argument(
        "--flights-csv",
        default="output/flights_20230717_0000-2359.csv",
        help="CSV file containing flight trajectory segments",
    )
    parser.add_argument("--tvtw-indexer", default="output/tvtw_indexer.json")
    parser.add_argument(
        "--type1-demand",
        default="/mnt/d/project-gemini/data/per_flight_demand.csv.gz",
        help="CSV or CSV.GZ file with per-flight Type-1 demand statistics",
    )
    parser.add_argument("--ground-jitter-config", help="Optional JSON file overriding the synthetic config")
    parser.add_argument("--flight-limit", type=int, help="Only process the first N flights with Type-1 data")
    parser.add_argument("--cdf-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-shift-bins", type=int, help="Optional cap on the number of bins a flight can slip into")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top-demand TV/time slices to print")
    parser.add_argument("--sample-seed", type=int, help="Optional RNG seed for sampling ground jitter")
    parser.add_argument(
        "--sample-preview",
        type=int,
        default=5,
        help="Number of sampled delay assignments to preview",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = _load_config(args.ground_jitter_config)

    logging.info("Loading flight list from %s", args.flights_csv)
    flight_list = FlightListGemini(args.flights_csv)
    logging.info("Loading Type-1 demand records from %s", args.type1_demand)
    demand_store = GeminiType1DemandStore(args.type1_demand, tvtw_indexer_path=args.tvtw_indexer)

    selected_flights = None
    if args.flight_limit:
        target_ids = demand_store.type1_flight_ids
        selected_flights = tuple(target_ids[: args.flight_limit])
        logging.info("Restricting computation to the first %d flights", len(selected_flights))

    operator = GroundJitterOperator(
        flight_list,
        demand_store,
        config,
        cdf_tolerance=args.cdf_tolerance,
        max_shift_bins=args.max_shift_bins,
    )

    counts = operator.run_aggregated_counts(selected_flights)
    # operator.run() remains as a backwards-compatible alias.

    print("=== Ground Jitter Smoke Test ===")
    print(f"Total expected demand: {counts.total_mean():,.2f} flights")
    print(f"Aggregate variance: {counts.total_variance():,.2f}")
    print(f"Traffic volumes covered: {counts.num_traffic_volumes}")
    print(f"Time bin length: {counts.time_bin_minutes} minutes")
    print("")
    print(f"Top {args.top_n} stochastic slices:")
    for tv_id, time_bin, mu, var in counts.top_slices(args.top_n):
        clock = _bin_to_clock(time_bin, counts.time_bin_minutes)
        print(f"  {tv_id} @ {clock}: mean={mu:.3f}, var={var:.3f}")

    # Per-flight counts example (documentation only; uncomment for inspection):
    # per_flight = operator.run_per_flight_counts(selected_flights)
    # Example shape:
    # {
    #   "AAL123": {"LFPGZ1": {32: (0.12, 0.09), 33: (0.08, 0.06)}},
    #   "BAW456": {"EGLLR1": {15: (0.05, 0.04)}},
    # }

    sample = operator.sample_flight_delays(selected_flights, seed=args.sample_seed)
    # operator.sample_ground_jitter() remains as a backwards-compatible alias.
    assignments = list(sample.nonzero_items())
    print("")
    print(
        f"Sampled delays: {sample.num_delayed_flights} flights delayed, total {sample.total_delay_minutes} minutes"
    )
    preview = min(args.sample_preview, len(assignments))
    if preview:
        print(f"Preview of {preview} sampled flights:")
        for flight_id, delay in assignments[:preview]:
            print(f"  {flight_id}: {delay} minutes")


if __name__ == "__main__":
    main()
