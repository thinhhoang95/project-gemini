"""CLI entry point for empirical hourly-kernel estimation."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from time import perf_counter
from typing import Iterable

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    from gemini.arr.fr_artifacts_loader import load_fr_segments
    from gemini.arrivals.flight_list_gemini import FlightListGemini
    from gemini.propagation.fr_traversal import (
        FlightTakeoffLookup,
        build_traversals_for_fr_route,
        group_fr_segments_by_route,
        tag_route_groups,
    )
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.tvtw_indexer import TVTWIndexer
except ModuleNotFoundError:  # Allows running the CLI as a standalone script.
    import pathlib
    import sys

    PROJECT_SRC = pathlib.Path(__file__).resolve().parents[2]
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from gemini.arr.fr_artifacts_loader import load_fr_segments
    from gemini.arrivals.flight_list_gemini import FlightListGemini
    from gemini.propagation.fr_traversal import (
        FlightTakeoffLookup,
        build_traversals_for_fr_route,
        group_fr_segments_by_route,
        tag_route_groups,
    )
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.tvtw_indexer import TVTWIndexer

DEFAULT_MASTER = "/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv"
DEFAULT_FR_DEMAND = "data/fr/gem_artifacts_demand_all"
DEFAULT_FR_ROUTE_CATALOGUE = "data/fr/gem_artifacts_route_catalogue_all"
DEFAULT_TVTW = "/mnt/d/project-tailwind/output/tvtw_indexer.json"
DEFAULT_OUTPUT = "/mnt/d/project-gemini/data/hourly_kernels.csv"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical hourly kernels from FR route artifacts."
    )
    parser.add_argument("--master-flights", default=DEFAULT_MASTER, help="Master flights CSV path.")
    parser.add_argument(
        "--fr-demand",
        default=DEFAULT_FR_DEMAND,
        help="Path to gem_artifacts_demand_all CSV/CSV.GZ.",
    )
    parser.add_argument(
        "--fr-route-catalogue",
        default=DEFAULT_FR_ROUTE_CATALOGUE,
        help="Path to gem_artifacts_route_catalogue_all CSV/CSV.GZ.",
    )
    parser.add_argument(
        "--tvtw-indexer",
        default=DEFAULT_TVTW,
        help="Serialized TVTW indexer JSON (for bin length metadata).",
    )
    parser.add_argument(
        "--output-kernels",
        default=DEFAULT_OUTPUT,
        help="Destination CSV path for the hourly kernel table.",
    )
    parser.add_argument(
        "--planning-day",
        default="2023-07-17",
        help="Planning day in YYYY-MM-DD used to anchor local times.",
    )
    parser.add_argument(
        "--max-lag-bins",
        type=int,
        default=None,
        help="Maximum lag (in bins) kept per edge (defaults to full horizon).",
    )
    parser.add_argument(
        "--shrinkage-M",
        type=float,
        default=75.0,
        help="Shrinkage constant M used in Step K3.",
    )
    parser.add_argument(
        "--min-traversals-per-edge",
        type=int,
        default=5,
        help="Drop edges with fewer traversals than this threshold.",
    )
    parser.add_argument(
        "--emit-empty-hours",
        action="store_true",
        help="Emit per-hour kernels even when a specific hour has zero traversals.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Log progress after processing this many flight-route trajectories.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for the CLI logger.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    planning_day = datetime.strptime(args.planning_day, "%Y-%m-%d").date()
    logging.info("Anchoring planning day at %s", planning_day.isoformat())
    logging.info("Loading TVTW indexer from %s", args.tvtw_indexer)
    tvtw_indexer = TVTWIndexer.load(args.tvtw_indexer)
    max_lag_bins = args.max_lag_bins or tvtw_indexer.num_bins

    estimator = HourlyKernelEstimator(
        delta_minutes=tvtw_indexer.time_bin_minutes,
        num_bins=tvtw_indexer.num_bins,
        bins_per_hour=tvtw_indexer.bins_per_hour,
        max_lag_bins=max_lag_bins,
        shrinkage_M=args.shrinkage_M,
        min_traversals_per_edge=args.min_traversals_per_edge,
        emit_empty_hours=args.emit_empty_hours,
    )

    logging.info("Loading master flight metadata from %s", args.master_flights)
    flights = FlightListGemini(args.master_flights)
    takeoff_lookup = FlightTakeoffLookup(flights)

    logging.info(
        "Loading FR artifacts from demand=%s and route catalogue=%s",
        args.fr_demand,
        args.fr_route_catalogue,
    )
    segments = load_fr_segments(args.fr_demand, args.fr_route_catalogue)
    if not segments:
        raise SystemExit("No FR segments available; cannot build kernels.")

    segments.sort(key=lambda seg: (seg.flight_id, seg.route_index, seg.entry_offset_min))
    route_groups = group_fr_segments_by_route(segments)
    del segments
    if not route_groups:
        raise SystemExit("FR artifacts did not yield any route groups.")
    tag_route_groups(route_groups)

    route_total = len(route_groups)
    unique_flights = len({group.flight_id for group in route_groups})
    progress_console = Console(stderr=True)
    progress_columns = [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed:,} routes", justify="right"),
        TimeElapsedColumn(),
    ]
    progress = Progress(
        *progress_columns,
        console=progress_console,
        transient=True,
        disable=not progress_console.is_terminal,
    )

    total_routes = 0
    total_traversals = 0
    missing_metadata = 0
    dropped_departures = 0

    with progress:
        task_id = progress.add_task(
            f"FR routes ({route_total:,})" if route_total else "FR routes",
            total=route_total or None,
        )

        logging.info(
            "Processing %s FR routes covering %s flights.",
            f"{route_total:,}",
            f"{unique_flights:,}",
        )
        for route_group in route_groups:
            segment_count = len(route_group.segments)
            route_start = perf_counter()
            try:
                takeoff_minute = takeoff_lookup.get_takeoff_minute_of_day(route_group.flight_id)
            except KeyError:
                missing_metadata += 1
                logging.warning(
                    "Skipping flight %s route %s due to missing takeoff metadata.",
                    route_group.flight_id,
                    route_group.route_index,
                )
                progress.advance(task_id, 1)
                continue

            traversal_result = build_traversals_for_fr_route(
                route_group=route_group,
                tvtw_indexer=tvtw_indexer,
                takeoff_minute_of_day=takeoff_minute,
                max_lag_bins=max_lag_bins,
                route_label=str(route_group.route_index),
            )
            traversal_count = len(traversal_result.traversals)
            dropped_departures += traversal_result.dropped_departures
            for traversal in traversal_result.traversals:
                estimator.add_traversal(traversal)
                total_traversals += 1
            total_routes += 1
            elapsed = perf_counter() - route_start
            group_tag = route_group.group or "FR"

            if traversal_count == 0:
                logging.warning(
                    "%s flight %s route %s yielded zero traversals (segments=%s, dropped=%s, elapsed=%.2fs)",
                    group_tag,
                    route_group.flight_id,
                    route_group.route_index,
                    segment_count,
                    traversal_result.dropped_departures,
                    elapsed,
                )
            else:
                logging.debug(
                    "Finished %s flight %s route %s | traversals=%s | elapsed=%.2fs",
                    group_tag,
                    route_group.flight_id,
                    route_group.route_index,
                    traversal_count,
                    elapsed,
                )

            if args.log_every and total_routes % args.log_every == 0:
                logging.info(
                    "Processed %s routes | last flight=%s route=%s | traversals=%s | total traversals=%s | last elapsed=%.2fs",
                    total_routes,
                    route_group.flight_id,
                    route_group.route_index,
                    traversal_count,
                    total_traversals,
                    elapsed,
                )
            progress.advance(task_id, 1)

    logging.info(
        "Finished traversal extraction: %s traversals retained (%s censored, %s dropped).",
        estimator.total_records,
        estimator.total_censored,
        estimator.dropped_records,
    )
    logging.info(
        "Routes processed: %s | flights covered: %s | missing metadata: %s | departures dropped=%s",
        f"{total_routes:,}",
        f"{unique_flights:,}",
        f"{missing_metadata:,}",
        f"{dropped_departures:,}",
    )
    rows = estimator.finalize_kernels()
    logging.info("Kernel table contains %s rows for %s edges.", len(rows), len(estimator.edge_totals))

    out_dir = os.path.dirname(args.output_kernels)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_kernels, index=False)
    logging.info("Hourly kernels written to %s", args.output_kernels)

    logging.info("FR traversal processing complete.")


if __name__ == "__main__":
    main()
