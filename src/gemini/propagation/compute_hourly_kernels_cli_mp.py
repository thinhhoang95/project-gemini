"""Multiprocessing CLI for empirical hourly-kernel estimation from FR artifacts."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from datetime import datetime
from time import perf_counter
from typing import Dict, Iterable

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
        FRRouteGroup,
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
        FRRouteGroup,
        build_traversals_for_fr_route,
        group_fr_segments_by_route,
        tag_route_groups,
    )
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.tvtw_indexer import TVTWIndexer

DEFAULT_MASTER = "/Volumes/CrucialX/project-tailwind/output/flights_20230717_0000-2359.csv"
DEFAULT_FR_DEMAND = "data/fr/gem_artifacts_demand_all"
DEFAULT_FR_ROUTE_CATALOGUE = "data/fr/gem_artifacts_route_catalogue_all"
DEFAULT_TVTW = "/Volumes/CrucialX/project-tailwind/output/tvtw_indexer.json"
DEFAULT_OUTPUT = "/Volumes/CrucialX/project-gemini/data/hourly_kernels.csv"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical hourly kernels from FR route artifacts (multiprocess)."
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
        help="Emit per-hour kernels even when an hour has zero traversals.",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes to use (defaults to n_cpu - 1).",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


WORKER_TVTW: TVTWIndexer | None = None
WORKER_MAX_LAG: int = 0
WORKER_MINUTES: Dict[str, float] = {}


def _init_worker(payload: dict) -> None:
    """Initialise shared TVTW state inside each worker process."""

    global WORKER_TVTW, WORKER_MAX_LAG, WORKER_MINUTES
    WORKER_TVTW = payload["tvtw_indexer"]
    WORKER_MAX_LAG = int(payload["max_lag_bins"])
    WORKER_MINUTES = dict(payload["minutes_of_day"])


def _process_fr_route(route_group: FRRouteGroup) -> dict:
    """Worker target that builds traversals for one FR route group."""

    if WORKER_TVTW is None:
        raise RuntimeError("Worker TVTW indexer not initialised.")
    minute_map = WORKER_MINUTES
    route_start = perf_counter()
    takeoff_minute = minute_map.get(route_group.flight_id)
    if takeoff_minute is None:
        return {
            "flight_id": route_group.flight_id,
            "route_index": route_group.route_index,
            "group": route_group.group,
            "segment_count": len(route_group.segments),
            "traversals": [],
            "dropped": 0,
            "censored": 0,
            "elapsed": 0.0,
            "missing_metadata": True,
        }

    traversal_result = build_traversals_for_fr_route(
        route_group=route_group,
        tvtw_indexer=WORKER_TVTW,
        takeoff_minute_of_day=takeoff_minute,
        max_lag_bins=WORKER_MAX_LAG,
        route_label=str(route_group.route_index),
    )
    elapsed = perf_counter() - route_start
    return {
        "flight_id": route_group.flight_id,
        "route_index": route_group.route_index,
        "group": route_group.group,
        "segment_count": len(route_group.segments),
        "traversals": traversal_result.traversals,
        "dropped": traversal_result.dropped_departures,
        "censored": traversal_result.censored_traversals,
        "elapsed": elapsed,
        "missing_metadata": False,
    }


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

    cpu_total = os.cpu_count() or 1
    default_workers = max(1, cpu_total - 1)
    worker_count = max(1, args.num_workers or default_workers)
    logging.info("Using %s worker processes for traversal extraction.", worker_count)

    worker_payload = {
        "tvtw_indexer": tvtw_indexer,
        "max_lag_bins": max_lag_bins,
        "minutes_of_day": takeoff_lookup.as_minutes_dict(),
    }

    total_routes = 0
    total_traversals = 0
    missing_metadata = 0
    dropped_departures = 0
    censored_traversals = 0

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

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(worker_payload,),
    ) as pool:
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
            for result in pool.imap_unordered(_process_fr_route, route_groups, chunksize=1):
                progress.advance(task_id, 1)
                if result.get("missing_metadata"):
                    missing_metadata += 1
                    logging.warning(
                        "Skipping flight %s route %s due to missing takeoff metadata.",
                        result.get("flight_id"),
                        result.get("route_index"),
                    )
                    continue

                traversals = result.get("traversals", [])
                traversal_count = len(traversals)
                for traversal in traversals:
                    estimator.add_traversal(traversal)
                total_routes += 1
                total_traversals += traversal_count
                dropped_departures += int(result.get("dropped", 0))
                censored_traversals += int(result.get("censored", 0))

                group_tag = (result.get("group") or "FR").upper()
                flight_id = result.get("flight_id")
                route_index = result.get("route_index")
                segment_count = result.get("segment_count", 0)
                elapsed = result.get("elapsed", 0.0)

                if traversal_count == 0:
                    logging.warning(
                        "%s flight %s route %s yielded zero traversals (segments=%s, dropped=%s, elapsed=%.2fs)",
                        group_tag,
                        flight_id,
                        route_index,
                        segment_count,
                        result.get("dropped", 0),
                        elapsed,
                    )
                else:
                    logging.debug(
                        "Finished %s flight %s route %s | traversals=%s | elapsed=%.2fs",
                        group_tag,
                        flight_id,
                        route_index,
                        traversal_count,
                        elapsed,
                    )

                if args.log_every and total_routes % args.log_every == 0:
                    logging.info(
                        "Processed %s routes | last flight=%s route=%s | traversals=%s | total traversals=%s | last elapsed=%.2fs",
                        total_routes,
                        flight_id,
                        route_index,
                        traversal_count,
                        total_traversals,
                        elapsed,
                    )

    logging.info(
        "Finished traversal extraction: %s traversals retained (%s censored, %s dropped).",
        estimator.total_records,
        estimator.total_censored,
        estimator.dropped_records,
    )
    logging.info(
        "Routes processed: %s | flights covered: %s | missing metadata: %s | departures dropped=%s | censored traversals=%s",
        f"{total_routes:,}",
        f"{unique_flights:,}",
        f"{missing_metadata:,}",
        f"{dropped_departures:,}",
        f"{censored_traversals:,}",
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
