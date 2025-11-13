"""CLI entry point for empirical hourly-kernel estimation."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
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
    from gemini.propagation.data_sources import iter_nonorig_segments, iter_original_segments
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.routes import RouteCatalog
    from gemini.propagation.traversal_extractor import TraversalExtractor
    from gemini.propagation.tvtw_indexer import TVTWIndexer
    from gemini.propagation.volume_graph import VolumeGraph
except ModuleNotFoundError:  # Allows running the CLI as a standalone script.
    import pathlib
    import sys

    PROJECT_SRC = pathlib.Path(__file__).resolve().parents[2]
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from gemini.propagation.data_sources import iter_nonorig_segments, iter_original_segments
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.routes import RouteCatalog
    from gemini.propagation.traversal_extractor import TraversalExtractor
    from gemini.propagation.tvtw_indexer import TVTWIndexer
    from gemini.propagation.volume_graph import VolumeGraph

DEFAULT_MASTER = "/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv"
DEFAULT_ROUTES = "/mnt/d/project-gemini/data/strictly_better_routes.csv"
DEFAULT_SEGMENTS_DIR = "/mnt/d/project-silverdrizzle/tmp/all_segs"
DEFAULT_GEOJSON = "/mnt/d/project-tailwind/output/wxm_sm_ih_maxpool.geojson"
DEFAULT_TVTW = "/mnt/d/project-tailwind/output/tvtw_indexer.json"
DEFAULT_OUTPUT = "/mnt/d/project-gemini/data/hourly_kernels.csv"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute empirical hourly kernels from 4D trajectory segments."
    )
    parser.add_argument("--master-flights", default=DEFAULT_MASTER, help="Master flights CSV path.")
    parser.add_argument(
        "--strictly-better-routes",
        default=DEFAULT_ROUTES,
        help="CSV mapping each flight to ORIGINAL vs candidate routes.",
    )
    parser.add_argument(
        "--nonorig-4d-segments-dir",
        default=DEFAULT_SEGMENTS_DIR,
        help="Directory containing partitioned non-ORIGINAL segments CSV.GZ files.",
    )
    parser.add_argument(
        "--volumes-geojson",
        default=DEFAULT_GEOJSON,
        help="GeoJSON describing regulated traffic volumes.",
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
        "--sampling-distance-km",
        type=float,
        default=10.0,
        help="Target spatial spacing between sampled 4D points along each segment.",
    )
    parser.add_argument(
        "--sampling-time-seconds",
        type=float,
        default=120.0,
        help="Target temporal spacing between sampled 4D points along each segment.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Chunk size for streaming CSV ingestion.",
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
    logging.info("Loading TVTW indexer from %s", args.tvtw_indexer)
    tvtw_indexer = TVTWIndexer.load(args.tvtw_indexer)
    max_lag_bins = args.max_lag_bins or tvtw_indexer.num_bins

    logging.info("Loading traffic volumes from %s", args.volumes_geojson)
    volume_graph = VolumeGraph.from_geojson(args.volumes_geojson)

    logging.info("Loading route catalog from %s", args.strictly_better_routes)
    route_catalog = RouteCatalog.from_csv(args.strictly_better_routes)

    extractor = TraversalExtractor(
        tvtw_indexer=tvtw_indexer,
        volume_locator=volume_graph.locator,
        planning_day=planning_day,
        max_lag_bins=max_lag_bins,
        sampling_distance_km=args.sampling_distance_km,
        sampling_time_seconds=args.sampling_time_seconds,
        logger=logging.getLogger("TraversalExtractor"),
    )

    estimator = HourlyKernelEstimator(
        delta_minutes=tvtw_indexer.time_bin_minutes,
        num_bins=tvtw_indexer.num_bins,
        bins_per_hour=tvtw_indexer.bins_per_hour,
        max_lag_bins=max_lag_bins,
        shrinkage_M=args.shrinkage_M,
        min_traversals_per_edge=args.min_traversals_per_edge,
        emit_empty_hours=args.emit_empty_hours,
    )

    total_routes = 0
    total_traversals = 0

    original_total = len(route_catalog.original_flights)
    nonorig_total = sum(len(routes) for routes in route_catalog.nonorig_routes.values())
    progress_console = Console(stderr=True)
    progress_columns = [
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("{task.completed:,} flights", justify="right"),
        TimeElapsedColumn(),
    ]
    progress = Progress(
        *progress_columns,
        console=progress_console,
        transient=True,
        disable=not progress_console.is_terminal,
    )

    with progress:
        original_task = progress.add_task(
            f"ORIGINAL flights ({original_total:,})" if original_total else "ORIGINAL flights",
            total=original_total or None,
        )
        nonorig_task = progress.add_task(
            f"NON-ORIGINAL flights ({nonorig_total:,})" if nonorig_total else "NON-ORIGINAL flights",
            total=nonorig_total or None,
        )

        logging.info("Processing ORIGINAL flights...")
        for flight_route in iter_original_segments(
            args.master_flights, route_catalog, chunksize=args.chunk_size
        ):
            total_routes += 1
            for traversal in extractor.extract_traversals(flight_route):
                estimator.add_traversal(traversal)
                total_traversals += 1
            progress.advance(original_task, 1)
            if args.log_every and total_routes % args.log_every == 0:
                logging.info(
                    "Processed %s ORIGINAL trajectories | total traversals=%s",
                    total_routes,
                    total_traversals,
                )

        logging.info("Processing NON-ORIGINAL trajectories...")
        for flight_route in iter_nonorig_segments(
            args.nonorig_4d_segments_dir, route_catalog, chunksize=args.chunk_size
        ):
            total_routes += 1
            for traversal in extractor.extract_traversals(flight_route):
                estimator.add_traversal(traversal)
                total_traversals += 1
            progress.advance(nonorig_task, 1)
            if args.log_every and total_routes % args.log_every == 0:
                logging.info(
                    "Processed %s total trajectories | total traversals=%s",
                    total_routes,
                    total_traversals,
                )

    logging.info(
        "Finished traversal extraction: %s traversals retained (%s dropped).",
        estimator.total_records,
        estimator.dropped_records,
    )
    rows = estimator.finalize_kernels()
    logging.info("Kernel table contains %s rows for %s edges.", len(rows), len(estimator.edge_totals))

    out_dir = os.path.dirname(args.output_kernels)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_kernels, index=False)
    logging.info("Hourly kernels written to %s", args.output_kernels)

    extractor_stats = extractor.get_stats()
    logging.info("Extractor stats: %s", extractor_stats)


if __name__ == "__main__":
    main()
