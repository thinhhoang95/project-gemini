"""CLI entry point for empirical hourly-kernel estimation."""
"""CLI entry point for empirical hourly-kernel estimation.

This script processes 4D trajectory segments (both ORIGINAL flights and candidate
routes) to compute empirical hourly kernels that model propagation probabilities
through regulated airspace volumes. The kernels estimate the probability distribution
of traversal times (lags) between volume edges, conditional on the hour of day.

Inputs:
--------
1. Master flights CSV (--master-flights)
   Path to CSV containing ORIGINAL flight trajectory segments.
   Expected columns:
   - flight_identifier: unique flight ID (str)
   - time_begin_segment, time_end_segment: segment time bounds
   - date_begin_segment, date_end_segment: segment date bounds
   - _start_datetime: flight start timestamp
   - latitude_begin, longitude_begin: segment start position
   - latitude_end, longitude_end: segment end position
   - flight_level_begin, flight_level_end: altitude bounds
   - sequence: segment ordering
   
   Example: /Volumes/CrucialX/project-tailwind/output/flights_20230717_0000-2359.csv

2. Strictly better routes CSV (--strictly-better-routes)
   Path to CSV mapping each flight to ORIGINAL vs candidate routes.
   Expected columns:
   - flight_identifier: unique flight ID (str)
   - route: route label, either "ORIGINAL" or a candidate route identifier (str)
   
   Example:
   flight_identifier,route
   ABC123,ORIGINAL
   ABC123,ROUTE_A
   ABC123,ROUTE_B
   
   Example path: /Volumes/CrucialX/project-gemini/data/strictly_better_routes.csv

3. Non-ORIGINAL 4D segments directory (--nonorig-4d-segments-dir)
   Directory containing partitioned CSV.GZ files with candidate route segments.
   Each file should contain segments with columns:
   - flight_identifier: unique flight ID (str)
   - route: candidate route label (str)
   - time_begin_segment, time_end_segment: segment time bounds
   - latitude_begin, longitude_begin: segment start position
   - latitude_end, longitude_end: segment end position
   - flight_level_begin, flight_level_end: altitude bounds
   
   Example: /Volumes/CrucialX/project-silverdrizzle/tmp/all_segs_unsharded/

4. Traffic volumes GeoJSON (--volumes-geojson)
   GeoJSON file describing regulated traffic volumes (sectors/regions).
   Used to map trajectory points to volume identifiers for edge traversal.
   
   Example: /Volumes/CrucialX/project-tailwind/output/wxm_sm_ih_maxpool.geojson

5. TVTW indexer JSON (--tvtw-indexer)
   Serialized TVTWIndexer JSON containing temporal binning metadata:
   - time_bin_minutes: duration of each time bin (int)
   - num_bins: total number of bins in the planning horizon (int)
   - bins_per_hour: number of bins per hour (int)
   
   Example: /Volumes/CrucialX/project-tailwind/output/tvtw_indexer.json

6. Planning day (--planning-day)
   Date string in YYYY-MM-DD format used to anchor local time calculations.
   
   Example: "2023-07-17"

Processing Parameters:
----------------------
- --max-lag-bins: Maximum lag (in bins) kept per edge (defaults to full horizon)
- --shrinkage-M: Shrinkage constant M used in Step K3 for empirical Bayes (default: 75.0)
- --min-traversals-per-edge: Minimum traversals required per edge; edges below threshold are dropped (default: 5)
- --sampling-distance-km: Target spatial spacing between sampled 4D points along segments (default: 10.0)
- --sampling-time-seconds: Target temporal spacing between sampled 4D points (default: 120.0)
- --chunk-size: Chunk size for streaming CSV ingestion (default: 200,000)
- --log-every: Log progress after processing N flight-route trajectories (default: 250)
- --num-workers: Number of worker processes for parallel processing (default: n_cpu - 1)

Outputs:
--------
1. Hourly kernels CSV (--output-kernels)
   CSV file containing the empirical hourly kernel table.
   Each row represents a kernel value for a specific (edge, hour, lag) combination.
   
   Columns:
   - edge_u: upstream volume identifier (str)
   - edge_v: downstream volume identifier (str)
   - edge_id: string representation of edge (str)
   - hour_index: hour of day (0-23) (int)
   - lag_bins: traversal lag in bins (int)
   - lag_minutes: traversal lag in minutes (float)
   - kernel_value: empirical kernel probability [0.0, 1.0] (float)
   - traversal_count_hour: number of traversals for this edge-hour (int)
   - lag_count_hour: count for this specific lag in this hour (int)
   - traversal_count_edge: total traversals for this edge across all hours (int)
   - lost_count_hour: lost/censored traversals for this edge-hour (int)
   - lost_fraction_hour: fraction of traversals lost/censored in this hour (float)
   - lost_count_edge: total lost/censored traversals for this edge (int)
   - alpha: shrinkage weight used in empirical Bayes [0.0, 1.0] (float)
   - delta_minutes: time bin duration in minutes (int)
   
   Example output row:
   edge_u,edge_v,edge_id,hour_index,lag_bins,lag_minutes,kernel_value,traversal_count_hour,lag_count_hour,traversal_count_edge,lost_count_hour,lost_fraction_hour,lost_count_edge,alpha,delta_minutes
   VOL_001,VOL_002,VOL_001->VOL_002,14,5,60.0,0.15,100,15,500,0,0.0,5,0.571,12.0
   
   Example path: /Volumes/CrucialX/project-gemini/data/hourly_kernels.csv

2. Log output (stdout/stderr)
   Progress logs including:
   - Loading status for each input file
   - Processing statistics (traversals retained, censored, dropped)
   - Final kernel table statistics (rows, edges)
   - Extractor statistics (counters for various processing events)

The kernel_value represents the empirical probability that a flight traversing edge
(edge_u -> edge_v) during hour_index will have a lag of lag_bins bins between
departure and arrival. Values are computed using empirical Bayes shrinkage (Step K3)
that blends hour-specific empirical distributions with global edge-level priors.

Example Usage:
--------------
    python -m gemini.propagation.compute_hourly_kernels_cli_mp \\
        --master-flights /path/to/flights.csv \\
        --strictly-better-routes /path/to/routes.csv \\
        --nonorig-4d-segments-dir /path/to/segments/ \\
        --volumes-geojson /path/to/volumes.geojson \\
        --tvtw-indexer /path/to/tvtw_indexer.json \\
        --output-kernels /path/to/output.csv \\
        --planning-day 2023-07-17 \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from collections import Counter
from datetime import datetime
from time import perf_counter
from typing import Iterable

from multiprocessing.pool import Pool

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
    from gemini.propagation.traversal_extractor import FlightRouteSegments, TraversalExtractor
    from gemini.propagation.tvtw_indexer import TVTWIndexer
    from gemini.propagation.volume_graph import VolumeGraph, VolumeLocator
except ModuleNotFoundError:  # Allows running the CLI as a standalone script.
    import pathlib
    import sys

    PROJECT_SRC = pathlib.Path(__file__).resolve().parents[2]
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from gemini.propagation.data_sources import iter_nonorig_segments, iter_original_segments
    from gemini.propagation.hourly_kernel import HourlyKernelEstimator
    from gemini.propagation.routes import RouteCatalog
    from gemini.propagation.traversal_extractor import FlightRouteSegments, TraversalExtractor
    from gemini.propagation.tvtw_indexer import TVTWIndexer
    from gemini.propagation.volume_graph import VolumeGraph, VolumeLocator

DEFAULT_MASTER = "/Volumes/CrucialX/project-tailwind/output/flights_20230717_0000-2359.csv"
DEFAULT_ROUTES = "/Volumes/CrucialX/project-gemini/data/strictly_better_routes.csv"
DEFAULT_SEGMENTS_DIR = "/Volumes/CrucialX/project-silverdrizzle/tmp/all_segs_unsharded"
DEFAULT_GEOJSON = "/Volumes/CrucialX/project-tailwind/output/wxm_sm_ih_maxpool.geojson"
DEFAULT_TVTW = "/Volumes/CrucialX/project-tailwind/output/tvtw_indexer.json"
DEFAULT_OUTPUT = "/Volumes/CrucialX/project-gemini/data/hourly_kernels.csv"


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
        "--log-every",
        type=int,
        default=250,
        help="Log progress after processing this many flight-route trajectories.",
    )
    parser.add_argument(
        "--log-level",
        default="ERROR",
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


WORKER_EXTRACTOR: TraversalExtractor | None = None


def _init_worker(payload: dict) -> None:
    """Initialise a TraversalExtractor inside each worker process."""
    global WORKER_EXTRACTOR
    geo_df = payload["geo_dataframe"]
    locator = VolumeLocator(geo_df)
    WORKER_EXTRACTOR = TraversalExtractor(
        tvtw_indexer=payload["tvtw_indexer"],
        volume_locator=locator,
        planning_day=payload["planning_day"],
        max_lag_bins=payload["max_lag_bins"],
        sampling_distance_km=payload["sampling_distance_km"],
        sampling_time_seconds=payload["sampling_time_seconds"],
        logger=logging.getLogger("TraversalExtractor"),
    )


def _extract_flight_route(flight_route: FlightRouteSegments) -> dict:
    """Worker target that extracts traversals for a single flight route."""
    if WORKER_EXTRACTOR is None:
        raise RuntimeError("TraversalExtractor was not initialised in worker.")
    route_start = perf_counter()
    stats_before = WORKER_EXTRACTOR.get_stats()
    traversals = list(WORKER_EXTRACTOR.extract_traversals(flight_route))
    stats_after = WORKER_EXTRACTOR.get_stats()
    stats_delta = {
        key: stats_after.get(key, 0) - stats_before.get(key, 0)
        for key in set(stats_after) | set(stats_before)
    }
    stats_delta = {key: value for key, value in stats_delta.items() if value}
    elapsed = perf_counter() - route_start
    segments = getattr(flight_route, "segments", None)
    segment_count = len(segments) if segments is not None else 0
    return {
        "flight_id": flight_route.flight_id,
        "route_label": flight_route.route_label,
        "group": flight_route.group,
        "segment_count": segment_count,
        "traversals": traversals,
        "elapsed": elapsed,
        "stats_delta": stats_delta,
    }


def _process_route_group(
    *,
    pool: Pool,
    routes: Iterable[FlightRouteSegments],
    estimator: HourlyKernelEstimator,
    stats_counter: Counter,
    progress: Progress,
    progress_task: int,
    log_every: int,
    total_routes: int,
    total_traversals: int,
) -> tuple[int, int]:
    """Fan out traversal extraction for a specific route group."""
    group_processed = 0
    for result in pool.imap_unordered(_extract_flight_route, routes, chunksize=1):
        group_processed += 1
        total_routes += 1
        traversals = result["traversals"]
        traversal_count = len(traversals)
        total_traversals += traversal_count
        for traversal in traversals:
            estimator.add_traversal(traversal)
        stats_counter.update(result["stats_delta"])
        _log_route_result(
            result=result,
            traversal_count=traversal_count,
            group_processed=group_processed,
            total_routes=total_routes,
            total_traversals=total_traversals,
            log_every=log_every,
        )
        progress.advance(progress_task, 1)
    return total_routes, total_traversals


def _log_route_result(
    *,
    result: dict,
    traversal_count: int,
    group_processed: int,
    total_routes: int,
    total_traversals: int,
    log_every: int,
) -> None:
    """Emit debug/progress logs mirroring the single-process behaviour."""
    group = (result.get("group") or "").upper()
    flight_id = result.get("flight_id")
    route_label = result.get("route_label")
    segment_count = result.get("segment_count", 0)
    elapsed = result.get("elapsed", 0.0)
    is_original = group == "ORIGINAL"

    if traversal_count == 0:
        if is_original:
            logging.warning(
                "ORIGINAL flight %s yielded zero traversals (segments=%s, elapsed=%.2fs)",
                flight_id,
                segment_count,
                elapsed,
            )
        else:
            logging.warning(
                "NON-ORIGINAL flight %s (%s) yielded zero traversals (segments=%s, elapsed=%.2fs)",
                flight_id,
                route_label,
                segment_count,
                elapsed,
            )
    else:
        if is_original:
            logging.debug(
                "Finished ORIGINAL flight %s | traversals=%s | elapsed=%.2fs",
                flight_id,
                traversal_count,
                elapsed,
            )
        else:
            logging.debug(
                "Finished NON-ORIGINAL flight %s (%s) | traversals=%s | elapsed=%.2fs",
                flight_id,
                route_label,
                traversal_count,
                elapsed,
            )

    if not log_every:
        return
    if is_original:
        if group_processed % log_every == 0:
            logging.info(
                "Processed %s ORIGINAL trajectories | last flight=%s | traversals=%s | total traversals=%s | last elapsed=%.2fs",
                group_processed,
                flight_id,
                traversal_count,
                total_traversals,
                elapsed,
            )
    else:
        if total_routes % log_every == 0:
            logging.info(
                "Processed %s total trajectories | last flight=%s (%s) | traversals=%s | total traversals=%s | last elapsed=%.2fs",
                total_routes,
                flight_id,
                route_label,
                traversal_count,
                total_traversals,
                elapsed,
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

    estimator = HourlyKernelEstimator(
        delta_minutes=tvtw_indexer.time_bin_minutes,
        num_bins=tvtw_indexer.num_bins,
        bins_per_hour=tvtw_indexer.bins_per_hour,
        max_lag_bins=max_lag_bins,
        shrinkage_M=args.shrinkage_M,
        min_traversals_per_edge=args.min_traversals_per_edge,
        emit_empty_hours=True
    )

    cpu_total = os.cpu_count() or 1
    default_workers = max(1, cpu_total - 1)
    worker_count = max(1, args.num_workers or default_workers)
    logging.info("Using %s worker processes for traversal extraction.", worker_count)

    extractor_payload = {
        "geo_dataframe": volume_graph.geo_dataframe,
        "tvtw_indexer": tvtw_indexer,
        "planning_day": planning_day,
        "max_lag_bins": max_lag_bins,
        "sampling_distance_km": args.sampling_distance_km,
        "sampling_time_seconds": args.sampling_time_seconds,
    }
    stats_counter: Counter = Counter()

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

    ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(extractor_payload,),
    ) as pool:
        with progress:
            nonorig_task = progress.add_task(
                f"NON-ORIGINAL flights ({nonorig_total:,})"
                if nonorig_total
                else "NON-ORIGINAL flights",
                total=nonorig_total or None,
            )
            original_task = progress.add_task(
                f"ORIGINAL flights ({original_total:,})" if original_total else "ORIGINAL flights",
                total=original_total or None,
            )

            logging.info(
                "Processing NON-ORIGINAL trajectories (route catalog has %s candidate routes).",
                f"{nonorig_total:,}",
            )
            if nonorig_total:
                total_routes, total_traversals = _process_route_group(
                    pool=pool,
                    routes=iter_nonorig_segments(
                        args.nonorig_4d_segments_dir,
                        route_catalog,
                        chunksize=args.chunk_size,
                    ),
                    estimator=estimator,
                    stats_counter=stats_counter,
                    progress=progress,
                    progress_task=nonorig_task,
                    log_every=args.log_every,
                    total_routes=total_routes,
                    total_traversals=total_traversals,
                )
            else:
                logging.info("Route catalog contains no NON-ORIGINAL candidates; skipping.")

            logging.info(
                "Processing ORIGINAL flights (route catalog has %s entries).", f"{original_total:,}"
            )
            if original_total:
                total_routes, total_traversals = _process_route_group(
                    pool=pool,
                    routes=iter_original_segments(
                        args.master_flights, route_catalog, chunksize=args.chunk_size
                    ),
                    estimator=estimator,
                    stats_counter=stats_counter,
                    progress=progress,
                    progress_task=original_task,
                    log_every=args.log_every,
                    total_routes=total_routes,
                    total_traversals=total_traversals,
                )
            else:
                logging.info("Route catalog contains no ORIGINAL flights; skipping.")

    logging.info(
        "Finished traversal extraction: %s traversals retained (%s censored, %s dropped).",
        estimator.total_records,
        estimator.total_censored,
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

    extractor_stats = dict(stats_counter)
    logging.info("Extractor stats: %s", extractor_stats)


if __name__ == "__main__":
    main()
