"""CLI helper that builds arrival moments from FR artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

from gemini.arr.fr_arrivals_service import FRArrivalArtifacts, FRArrivalMomentsService
from gemini.arrivals.ground_hold_config import GroundHoldConfig
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fr-demand",
    required=False,
    default="data/fr/gem_artifacts_demand_all",
    help="Path to gem_artifacts_demand_all CSV.")
    parser.add_argument(
        "--fr-route-catalogue",
        required=False,
        default="data/fr/gem_artifacts_route_catalogue_all",
        help="Path to gem_artifacts_route_catalogue_all CSV.",
    )
    parser.add_argument(
        "--flights-csv",
        required=False,
        default="/mnt/d/project-tailwind/output/flights_20230717_0000-2359.csv",
        help="Flight metadata CSV consumed by FlightListGemini.",
    )
    parser.add_argument(
        "--ground-jitter-config",
        required=False,
        default="data/fr/ground_jitter_default.json",
        help="Ground jitter configuration JSON.",
    )
    parser.add_argument(
        "--ground-hold-config",
        required=False,
        default=None,
        help=(
            "Optional YAML file describing deterministic ground-hold windows. "
            "Flights with takeoff times inside a window's [start, end) range join its FCFS queue, "
            "and the provided rate_fph defines the post-regulation release rate in flights/hour."
        ),
    )
    parser.add_argument(
        "--tvtw-indexer",
        required=False,
        default="/mnt/d/project-tailwind/output/tvtw_indexer.json",
        help="TVTW indexer JSON describing the binning horizon.",
    )
    parser.add_argument("--output-csv",
    required=False,
    default="data/fr/arrival_moments.csv",
    help="Destination CSV for arrival moments.")
    parser.add_argument(
        "--tail-tolerance",
        type=float,
        default=1e-6,
        help="Tail probability cutoff when integrating jitter distributions.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    artifacts = FRArrivalArtifacts(
        fr_demand_path=args.fr_demand,
        fr_route_catalogue_path=args.fr_route_catalogue,
        flights_csv_path=args.flights_csv,
        tvtw_indexer_path=args.tvtw_indexer,
    )
    service = FRArrivalMomentsService(artifacts)
    try:
        segments = service.load_segments()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    jitter_config = GroundJitterConfig.from_json(args.ground_jitter_config)
    hold_config = None
    if args.ground_hold_config:
        hold_config = GroundHoldConfig.from_yaml(args.ground_hold_config)
    dataframe = _build_arrival_moments_with_progress(
        service,
        segments,
        jitter_config,
        ground_hold_config=hold_config,
        tail_tolerance=args.tail_tolerance,
    )
    _write_arrival_moments_csv(args.output_csv, dataframe)
    logger.info("Wrote arrival CSV with %d entries to %s", len(dataframe), args.output_csv)


def _write_arrival_moments_csv(path: str | Path, dataframe) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def _build_arrival_moments_with_progress(
    service: FRArrivalMomentsService,
    segments: Sequence,
    jitter_config: GroundJitterConfig,
    *,
    ground_hold_config: GroundHoldConfig | None,
    tail_tolerance: float,
):
    """Build arrival moments while displaying a progress bar for the FR segments."""

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )

    with progress:
        task_id = progress.add_task("Processing FR segments", total=len(segments))
        volume_ids = service.get_volume_ids()

        def iter_with_progress() -> Iterable:
            for segment in segments:
                yield segment
                progress.advance(task_id)

        return service.get_arrival_moments(
            jitter_config,
            ground_hold_config=ground_hold_config,
            tail_tolerance=tail_tolerance,
            volume_ids=volume_ids,
            segments=iter_with_progress(),
        )


if __name__ == "__main__":
    main()
