"""CLI helper that builds arrival moments from FR artifacts."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from gemini.arrivals.flight_list_gemini import FlightListGemini
from gemini.arrivals.ground_jitter_config import GroundJitterConfig
from gemini.gem.arrival_moments import ArrivalMoments
from gemini.propagation.tvtw_indexer import TVTWIndexer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gemini.arr.flight_metadata_provider import FlightMetadataProvider
from gemini.arr.fr_arrival_moments_builder import build_arrival_moments_from_fr
from gemini.arr.fr_artifacts_loader import FRSegment, load_fr_segments

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
    segments = load_fr_segments(args.fr_demand, args.fr_route_catalogue)
    if not segments:
        raise SystemExit("No FR segments available; cannot build arrival moments.")
    volume_ids = _collect_volume_ids(segments)
    tvtw = TVTWIndexer.load(args.tvtw_indexer)
    flights = FlightListGemini(args.flights_csv)
    jitter_config = GroundJitterConfig.from_json(args.ground_jitter_config)
    metadata_provider = FlightMetadataProvider(flights, jitter_config)
    moments = _build_arrival_moments_with_progress(
        segments,
        metadata_provider,
        tvtw,
        volume_ids=volume_ids,
        tail_tolerance=args.tail_tolerance,
    )
    _write_arrival_moments_csv(args.output_csv, moments)
    logger.info("Wrote arrival CSV with %d entries to %s", len(moments.lambda_ext), args.output_csv)


def _collect_volume_ids(segments: Iterable[FRSegment]) -> List[str]:
    seen: Dict[str, None] = {}
    for segment in segments:
        if segment.volume_id not in seen:
            seen[segment.volume_id] = None
    return list(seen.keys())


def _write_arrival_moments_csv(path: str | Path, moments: ArrivalMoments) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["volume_id", "time_bin", "lambda_mean", "lambda_var", "gamma_lag1"]
    rows = _moment_rows(moments)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _moment_rows(moments: ArrivalMoments) -> List[Dict[str, float | str | int]]:
    keys = set(moments.lambda_ext.keys())
    keys.update(moments.nu_ext.keys())
    keys.update(moments.gamma_ext.keys())
    sorted_keys = sorted(keys, key=lambda item: (item[0], item[1]))
    rows: List[Dict[str, float | str | int]] = []
    for volume_id, time_bin in sorted_keys:
        rows.append(
            {
                "volume_id": volume_id,
                "time_bin": int(time_bin),
                "lambda_mean": moments.lambda_ext.get((volume_id, time_bin), 0.0),
                "lambda_var": moments.nu_ext.get((volume_id, time_bin), 0.0),
                "gamma_lag1": moments.gamma_ext.get((volume_id, time_bin), 0.0),
            }
        )
    return rows


def _build_arrival_moments_with_progress(
    segments: List[FRSegment],
    metadata_provider: FlightMetadataProvider,
    tvtw: TVTWIndexer,
    *,
    volume_ids: Sequence[str] | None,
    tail_tolerance: float,
) -> ArrivalMoments:
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

        def iter_with_progress() -> Iterable[FRSegment]:
            for segment in segments:
                yield segment
                progress.advance(task_id)

        return build_arrival_moments_from_fr(
            iter_with_progress(),
            metadata_provider,
            tvtw,
            volume_ids=volume_ids,
            tail_tolerance=tail_tolerance,
        )


if __name__ == "__main__":
    main()
