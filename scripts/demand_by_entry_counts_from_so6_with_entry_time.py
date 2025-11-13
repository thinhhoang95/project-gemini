"""
Note that this script works with entry counts, not occupancy anymore.
"""

import argparse
import datetime as dt
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic
from shapely.geometry import LineString, Point

from sdrizzle.occupancy.tvtw_indexer import TVTWIndexer

warnings.filterwarnings("ignore")


REQUIRED_ROUTE_COLUMNS = [
    "time_begin_segment",
    "time_end_segment",
    "latitude_begin",
    "longitude_begin",
    "latitude_end",
    "longitude_end",
    "flight_level_begin",
    "flight_level_end",
    "route",
]


def validate_required_files(file_requirements: List[Tuple[str, str]]) -> None:
    """Raise an error if any required file path is missing on disk."""
    missing = [
        f"{description} ({path})"
        for path, description in file_requirements
        if not path or not os.path.exists(path)
    ]
    if missing:
        formatted = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "The following required inputs are missing:" + formatted
        )


def load_traffic_volumes(tv_path: str) -> gpd.GeoDataFrame:
    """
    Load traffic volumes from GeoJSON and ensure spatial index exists.
    """
    tv_gdf = gpd.read_file(tv_path)
    if "traffic_volume_id" not in tv_gdf.columns:
        raise ValueError(
            "GeoJSON file must have a 'traffic_volume_id' property in each feature."
        )
    if "min_fl" not in tv_gdf.columns or "max_fl" not in tv_gdf.columns:
        raise ValueError(
            "GeoJSON file must have 'min_fl' and 'max_fl' properties in each feature."
        )
    # Trigger spatial index construction once so repeated queries stay fast.
    tv_gdf.sindex
    return tv_gdf


def find_tvs_for_point(point: Point, fl: float, tv_gdf: gpd.GeoDataFrame) -> List[str]:
    """
    Find all traffic volume IDs that contain a given point at a specific flight level.
    """
    if point.is_empty:
        return []

    possible_matches_idx = list(tv_gdf.sindex.query(point, predicate="intersects"))
    if not possible_matches_idx:
        return []

    possible_matches = tv_gdf.iloc[possible_matches_idx]
    precise_matches = possible_matches[possible_matches.geometry.covers(point)]
    vertically_relevant_tvs = precise_matches[
        (precise_matches["min_fl"] * 100 <= fl)
        & (precise_matches["max_fl"] * 100 >= fl)
    ]
    return vertically_relevant_tvs["traffic_volume_id"].tolist()


def parse_hhmmss(value: Any, reference_date: dt.date = dt.date(1970, 1, 1)) -> dt.datetime:
    """
    Parse an HHMMSS-style value (possibly shorter) into a datetime on a reference date.
    """
    if value is None:
        raise ValueError("Cannot parse None as HHMMSS.")

    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("Cannot parse NaN as HHMMSS.")
        value = int(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Empty string is not a valid HHMMSS value.")
    if "." in text:
        text = text.split(".", 1)[0]

    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Value '{value}' does not contain digits for HHMMSS parsing.")

    if len(digits) > 6:
        digits = digits[-6:]

    digits = digits.zfill(6)

    hour = int(digits[0:2])
    minute = int(digits[2:4])
    second = int(digits[4:6])

    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError(f"HHMMSS value '{digits}' is outside valid time bounds.")

    return dt.datetime.combine(reference_date, dt.time(hour, minute, second))


def canonicalize_route(route: Any) -> Tuple[str, ...]:
    """
    Produce a canonical tuple of waypoints from TXT or CSV representations.
    """

    def _looks_like_cost(text: str) -> bool:
        """Return True when the fragment appears to be a numeric cost."""
        cleaned = str(text).strip()
        if not cleaned:
            return False
        try:
            float(cleaned)
        except (TypeError, ValueError):
            return False
        return True

    def _strip_cost_suffix(text: str) -> str:
        """Drop a trailing ', cost' suffix emitted by trajectory sampling."""
        head, sep, tail = text.rpartition(",")
        if sep and _looks_like_cost(tail):
            return head
        return text

    if route is None:
        return tuple()

    if isinstance(route, (list, tuple)):
        tokens_iter = route
    else:
        text = str(route).strip()
        if not text:
            return tuple()
        text = _strip_cost_suffix(text)
        if "->" in text:
            tokens_iter = text.split("->")
        else:
            tokens_iter = text.split()

    tokens: List[str] = []
    for token in tokens_iter:
        token_str = str(token).strip()
        if token_str.endswith(","):
            token_str = token_str.rstrip(",").strip()
        if token_str:
            tokens.append(token_str)

    while len(tokens) > 1 and _looks_like_cost(tokens[-1]):
        tokens.pop()

    return tuple(tokens)


def load_route_probabilities(
    samples_path: str, available_routes: Iterable[Tuple[str, ...]]
) -> Dict[Tuple[str, ...], float]:
    """
    Count route samples and normalize into probabilities over routes seen in the CSV.
    """
    available_set = set(available_routes)
    counts: Counter = Counter()

    with open(samples_path, "r") as handle:
        for line in handle:
            route_key = canonicalize_route(line)
            if route_key:
                counts[route_key] += 1

    if not counts:
        raise ValueError(f"No valid route samples were found in {samples_path}.")

    missing = set(counts.keys()) - available_set
    if missing:
        warnings.warn(
            f"Ignoring {len(missing)} route(s) present in samples but missing from the CSV.",
            RuntimeWarning,
        )

    matched_counts = {route: counts[route] for route in available_set if route in counts}
    total = sum(matched_counts.values())
    if total == 0:
        raise ValueError(
            "No overlapping routes between the TXT samples and the CSV trajectories."
        )

    return {route: count / total for route, count in matched_counts.items()}


def compute_route_entry_counts(
    route_df: pd.DataFrame,
    tv_gdf: gpd.GeoDataFrame,
    tvtw_indexer: TVTWIndexer,
    sampling_dist_nm: float = 5.0,
) -> Dict[int, int]:
    """
    Compute a sparse entry-count dict for a single canonical route.

    Each key is a TVTW index whose value represents how many times the route
    newly enters the corresponding traffic volume within a given time bin.
    """
    if route_df.empty:
        return {}

    route_df = route_df.sort_values(
        by=["time_begin_segment", "time_end_segment"], kind="mergesort"
    )
    points: List[Tuple[dt.datetime, Point, float]] = []

    for row in route_df.itertuples(index=False):
        try:
            start_dt = parse_hhmmss(getattr(row, "time_begin_segment"))
            end_dt = parse_hhmmss(getattr(row, "time_end_segment"))
        except ValueError:
            continue

        if end_dt <= start_dt:
            continue

        try:
            start_point = Point(
                float(getattr(row, "longitude_begin")),
                float(getattr(row, "latitude_begin")),
            )
            end_point = Point(
                float(getattr(row, "longitude_end")),
                float(getattr(row, "latitude_end")),
            )
            start_alt_ft = float(getattr(row, "flight_level_begin")) * 100.0
            end_alt_ft = float(getattr(row, "flight_level_end")) * 100.0
        except (TypeError, ValueError):
            continue

        if start_point.is_empty or end_point.is_empty:
            continue

        points.append((start_dt, start_point, start_alt_ft))

        distance_nm = geodesic(
            (start_point.y, start_point.x), (end_point.y, end_point.x)
        ).nm
        num_samples = int(distance_nm / sampling_dist_nm) if sampling_dist_nm > 0 else 0

        if num_samples > 0:
            line = LineString([start_point, end_point])
            if not line.is_empty:
                for idx in range(1, num_samples + 1):
                    fraction = idx / (num_samples + 1)
                    sample_point = line.interpolate(fraction, normalized=True)
                    sample_time = start_dt + fraction * (end_dt - start_dt)
                    sample_alt = start_alt_ft + fraction * (end_alt_ft - start_alt_ft)
                    points.append((sample_time, sample_point, sample_alt))

        points.append((end_dt, end_point, end_alt_ft))

    if not points:
        return {}

    points.sort(key=lambda sample: sample[0])
    known_tvs = set(tvtw_indexer.tv_id_to_idx.keys())

    entry_counts: Dict[int, int] = defaultdict(int)
    active_tvs: Set[str] = set()

    for current_time, geom, alt_ft in points:
        tvs = {
            tv
            for tv in find_tvs_for_point(geom, alt_ft, tv_gdf)
            if tv in known_tvs
        }

        if active_tvs:
            exited = {tv for tv in active_tvs if tv not in tvs}
            if exited:
                active_tvs.difference_update(exited)

        newly_entered = tvs - active_tvs
        if newly_entered:
            bin_idx = tvtw_indexer.bin_of_datetime(current_time)
            for tv in newly_entered:
                tvtw_index = tvtw_indexer.get_tvtw_index(tv, bin_idx)
                if tvtw_index is not None:
                    entry_counts[tvtw_index] += 1
            active_tvs.update(newly_entered)

    return {idx: count for idx, count in entry_counts.items() if count > 0}


def load_allowed_routes(path: str) -> Set[Tuple[str, ...]]:
    """
    Load a set of canonical routes that should be kept in the computation.

    The file can contain either:
        * JSON (array of routes, or object with a 'routes' array)
        * Plain text with one route per line
    """
    if not path:
        return set()

    with open(path, "r") as handle:
        raw = handle.read()

    if not raw.strip():
        return set()

    routes_iter: Iterable[str]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        routes_iter = (line.strip() for line in raw.splitlines() if line.strip())
    else:
        if isinstance(parsed, dict) and "routes" in parsed:
            parsed = parsed["routes"]
        if isinstance(parsed, (list, tuple, set)):
            routes_iter = (str(item).strip() for item in parsed if str(item).strip())
        else:
            raise ValueError(
                f"Unsupported JSON structure in allowed routes file {path!r}. "
                "Expected an array or an object with a 'routes' key."
            )

    allowed: Set[Tuple[str, ...]] = set()
    skipped = 0
    for route_text in routes_iter:
        canonical = canonicalize_route(route_text)
        if canonical:
            allowed.add(canonical)
        else:
            skipped += 1

    if skipped:
        warnings.warn(
            f"Skipped {skipped} empty/invalid route(s) while loading {path}.",
            RuntimeWarning,
        )

    return allowed


def load_or_create_indexer(
    indexer_path: str, tv_path: str, time_bin_minutes: int
) -> TVTWIndexer:
    """
    Load an existing TVTW indexer or build and persist a new one.
    """
    if os.path.exists(indexer_path):
        indexer = TVTWIndexer.load(indexer_path)
        if indexer.time_bin_minutes != int(time_bin_minutes):
            raise ValueError(
                "Existing TVTW indexer uses a different bin size than requested."
            )
        return indexer

    indexer = TVTWIndexer(time_bin_minutes=time_bin_minutes)
    indexer.build_from_tv_geojson(tv_path)
    out_dir = os.path.dirname(indexer_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    indexer.save(indexer_path)
    print(f"TVTW indexer saved to {indexer_path}")
    return indexer


def compute_flight_entry_moments(
    route_entry_counts: Dict[Tuple[str, ...], Dict[int, int]],
    route_probabilities: Dict[Tuple[str, ...], float],
    tvtw_indexer: TVTWIndexer,
) -> pd.DataFrame:
    """
    Combine per-route entry counts into per-flight first and second moments.

    The result contains only non-zero cells with:
        - `time_bin`: integer index returned by `TVTWIndexer.bin_of_datetime`
        - `mu_i`: expected entry count for the flight in the TV/time bin
        - `var_i`: entry-count variance for the flight in the TV/time bin
    """
    mu: Dict[int, float] = defaultdict(float)
    second_moment: Dict[int, float] = defaultdict(float)

    for route_key, probability in route_probabilities.items():
        entry_counts = route_entry_counts.get(route_key, {})
        if not entry_counts or probability <= 0:
            continue
        for tvtw_index, count in entry_counts.items():
            if count <= 0:
                continue
            mu[tvtw_index] += probability * count
            second_moment[tvtw_index] += probability * (count ** 2)

    records: List[Dict[str, Any]] = []
    relevant_indices = set(mu.keys()) | set(second_moment.keys())

    for tvtw_index in sorted(relevant_indices):
        mean = mu.get(tvtw_index, 0.0)
        variance = second_moment.get(tvtw_index, 0.0) - (mean ** 2)
        if variance < 0 and abs(variance) < 1e-9:
            variance = 0.0
        if mean == 0.0 and variance == 0.0:
            continue
        tvtw = tvtw_indexer.get_tvtw_from_index(tvtw_index)
        if tvtw is None:
            continue
        volume_id, time_bin = tvtw
        records.append(
            {
                "volume_id": volume_id,
                "time_bin": int(time_bin),
                "mu_i": float(mean),
                "var_i": float(max(variance, 0.0)),
            }
        )

    if not records:
        return pd.DataFrame(columns=["volume_id", "time_bin", "mu_i", "var_i"])

    return pd.DataFrame.from_records(
        records, columns=["volume_id", "time_bin", "mu_i", "var_i"]
    )


def run_per_flight(args: argparse.Namespace) -> None:
    """
    Execute the per-flight entry-count pipeline and write Parquet output.
    """
    validate_required_files(
        [
            (args.routes_csv, "Routes CSV"),
            (args.route_samples_txt, "Route samples TXT"),
            (args.tv_path, "Traffic volumes GeoJSON"),
        ]
    )

    tv_gdf = load_traffic_volumes(args.tv_path)
    tvtw_indexer = load_or_create_indexer(
        args.tvtw_indexer_path, args.tv_path, args.time_bin_minutes
    )

    allowed_routes: Optional[Set[Tuple[str, ...]]] = None
    if getattr(args, "allowed_routes_path", None):
        allowed_routes = load_allowed_routes(args.allowed_routes_path)
        if not allowed_routes:
            raise ValueError(
                f"Allowed routes file '{args.allowed_routes_path}' did not "
                "produce any canonical routes."
            )

    routes_df = pd.read_csv(args.routes_csv)
    for column in REQUIRED_ROUTE_COLUMNS:
        if column not in routes_df.columns:
            raise ValueError(f"Routes CSV missing required column '{column}'.")

    routes_df = routes_df.dropna(subset=REQUIRED_ROUTE_COLUMNS).copy()
    routes_df["canonical_route"] = routes_df["route"].apply(canonicalize_route)
    routes_df = routes_df[routes_df["canonical_route"].map(bool)]

    if routes_df.empty:
        raise ValueError("Routes CSV does not contain any usable segments.")

    if allowed_routes is not None:
        before_unique = routes_df["canonical_route"].nunique()
        routes_df = routes_df[routes_df["canonical_route"].isin(allowed_routes)].copy()
        if routes_df.empty:
            raise ValueError(
                "After applying the allowed routes filter, no trajectory segments remain."
            )
        after_unique = routes_df["canonical_route"].nunique()
        if after_unique < before_unique:
            missing = allowed_routes - set(routes_df["canonical_route"])
            if missing:
                warnings.warn(
                    f"{len(missing)} allowed route(s) were not present in the routes CSV.",
                    RuntimeWarning,
                )


    route_entry_counts: Dict[Tuple[str, ...], Dict[int, int]] = {}
    for canonical_route, group in routes_df.groupby("canonical_route"):
        route_entry_counts[canonical_route] = compute_route_entry_counts(
            group, tv_gdf, tvtw_indexer, sampling_dist_nm=args.sampling_dist_nm
        )

    print(f"Computed sparse entry counts for {len(route_entry_counts)} canonical route(s).")

    route_probabilities = load_route_probabilities(
        args.route_samples_txt, route_entry_counts.keys()
    )
    print(
        f"Matched {len(route_probabilities)} route(s) between samples and trajectories."
    )

    flight_df = compute_flight_entry_moments(route_entry_counts, route_probabilities, tvtw_indexer)

    if args.output_parquet:
        output_path = args.output_parquet
    else:
        base = os.path.splitext(os.path.basename(args.routes_csv))[0]
        output_path = os.path.join(os.path.dirname(args.routes_csv), f"{base}.parquet")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    flight_df.to_parquet(output_path, engine="pyarrow", index=False)
    print(flight_df.head())
    print(f"Wrote {len(flight_df)} entry cell(s) to {output_path}.")


def build_parser() -> argparse.ArgumentParser:
    """
    Configure the CLI parser for the per-flight entry-count workflow.
    """
    parser = argparse.ArgumentParser(
        description="Compute per-flight entry-count moments and persist them as Parquet."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    per_flight_parser = subparsers.add_parser(
        "per-flight", help="Generate per-flight entry-count moments (Parquet)."
    )
    per_flight_parser.add_argument("--routes_csv", required=True, help="Path to the CSV with trajectory segments.")
    per_flight_parser.add_argument("--route_samples_txt", required=True, help="Path to the TXT file containing route samples.")
    per_flight_parser.add_argument("--tv_path", required=False, default="/mnt/d/project-tailwind/output/wxm_sm_ih_maxpool.geojson", help="Path to the traffic volumes GeoJSON.")
    per_flight_parser.add_argument("--tvtw_indexer_path", required=False, default="/mnt/d/project-tailwind/output/tvtw_indexer.json", help="Path to load/save the TVTW indexer JSON.")
    per_flight_parser.add_argument(
        "--output_parquet",
        required=False,
        help="Optional output path for the Parquet file (defaults to <routes_csv_basename>.parquet).",
    )
    per_flight_parser.add_argument(
        "--allowed_routes_path",
        required=False,
        help="Optional path to a JSON/line-delimited file listing canonical routes to include.",
    )
    per_flight_parser.add_argument(
        "--time_bin_minutes",
        type=int,
        default=15,
        help="Time bin duration in minutes (default: 15).",
    )
    per_flight_parser.add_argument(
        "--sampling_dist_nm",
        type=float,
        default=5.0,
        help="Great-circle sampling distance in nautical miles (default: 5.0).",
    )
    per_flight_parser.set_defaults(func=run_per_flight)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
