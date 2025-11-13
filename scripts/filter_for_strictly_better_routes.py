"""Filter alternate routes whose cost is close to the filed cost.

File formats
------------
Filed cost snapshot (example: ``data/default_cost_230717.csv``):
    task_id,flight_identifier,origin,destination,takeoff_time,landing_time,cost
    1500013765,1500013765,EGYE,EGNT,2023-07-17 11:00:00,2023-07-17 11:54:24,0.539691

Potential routes (example: ``data/potential_cost_230717.csv.gz``):
    flight_identifier,takeoff_time,landing_time,origin_icao,destination_icao,route,utility_cost,probability
    263362880,005948,20948,DAAG,LFML,DAAG -> LESL -> LUMAS_45 -> POMEG -> LFML,0.835094,0.013157894736842105

The script keeps potential routes whose ``utility_cost`` is between
``filed_cost - tolerance`` and ``filed_cost`` (inclusive). The tolerance
comes from ``lower_acceptance_threshold`` which can be interpreted either
as an absolute value or a percentage of the filed cost.

Examples
--------
Compare routes while allowing a 5% tolerance below the filed cost and
write the filtered alternatives to ``data/filtered_routes.csv``::

    /home/hoang/anaconda3/envs/silverdrizzle/bin/python scripts/filter_routes.py \
        --output data/filtered_routes.csv \
        --unit percentage \
        --threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from rich.console import Console
from rich.table import Table

# Column order expected in the potential cost export.
POTENTIAL_COLUMNS: List[str] = [
    "flight_identifier",
    "takeoff_time",
    "landing_time",
    "origin_icao",
    "destination_icao",
    "route",
    "utility_cost",
    "probability",
]


@dataclass
class FilterResult:
    routes: pd.DataFrame
    kept_potential_rows: int
    flights_with_alternatives: int
    flights_with_alternatives_in_potential: int
    flights_copied: int
    total_flights_processed: int
    flights_with_better_cost: int
    sample_10_flight_ids: List[int]
    sample_10_alt_count: int


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter alternative routes whose utility cost is roughly equal "
            "to the filed cost for each flight."
        )
    )
    parser.add_argument(
        "--filed",
        default="data/default_cost_230717.csv",
        help="Path to the filed cost CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--potential",
        default="data/potential_cost_230717.csv.gz",
        help="Path to the potential routes CSV/CSV.GZ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="data/strictly_better_routes.csv",
        help="Destination CSV path for the filtered routes (default: %(default)s).",
    )
    parser.add_argument(
        "--unit",
        choices=("percentage", "absolute"),
        default="percentage",
        help="Interpret threshold as a percentage (0-1) or an absolute delta.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help=(
            "Lower acceptance threshold. If --unit=percentage this value must be "
            "between 0 and 1 (inclusive)."
        ),
    )
    return parser.parse_args(argv)


def validate_threshold(unit: str, threshold: float) -> None:
    if threshold < 0:
        raise ValueError("lower_acceptance_threshold must be non-negative.")
    if unit == "percentage" and threshold > 1:
        raise ValueError("When --unit=percentage, the threshold must be between 0 and 1.")


def load_csv(path: str) -> pd.DataFrame:
    read_kwargs = {"compression": "infer"} if path.endswith(".gz") else {}
    return pd.read_csv(path, **read_kwargs)


def compute_filtered_routes(
    filed_df: pd.DataFrame,
    potential_df: pd.DataFrame,
    unit: str,
    threshold: float,
) -> FilterResult:
    if "flight_identifier" not in filed_df.columns or "cost" not in filed_df.columns:
        raise KeyError("Filed cost CSV must contain 'flight_identifier' and 'cost' columns.")
    missing_cols = [col for col in POTENTIAL_COLUMNS if col not in potential_df.columns]
    if missing_cols:
        raise KeyError(f"Potential routes CSV is missing columns: {missing_cols}")

    filed_columns = [
        "flight_identifier",
        "takeoff_time",
        "landing_time",
        "origin",
        "destination",
        "cost",
    ]
    available_cols = [col for col in filed_columns if col in filed_df.columns]
    filed_details = filed_df[available_cols].copy()
    for missing_col in set(filed_columns) - set(available_cols):
        filed_details[missing_col] = pd.NA

    duplicate_counts = filed_details["flight_identifier"].value_counts()
    duplicate_flights = int((duplicate_counts > 1).sum())
    if duplicate_flights:
        print(
            f"Notice: {duplicate_flights} flights have multiple filed cost entries; "
            "using the minimum cost for each.",
            file=sys.stderr,
        )
        filed_details = filed_details.sort_values("cost", kind="mergesort").drop_duplicates(
            subset="flight_identifier", keep="first"
        )
    else:
        filed_details = filed_details.drop_duplicates(subset="flight_identifier", keep="first")

    filed_costs = filed_details[["flight_identifier", "cost"]]

    merged = potential_df.merge(
        filed_costs.rename(columns={"cost": "filed_cost"}),
        on="flight_identifier",
        how="left",
        validate="many_to_one",
    )

    unmatched = merged["filed_cost"].isna()
    if unmatched.any():
        missing_count = int(unmatched.sum())
        unique_missing = merged.loc[unmatched, "flight_identifier"].nunique()
        print(
            f"Warning: {missing_count} potential routes "
            f"({unique_missing} distinct flights) missing a filed cost and will be skipped.",
            file=sys.stderr,
        )
        merged = merged.loc[~unmatched].copy()

    if merged.empty:
        original_rows = filed_details.rename(
            columns={
                "origin": "origin_icao",
                "destination": "destination_icao",
                "cost": "utility_cost",
            }
        )
        original_rows["route"] = "ORIGINAL"
        original_rows["probability"] = 1.0
        original_rows = original_rows[POTENTIAL_COLUMNS]
        flights_copied = original_rows["flight_identifier"].nunique()
        total_flights = filed_details["flight_identifier"].nunique()
        
        # Calculate real flights with alternatives even if merged is empty
        filed_ids = set(filed_details["flight_identifier"])
        potential_flight_ids = set(potential_df["flight_identifier"])
        flights_with_alternatives_in_potential = len(filed_ids & potential_flight_ids)
        
        return FilterResult(
            routes=original_rows,
            kept_potential_rows=0,
            flights_with_alternatives=0,
            flights_with_alternatives_in_potential=flights_with_alternatives_in_potential,
            flights_copied=flights_copied,
            total_flights_processed=total_flights,
            flights_with_better_cost=0,
            sample_10_flight_ids=[],
            sample_10_alt_count=0,
        )

    tolerance = threshold if unit == "absolute" else merged["filed_cost"] * threshold
    min_allowed = merged["filed_cost"] - tolerance

    mask = (merged["utility_cost"] >= min_allowed) & (merged["utility_cost"] <= merged["filed_cost"])
    filtered_candidates = merged.loc[mask, POTENTIAL_COLUMNS].copy()

    flights_with_alternatives = filtered_candidates["flight_identifier"].nunique()
    filtered_ids = set(filtered_candidates["flight_identifier"])
    filed_ids = set(filed_details["flight_identifier"])
    copied_ids = filed_ids - filtered_ids

    # Calculate real flights with alternatives: flights from filed list that appear in potential list
    potential_flight_ids = set(potential_df["flight_identifier"])
    flights_with_alternatives_in_potential = len(filed_ids & potential_flight_ids)

    # Calculate flights with better cost (utility_cost < filed_cost)
    better_mask = merged["utility_cost"] < merged["filed_cost"]
    flights_with_better = merged.loc[better_mask & mask, "flight_identifier"].nunique()

    # Sample 10 flights and count their alternatives
    all_flight_ids = list(filed_ids)
    sample_10 = all_flight_ids[:10] if len(all_flight_ids) >= 10 else all_flight_ids
    sample_10_alternatives = filtered_candidates[
        filtered_candidates["flight_identifier"].isin(sample_10)
    ]
    sample_10_count = len(sample_10_alternatives)

    if copied_ids:
        original_rows = filed_details[
            filed_details["flight_identifier"].isin(copied_ids)
        ].rename(
            columns={
                "origin": "origin_icao",
                "destination": "destination_icao",
                "cost": "utility_cost",
            }
        )
        original_rows["route"] = "ORIGINAL"
        original_rows["probability"] = 1.0
        original_rows = original_rows[POTENTIAL_COLUMNS]
        combined = pd.concat([filtered_candidates, original_rows], ignore_index=True)
    else:
        combined = filtered_candidates

    total_flights = filed_details["flight_identifier"].nunique()

    return FilterResult(
        routes=combined,
        kept_potential_rows=len(filtered_candidates),
        flights_with_alternatives=flights_with_alternatives,
        flights_with_alternatives_in_potential=flights_with_alternatives_in_potential,
        flights_copied=len(copied_ids),
        total_flights_processed=total_flights,
        flights_with_better_cost=flights_with_better,
        sample_10_flight_ids=sample_10,
        sample_10_alt_count=sample_10_count,
    )


def print_summary_table(result: FilterResult, potential_df_size: int) -> None:
    """Print a rich table summarizing the filtering results."""
    console = Console()
    
    table = Table(title="Flight Route Filtering Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold", width=50)
    table.add_column("Value", justify="right", style="green")
    
    # Add rows with statistics
    table.add_row("Total Flights Processed", f"{result.total_flights_processed:,}")
    table.add_row(
        "Flights with Alternatives (in potential_cost)",
        f"{result.flights_with_alternatives_in_potential:,}"
    )
    table.add_row(
        "Flights with Filtered Routes (passed threshold)",
        f"{result.flights_with_alternatives:,}"
    )
    table.add_row(
        "Total Rows in potential_cost",
        f"{potential_df_size:,}"
    )
    table.add_row(
        "Flights with Roughly Better Cost",
        f"{result.flights_with_better_cost:,}"
    )
    table.add_row(
        "Alternative Routes for First 10 Flights",
        f"{result.sample_10_alt_count:,}"
    )
    table.add_row("", "")  # Separator
    table.add_row("Total Routes in Output", f"{len(result.routes):,}", style="bold yellow")
    table.add_row(
        "Potential Routes Kept",
        f"{result.kept_potential_rows:,} / {potential_df_size:,}"
    )
    table.add_row("Flights Copied as ORIGINAL", f"{result.flights_copied:,}")
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Print the 10 sample flight IDs
    if result.sample_10_flight_ids:
        console.print(
            f"[bold]Sample 10 Flight IDs:[/bold] {', '.join(map(str, result.sample_10_flight_ids))}"
        )
        console.print("\n")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    validate_threshold(args.unit, args.threshold)

    filed_df = load_csv(args.filed)
    potential_df = load_csv(args.potential)

    print(f"Loaded {len(filed_df)} filed cost rows and {len(potential_df)} potential cost rows.")
    

    result = compute_filtered_routes(
        filed_df=filed_df,
        potential_df=potential_df,
        unit=args.unit,
        threshold=args.threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.routes.to_csv(output_path, index=False)

    print(
        f"Wrote {len(result.routes)} routes "
        f"(kept {result.kept_potential_rows}/{len(potential_df)} potential rows) "
        f"to {output_path}.",
        file=sys.stderr,
    )
    print(
        f"Flights with better alternatives: {result.flights_with_alternatives}. "
        f"Flights copied as ORIGINAL: {result.flights_copied}.",
        file=sys.stderr,
    )
    
    # Print the summary table
    print_summary_table(result, len(potential_df))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
