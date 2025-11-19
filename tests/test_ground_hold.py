from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import textwrap

from gemini.arr.fr_arrivals_cli import main as fr_arrivals_main
from gemini.arr.flight_metadata_provider import FlightMetadataProvider
from gemini.arrivals.ground_hold_config import GroundHoldConfig, GroundHoldWindow
from gemini.arrivals.ground_hold_operator import GroundHoldOperator


class DummyFlightList:
    def __init__(self, flights: Dict[str, Dict[str, object]]):
        self._flights = {str(fid): meta for fid, meta in flights.items()}
        self.flight_ids = list(self._flights.keys())

    def get_flight_metadata(self, flight_id: str) -> Dict[str, object]:
        return self._flights.get(str(flight_id), {})


class DummyJitterConfig:
    def __init__(self, distribution: object):
        self._distribution = distribution
        self.calls: List[Tuple[str | None, int]] = []

    def get_distribution(self, origin: str | None, minute_of_day: int):
        self.calls.append((origin, minute_of_day))
        return self._distribution


def test_ground_hold_config_roundtrip(tmp_path):
    yaml_text = textwrap.dedent(
        """
        version: test
        default_rate_fph: 12
        airports:
          lfpg:
            - start: '2023-07-01T09:00:00'
              end: '2023-07-01T10:00:00'
              rate_fph: 6
              regulation_id: REG-1
          eham:
            - start: '2023-07-01T08:00:00'
              end: '2023-07-01T09:00:00'
              regulation_id: REG-2
        """
    ).strip()
    config_path = tmp_path / "hold.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    config = GroundHoldConfig.from_yaml(config_path)
    lfpg_windows = config.windows_for_airport("lfpg")
    assert len(lfpg_windows) == 1
    assert lfpg_windows[0].rate_fph == 6
    eham_windows = config.windows_for_airport("EHAM")
    assert len(eham_windows) == 1
    # Default rate applied to entries that omit rate_fph
    assert eham_windows[0].rate_fph == 12

    roundtrip_path = tmp_path / "roundtrip.yaml"
    config.to_yaml(roundtrip_path)
    roundtrip = GroundHoldConfig.from_yaml(roundtrip_path)
    assert roundtrip.windows_for_airport("LFPG")[0].regulation_id == "REG-1"


def test_ground_hold_config_rejects_missing_rate(tmp_path):
    yaml_text = textwrap.dedent(
        """
        airports:
          lfpg:
            - start: '2023-07-01T09:00:00'
              end: '2023-07-01T09:30:00'
        """
    ).strip()
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        GroundHoldConfig.from_yaml(config_path)


def test_ground_hold_operator_computes_fcfs_delays():
    base = datetime(2023, 7, 1, 9, 0, 0)
    flights = {
        "F1": {"takeoff_time": base + timedelta(minutes=5), "origin": "LFPG"},
        "F2": {"takeoff_time": base + timedelta(minutes=10), "origin": "LFPG"},
        "F3": {"takeoff_time": base + timedelta(minutes=15), "origin": "LFPG"},
        "IGNORED": {"takeoff_time": base + timedelta(minutes=70), "origin": "LFPG"},
    }
    window = GroundHoldWindow(
        start=base,
        end=base + timedelta(minutes=30),
        rate_fph=2.0,
        airport="LFPG",
        regulation_id="REG-A",
    )
    config = GroundHoldConfig(windows_by_airport={"LFPG": [window]})
    operator = GroundHoldOperator(DummyFlightList(flights), config)
    assignment = operator.compute_flight_delays()
    assert assignment.get("F1") == 25
    assert assignment.get("F2") == 50
    assert assignment.get("F3") == 75
    assert assignment.get("IGNORED") == 0


def test_flight_metadata_provider_applies_ground_hold_delays():
    takeoff = datetime(2023, 7, 1, 10, 0, 0)
    flights = {"TEST": {"takeoff_time": takeoff, "origin": "LFPO"}}
    flight_list = DummyFlightList(flights)
    jitter_config = DummyJitterConfig(distribution=object())
    provider = FlightMetadataProvider(
        flight_list,
        jitter_config,
        ground_hold_delays={"TEST": 30},
    )
    context = provider.get_entry_context("TEST")
    assert context.ground_hold_delay_min == 30
    assert context.effective_takeoff_time == takeoff + timedelta(minutes=30)
    assert context.minute_of_day == pytest.approx(630.0)


def test_fr_arrivals_cli_shifts_bins_when_ground_hold_applied(tmp_path):
    flights_csv = tmp_path / "flights.csv"
    demand_csv = tmp_path / "demand.csv"
    route_csv = tmp_path / "routes.csv"
    jitter_json = tmp_path / "jitter.json"
    tvtw_json = tmp_path / "tvtw.json"
    output_no_hold = tmp_path / "arrival_no_hold.csv"
    output_hold = tmp_path / "arrival_with_hold.csv"
    hold_yaml = tmp_path / "hold.yaml"

    _write_flights_csv(flights_csv)
    _write_demand_csv(demand_csv)
    _write_route_csv(route_csv)
    jitter_json.write_text(
        json.dumps(
            {
                "default": {
                    "00:00-24:00": {
                        "p_hurdle": 0.0,
                        "mean": 1.0,
                        "std": 1.0,
                        "threshold": 5.0,
                        "shift": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tvtw_json.write_text(
        json.dumps({"time_bin_minutes": 60, "tv_id_to_idx": {"VOL1": 0}}),
        encoding="utf-8",
    )
    hold_yaml.write_text(
        "airports:\n  LFPG:\n    - start: '2023-07-01T09:00:00'\n"
        "      end: '2023-07-01T11:00:00'\n"
        "      rate_fph: 2.0\n",
        encoding="utf-8",
    )

    base_args = [
        "--fr-demand",
        str(demand_csv),
        "--fr-route-catalogue",
        str(route_csv),
        "--flights-csv",
        str(flights_csv),
        "--ground-jitter-config",
        str(jitter_json),
        "--tvtw-indexer",
        str(tvtw_json),
        "--log-level",
        "ERROR",
    ]

    fr_arrivals_main(base_args + ["--output-csv", str(output_no_hold)])
    fr_arrivals_main(
        base_args
        + [
            "--ground-hold-config",
            str(hold_yaml),
            "--output-csv",
            str(output_hold),
        ]
    )

    no_hold_bins = _load_lambda_by_bin(output_no_hold)
    hold_bins = _load_lambda_by_bin(output_hold)
    assert ("VOL1", 10) in no_hold_bins
    assert ("VOL1", 10) not in hold_bins
    assert ("VOL1", 11) in hold_bins


def _write_flights_csv(path: Path) -> None:
    fieldnames = [
        "flight_identifier",
        "date_begin_segment",
        "time_begin_segment",
        "origin_aerodrome",
        "date_end_segment",
        "time_end_segment",
        "destination_aerodrome",
        "flight_level_begin",
        "flight_level_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "flight_identifier": "FLT1",
                "date_begin_segment": "230701",
                "time_begin_segment": "100000",
                "origin_aerodrome": "LFPG",
                "date_end_segment": "230701",
                "time_end_segment": "120000",
                "destination_aerodrome": "KJFK",
                "flight_level_begin": "300",
                "flight_level_end": "320",
            }
        )


def _write_demand_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "flight_id",
                "route_index",
                "traffic_volume_name",
                "entry_time_from_takeoff_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "flight_id": "FLT1",
                "route_index": 0,
                "traffic_volume_name": "VOL1",
                "entry_time_from_takeoff_s": 600,
            }
        )


def _write_route_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["flight_id", "route_index", "probability"])
        writer.writeheader()
        writer.writerow({"flight_id": "FLT1", "route_index": 0, "probability": 1.0})


def _load_lambda_by_bin(path: Path) -> Dict[tuple[str, int], float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[tuple[str, int], float] = {}
        for row in reader:
            lam = float(row["lambda_mean"])
            if lam <= 0.0:
                continue
            key = (row["volume_id"], int(row["time_bin"]))
            mapping[key] = lam
        return mapping
